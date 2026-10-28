from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from data.domain import filter_skus, require_columns


class StageArtifactStore:
    """The only reader/writer for standardized inter-stage artifacts.

    Algorithms receive DataFrames and never open CSV, JSON, Parquet, or raw source
    files themselves. Writes use a temporary sibling followed by atomic replace.
    """

    @staticmethod
    def read_frame(
        path: Path,
        *,
        required: Iterable[str] = (),
        sku_codes: Sequence[str] | None = None,
        sku_column: str = "sku_code",
        limit_skus: int | None = None,
        columns: Sequence[str] | None = None,
        optional_columns: Sequence[str] = (),
        filters: Mapping[str, Any | Sequence[Any]] | None = None,
        chunksize: int | None = None,
    ) -> pd.DataFrame:
        suffix = path.suffix.lower()
        read_columns = list(dict.fromkeys(columns or ())) or None
        if read_columns is not None and optional_columns:
            if suffix == ".csv":
                available_columns = set(pd.read_csv(path, nrows=0).columns)
            elif suffix in {".parquet", ".pq"}:
                try:
                    import pyarrow.parquet as parquet

                    available_columns = set(parquet.ParquetFile(path).schema.names)
                except ImportError:
                    available_columns = set(pd.read_parquet(path).columns)
            else:
                available_columns = set()
            for column in optional_columns:
                if column in available_columns and column not in read_columns:
                    read_columns.append(column)
        if read_columns is not None:
            for column in (*required, *(filters or {}).keys()):
                if column not in read_columns:
                    read_columns.append(column)
        identifiers = {
            key: str
            for key in (
                sku_column, "sku_code", "target_sku_code",
                "candidate_sku_code", "store_code", "store_tier",
            )
            if read_columns is None or key in read_columns
        }

        def apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
            if not filters:
                return frame
            keep = pd.Series(True, index=frame.index)
            for column, values in filters.items():
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    keep &= frame[column].isin(values)
                else:
                    keep &= frame[column].eq(values)
            return frame.loc[keep]

        if suffix == ".csv":
            if chunksize:
                pieces: list[pd.DataFrame] = []
                for chunk in pd.read_csv(
                    path, dtype=identifiers, usecols=read_columns,
                    low_memory=False, chunksize=chunksize,
                ):
                    filtered = apply_filters(chunk)
                    if not filtered.empty:
                        pieces.append(filtered)
                frame = (
                    pd.concat(pieces, ignore_index=True)
                    if pieces
                    else pd.DataFrame(columns=read_columns)
                )
            else:
                frame = apply_filters(
                    pd.read_csv(
                        path, dtype=identifiers, usecols=read_columns,
                        low_memory=False,
                    )
                )
        elif suffix in {".parquet", ".pq"}:
            parquet_filters = None
            if filters:
                parquet_filters = []
                for column, values in filters.items():
                    if isinstance(values, Sequence) and not isinstance(
                        values, (str, bytes)
                    ):
                        parquet_filters.append((column, "in", list(values)))
                    else:
                        parquet_filters.append((column, "==", values))
            frame = apply_filters(
                pd.read_parquet(
                    path,
                    columns=read_columns,
                    filters=parquet_filters,
                )
            )
        else:
            raise ValueError(f"unsupported standardized artifact format: {path}")
        require_columns(frame, required, path.name)
        if sku_codes or limit_skus is not None:
            frame = filter_skus(
                frame, sku_codes, column=sku_column, limit_skus=limit_skus
            )
        return frame

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"JSON artifact must contain an object: {path}")
        return payload

    @staticmethod
    def iter_frame_parts(
        path: Path,
        *,
        required: Iterable[str] = (),
        columns: Sequence[str] | None = None,
        sku_codes: Sequence[str] | None = None,
        sku_column: str = "sku_code",
        csv_chunksize: int = 250_000,
        require_all_skus: bool = True,
    ) -> Iterable[pd.DataFrame]:
        """Yield projected standardized artifact parts without loading all rows.

        Round 4 writes a partitioned Parquet directory next to its large CSV.
        Reading one part at a time bounds peak memory while keeping every file
        operation inside the standardized data layer.
        """

        if path.is_dir():
            parts = sorted(
                item
                for item in path.iterdir()
                if item.is_file() and item.suffix.lower() in {".parquet", ".pq"}
            )
            if not parts:
                raise ValueError(f"standardized artifact directory has no parquet parts: {path}")
            selected_skus = (
                None if not sku_codes else {str(value) for value in sku_codes}
            )
            observed_skus: set[str] = set()
            for part in parts:
                frame = StageArtifactStore.read_frame(
                    part,
                    required=required,
                    columns=columns,
                    sku_column=sku_column,
                )
                if selected_skus is not None:
                    frame_skus = frame[sku_column].astype(str)
                    frame = frame.loc[frame_skus.isin(selected_skus)]
                    observed_skus.update(frame[sku_column].astype(str).unique())
                if not frame.empty:
                    yield frame.reset_index(drop=True)
            if selected_skus is not None and require_all_skus:
                missing = sorted(selected_skus - observed_skus)
                if missing:
                    raise ValueError(
                        f"requested SKUs are absent from partitioned artifact: {missing}"
                    )
            return
        if path.suffix.lower() == ".csv":
            read_columns = list(dict.fromkeys(columns or ())) or None
            if read_columns is not None:
                for column in required:
                    if column not in read_columns:
                        read_columns.append(column)
            identifiers = {
                key: str
                for key in (
                    sku_column,
                    "sku_code",
                    "target_sku_code",
                    "candidate_sku_code",
                    "store_code",
                    "store_tier",
                )
                if read_columns is None or key in read_columns
            }
            selected_skus = (
                None if not sku_codes else {str(value) for value in sku_codes}
            )
            for chunk in pd.read_csv(
                path,
                dtype=identifiers,
                usecols=read_columns,
                low_memory=False,
                chunksize=csv_chunksize,
            ):
                require_columns(chunk, required, path.name)
                if selected_skus is not None:
                    chunk = chunk.loc[
                        chunk[sku_column].astype(str).isin(selected_skus)
                    ]
                if not chunk.empty:
                    yield chunk.reset_index(drop=True)
            return
        frame = StageArtifactStore.read_frame(
            path,
            required=required,
            columns=columns,
            sku_codes=sku_codes,
            sku_column=sku_column,
        )
        if not frame.empty:
            yield frame

    @staticmethod
    def write_frame(path: Path, frame: pd.DataFrame) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        if path.suffix.lower() == ".csv":
            frame.to_csv(temporary, index=False)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frame.to_parquet(temporary, index=False)
        else:
            raise ValueError(f"unsupported standardized artifact format: {path}")
        temporary.replace(path)
        return path

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path
