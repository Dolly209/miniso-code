from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    ) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path, dtype={sku_column: str})
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
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
