from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from config.settings import TableSpec
from data.point_in_time import PointInTimeContext
from data.server_probe import resolve_table_path


class DataLayerError(RuntimeError):
    """Base error for safe source-data reads."""


class SourceNotFoundError(DataLayerError):
    pass


class SchemaDriftError(DataLayerError):
    pass


class UnsafeReadError(DataLayerError):
    pass


class RowLimitExceeded(DataLayerError):
    pass


@dataclass(frozen=True)
class ReadResult:
    source_name: str
    frame: Any
    selected_partitions: tuple[date, ...]
    projected_columns: tuple[str, ...]
    sku_filter: str | None
    bounded_sample: bool = False
    sku_filters: tuple[str, ...] = ()

    @property
    def row_count(self) -> int:
        return len(self.frame)


def _parse_partition_date(value: str) -> date | None:
    for pattern in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            pass
    return None


class ParquetDataRepository:
    """Read-only, bounded Parquet access with partition and predicate pushdown.

    The repository never discovers a global MAX(ds) without a cutoff. Snapshot
    reads choose the latest physical partition on or before the decision cutoff;
    history reads require an explicit start date and reject a future end date.
    """

    def __init__(
        self,
        *,
        data_root: Path,
        table_specs: Sequence[TableSpec],
        max_rows_per_read: int = 1_000_000,
    ) -> None:
        if max_rows_per_read <= 0:
            raise ValueError("max_rows_per_read must be positive")
        self.data_root = data_root
        self.max_rows_per_read = max_rows_per_read
        self._specs = {spec.table_name: spec for spec in table_specs}
        if len(self._specs) != len(table_specs):
            raise ValueError("table_specs contains duplicate table names")
        # Metadata only: these caches never retain business rows.  M2 repeatedly
        # visits overlapping date windows, so rescanning the same directories for
        # every SKU is pure filesystem overhead.
        self._partition_cache: dict[str, tuple[date, ...]] = {}
        self._partition_path_cache: dict[str, dict[date, Path]] = {}
        self._partition_file_cache: dict[tuple[str, date], tuple[Path, ...]] = {}

    def table_spec(self, source_name: str) -> TableSpec:
        try:
            return self._specs[source_name]
        except KeyError as exc:
            raise SourceNotFoundError(f"source is not configured: {source_name}") from exc

    def table_path(self, source_name: str) -> Path:
        path, _ = resolve_table_path(self.data_root, source_name)
        if not path.is_dir():
            raise SourceNotFoundError(f"source directory not found: {path}")
        return path

    def available_partitions(self, source_name: str) -> tuple[date, ...]:
        cached = self._partition_cache.get(source_name)
        if cached is not None:
            return cached
        spec = self.table_spec(source_name)
        table_path = self.table_path(source_name)
        prefix = f"{spec.partition_column}="
        parsed: dict[date, Path] = {}
        for child in table_path.iterdir():
            if not child.is_dir() or not child.name.startswith(prefix):
                continue
            value = child.name[len(prefix) :]
            parsed_value = _parse_partition_date(value)
            if parsed_value is not None:
                parsed[parsed_value] = child
        if not parsed:
            raise SourceNotFoundError(
                f"no parseable {spec.partition_column} partitions found for {source_name}"
            )
        partitions = tuple(sorted(parsed))
        self._partition_cache[source_name] = partitions
        self._partition_path_cache[source_name] = parsed
        return partitions

    def _partition_directories(
        self, source_name: str, partitions: Sequence[date]
    ) -> list[Path]:
        self.available_partitions(source_name)
        paths = self._partition_path_cache[source_name]
        return [paths[value] for value in sorted(set(partitions)) if value in paths]

    def _selected_data_files(
        self, source_name: str, partitions: Sequence[date]
    ) -> list[Path]:
        files: list[Path] = []
        self.available_partitions(source_name)
        directories = self._partition_path_cache[source_name]
        for partition_value in sorted(set(partitions)):
            cache_key = (source_name, partition_value)
            cached = self._partition_file_cache.get(cache_key)
            if cached is not None:
                files.extend(cached)
                continue
            partition_dir = directories.get(partition_value)
            if partition_dir is None:
                continue
            partition_files: list[Path] = []
            for root, directory_names, file_names in os.walk(
                partition_dir, followlinks=False
            ):
                directory_names[:] = sorted(
                    name for name in directory_names if not name.startswith(".")
                )
                for file_name in sorted(file_names):
                    if (
                        file_name.startswith((".", "_"))
                        or file_name.endswith((".crc", ".tmp"))
                    ):
                        continue
                    path = Path(root) / file_name
                    if path.is_file():
                        partition_files.append(path)
            partition_files.sort()
            self._partition_file_cache[cache_key] = tuple(partition_files)
            files.extend(partition_files)
        return files

    @staticmethod
    def _require_aware(cutoff_time: datetime) -> None:
        if cutoff_time.tzinfo is None or cutoff_time.utcoffset() is None:
            raise UnsafeReadError("cutoff_time must be timezone-aware")

    def latest_partition(self, source_name: str, cutoff_time: datetime) -> date:
        self._require_aware(cutoff_time)
        eligible = [
            value
            for value in self.available_partitions(source_name)
            if value <= cutoff_time.date()
        ]
        if not eligible:
            raise SourceNotFoundError(
                f"no partition for {source_name} on or before {cutoff_time.date()}"
            )
        return eligible[-1]

    def _pinned_partition(
        self, context: PointInTimeContext, source_name: str
    ) -> date | None:
        raw = context.source_versions.get(source_name)
        if raw is None:
            return None
        parsed = _parse_partition_date(raw)
        if parsed is None:
            raise UnsafeReadError(
                f"source_versions[{source_name!r}] must be a ds date, got {raw!r}"
            )
        if parsed > context.decision_cutoff_time.date():
            raise UnsafeReadError(
                f"pinned partition {parsed} for {source_name} is after the cutoff"
            )
        if parsed not in self.available_partitions(source_name):
            raise SourceNotFoundError(
                f"pinned partition {parsed} does not exist for {source_name}"
            )
        return parsed

    @staticmethod
    def _load_arrow_dataset(table_path: Path, files: Sequence[Path]) -> Any:
        try:
            import pyarrow.dataset as ds
        except ImportError as exc:
            raise DataLayerError("pyarrow is required for the confirmed Parquet sources") from exc
        try:
            return ds.dataset(
                [str(path) for path in files],
                format="parquet",
                partitioning="hive",
                partition_base_dir=str(table_path),
                exclude_invalid_files=False,
            )
        except Exception as exc:
            raise DataLayerError(
                f"failed to open Parquet dataset {table_path}: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _projection(
        spec: TableSpec, dataset: Any, columns: Sequence[str] | None
    ) -> tuple[str, ...]:
        requested = tuple(dict.fromkeys(columns or spec.required_fields))
        if spec.partition_column not in requested:
            requested = requested + (spec.partition_column,)
        available = set(dataset.schema.names)
        missing = sorted(set(requested) - available)
        if missing:
            raise SchemaDriftError(
                f"{spec.table_name} is missing projected columns: {', '.join(missing)}"
            )
        return requested

    def _scan(
        self,
        *,
        source_name: str,
        partitions: Sequence[date],
        columns: Sequence[str] | None,
        sku_code: str | None,
        sku_codes: Sequence[str] | None,
        equality_filters: Mapping[str, object] | None,
        max_rows: int | None,
        bounded_sample: bool,
    ) -> ReadResult:
        import pandas as pd
        import pyarrow.dataset as ds

        spec = self.table_spec(source_name)
        if sku_code is not None and sku_codes is not None:
            raise ValueError("sku_code and sku_codes are mutually exclusive")
        normalized_skus = tuple(dict.fromkeys(str(value) for value in (sku_codes or ())))
        if sku_codes is not None and not normalized_skus:
            raise ValueError("sku_codes must not be empty")
        limit = self.max_rows_per_read if max_rows is None else max_rows
        if limit <= 0:
            raise ValueError("max_rows must be positive")
        if not partitions:
            projection = tuple(dict.fromkeys(columns or spec.required_fields))
            if spec.partition_column not in projection:
                projection = projection + (spec.partition_column,)
            return ReadResult(
                source_name=source_name,
                frame=pd.DataFrame(columns=projection),
                selected_partitions=(),
                projected_columns=projection,
                sku_filter=sku_code,
                bounded_sample=bounded_sample,
                sku_filters=normalized_skus,
            )
        table_path = self.table_path(source_name)
        selected_files = self._selected_data_files(source_name, partitions)
        if not selected_files:
            raise SourceNotFoundError(
                f"no data files found in selected partitions for {source_name}"
            )
        dataset = self._load_arrow_dataset(table_path, selected_files)
        projection = self._projection(spec, dataset, columns)

        expression = ds.field(spec.partition_column).isin(
            [value.isoformat() for value in partitions]
        )
        if sku_code is not None:
            if not spec.sku_column:
                raise UnsafeReadError(f"{source_name} has no configured SKU column")
            expression = expression & (ds.field(spec.sku_column) == sku_code)
        elif normalized_skus:
            if not spec.sku_column:
                raise UnsafeReadError(f"{source_name} has no configured SKU column")
            expression = expression & ds.field(spec.sku_column).isin(
                list(normalized_skus)
            )
        for name, value in (equality_filters or {}).items():
            if name not in dataset.schema.names:
                raise SchemaDriftError(
                    f"{source_name} filter column does not exist: {name}"
                )
            expression = expression & (ds.field(name) == value)

        scanner = dataset.scanner(
            columns=list(projection),
            filter=expression,
            use_threads=True,
        )
        arrow_table = scanner.head(limit if bounded_sample else limit + 1)
        if not bounded_sample and arrow_table.num_rows > limit:
            raise RowLimitExceeded(
                f"{source_name} read exceeds {limit} rows; narrow the partition window "
                "or add a SKU filter"
            )
        return ReadResult(
            source_name=source_name,
            frame=arrow_table.to_pandas(),
            selected_partitions=tuple(partitions),
            projected_columns=projection,
            sku_filter=sku_code,
            bounded_sample=bounded_sample,
            sku_filters=normalized_skus,
        )

    def read_latest_partition(
        self,
        *,
        source_name: str,
        cutoff_time: datetime,
        columns: Sequence[str] | None = None,
        sku_code: str | None = None,
        equality_filters: Mapping[str, object] | None = None,
        max_rows: int | None = None,
        sku_codes: Sequence[str] | None = None,
    ) -> ReadResult:
        selected = self.latest_partition(source_name, cutoff_time)
        return self._scan(
            source_name=source_name,
            partitions=(selected,),
            columns=columns,
            sku_code=sku_code,
            sku_codes=sku_codes,
            equality_filters=equality_filters,
            max_rows=max_rows,
            bounded_sample=False,
        )

    def sample_latest_partition(
        self,
        *,
        source_name: str,
        cutoff_time: datetime,
        columns: Sequence[str] | None = None,
        sample_rows: int = 100,
    ) -> ReadResult:
        selected = self.latest_partition(source_name, cutoff_time)
        return self._scan(
            source_name=source_name,
            partitions=(selected,),
            columns=columns,
            sku_code=None,
            sku_codes=None,
            equality_filters=None,
            max_rows=sample_rows,
            bounded_sample=True,
        )

    def read_partition_range(
        self,
        *,
        source_name: str,
        start_date: date,
        end_date: date,
        cutoff_time: datetime,
        columns: Sequence[str] | None = None,
        sku_code: str | None = None,
        equality_filters: Mapping[str, object] | None = None,
        max_rows: int | None = None,
        sku_codes: Sequence[str] | None = None,
    ) -> ReadResult:
        self._require_aware(cutoff_time)
        if start_date > end_date:
            raise UnsafeReadError("start_date must be on or before end_date")
        if end_date > cutoff_time.date():
            raise UnsafeReadError(
                f"end_date {end_date} is after decision cutoff {cutoff_time.date()}"
            )
        partitions = tuple(
            value
            for value in self.available_partitions(source_name)
            if start_date <= value <= end_date
        )
        return self._scan(
            source_name=source_name,
            partitions=partitions,
            columns=columns,
            sku_code=sku_code,
            sku_codes=sku_codes,
            equality_filters=equality_filters,
            max_rows=max_rows,
            bounded_sample=False,
        )

    def iter_partition_batches(
        self,
        *,
        source_name: str,
        start_date: date,
        end_date: date,
        cutoff_time: datetime,
        columns: Sequence[str],
        batch_size: int = 50_000,
        sku_code: str | None = None,
        sku_codes: Sequence[str] | None = None,
        equality_filters: Mapping[str, object] | None = None,
    ) -> Any:
        """Yield Arrow batches for an explicit bounded partition window.

        This is the only unmaterialized bulk-read interface. It constructs the
        dataset from selected partition files only and never calls ``to_table``.
        """

        import pyarrow.dataset as ds

        self._require_aware(cutoff_time)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if start_date > end_date:
            raise UnsafeReadError("start_date must be on or before end_date")
        if end_date > cutoff_time.date():
            raise UnsafeReadError(
                f"end_date {end_date} is after decision cutoff {cutoff_time.date()}"
            )
        spec = self.table_spec(source_name)
        if sku_code is not None and sku_codes is not None:
            raise ValueError("sku_code and sku_codes are mutually exclusive")
        normalized_skus = tuple(dict.fromkeys(str(value) for value in (sku_codes or ())))
        if sku_codes is not None and not normalized_skus:
            raise ValueError("sku_codes must not be empty")
        partitions = tuple(
            value
            for value in self.available_partitions(source_name)
            if start_date <= value <= end_date
        )
        if not partitions:
            return
        table_path = self.table_path(source_name)
        selected_files = self._selected_data_files(source_name, partitions)
        if not selected_files:
            raise SourceNotFoundError(
                f"no data files found in selected partitions for {source_name}"
            )
        dataset = self._load_arrow_dataset(table_path, selected_files)
        projection = self._projection(spec, dataset, columns)
        expression = ds.field(spec.partition_column).isin(
            [value.isoformat() for value in partitions]
        )
        if sku_code is not None:
            if not spec.sku_column:
                raise UnsafeReadError(f"{source_name} has no configured SKU column")
            expression = expression & (ds.field(spec.sku_column) == sku_code)
        elif normalized_skus:
            if not spec.sku_column:
                raise UnsafeReadError(f"{source_name} has no configured SKU column")
            expression = expression & ds.field(spec.sku_column).isin(
                list(normalized_skus)
            )
        for name, value in (equality_filters or {}).items():
            if name not in dataset.schema.names:
                raise SchemaDriftError(
                    f"{source_name} filter column does not exist: {name}"
                )
            expression = expression & (ds.field(name) == value)
        scanner = dataset.scanner(
            columns=list(projection),
            filter=expression,
            batch_size=batch_size,
            use_threads=True,
        )
        yield from scanner.to_batches()

    def read_latest(
        self,
        context: PointInTimeContext,
        *,
        source_name: str,
        columns: Sequence[str] | None = None,
        max_rows: int | None = None,
    ) -> ReadResult:
        spec = self.table_spec(source_name)
        pinned = self._pinned_partition(context, source_name)
        if pinned is not None:
            return self._scan(
                source_name=source_name,
                partitions=(pinned,),
                columns=columns,
                sku_code=context.sku_code if spec.sku_column else None,
                sku_codes=None,
                equality_filters=None,
                max_rows=max_rows,
                bounded_sample=False,
            )
        return self.read_latest_partition(
            source_name=source_name,
            cutoff_time=context.decision_cutoff_time,
            columns=columns,
            sku_code=context.sku_code if spec.sku_column else None,
            max_rows=max_rows,
        )

    def read_history(
        self,
        context: PointInTimeContext,
        *,
        source_name: str,
        start_date: date,
        columns: Sequence[str] | None = None,
        max_rows: int | None = None,
    ) -> ReadResult:
        spec = self.table_spec(source_name)
        if spec.read_mode != "partition_range":
            raise UnsafeReadError(
                f"{source_name} is configured as {spec.read_mode}; range reads are disabled"
            )
        pinned = self._pinned_partition(context, source_name)
        end_date = pinned or context.decision_cutoff_time.date()
        return self.read_partition_range(
            source_name=source_name,
            start_date=start_date,
            end_date=end_date,
            cutoff_time=context.decision_cutoff_time,
            columns=columns,
            sku_code=context.sku_code if spec.sku_column else None,
            max_rows=max_rows,
        )

    def fetch_as_of(
        self,
        context: PointInTimeContext,
        *,
        source_name: str,
        columns: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        spec = self.table_spec(source_name)
        if spec.read_mode != "latest_snapshot":
            raise UnsafeReadError(
                f"{source_name} is historical; use read_history with an explicit start_date"
            )
        result = self.read_latest(context, source_name=source_name, columns=columns)
        return result.frame.to_dict(orient="records")
