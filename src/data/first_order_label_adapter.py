from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Iterator, Mapping

import pandas as pd

from config.settings import AppConfig, load_table_specs
from data.decision_adapter import SOURCES
from data.parquet_repository import ParquetDataRepository
from data.restoration_adapter import SALES_COLUMNS


ProgressCallback = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class FirstOrderLabelSalesBatch:
    batch_index: int
    batch_count: int
    windows: pd.DataFrame
    raw_sales: pd.DataFrame
    selected_partitions: tuple[date, ...]


class FirstOrderLabelDataAdapter:
    """Read sales rows for observed-volume audits, not formal labels.

    The formal V3.2 ``D_first_ref`` label is built from Round 4
    ``restored_natural_sales`` and therefore cannot be produced by this
    sales-only adapter.  The adapter remains available for signed-order and
    zero-week audit tests.
    """

    def __init__(self, repository: ParquetDataRepository) -> None:
        self.repository = repository

    @classmethod
    def from_project(
        cls, *, project_root: Path, data_root: Path, config: AppConfig
    ) -> "FirstOrderLabelDataAdapter":
        return cls(
            ParquetDataRepository(
                data_root=data_root,
                table_specs=load_table_specs(config.probe.table_specs_path),
                max_rows_per_read=config.data_reader.max_rows_per_read,
            )
        )

    def iter_sales_batches(
        self,
        *,
        windows: pd.DataFrame,
        cutoff_time: datetime,
        batch_size: int,
        progress: ProgressCallback | None = None,
    ) -> Iterator[FirstOrderLabelSalesBatch]:
        if windows.empty:
            raise ValueError("no W_cover label windows were supplied")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        required = {"sku_code", "label_start_date", "label_end_date"}
        missing = sorted(required - set(windows.columns))
        if missing:
            raise ValueError(f"label windows are missing columns: {missing}")
        work = windows.copy()
        work["sku_code"] = work["sku_code"].astype(str).str.strip()
        work["label_start_date"] = pd.to_datetime(
            work["label_start_date"], errors="raise"
        ).dt.date
        work["label_end_date"] = pd.to_datetime(
            work["label_end_date"], errors="raise"
        ).dt.date
        if work["sku_code"].duplicated().any():
            raise ValueError("label windows must contain one row per SKU")
        batch_count = (len(work) + batch_size - 1) // batch_size
        for zero_index in range(batch_count):
            batch_index = zero_index + 1
            selected = work.iloc[
                zero_index * batch_size : (zero_index + 1) * batch_size
            ].copy()
            start = selected["label_start_date"].min()
            end = min(selected["label_end_date"].max(), cutoff_time.date())
            if start > end:
                raise ValueError(
                    "W_cover label window starts after the available-through cutoff"
                )
            skus = selected["sku_code"].tolist()
            if progress is not None:
                progress(
                    "label_sales_read_start",
                    {
                        "batch": batch_index,
                        "batches": batch_count,
                        "skus": len(skus),
                        "start_date": start.isoformat(),
                        "end_date": end.isoformat(),
                    },
                )
            started = monotonic()
            result = self.repository.read_partition_range(
                source_name=SOURCES["sales"],
                start_date=start,
                end_date=end,
                cutoff_time=cutoff_time,
                columns=SALES_COLUMNS,
                sku_codes=skus,
                max_rows=self.repository.max_rows_per_read * max(len(skus), 1),
            )
            if progress is not None:
                progress(
                    "label_sales_read_complete",
                    {
                        "batch": batch_index,
                        "batches": batch_count,
                        "rows": len(result.frame),
                        "partitions": len(result.selected_partitions),
                        "elapsed_seconds": round(monotonic() - started, 3),
                    },
                )
            yield FirstOrderLabelSalesBatch(
                batch_index=batch_index,
                batch_count=batch_count,
                windows=selected,
                raw_sales=result.frame,
                selected_partitions=result.selected_partitions,
            )
