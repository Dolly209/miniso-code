from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from data.decision_adapter import SOURCES
from data.parquet_repository import ReadResult
from data.restoration_adapter import (
    PRODUCT_COLUMNS,
    STORE_COLUMNS,
    RestorationDataAdapter,
)


class FakeRepository:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.max_rows_per_read = 100
        self.range_calls = 0
        self.store_calls = 0
        self.product_calls = 0

    def read_latest_partition(
        self,
        *,
        source_name: str,
        cutoff_time: datetime,
        columns: tuple[str, ...],
        sku_codes: list[str] | None = None,
        max_rows: int | None = None,
        **_: object,
    ) -> ReadResult:
        if source_name == SOURCES["store"]:
            self.store_calls += 1
            row = {name: None for name in STORE_COLUMNS}
            row.update(
                {
                    "ds": cutoff_time.date().isoformat(),
                    "store_code": "STORE1",
                    "store_business_state": "normal",
                    "store_level": "A",
                    "sts_state": "active",
                }
            )
            frame = pd.DataFrame([row], columns=columns)
        else:
            assert source_name == SOURCES["product"]
            self.product_calls += 1
            frame = pd.DataFrame(
                [
                    {
                        "ds": cutoff_time.date().isoformat(),
                        "sku_code": sku,
                        "cate_level4_code": "C1",
                        "retail_price": 10.0,
                    }
                    for sku in (sku_codes or [])
                ],
                columns=PRODUCT_COLUMNS,
            )
        return ReadResult(
            source_name=source_name,
            frame=frame,
            selected_partitions=(cutoff_time.date(),),
            projected_columns=tuple(columns),
            sku_filter=None,
            sku_filters=tuple(sku_codes or ()),
        )

    def read_partition_range(
        self,
        *,
        source_name: str,
        start_date: date,
        end_date: date,
        columns: tuple[str, ...],
        sku_codes: list[str],
        **_: object,
    ) -> ReadResult:
        self.range_calls += 1
        rows = []
        current = start_date
        while current <= end_date:
            for sku in sku_codes:
                row = {name: None for name in columns}
                row.update(
                    {
                        "ds": current.isoformat(),
                        "sku_code": sku,
                        "store_code": "STORE1",
                    }
                )
                rows.append(row)
            current += timedelta(days=1)
        return ReadResult(
            source_name=source_name,
            frame=pd.DataFrame(rows, columns=columns),
            selected_partitions=tuple(),
            projected_columns=tuple(columns),
            sku_filter=None,
            sku_filters=tuple(sku_codes),
        )


def _selected() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_version": "round3-test",
                "sku_code": "SKU1",
                "first_actual_sale_date": date(2025, 1, 2),
                "split": "Train",
                "long_tail_weight": 1.0,
            },
            {
                "dataset_version": "round3-test",
                "sku_code": "SKU2",
                "first_actual_sale_date": date(2025, 1, 3),
                "split": "Train",
                "long_tail_weight": 1.0,
            },
            {
                "dataset_version": "round3-test",
                "sku_code": "SKU3",
                "first_actual_sale_date": date(2025, 1, 4),
                "split": "Test",
                "long_tail_weight": 0.5,
            },
        ]
    )


def test_batched_restore_reads_each_source_once_per_batch_and_resumes(
    tmp_path: Path,
) -> None:
    repository = FakeRepository(tmp_path / "data")
    adapter = RestorationDataAdapter(repository)  # type: ignore[arg-type]
    events: list[str] = []
    kwargs = {
        "selected": _selected(),
        "cutoff_time": datetime.fromisoformat("2025-01-10T23:59:59+08:00"),
        "window_days": 2,
        "batch_size": 2,
        "checkpoint_dir": tmp_path / "checkpoints",
        "progress": lambda event, details: events.append(event),
    }

    first = adapter.load_batched(**kwargs)

    assert repository.range_calls == 6  # 2 sequential batches x 3 history sources
    assert repository.store_calls == 1
    assert repository.product_calls == 2
    assert len(first.raw_sales) == 6  # 3 SKUs x their exact 2-day windows
    assert first.computed_batch_count == 2
    assert first.reused_batch_count == 0
    assert "source_read_complete" in events

    calls_before_resume = (
        repository.range_calls,
        repository.store_calls,
        repository.product_calls,
    )
    second = adapter.load_batched(**kwargs)

    assert (
        repository.range_calls,
        repository.store_calls,
        repository.product_calls,
    ) == calls_before_resume
    assert second.computed_batch_count == 0
    assert second.reused_batch_count == 2
    assert len(second.raw_sales) == len(first.raw_sales)


def test_batched_restore_reuses_unchanged_batches_when_sku_scope_expands(
    tmp_path: Path,
) -> None:
    repository = FakeRepository(tmp_path / "data")
    adapter = RestorationDataAdapter(repository)  # type: ignore[arg-type]
    common = {
        "cutoff_time": datetime.fromisoformat("2025-01-10T23:59:59+08:00"),
        "window_days": 2,
        "batch_size": 2,
        "checkpoint_dir": tmp_path / "checkpoints",
    }

    first = adapter.load_batched(selected=_selected().iloc[:2].copy(), **common)
    assert first.computed_batch_count == 1
    assert repository.range_calls == 3
    assert repository.product_calls == 1

    expanded = adapter.load_batched(selected=_selected(), **common)

    assert expanded.reused_batch_count == 1
    assert expanded.computed_batch_count == 1
    assert repository.range_calls == 6
    assert repository.product_calls == 2
