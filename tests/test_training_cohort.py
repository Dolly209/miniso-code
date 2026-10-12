from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from config.settings import TableSpec
from data.domestic import SALES_SOURCE, DataContractViolation
from data.parquet_repository import ParquetDataRepository
from training.cohort import build_first_order_cohort


def _spec() -> TableSpec:
    return TableSpec(
        table_name=SALES_SOURCE,
        partition_column="ds",
        required_fields=(
            "ds",
            "order_id",
            "order_detail_id",
            "order_time",
            "sku_code",
            "sale_qty",
            "is_return",
            "gift_flag",
        ),
        read_mode="partition_range",
        business_date_column="order_time",
        grain_columns=("order_id", "order_detail_id"),
        sku_column="sku_code",
    )


def _row(
    order_id: str,
    order_time: str,
    sku: str,
    quantity: float,
) -> dict[str, object]:
    return {
        "order_id": order_id,
        "order_detail_id": "1",
        "order_time": order_time,
        "sku_code": sku,
        "sale_qty": quantity,
        "is_return": "0",
        "gift_flag": "N",
    }


def _write(
    root: Path, ds: str, rows: list[dict[str, object]], index: int = 0
) -> None:
    target = root / SALES_SOURCE / f"ds={ds}"
    target.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), target / f"part-{index:05d}.parquet")


def test_streaming_cohort_uses_latest_line_version_and_only_mature_skus(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "2025-01-01", [_row("A", "2025-01-01 10:00:00", "SKU1", 2)])
    _write(tmp_path, "2025-01-02", [_row("A", "2025-01-01 10:00:00", "SKU1", 3)])
    _write(tmp_path, "2025-01-03", [_row("B", "2025-01-03 10:00:00", "SKU1", 4)])
    _write(tmp_path, "2025-01-04", [_row("E", "2025-01-04 10:00:00", "SKU1", -1)])
    _write(tmp_path, "2025-01-05", [_row("C", "2025-01-05 10:00:00", "SKU2", 9)])
    _write(tmp_path, "2025-01-08", [_row("D", "2025-01-08 10:00:00", "SKU3", 0)])
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[_spec()])

    result = build_first_order_cohort(
        repository=repository,
        cutoff_time=datetime.fromisoformat("2025-01-08T23:59:59+08:00"),
        cache_path=tmp_path / "cohort.sqlite",
        window_weeks=1,
        batch_size=2,
    )

    assert len(result.rows) == 1
    assert result.rows[0].sku_code == "SKU1"
    assert result.rows[0].first_actual_sale_date == date(2025, 1, 1)
    assert result.rows[0].sales_26w_qty == 6
    assert result.stats.source_partition_min == date(2025, 1, 1)
    assert result.stats.label_as_of_date == date(2025, 1, 8)
    assert result.stats.sku_with_positive_sales == 2
    assert result.stats.mature_sku_count == 1
    assert result.stats.immature_sku_count == 1
    assert result.stats.first_sale_on_source_min_count == 1
    assert result.stats.negative_sale_qty_versions == 1
    assert result.stats.negative_sale_qty_sum == -1
    assert result.stats.negative_sale_qty_min == -1
    assert result.stats.latest_negative_order_lines == 1
    assert result.stats.latest_negative_sale_qty_sum == -1
    assert result.stats.latest_negative_sale_qty_min == -1


def test_cohort_collapses_exact_duplicate_order_line_in_same_ds(
    tmp_path: Path,
) -> None:
    duplicate = _row("A", "2025-01-01 10:00:00", "SKU1", 2)
    _write(tmp_path, "2025-01-01", [duplicate, duplicate])
    _write(tmp_path, "2025-01-08", [_row("B", "2025-01-08 10:00:00", "SKU2", 1)])
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[_spec()])

    result = build_first_order_cohort(
        repository=repository,
        cutoff_time=datetime.fromisoformat("2025-01-08T23:59:59+08:00"),
        cache_path=tmp_path / "cohort.sqlite",
        window_weeks=1,
        batch_size=1,
    )

    assert result.rows[0].sales_26w_qty == 2
    assert result.stats.same_ds_duplicate_rows == 1
    assert result.stats.same_ds_exact_duplicate_rows == 1
    assert result.stats.same_ds_signed_aggregation_extra_rows == 0


def test_cohort_net_aggregates_same_sign_rows_with_distinct_event_times(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "2025-01-01",
        [
            _row("A", "2025-01-01 10:00:00", "SKU1", 2),
            _row("A", "2025-01-01 10:05:00", "SKU1", 3),
        ],
    )
    _write(tmp_path, "2025-01-08", [_row("B", "2025-01-08 10:00:00", "SKU2", 0)])
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[_spec()])

    result = build_first_order_cohort(
        repository=repository,
        cutoff_time=datetime.fromisoformat("2025-01-08T23:59:59+08:00"),
        cache_path=tmp_path / "cohort.sqlite",
        window_weeks=1,
        batch_size=1,
    )

    assert result.rows[0].sales_26w_qty == 5
    assert result.stats.same_ds_signed_aggregation_groups == 1
    assert result.stats.same_ds_nonzero_net_groups == 1


def test_cohort_net_aggregates_signed_same_ds_reversals(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "2025-01-01",
        [
            _row("A", "2025-01-01 10:00:00", "SKU1", 1),
            _row("A", "2025-01-01 10:05:00", "SKU1", -1),
            _row("B", "2025-01-01 11:00:00", "SKU2", 2),
            _row("B", "2025-01-01 11:05:00", "SKU2", -1),
        ],
    )
    _write(tmp_path, "2025-01-08", [_row("C", "2025-01-08 10:00:00", "SKU3", 0)])
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[_spec()])

    result = build_first_order_cohort(
        repository=repository,
        cutoff_time=datetime.fromisoformat("2025-01-08T23:59:59+08:00"),
        cache_path=tmp_path / "cohort.sqlite",
        window_weeks=1,
        batch_size=1,
    )

    assert [(row.sku_code, row.sales_26w_qty) for row in result.rows] == [
        ("SKU2", 1.0)
    ]
    assert result.stats.same_ds_signed_aggregation_extra_rows == 2
    assert result.stats.same_ds_signed_aggregation_groups == 2
    assert result.stats.same_ds_zero_net_groups == 1
    assert result.stats.same_ds_nonzero_net_groups == 1


def test_cohort_rejects_same_key_with_different_skus(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "2025-01-01",
        [
            _row("A", "2025-01-01 10:00:00", "SKU1", 1),
            _row("A", "2025-01-01 10:05:00", "SKU2", -1),
        ],
    )
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[_spec()])

    with pytest.raises(DataContractViolation, match="different SKUs"):
        build_first_order_cohort(
            repository=repository,
            cutoff_time=datetime.fromisoformat("2025-01-08T23:59:59+08:00"),
            cache_path=tmp_path / "cohort.sqlite",
            window_weeks=1,
            batch_size=1,
        )


def test_latest_version_after_cutoff_does_not_fall_back_to_older_version(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "2025-01-01", [_row("A", "2025-01-01 10:00:00", "SKU1", 2)])
    _write(tmp_path, "2025-01-08", [_row("A", "2025-01-08 18:00:00", "SKU1", 3)])
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[_spec()])

    result = build_first_order_cohort(
        repository=repository,
        cutoff_time=datetime.fromisoformat("2025-01-08T12:00:00+08:00"),
        cache_path=tmp_path / "cohort.sqlite",
        window_weeks=1,
        batch_size=1,
    )

    assert result.rows == ()
    assert result.stats.sku_with_positive_sales == 0
    assert result.stats.visible_rows_scanned == 1
    assert result.stats.order_line_versions == 2
