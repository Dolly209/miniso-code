from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pytest

from config.settings import TableSpec
from data.decision_adapter import (
    DecisionDataAdapter,
    PLACEMENT_LAUNCH_COLUMNS,
    ProductDomain,
    SOURCES,
)
from data.domestic import (
    DataContractViolation,
    ProductDataService,
    SalesDataService,
    safe_join_sku_attributes,
)
from data.parquet_repository import ParquetDataRepository, UnsafeReadError
from data.point_in_time import DecisionType, PointInTimeContext
from data.restoration_adapter import PRODUCT_COLUMNS, RestorationDataAdapter


TZ = ZoneInfo("Asia/Shanghai")


def _write_partition(
    root: Path, table_name: str, ds: str, rows: list[dict[str, object]], index: int = 0
) -> None:
    target = root / table_name / f"ds={ds}"
    target.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), target / f"part-{index:05d}")


def _context(cutoff: str = "2025-01-02T12:00:00+08:00") -> PointInTimeContext:
    return PointInTimeContext(
        run_id="RUN-1",
        sku_code="SKU1",
        decision_type=DecisionType.FIRST_ORDER,
        decision_cutoff_time=datetime.fromisoformat(cutoff),
    )


def _context_with_versions(versions: dict[str, str]) -> PointInTimeContext:
    return PointInTimeContext(
        run_id="RUN-1",
        sku_code="SKU1",
        decision_type=DecisionType.FIRST_ORDER,
        decision_cutoff_time=datetime.fromisoformat("2025-01-02T12:00:00+08:00"),
        source_versions=versions,
    )


def _sales_spec() -> TableSpec:
    fields = SalesDataService.REQUIRED_COLUMNS
    return TableSpec(
        table_name="scm.dwd_trd_order_all_flow_di_ip_forecast",
        partition_column="ds",
        required_fields=fields,
        read_mode="partition_range",
        business_date_column="order_time",
        grain_columns=("order_id", "order_detail_id"),
        sku_column="sku_code",
    )


def _sales_row(
    *,
    order_id: str,
    qty: float,
    sku: str = "SKU1",
    is_return: str = "0",
    gift_flag: str = "0",
) -> dict[str, object]:
    return {
        "order_id": order_id,
        "order_detail_id": "1",
        "order_time": "2025-01-01 10:00:00",
        "sku_code": sku,
        "store_code": "S1",
        "sale_qty": qty,
        "order_amt": qty * 10,
        "paid_amt": qty * 10,
        "achievement_amt": qty * 10,
        "is_return": is_return,
        "gift_flag": gift_flag,
    }


def test_repository_prunes_future_partition_and_pushes_sku_filter(tmp_path: Path) -> None:
    spec = _sales_spec()
    _write_partition(tmp_path, spec.table_name, "2025-01-01", [_sales_row(order_id="A", qty=1)])
    _write_partition(
        tmp_path,
        spec.table_name,
        "2025-01-02",
        [_sales_row(order_id="B", qty=2), _sales_row(order_id="C", qty=9, sku="SKU2")],
    )
    _write_partition(tmp_path, spec.table_name, "2025-01-03", [_sales_row(order_id="D", qty=3)])
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[spec])

    result = repository.read_history(
        _context(), source_name=spec.table_name, start_date=date(2025, 1, 1)
    )

    assert result.selected_partitions == (date(2025, 1, 1), date(2025, 1, 2))
    assert set(result.frame["sku_code"]) == {"SKU1"}
    assert set(result.frame["order_id"]) == {"A", "B"}


def test_repository_pushes_multi_sku_filter(tmp_path: Path) -> None:
    spec = _sales_spec()
    _write_partition(
        tmp_path,
        spec.table_name,
        "2025-01-01",
        [
            _sales_row(order_id="A", qty=1, sku="SKU1"),
            _sales_row(order_id="B", qty=2, sku="SKU2"),
            _sales_row(order_id="C", qty=3, sku="SKU3"),
        ],
    )
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[spec])

    result = repository.read_partition_range(
        source_name=spec.table_name,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 1),
        cutoff_time=datetime.fromisoformat("2025-01-01T23:59:59+08:00"),
        sku_codes=["SKU1", "SKU2"],
    )

    assert set(result.frame["sku_code"]) == {"SKU1", "SKU2"}
    assert result.sku_filters == ("SKU1", "SKU2")


def test_repository_caches_partition_metadata_for_process_lifetime(
    tmp_path: Path,
) -> None:
    spec = _sales_spec()
    _write_partition(tmp_path, spec.table_name, "2025-01-01", [_sales_row(order_id="A", qty=1)])
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[spec])

    first = repository.available_partitions(spec.table_name)
    _write_partition(tmp_path, spec.table_name, "2025-01-02", [_sales_row(order_id="B", qty=1)])
    second = repository.available_partitions(spec.table_name)

    assert first is second
    assert second == (date(2025, 1, 1),)


def test_restoration_product_snapshot_falls_back_per_sku_before_cutoff(
    tmp_path: Path,
) -> None:
    table_name = SOURCES["product"]
    spec = TableSpec(
        table_name=table_name,
        partition_column="ds",
        required_fields=PRODUCT_COLUMNS,
        read_mode="latest_snapshot",
        business_date_column="ds",
        grain_columns=("ds", "sku_code"),
        sku_column="sku_code",
    )
    _write_partition(
        tmp_path,
        table_name,
        "2025-01-01",
        [
            {"sku_code": "OLD", "cate_level4_code": "C1", "retail_price": 10.0},
            {"sku_code": "CURRENT", "cate_level4_code": "C2", "retail_price": 20.0},
        ],
    )
    _write_partition(
        tmp_path,
        table_name,
        "2025-01-02",
        [
            {"sku_code": "CURRENT", "cate_level4_code": "C2", "retail_price": 20.0},
        ],
    )
    _write_partition(
        tmp_path,
        table_name,
        "2025-01-03",
        [
            {"sku_code": "OLD", "cate_level4_code": "FUTURE", "retail_price": 99.0},
        ],
    )
    adapter = RestorationDataAdapter(
        ParquetDataRepository(data_root=tmp_path, table_specs=[spec])
    )
    events: list[str] = []

    products = adapter._read_products(
        ["OLD", "CURRENT"],
        datetime.fromisoformat("2025-01-02T23:59:59+08:00"),
        progress=lambda event, _details: events.append(event),
    ).set_index("sku_code")

    assert set(products.index) == {"OLD", "CURRENT"}
    assert products.loc["OLD", "cate_level4_code"] == "C1"
    assert products.loc["OLD", "retail_price"] == 10.0
    assert "product_snapshot_fallback_complete" in events


def test_historical_source_refuses_implicit_unbounded_fetch(tmp_path: Path) -> None:
    spec = _sales_spec()
    _write_partition(tmp_path, spec.table_name, "2025-01-01", [_sales_row(order_id="A", qty=1)])
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[spec])
    with pytest.raises(UnsafeReadError, match="explicit start_date"):
        repository.fetch_as_of(_context(), source_name=spec.table_name)


def test_context_can_pin_a_reproducible_source_partition(tmp_path: Path) -> None:
    table_name = "scm.snapshot"
    spec = TableSpec(
        table_name=table_name,
        partition_column="ds",
        required_fields=("ds", "sku_code", "value"),
        sku_column="sku_code",
    )
    _write_partition(tmp_path, table_name, "2025-01-01", [{"sku_code": "SKU1", "value": 1}])
    _write_partition(tmp_path, table_name, "2025-01-02", [{"sku_code": "SKU1", "value": 2}])
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[spec])

    result = repository.read_latest(
        _context_with_versions({table_name: "2025-01-01"}), source_name=table_name
    )

    assert result.selected_partitions == (date(2025, 1, 1),)
    assert result.frame.loc[0, "value"] == 1


def test_sales_keeps_latest_ds_version_and_never_subtracts_refund(tmp_path: Path) -> None:
    spec = _sales_spec()
    _write_partition(tmp_path, spec.table_name, "2025-01-01", [_sales_row(order_id="A", qty=2)])
    _write_partition(
        tmp_path,
        spec.table_name,
        "2025-01-02",
        [_sales_row(order_id="A", qty=3, gift_flag="N")],
    )
    service = SalesDataService(
        ParquetDataRepository(data_root=tmp_path, table_specs=[spec])
    )

    lines = service.read_order_lines(_context(), start_date=date(2025, 1, 1))
    daily = service.aggregate_sku_store_day(lines)

    assert len(lines.frame) == 1
    assert daily.loc[0, "sale_qty"] == 3
    assert service.actual_first_sale_date(lines) == date(2025, 1, 1)


def test_sales_contract_rejects_return_rows(tmp_path: Path) -> None:
    spec = _sales_spec()
    _write_partition(
        tmp_path,
        spec.table_name,
        "2025-01-01",
        [_sales_row(order_id="A", qty=-1, is_return="1")],
    )
    service = SalesDataService(
        ParquetDataRepository(data_root=tmp_path, table_specs=[spec])
    )
    with pytest.raises(DataContractViolation, match="return rows"):
        service.read_order_lines(_context(), start_date=date(2025, 1, 1))


def test_sales_contract_rejects_confirmed_gift_rows(tmp_path: Path) -> None:
    spec = _sales_spec()
    _write_partition(
        tmp_path,
        spec.table_name,
        "2025-01-01",
        [_sales_row(order_id="A", qty=1, gift_flag="Y")],
    )
    service = SalesDataService(
        ParquetDataRepository(data_root=tmp_path, table_specs=[spec])
    )
    with pytest.raises(DataContractViolation, match="gift rows"):
        service.read_order_lines(_context(), start_date=date(2025, 1, 1))


def test_product_multi_ip_is_separated_from_fact_join(tmp_path: Path) -> None:
    table_name = "scm.dim_goods_info_ip_forecast"
    fields = ("ds", "sku_code", "sku_name", "ip_code", "ip_name", "retail_price")
    spec = TableSpec(
        table_name=table_name,
        partition_column="ds",
        required_fields=fields,
        read_mode="latest_snapshot",
        business_date_column="ds",
        grain_columns=("ds", "sku_code", "ip_code"),
        sku_column="sku_code",
        ip_column="ip_code",
    )
    _write_partition(
        tmp_path,
        table_name,
        "2025-01-01",
        [
            {"sku_code": "SKU1", "sku_name": "Toy", "ip_code": "A", "ip_name": "A", "retail_price": 10.0},
            {"sku_code": "SKU1", "sku_name": "Toy", "ip_code": "B", "ip_name": "B", "retail_price": 10.0},
        ],
    )
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[spec])
    snapshot = ProductDataService(repository).read_snapshot(_context())
    membership = ProductDataService.ip_membership(snapshot)
    attributes = ProductDataService.one_row_per_sku_attributes(snapshot)

    assert len(membership) == 2
    assert len(attributes) == 1
    facts = snapshot.frame[["sku_code"]].head(1).assign(sale_qty=4)
    joined = safe_join_sku_attributes(facts, attributes)
    assert len(joined) == 1
    assert joined.loc[0, "sale_qty"] == 4


def test_cutoff_must_be_timezone_aware(tmp_path: Path) -> None:
    spec = _sales_spec()
    _write_partition(tmp_path, spec.table_name, "2025-01-01", [_sales_row(order_id="A", qty=1)])
    repository = ParquetDataRepository(data_root=tmp_path, table_specs=[spec])
    with pytest.raises(UnsafeReadError, match="timezone-aware"):
        repository.latest_partition(spec.table_name, datetime(2025, 1, 1))


def test_placement_adapter_builds_28_day_arrival_universe(tmp_path: Path) -> None:
    table_name = SOURCES["placement"]
    spec = TableSpec(
        table_name=table_name,
        partition_column="ds",
        required_fields=PLACEMENT_LAUNCH_COLUMNS,
        read_mode="partition_range",
        business_date_column="ds",
        grain_columns=("ds", "sku_code", "store_code"),
        sku_column="sku_code",
    )
    _write_partition(
        tmp_path,
        table_name,
        "2025-01-02",
        [
            {
                "sku_code": "C1",
                "store_code": "S1",
                "first_store_arrival_time": "2025-01-02 10:00:00",
            }
        ],
    )
    adapter = DecisionDataAdapter(
        ParquetDataRepository(data_root=tmp_path, table_specs=[spec])
    )
    domain = adapter.placement_launch_evidence(
        datetime.fromisoformat("2025-01-30T23:59:59+08:00"),
        candidate_products=pd.DataFrame(
            [
                {
                    "sku_code": "C1",
                    "first_sale_date": "2025-01-01",
                    "release_dt": "2024-12-20",
                }
            ]
        ),
        eligible_stores=pd.DataFrame(
            [
                {"store_code": "S1", "store_tier": "1|S"},
                {"store_code": "S2", "store_tier": "1|A"},
            ]
        ),
        restored_store_weekly=pd.DataFrame(
            [
                {
                    "sku_code": "C1",
                    "store_code": "S1",
                    "store_tier": "1|S",
                    "v_neutral": 2.0,
                    "stockout_days": 1,
                    "source_day_rows": 7,
                }
            ]
        ),
        launch_window_days=28,
    )

    evidence = domain.evidence.set_index("store_code")
    assert len(evidence) == 2
    assert evidence.loc["S1", "arrived_in_launch_window"] == 1
    assert evidence.loc["S2", "arrived_in_launch_window"] == 0
    assert evidence.loc["S1", "launch_window_end"] == "2025-01-28"
    assert evidence.loc["S1", "stock_pressure"] == pytest.approx(1 / 7)


def test_factor_features_use_resolved_ip_and_preserve_unknown_as_missing() -> None:
    adapter = DecisionDataAdapter.__new__(DecisionDataAdapter)
    products = pd.DataFrame(
        [
            {
                "sku_code": sku,
                "retail_price": 20.0,
                "lifecycle": "补货期",
                "copyright_category": "授权IP",
                "cate_level4_code": "C4",
            }
            for sku in ("MAPPED", "UNKNOWN")
        ]
    )
    # Both raw rows carry the same coarse theme code. M5 must not use it.
    raw_ip = pd.DataFrame(
        [
            {"sku_code": "MAPPED", "ip_code": "ip000057", "ip_name": "盲盒专区"},
            {"sku_code": "UNKNOWN", "ip_code": "ip000057", "ip_name": "盲盒专区"},
        ]
    )
    adapter.products = lambda *_args, **_kwargs: ProductDomain(products, raw_ip, "2026-09-30")

    supplier = pd.DataFrame(
        [
            {
                "ds": "2026-09-30",
                "supplier_code": "S1",
                "sku_code": "MAPPED",
                "target_qty": 10,
                "received_qty": 9,
                "receipt_fulfillment_rate": 0.9,
                "purchase_order_count": 1,
                "multi_supplier_flag": 0,
            }
        ]
    )

    def latest(source_key, *_args, **_kwargs):
        if source_key == "supplier":
            return supplier, "2026-09-30"
        return pd.DataFrame(), "2026-09-30"

    adapter._latest = latest
    resolved_ip = pd.DataFrame(
        [
            {"sku_code": "MAPPED", "ip_code": "迪士尼::米奇"},
            {"sku_code": "MAPPED", "ip_code": "迪士尼::米妮"},
        ]
    )
    restored = pd.DataFrame(
        [
            {
                "sku_code": "MAPPED",
                "store_tier": "1|A",
                "lifecycle_week": week,
                "v_neutral": 1.0,
                "n_eff": 7.0,
                "restored_exposure_days": 7.0,
            }
            for week in range(1, 11)
        ]
    )
    features, _ = adapter.factor_features(
        datetime.fromisoformat("2026-09-30T23:59:59+08:00"),
        sku_codes=["MAPPED", "UNKNOWN"],
        standard_ip_membership=resolved_ip,
        restored_weekly=restored,
    )
    by_sku = features.set_index("sku_code")

    assert by_sku.loc["MAPPED", "ip_breadth"] == 2
    assert by_sku.loc["MAPPED", "multi_supplier"] == 0
    assert by_sku.loc["MAPPED", "lifecycle_status"] == "补货期"
    assert by_sku.loc["MAPPED", "lifecycle_persistence"] == pytest.approx(0.6)
    assert (
        by_sku.loc["MAPPED", "lifecycle_persistence_status"]
        == "v3_2_pr04_weeks_5_10_over_1_10"
    )
    assert pd.isna(by_sku.loc["UNKNOWN", "ip_breadth"])
    assert pd.isna(by_sku.loc["UNKNOWN", "multi_supplier"])
    assert pd.isna(by_sku.loc["UNKNOWN", "lifecycle_persistence"])
    assert by_sku.loc["UNKNOWN", "ip_breadth_status"] == "missing_standard_ip"
    assert by_sku.loc["UNKNOWN", "multi_supplier_status"] == "missing_supplier_flag"
