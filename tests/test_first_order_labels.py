from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from data.first_order_label_adapter import FirstOrderLabelDataAdapter
from data.parquet_repository import ReadResult
from training.first_order_labels import (
    actual_sales_weekly_from_raw,
    aggregate_weekly_label_inputs,
    build_first_order_wcover_labels,
    build_wcover_windows,
)


CONFIG = {
    "decision_weeks": 2,
    "buffer_weeks": 2,
    "lead_weeks": 6,
    "lifecycle_weeks": 26,
}


def _schedule() -> pd.DataFrame:
    rows = []
    for sku, first_sale, sales_26w in (
        ("A", "2025-01-01", 400.0),
        ("B", "2025-02-01", 900.0),
        ("C", "2025-03-01", 12_000.0),
        ("D", "2025-04-01", 20_000.0),
    ):
        for lead in (5, 4, 3):
            rows.append(
                {
                    "dataset_version": "v1",
                    "sku_code": sku,
                    "first_actual_sale_date": first_sale,
                    "sales_26w_qty": sales_26w,
                    "lead_months": lead,
                    "prediction_checkpoint": f"M{lead}",
                    "prediction_checkpoint_key": f"{sku}@M{lead}",
                    "decision_cutoff_time": f"2024-0{6-lead}-01T23:59:59+08:00",
                }
            )
    return pd.DataFrame(rows)


def _weekly() -> pd.DataFrame:
    rows = []
    for sku_index, sku in enumerate(("A", "B", "C", "D"), start=1):
        first = pd.Timestamp(
            {"A": "2025-01-01", "B": "2025-02-01", "C": "2025-03-01", "D": "2025-04-01"}[sku]
        )
        for week in range(1, 11):
            rows.append(
                {
                    "sku_code": sku,
                    "store_code": "S1",
                    "lifecycle_week": week,
                    "week_end": first + timedelta(days=week * 7 - 1),
                    "clean_sales_qty": float(sku_index),
                    # Deliberately different: formal demand must use restoration.
                    "restored_natural_sales": float(sku_index + 10),
                    "restored_lost_sales": 10.0,
                    "stock_grain_unreliable": sku == "B",
                    "unknown_exposure_days": 0.0,
                    "source_day_rows": 7.0,
                    "restored_exposure_days": 7.0,
                    "long_stockout_quality_weight": 1.0,
                    "complete_week": True,
                }
            )
    return pd.DataFrame(rows)


def test_wcover_label_uses_restored_demand_and_excludes_unrestorable_sku() -> None:
    weekly = aggregate_weekly_label_inputs(_weekly())
    result = build_first_order_wcover_labels(
        schedule=_schedule(),
        weekly=weekly,
        first_order_config=CONFIG,
        available_through=datetime(2026, 9, 30, 23, 59, tzinfo=timezone.utc),
    )
    labels = result.labels.set_index("sku_code")

    assert labels.loc["A", "W_cover"] == 10
    assert labels.loc["A", "D_first_ref"] == 110.0
    assert labels.loc["A", "observed_clean_sales_wcover"] == 10.0
    assert labels.loc["A", "restored_demand_wcover"] == 110.0
    assert labels.loc["A", "restoration_uplift_wcover"] == 100.0
    assert labels.loc["B", "D_first_ref"] == 120.0
    assert not labels.loc["B", "label_eligible"]
    assert labels.loc["B", "label_quality_status"] == "held_pending_inventory_risk_review"
    assert labels.loc["B", "inventory_risk_pending_review"]
    assert not labels["inventory_confirmed_unrestorable"].any()
    assert result.formal_schedule.groupby("sku_code").size().eq(3).all()
    assert set(result.formal_schedule["sku_code"]) == {"A", "C", "D"}
    assert result.report["train_skus"] == 2
    assert result.report["test_skus"] == 1


def test_wcover_is_minimum_of_supply_lifecycle_and_hard_window() -> None:
    constraints = pd.DataFrame(
        [
            {
                "sku_code": "A",
                "decision_weeks": 2,
                "buffer_weeks": 1,
                "lead_weeks": 3,
                "lifecycle_weeks": 5,
                "hard_life_weeks": 4,
            }
        ]
    )
    result = build_first_order_wcover_labels(
        schedule=_schedule(),
        weekly=aggregate_weekly_label_inputs(_weekly()),
        first_order_config=CONFIG,
        available_through=datetime(2026, 9, 30, 23, 59, tzinfo=timezone.utc),
        constraints=constraints,
    )
    row = result.labels.set_index("sku_code").loc["A"]

    assert row["W_supply"] == 6
    assert row["W_life"] == 5
    assert row["W_hard"] == 4
    assert row["W_cover"] == 4
    assert row["D_first_ref"] == 44.0


def test_sales_only_label_path_zero_fills_weeks_without_orders() -> None:
    schedule = _schedule().loc[lambda frame: frame["sku_code"].eq("A")]
    windows = build_wcover_windows(
        schedule=schedule,
        first_order_config=CONFIG,
    )
    raw = pd.DataFrame(
        [
            {
                "ds": "2025-01-01",
                "order_id": "O1",
                "order_detail_id": "L1",
                "order_time": "2025-01-01 10:00:00",
                "sku_code": "A",
                "store_code": "S1",
                "sale_qty": 2,
                "order_amt": 20,
                "paid_amt": 20,
                "unit_paid_price": 10,
                "achievement_amt": 20,
                "is_return": "0",
                "gift_flag": "N",
            },
            {
                "ds": "2025-01-08",
                "order_id": "O2",
                "order_detail_id": "L2",
                "order_time": "2025-01-08 10:00:00",
                "sku_code": "A",
                "store_code": "S1",
                "sale_qty": -1,
                "order_amt": -10,
                "paid_amt": -10,
                "unit_paid_price": 10,
                "achievement_amt": -10,
                "is_return": "0",
                "gift_flag": "N",
            },
        ]
    )
    weekly, audit = actual_sales_weekly_from_raw(
        raw_sales=raw,
        windows=windows,
        cutoff_time=datetime(2026, 9, 30, 23, 59, tzinfo=timezone.utc),
    )

    assert len(weekly) == 10
    assert weekly.loc[weekly["lifecycle_week"].eq(1), "clean_sales_qty"].item() == 2
    assert weekly.loc[weekly["lifecycle_week"].eq(2), "clean_sales_qty"].item() == -1
    assert weekly.loc[weekly["lifecycle_week"].ge(3), "clean_sales_qty"].eq(0).all()
    assert audit["latest_order_lines"] == 2


def test_label_adapter_scans_only_sales_source() -> None:
    class FakeRepository:
        max_rows_per_read = 100

        def __init__(self) -> None:
            self.sources: list[str] = []

        def read_partition_range(self, **kwargs: object) -> ReadResult:
            self.sources.append(str(kwargs["source_name"]))
            return ReadResult(
                source_name=str(kwargs["source_name"]),
                frame=pd.DataFrame(),
                selected_partitions=(pd.Timestamp("2025-01-01").date(),),
                projected_columns=(),
                sku_filter=None,
            )

    repository = FakeRepository()
    adapter = FirstOrderLabelDataAdapter(repository)  # type: ignore[arg-type]
    windows = build_wcover_windows(
        schedule=_schedule().loc[lambda frame: frame["sku_code"].eq("A")],
        first_order_config=CONFIG,
    )
    batches = list(
        adapter.iter_sales_batches(
            windows=windows,
            cutoff_time=datetime(2026, 9, 30, 23, 59, tzinfo=timezone.utc),
            batch_size=25,
        )
    )

    assert len(batches) == 1
    assert repository.sources == ["scm.dwd_trd_order_all_flow_di_ip_forecast"]
