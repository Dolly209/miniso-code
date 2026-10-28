from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from data.prelaunch_training_audit import audit_prelaunch_source_partitions
from training.decision_schedule import build_prelaunch_decision_schedule


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sku_code": "A",
                "release_dt": "2026-06-30",
                "first_actual_sale_date": "2026-05-31",
                "split": "Train",
                "time_block": "B1",
            },
            {
                "sku_code": "B",
                "release_dt": "2026-07-15",
                "first_actual_sale_date": "2026-06-15",
                "split": "Test",
                "time_block": "",
            },
        ]
    )


class _Repository:
    def available_partitions(self, source_name: str) -> tuple[date, ...]:
        if source_name.endswith("dim_goods_info_ip_forecast"):
            return (date(2026, 1, 1), date(2026, 2, 28), date(2026, 3, 31))
        return (date(2026, 3, 1), date(2026, 3, 31))


def test_schedule_uses_release_date_calendar_months_and_preserves_sku_split() -> None:
    schedule = build_prelaunch_decision_schedule(
        _cohort(),
        lead_months=(3, 4, 5),
        available_through=datetime(
            2026, 9, 30, 23, 59, 59, tzinfo=timezone(timedelta(hours=8))
        ),
    )

    assert len(schedule) == 6
    sku_a = schedule.loc[schedule["sku_code"].eq("A")].set_index("lead_months")
    assert sku_a.loc[3, "decision_date"] == "2026-03-30"
    assert sku_a.loc[4, "decision_date"] == "2026-02-28"
    assert sku_a.loc[5, "decision_date"] == "2026-01-30"
    assert set(sku_a["release_dt"]) == {"2026-06-30"}
    assert set(sku_a["decision_time_source"]) == {
        "release_dt_from_fixed_product_snapshot_minus_calendar_months"
    }
    assert set(sku_a["split"]) == {"Train"}
    assert "node_weight_share" not in schedule
    assert set(schedule["forecast_scope"]) == {"complete_first_order_demand"}
    assert schedule["prediction_checkpoint_key"].is_unique
    assert schedule["within_available_through"].all()


def test_partition_audit_flags_nodes_before_a_source_begins() -> None:
    schedule = build_prelaunch_decision_schedule(
        _cohort(),
        lead_months=(3, 4, 5),
    )
    audit = audit_prelaunch_source_partitions(
        _Repository(),  # type: ignore[arg-type]
        schedule,
        source_keys=("product", "sales"),
    )

    sku_a_m5 = audit.schedule.set_index("prediction_checkpoint_key").loc["A@M5"]
    assert sku_a_m5["missing_sources"] == "sales"
    assert not bool(sku_a_m5["all_audited_sources_available"])
    sku_b_m3 = audit.schedule.set_index("prediction_checkpoint_key").loc["B@M3"]
    assert bool(sku_b_m3["all_audited_sources_available"])
    assert len(audit.source_detail) == len(schedule) * 2
    assert set(audit.source_summary["source_key"]) == {"product", "sales"}
