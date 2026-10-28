from __future__ import annotations

from datetime import date

import pandas as pd
import pyarrow as pa

from data.decision_adapter import SOURCES
from data.prelaunch_row_audit import (
    PrelaunchRowAuditAdapter,
    _temporal_resplit,
    select_partition_replay_candidates,
)


class _Repository:
    def __init__(self) -> None:
        self.rows = {
            "product": pd.DataFrame(
                [{"ds": "2025-01-01", "sku_code": "A"}]
            ),
            "store": pd.DataFrame(
                [{"ds": "2025-01-01", "store_code": "S1"}]
            ),
            "pack": pd.DataFrame(
                [{"ds": "2025-01-02", "sku_code": "A"}]
            ),
            "supplier": pd.DataFrame(
                [
                    {
                        "ds": "2025-01-02",
                        "sku_code": "A",
                        "supplier_code": "SUP1",
                    }
                ]
            ),
            "member": pd.DataFrame(
                [{"ds": "2025-01-03", "sku_code": "A", "stat_month": "2025-01"}]
            ),
            "sales": pd.DataFrame(
                [{"ds": "2025-01-05", "sku_code": "A"}]
            ),
            "inventory": pd.DataFrame(columns=["ds", "sku_code"]),
            "placement": pd.DataFrame(columns=["ds", "sku_code"]),
            "social": pd.DataFrame(
                [
                    {
                        "ds": "2025-01-09",
                        "date": "2025-01-09",
                        "ip_name": "社媒热门IP",
                        "platform_code": "douyin",
                    }
                ]
            ),
        }

    def available_partitions(self, source_name: str) -> tuple[date, ...]:
        return tuple(date(2025, 1, day) for day in range(1, 32))

    def iter_partition_batches(self, *, source_name: str, **_: object):
        reverse = {value: key for key, value in SOURCES.items()}
        frame = self.rows[reverse[source_name]]
        if frame.empty:
            return iter(())
        return iter((pa.RecordBatch.from_pandas(frame, preserve_index=False),))


def _schedule() -> pd.DataFrame:
    rows = []
    for sku in ("A", "B"):
        for lead, day in ((5, 10), (4, 20), (3, 30)):
            rows.append(
                {
                    "prediction_checkpoint_key": f"{sku}@M{lead}",
                    "sku_code": sku,
                    "split": "Train",
                    "release_dt": "2025-06-30",
                    "first_actual_sale_date": "2025-06-30",
                    "sales_26w_qty": 100,
                    "lead_months": lead,
                    "prediction_checkpoint": f"M{lead}",
                    "decision_date": f"2025-01-{day:02d}",
                    "decision_cutoff_time": (
                        f"2025-01-{day:02d}T23:59:59+08:00"
                    ),
                    "within_available_through": True,
                    "product_snapshot_partition": "2026-09-30",
                    "product_snapshot_status": "latest_partition",
                }
            )
    return pd.DataFrame(rows)


def _mapping() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sku_code": "A",
                "standard_ip_code": "IP-A",
                "standard_ip_name": "热门IP",
                "standard_ip_level1": "热门IP",
                "standard_ip_level2": "",
                "path1_eligible": 1,
            }
        ]
    )


def _social_alias() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "standard_ip_code": "IP-A",
                "social_ip_name": "社媒热门IP",
                "match_method": "reviewed_exact_alias",
                "enabled": 1,
            }
        ]
    )


def test_row_audit_uses_entity_last_value_and_does_not_require_prelaunch_sales() -> None:
    result = PrelaunchRowAuditAdapter(  # type: ignore[arg-type]
        _Repository(), batch_size=100
    ).audit(
        _schedule(),
        standard_ip_mapping=_mapping(),
        social_ip_alias_mapping=_social_alias(),
        social_window_days=30,
    )

    checkpoint = result.checkpoint_summary.set_index("prediction_checkpoint_key")
    assert bool(checkpoint.loc["A@M5", "core_row_ready_for_feature_replay"])
    assert bool(checkpoint.loc["B@M5", "core_row_ready_for_feature_replay"])
    assert checkpoint.loc["A@M5", "predecision_target_history_rows"] == 1

    detail = result.source_detail.set_index(
        ["prediction_checkpoint_key", "source_key"]
    )
    assert detail.loc[("A@M3", "product"), "last_observed_date"] == "2026-09-30"
    assert detail.loc[("A@M3", "product"), "row_audit_status"] == (
        "fixed_product_snapshot_available"
    )
    assert detail.loc[("A@M5", "sales"), "row_audit_status"].startswith(
        "target_rows_before"
    )
    assert detail.loc[("A@M5", "social"), "row_count"] == 1
    assert detail.loc[("B@M5", "social"), "row_audit_status"] == (
        "optional_no_reviewed_standard_ip_alias"
    )


def test_temporal_resplit_keeps_same_launch_day_together() -> None:
    rows = []
    launch_dates = [
        "2025-06-01",
        "2025-07-01",
        "2025-08-01",
        "2025-09-01",
        "2025-10-01",
        "2025-11-01",
        "2025-12-01",
        "2026-01-01",
        "2026-01-01",
        "2026-02-01",
    ]
    for index, launch in enumerate(launch_dates):
        for lead in (3, 4, 5):
            rows.append(
                {
                    "sku_code": f"S{index}",
                    "first_actual_sale_date": launch,
                    "split": "OldTrain" if index < 8 else "OldTest",
                    "sales_26w_qty": index + 1,
                    "lead_months": lead,
                    "decision_date": "2025-01-01",
                    "within_available_through": True,
                    "all_audited_sources_available": True,
                    "core_row_ready_for_feature_replay": True,
                    "label_available": True,
                }
            )
    _, resplit = _temporal_resplit(
        pd.DataFrame(rows), expected_lead_months=(3, 4, 5)
    )

    same_day = resplit.loc[resplit["first_actual_sale_date"].eq("2026-01-01")]
    assert same_day["split"].nunique() == 1
    assert set(resplit["split"]) == {"Train", "Test"}
    assert resplit["formal_replay_eligible"].all()
    assert set(resplit["eligibility_scope"]) == {
        "formal_for_historical_feature_replay"
    }
    assert not resplit["training_ready"].any()
    # 7/3 and 9/1 are equally close to 80/20. Keep the earlier date boundary,
    # as the universe builder does, instead of always rounding up a tied day.
    assert resplit["split"].value_counts().to_dict() == {"Train": 7, "Test": 3}


def test_partition_candidates_require_all_nodes_before_sale_and_mature_label() -> None:
    rows = []
    for sku, first_sale, partitions_ok in (
        ("READY", "2025-08-01", True),
        ("PARTITION_GAP", "2025-08-01", False),
        ("TIME_TRAVEL", "2025-01-15", True),
        ("IMMATURE", "2026-08-01", True),
    ):
        for lead, decision in (
            (5, "2025-01-01"),
            (4, "2025-02-01"),
            (3, "2025-03-01"),
        ):
            rows.append(
                {
                    "prediction_checkpoint_key": f"{sku}@M{lead}",
                    "sku_code": sku,
                    "first_actual_sale_date": first_sale,
                    "sales_26w_qty": 100,
                    "lead_months": lead,
                    "decision_date": decision,
                    "within_available_through": True,
                    "all_audited_sources_available": (
                        partitions_ok and not (sku == "PARTITION_GAP" and lead == 5)
                    ),
                }
            )

    result = select_partition_replay_candidates(
        pd.DataFrame(rows),
        expected_lead_months=(3, 4, 5),
        label_as_of_date=date(2026, 9, 30),
    )

    assert set(result.schedule["sku_code"]) == {"READY"}
    audit = result.sku_audit.set_index("sku_code")
    assert bool(audit.loc["READY", "partition_replay_candidate"])
    assert "one_or_more_source_partitions_unavailable" in audit.loc[
        "PARTITION_GAP", "partition_ineligibility_reasons"
    ]
    assert "decision_not_strictly_before_first_sale" in audit.loc[
        "TIME_TRAVEL", "partition_ineligibility_reasons"
    ]
    assert "sales_26w_label_not_mature" in audit.loc[
        "IMMATURE", "partition_ineligibility_reasons"
    ]
