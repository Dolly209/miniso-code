from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from data.domestic import DataContractViolation
from data.domain import load_algorithm_parameters
from reorder import calculate_reorder
from scripts.run_reorder import (
    _derive_historical_cutoffs,
    _filter_point_in_time_actuals,
    _validate_target_coverage,
)


PARAMETERS = load_algorithm_parameters(
    Path(__file__).resolve().parents[1] / "config" / "algorithm_v3_2.json"
)


def _skus(*values: str) -> pd.DataFrame:
    return pd.DataFrame({"sku_code": list(values)})


def test_validate_target_coverage_accepts_matching_inputs() -> None:
    result = _validate_target_coverage(
        target_skus=["001", "002"],
        store_targets=_skus("001", "002", "002"),
        curve=_skus("001", "002"),
        actual=_skus("001", "001", "002"),
    )

    assert result == {
        "first_order_store_targets": 2,
        "demand_curve": 2,
        "actual_daily": 2,
    }


def test_validate_target_coverage_reports_all_missing_sources() -> None:
    with pytest.raises(DataContractViolation) as captured:
        _validate_target_coverage(
            target_skus=["001", "002", "003"],
            store_targets=_skus("001", "002"),
            curve=_skus("001", "003"),
            actual=_skus("001"),
        )

    message = str(captured.value)
    assert "first_order_store_targets: missing=1" in message
    assert "demand_curve: missing=1" in message
    assert "actual_daily: missing=2" in message


def test_historical_cutoff_uses_first_positive_sale_and_excludes_future() -> None:
    available = datetime(2026, 9, 30, 23, 59, 59, tzinfo=timezone(timedelta(hours=8)))
    actual = pd.DataFrame(
        [
            {"sku_code": "A", "business_day": "2026-01-01", "actual_sales": 0},
            {"sku_code": "A", "business_day": "2026-01-03", "actual_sales": 2},
            {"sku_code": "A", "business_day": "2026-01-17", "actual_sales": 3},
            {"sku_code": "A", "business_day": "2026-01-18", "actual_sales": 4},
        ]
    )

    cutoffs, audit = _derive_historical_cutoffs(
        actual=actual,
        target_skus=["A"],
        decision_age_days=14,
        available_through=available,
    )
    visible = _filter_point_in_time_actuals(actual, cutoffs)

    assert cutoffs["A"].date().isoformat() == "2026-01-17"
    assert audit.iloc[0]["first_actual_sale_date"] == "2026-01-03"
    assert visible["business_day"].tolist() == ["2026-01-01", "2026-01-03", "2026-01-17"]


def test_reorder_preserves_sales_for_unplanned_tier_and_zero_base_stays_zero() -> None:
    cutoff = datetime(2026, 1, 14, 23, 59, 59, tzinfo=timezone(timedelta(hours=8)))
    first = pd.DataFrame(
        [{"sku_code": "A", "Q_feas": 10, "planned_store_count": 1, "cover_weeks": 2}]
    )
    store_targets = pd.DataFrame(
        [
            {
                "sku_code": "A",
                "store_code": "S1",
                "store_tier": "1|A",
                "neutral_demand": 10,
                "sellable_demand": 10,
                "safety_stock": 0,
            }
        ]
    )
    actual = pd.DataFrame(
        [
            {
                "sku_code": "A",
                "store_code": "S1",
                "store_tier": "1|A",
                "business_day": "2026-01-14",
                "actual_sales": 1,
                "e_stock": 1,
                "e_display": 1,
                "available_stock_qty": 100,
            },
            {
                "sku_code": "A",
                "store_code": "S2",
                "store_tier": "unplanned",
                "business_day": "2026-01-14",
                "actual_sales": 5,
                "e_stock": 0,
                "e_display": 0,
                "available_stock_qty": 100,
            },
        ]
    )
    curve = pd.DataFrame(
        [
            {
                "sku_code": "A",
                "store_tier": "1|A",
                "lifecycle_week": week,
                "lifecycle_factor": 1.0,
                "period_factor_h": 1.0,
            }
            for week in range(1, 9)
        ]
    )
    constraints = pd.DataFrame(
        [
            {
                "sku_code": "A",
                "launch_date": "2026-01-01",
                "reorder_moq": 1,
                "case_pack": 1,
                "reorder_lead_days": 1,
                "next_review_days": 7,
            }
        ]
    )

    result = calculate_reorder(
        decision_cutoff_time=cutoff,
        first_order_decisions=first,
        first_order_store_targets=store_targets,
        actual_daily=actual,
        demand_curve=curve,
        config=PARAMETERS["reorder"],
        constraints=constraints,
    )

    unplanned = result.diagnostics.loc[result.diagnostics["store_tier"].eq("unplanned")].iloc[0]
    decision = result.decisions.iloc[0]
    assert unplanned["actual_demand"] == unplanned["actual_sales"] == 5
    assert decision["QR_base"] == 0
    assert decision["QR_plan"] == 0
    assert decision["QR_feas"] == 0
    assert result.demand_forecast["window_demand_contribution"].sum() == pytest.approx(
        decision["D_window"]
    )
