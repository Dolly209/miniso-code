from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from data.point_in_time import (
    DecisionType,
    PointInTimeContext,
    PointInTimeViolation,
)


def context() -> PointInTimeContext:
    return PointInTimeContext(
        run_id="run-synthetic-001",
        sku_code="SKU001",
        decision_type=DecisionType.FIRST_ORDER,
        decision_cutoff_time=datetime(2025, 1, 1, 12, tzinfo=timezone.utc),
        source_versions={"sales": "v1"},
    )


def test_available_before_cutoff_is_visible() -> None:
    point_in_time = context()
    assert point_in_time.is_visible(
        available_time=point_in_time.decision_cutoff_time - timedelta(seconds=1)
    )


def test_future_available_time_is_rejected_even_if_event_is_old() -> None:
    point_in_time = context()
    with pytest.raises(PointInTimeViolation):
        point_in_time.require_visible(
            event_time=point_in_time.decision_cutoff_time - timedelta(days=30),
            available_time=point_in_time.decision_cutoff_time + timedelta(seconds=1),
        )


def test_naive_cutoff_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        PointInTimeContext(
            run_id="run",
            sku_code="SKU001",
            decision_type=DecisionType.REORDER,
            decision_cutoff_time=datetime(2025, 1, 1),
        )


def test_partition_selection_never_uses_future_max_ds() -> None:
    point_in_time = context()
    selected = point_in_time.latest_partition_on_or_before(
        [date(2024, 12, 31), date(2025, 1, 1), date(2025, 1, 2)]
    )
    assert selected == date(2025, 1, 1)
