from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.contracts import AvailabilityStatus, UnavailableFieldDecision


def test_proxy_requires_explicit_physical_field() -> None:
    with pytest.raises(ValueError, match="physical proxy field"):
        UnavailableFieldDecision(
            concept="planned_distribution_qty",
            status=AvailabilityStatus.PROXY,
            reason="synthetic test",
        )


def test_missing_decision_cannot_smuggle_in_substitute_field() -> None:
    with pytest.raises(ValueError, match="only valid for proxy"):
        UnavailableFieldDecision(
            concept="capacity_calendar",
            status=AvailabilityStatus.MISSING,
            reason="not delivered",
            physical_field="looks_similar",
        )


def test_v3_2_availability_registry_uses_only_explicit_statuses() -> None:
    project_root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (project_root / "config" / "data_availability_v3_2.json").read_text(
            encoding="utf-8"
        )
    )
    for row in payload["decisions"]:
        decision = UnavailableFieldDecision(
            concept=row["concept"],
            status=AvailabilityStatus(row["status"]),
            reason=row["reason"],
            physical_field=row["physical_field"],
        )
        assert decision.status.value in {
            "missing",
            "proxy",
            "manual/config",
            "neutral",
            "disabled",
        }
