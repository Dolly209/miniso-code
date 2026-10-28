from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from data.domestic import DataContractViolation
from training.checkpoint_replay import (
    align_candidate_cohort_to_schedule,
    attach_checkpoint_identity,
    candidate_pool_at_cutoff,
    normalize_replay_schedule,
    unique_cutoff_groups,
)


AVAILABLE_THROUGH = datetime.fromisoformat("2026-09-30T23:59:59+08:00")


def _schedule() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sku, split, first_sale in (
        ("A", "Train", "2025-09-01"),
        ("B", "Test", "2025-10-01"),
    ):
        for checkpoint, cutoff in (
            ("M5", "2025-03-01T23:59:59+08:00"),
            ("M4", "2025-04-01T23:59:59+08:00"),
            ("M3", "2025-05-01T23:59:59+08:00"),
        ):
            rows.append(
                {
                    "prediction_checkpoint_key": f"{sku}@{checkpoint}",
                    "prediction_checkpoint": checkpoint,
                    "lead_months": int(checkpoint[1:]),
                    "sku_code": sku,
                    "decision_cutoff_time": cutoff,
                    "decision_date": cutoff[:10],
                    "first_actual_sale_date": first_sale,
                    "release_dt": first_sale,
                    "split": split,
                    "dataset_version": "v-test",
                    "temporal_split_boundary_date": "2025-09-01",
                }
            )
    return pd.DataFrame(rows)


def _candidate_cohort() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sku_code": "OLD",
                "dataset_version": "v-test",
                "first_actual_sale_date": "2025-01-01",
                "split": "Train",
                "sales_26w_qty": 100.0,
                "long_tail_weight": 1.0,
            },
            {
                "sku_code": "TOO_RECENT",
                "dataset_version": "v-test",
                "first_actual_sale_date": "2025-04-20",
                "split": "Train",
                "sales_26w_qty": 500.0,
                "long_tail_weight": 1.0,
            },
            {
                "sku_code": "EARLIER_TEST",
                "dataset_version": "v-test",
                "first_actual_sale_date": "2025-02-01",
                "split": "Test",
                "sales_26w_qty": 10_000.0,
                "long_tail_weight": 1.0,
            },
        ]
    )


def test_schedule_selection_keeps_all_three_checkpoints_per_sku() -> None:
    selected = normalize_replay_schedule(
        _schedule(), available_through=AVAILABLE_THROUGH, limit_skus=1
    )

    assert selected["sku_code"].nunique() == 1
    assert set(selected["prediction_checkpoint"]) == {"M5", "M4", "M3"}
    assert len(list(unique_cutoff_groups(selected))) == 3


def test_schedule_rejects_incomplete_checkpoint_set() -> None:
    incomplete = _schedule().loc[
        lambda frame: ~(
            frame["sku_code"].eq("A")
            & frame["prediction_checkpoint"].eq("M4")
        )
    ]

    with pytest.raises(DataContractViolation, match="complete configured checkpoints"):
        normalize_replay_schedule(
            incomplete,
            available_through=AVAILABLE_THROUGH,
            sku_codes=["A"],
        )


def test_candidate_pool_requires_four_complete_weeks_before_cutoff() -> None:
    pool = candidate_pool_at_cutoff(
        _candidate_cohort(),
        cutoff_time=datetime.fromisoformat("2025-05-01T23:59:59+08:00"),
        minimum_mature_weeks=4,
    )

    assert set(pool["sku_code"]) == {"OLD", "EARLIER_TEST"}
    assert "TOO_RECENT" not in set(pool["sku_code"])


def test_candidate_split_and_weights_are_refit_from_final_schedule() -> None:
    source = _candidate_cohort().assign(
        dataset_version="old-version",
        split=["Test", "Test", "Test"],
        long_tail_weight=[99.0, 99.0, 99.0],
    )

    aligned, boundary = align_candidate_cohort_to_schedule(source, _schedule())

    assert str(boundary) == "2025-09-01"
    assert set(aligned["dataset_version"]) == {"v-test"}
    assert set(aligned["source_dataset_version"]) == {"old-version"}
    assert set(aligned["split"]) == {"Train"}
    assert aligned["long_tail_weight"].sum() == pytest.approx(1.0)
    assert not aligned["long_tail_weight"].eq(99.0).any()


def test_candidate_limit_is_deterministic_and_explicitly_smoke_only() -> None:
    pool = candidate_pool_at_cutoff(
        _candidate_cohort(),
        cutoff_time=datetime.fromisoformat("2025-05-01T23:59:59+08:00"),
        minimum_mature_weeks=4,
        candidate_limit=1,
    )

    assert pool["sku_code"].tolist() == ["OLD"]
    assert pool["split"].tolist() == ["Train"]


def test_checkpoint_identity_requires_exact_point_in_time_match() -> None:
    targets = _schedule().loc[
        lambda frame: frame["prediction_checkpoint_key"].eq("A@M5")
    ]
    output = pd.DataFrame(
        [
            {
                "sku_code": "A",
                "feature_cutoff_time": "2025-03-01T23:59:59+08:00",
                "feature": 1.0,
            }
        ]
    )

    attached = attach_checkpoint_identity(output, targets)
    assert attached.loc[0, "prediction_checkpoint_key"] == "A@M5"

    future = output.assign(
        feature_cutoff_time="2025-03-02T23:59:59+08:00"
    )
    with pytest.raises(DataContractViolation, match="does not equal"):
        attach_checkpoint_identity(future, targets)
