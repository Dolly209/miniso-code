from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from config.settings import TableSpec
from data.decision_adapter import SOURCES
from data.domestic import DataContractViolation
from scripts.build_historical_restore import (
    _read_cohort,
    _restoration_probe_specs,
    parse_args,
)


def _write_cohort(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "dataset_version": "test-v1",
                "sku_code": "TEST-SKU-001",
                "first_actual_sale_date": "2025-10-17",
                "split": "Train",
                "long_tail_weight": 0.5,
            },
            {
                "dataset_version": "test-v1",
                "sku_code": "TEST-SKU-002",
                "first_actual_sale_date": "2025-11-03",
                "split": "Test",
                "long_tail_weight": None,
            },
        ]
    ).to_csv(path, index=False)


def test_explicit_sku_selection_accepts_train_and_test_and_preserves_order(
    tmp_path: Path,
) -> None:
    cohort_path = tmp_path / "cohort.csv"
    _write_cohort(cohort_path)

    selected = _read_cohort(
        cohort_path,
        [" TEST-SKU-002 ", "TEST-SKU-001"],
        limit=1,
    )

    assert selected["sku_code"].tolist() == ["TEST-SKU-002", "TEST-SKU-001"]
    assert selected["split"].tolist() == ["Test", "Train"]


def test_explicit_sku_selection_rejects_test_only_sample(tmp_path: Path) -> None:
    cohort_path = tmp_path / "cohort.csv"
    _write_cohort(cohort_path)

    with pytest.raises(DataContractViolation, match="at least one Train SKU"):
        _read_cohort(cohort_path, ["TEST-SKU-002"], limit=5)


def test_frozen_model_application_accepts_test_only_targets(tmp_path: Path) -> None:
    cohort_path = tmp_path / "cohort.csv"
    _write_cohort(cohort_path)

    selected = _read_cohort(
        cohort_path,
        ["TEST-SKU-002"],
        limit=5,
        require_train=False,
    )

    assert selected["sku_code"].tolist() == ["TEST-SKU-002"]
    assert selected["split"].tolist() == ["Test"]


def test_explicit_test_targets_can_prepend_train_fit_anchors(tmp_path: Path) -> None:
    cohort_path = tmp_path / "cohort.csv"
    _write_cohort(cohort_path)

    selected = _read_cohort(
        cohort_path,
        ["TEST-SKU-002"],
        limit=5,
        fit_sku_limit=1,
    )

    assert selected["sku_code"].tolist() == ["TEST-SKU-001", "TEST-SKU-002"]
    assert selected["split"].tolist() == ["Train", "Test"]


def test_checkpoint_schedule_collapses_identical_sku_rows(tmp_path: Path) -> None:
    cohort_path = tmp_path / "checkpoint_schedule.csv"
    base = pd.DataFrame(
        [
            {
                "dataset_version": "test-v1",
                "sku_code": "TEST-SKU-001",
                "first_actual_sale_date": "2025-10-17",
                "split": "Train",
                "long_tail_weight": 1.0,
                "prediction_checkpoint": checkpoint,
            }
            for checkpoint in ("M5", "M4", "M3")
        ]
    )
    base.to_csv(cohort_path, index=False)

    selected = _read_cohort(
        cohort_path,
        ["TEST-SKU-001"],
        limit=5,
    )

    assert len(selected) == 1
    assert selected.iloc[0]["sku_code"] == "TEST-SKU-001"


def test_checkpoint_schedule_rejects_conflicting_sku_rows(tmp_path: Path) -> None:
    cohort_path = tmp_path / "checkpoint_schedule.csv"
    pd.DataFrame(
        [
            {
                "dataset_version": "test-v1",
                "sku_code": "TEST-SKU-001",
                "first_actual_sale_date": "2025-10-17",
                "split": split,
                "long_tail_weight": 1.0,
            }
            for split in ("Train", "Test")
        ]
    ).to_csv(cohort_path, index=False)

    with pytest.raises(DataContractViolation, match="conflicting duplicate SKU"):
        _read_cohort(cohort_path, ["TEST-SKU-001"], limit=5)


def test_round4_probe_scope_excludes_unread_social_source() -> None:
    names = (
        SOURCES["product"],
        SOURCES["store"],
        SOURCES["sales"],
        SOURCES["inventory"],
        SOURCES["placement"],
        SOURCES["social"],
    )
    specs = [
        TableSpec(table_name=name, required_fields=("ds",), partition_column="ds")
        for name in names
    ]

    selected = _restoration_probe_specs(specs)

    selected_names = [spec.table_name for spec in selected]
    assert SOURCES["social"] not in selected_names
    assert selected_names == list(names[:-1])


def test_round4_cli_accepts_b1_oof_seed_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_restore.py",
            "--data-root",
            str(tmp_path),
            "--decision-cutoff-time",
            "2025-03-08T23:59:59+08:00",
            "--restoration-model-file",
            str(tmp_path / "restoration_model.json"),
            "--model-application-scope",
            "oof_seed",
        ],
    )

    args = parse_args()

    assert args.model_application_scope == "oof_seed"
