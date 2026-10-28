from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
import pandas as pd

from scripts.replay_prelaunch_features import (
    _attach_restoration_models,
    _compact_completed_group,
    _model_aware_groups,
    _time_spread_split_skus,
)


def test_completed_group_compaction_preserves_downstream_and_report(
    tmp_path: Path,
) -> None:
    group = tmp_path / "group"
    round4 = group / "round4"
    round4.mkdir(parents=True)
    report = {"status": "ok", "data_quality": {"warnings": []}}
    (round4 / "round4_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    (round4 / "large_consumed_rows.parquet").write_bytes(b"x" * 4096)
    for stage, name in (
        ("round5", "comparable_summary.csv"),
        ("round6", "placement_summary.csv"),
        ("round7", "demand_adjustments.csv"),
        ("round7", "transformed_features.csv"),
    ):
        path = group / stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sku_code\nSKU1\n", encoding="utf-8")
    checkpoint = group / "round6" / "_checkpoints" / "checkpoint.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"y" * 1024)

    before, after = _compact_completed_group(group)

    assert before > after
    assert not round4.exists()
    assert not checkpoint.parent.exists()
    assert json.loads((group / "round4_report.json").read_text()) == report
    assert (group / "round5" / "comparable_summary.csv").is_file()
    assert (group / "round6" / "placement_summary.csv").is_file()
    assert (group / "round7" / "demand_adjustments.csv").is_file()
    assert (group / "round7" / "transformed_features.csv").is_file()


def test_time_spread_selection_preserves_locked_split_and_endpoints() -> None:
    schedule = pd.DataFrame(
        [
            {
                "sku_code": f"T{index}",
                "split": "Train",
                "first_actual_sale_date": f"2025-01-{index + 1:02d}",
            }
            for index in range(5)
        ]
        + [
            {
                "sku_code": f"E{index}",
                "split": "Test",
                "first_actual_sale_date": f"2026-01-{index + 1:02d}",
            }
            for index in range(3)
        ]
    )

    assert _time_spread_split_skus(schedule, split="Train", limit=3) == [
        "T0",
        "T2",
        "T4",
    ]
    assert _time_spread_split_skus(schedule, split="Test", limit=2) == [
        "E0",
        "E2",
    ]


def test_model_registry_is_point_in_time_checked_and_groups_by_model(
    tmp_path: Path,
) -> None:
    for name in ("fold1.json", "fold2.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    registry = pd.DataFrame(
        [
            {
                "sku_code": "A",
                "model_id": "fold-1",
                "model_file": "fold1.json",
                "model_train_end_time": "2025-01-01T23:59:59+08:00",
                "usage_scope": "oof",
            },
            {
                "sku_code": "B",
                "model_id": "fold-2",
                "model_file": "fold2.json",
                "model_train_end_time": "2025-02-01T23:59:59+08:00",
                "usage_scope": "locked_test",
            },
        ]
    )
    registry_path = tmp_path / "registry.csv"
    registry.to_csv(registry_path, index=False)
    schedule = pd.DataFrame(
        [
            {
                "sku_code": "A",
                "decision_cutoff_time": "2025-03-01T23:59:59+08:00",
            },
            {
                "sku_code": "B",
                "decision_cutoff_time": "2025-03-01T23:59:59+08:00",
            },
        ]
    )

    attached = _attach_restoration_models(schedule, registry_path)
    groups = _model_aware_groups(attached)

    assert len(groups) == 2
    assert {group[3] for group in groups} == {"fold-1", "fold-2"}
