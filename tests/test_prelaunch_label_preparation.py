from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from data.domestic import DataContractViolation
from data.prelaunch_artifacts import (
    merge_row_audit_schedule, normalize_prelaunch_audit_metadata,
    validate_label_restoration_provenance,
)
from data.launch_universe import build_verified_launch_universe
from training.first_order_labels import (
    aggregate_weekly_label_inputs, build_first_order_wcover_labels,
    build_wcover_windows, combine_weekly_label_inputs,
)
from test_first_order_labels import CONFIG, _schedule, _weekly
from test_verified_launch_universe import CUTOFF, inputs


def versioned_inputs():
    schedule = build_verified_launch_universe(**inputs()).schedule
    windows = build_wcover_windows(schedule=schedule, first_order_config=CONFIG)
    report = dict(
        round3_dataset_version=schedule.iloc[0].dataset_version,
        dataset_version="m2-outcome-label-source", decision_cutoff_time=CUTOFF.isoformat(),
        product_snapshot_time=CUTOFF.isoformat(),
        model_execution_mode="legacy_fit_and_apply",
        selected_skus=[dict(sku_code=row.sku_code, start_date=str(row.label_start_date), end_date=str(row.label_end_date))
                       for row in windows.itertuples(index=False)],
    )
    return schedule, windows, report


def test_old_suffixes_are_readable_and_current_label_boundary_wins():
    old = pd.DataFrame(dict(
        training_ready_x=[False], training_ready_y=[False],
        temporal_split_boundary_date_x=["2025-12-30"], temporal_split_boundary_date_y=["2025-12-30"],
        eligibility_scope_x=["preflight"], eligibility_scope_y=["row_audit"],
    ))
    new = normalize_prelaunch_audit_metadata(old)
    assert not any(key.endswith(("_x", "_y")) for key in new)
    assert new.iloc[0].eligibility_scope == "row_audit"
    assert new.iloc[0].preflight_eligibility_scope == "preflight"
    assert normalize_prelaunch_audit_metadata(old.assign(temporal_split_boundary_date="2026-01-01")).iloc[0].temporal_split_boundary_date == "2026-01-01"
    pd.testing.assert_frame_equal(normalize_prelaunch_audit_metadata(new), new)


def test_new_row_audit_merge_emits_canonical_fields_without_rescanning_data():
    schedule, _, _ = versioned_inputs()
    split = schedule[["sku_code", "split"]].assign(
        training_ready=False, eligibility_scope="formal_for_historical_feature_replay",
        temporal_split_boundary_date="2025-09-01",
    )
    merged = merge_row_audit_schedule(schedule, split)
    assert not any(key.endswith(("_x", "_y")) for key in merged)
    assert merged.eligibility_scope.eq("formal_for_historical_feature_replay").all()
    assert set(merged.launch_universe_version) == set(schedule.launch_universe_version)


def test_provenance_checks_full_per_sku_origin_and_window_without_loading_facts():
    schedule, windows, report = versioned_inputs()
    result = validate_label_restoration_provenance(schedule=schedule, windows=windows, report=report, available_through=CUTOFF)
    assert result["validated_skus"] == 3
    assert result["restoration_dataset_version"] == "m2-outcome-label-source"


@pytest.mark.parametrize("problem", ["missing_report", "version", "cutoff", "snapshot", "old_origin", "short_window", "fit_only", "missing_sku"])
def test_wrong_restoration_sources_are_rejected_before_streaming_large_artifacts(problem):
    schedule, windows, report = versioned_inputs()
    if problem == "missing_report": report = None
    elif problem == "version": report["round3_dataset_version"] = "old-universe"
    elif problem == "cutoff": report["decision_cutoff_time"] = "2026-10-08T23:59:59+08:00"
    elif problem == "snapshot": report["product_snapshot_time"] = "2026-10-08T23:59:59+08:00"
    elif problem == "old_origin": report["selected_skus"][0]["start_date"] = "2025-01-01"
    elif problem == "short_window": report["selected_skus"][0]["end_date"] = report["selected_skus"][0]["start_date"]
    elif problem == "fit_only": report["model_execution_mode"] = "fit_train_only"
    elif problem == "missing_sku": report["selected_skus"] = report["selected_skus"][1:]
    with pytest.raises(DataContractViolation):
        validate_label_restoration_provenance(schedule=schedule, windows=windows, report=report, available_through=CUTOFF)


def test_window_alignment_catches_a_bad_store_even_when_max_week_end_looks_valid():
    raw = _weekly()
    extra = raw.loc[raw.sku_code.eq("A") & raw.lifecycle_week.eq(1)].copy()
    extra["store_code"] = "S2"
    extra["week_end"] -= timedelta(days=1)
    grouped = combine_weekly_label_inputs([aggregate_weekly_label_inputs(raw), aggregate_weekly_label_inputs(extra)])
    result = build_first_order_wcover_labels(schedule=_schedule(), weekly=grouped, first_order_config=CONFIG, available_through=CUTOFF)
    labels = result.labels.set_index("sku_code")
    assert labels.loc["A", "origin_mismatch_weeks"] == 1
    assert not labels.loc["A", "label_eligible"]
    assert labels.loc["A", "label_quality_status"] == "restoration_lifecycle_origin_mismatch"


@pytest.mark.parametrize("bad_value", [None, float("inf")])
def test_missing_restored_demand_is_not_filled_with_zero(bad_value):
    raw = _weekly()
    raw.loc[0, "restored_natural_sales"] = bad_value
    with pytest.raises(ValueError, match="invalid"):
        aggregate_weekly_label_inputs(raw)


def test_insufficient_eligible_skus_preserves_quality_audit_but_emits_no_training_schedule():
    weekly = _weekly().assign(stock_grain_unreliable=True)
    result = build_first_order_wcover_labels(schedule=_schedule(), weekly=aggregate_weekly_label_inputs(weekly), first_order_config=CONFIG, available_through=CUTOFF)
    assert len(result.labels) == len(result.excluded_skus) == 4
    assert result.formal_schedule.empty
    assert result.report["split_ready"] is False
    assert result.report["train_skus"] == result.report["test_skus"] == 0


def test_safety_and_display_policies_do_not_change_the_sales_demand_label():
    weekly = aggregate_weekly_label_inputs(_weekly())
    base = build_first_order_wcover_labels(schedule=_schedule(), weekly=weekly, first_order_config=CONFIG, available_through=CUTOFF)
    policy = dict(CONFIG, minimum_display_units=1000, minimum_store_units=1000, safety_stock_quantile_by_tier={"S": .999})
    changed = build_first_order_wcover_labels(schedule=_schedule(), weekly=weekly, first_order_config=policy, available_through=CUTOFF)
    pd.testing.assert_series_equal(base.labels.D_first_ref, changed.labels.D_first_ref)
    pd.testing.assert_series_equal(base.labels.W_cover, changed.labels.W_cover)


def test_cli_streams_weekly_labels_and_normalizes_old_audit_fields(tmp_path, monkeypatch):
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path: sys.path.insert(0, scripts)
    import build_first_order_labels as cli
    schedule, windows, report = versioned_inputs()
    schedule = schedule.rename(columns={"training_ready": "training_ready_x", "eligibility_scope": "eligibility_scope_x", "temporal_split_boundary_date": "temporal_split_boundary_date_x"})
    schedule["training_ready_y"] = False
    schedule["eligibility_scope_y"] = "formal_for_historical_feature_replay"
    schedule["temporal_split_boundary_date_y"] = "2025-09-01"
    schedule_path = tmp_path / "schedule.csv"
    schedule.to_csv(schedule_path, index=False)
    rows = [dict(sku_code=r.sku_code, store_code="S1", lifecycle_week=w,
                 week_end=r.label_start_date + timedelta(days=7*w-1),
                 clean_sales_qty=1., restored_natural_sales=2., restored_lost_sales=1.,
                 stock_grain_unreliable=False, unknown_exposure_days=0., source_day_rows=7.,
                 restored_exposure_days=7., long_stockout_quality_weight=1., complete_week=True)
            for r in windows.itertuples(index=False) for w in range(1, 11)]
    weekly_path = tmp_path / "restored_store_weekly.csv"
    pd.DataFrame(rows).to_csv(weekly_path, index=False)
    (tmp_path / "round4_report.json").write_text(json.dumps(report))
    output = tmp_path / "new-labels"
    monkeypatch.setattr(sys, "argv", ["build_first_order_labels.py", "--schedule-file", str(schedule_path), "--restore-weekly", str(weekly_path), "--output-dir", str(output), "--decision-cutoff-time", CUTOFF.isoformat()])
    assert cli.main() == 0
    final = pd.read_csv(output / "formal_training_schedule.csv")
    assert not any(key.endswith(("_x", "_y")) for key in final)
    assert final.training_ready_for_feature_replay.all()
    labels = pd.read_csv(output / "first_order_wcover_labels.csv")
    assert labels.D_first_ref.eq(20.).all()
    actual_report = json.loads((output / "first_order_label_report.json").read_text())
    assert actual_report["restoration_provenance"]["validated_skus"] == 3


def test_restore_cli_window_override_keeps_existing_default(monkeypatch):
    import scripts.build_historical_restore as cli
    base = ["run_restore.py", "--data-root", "/data", "--decision-cutoff-time", CUTOFF.isoformat()]
    monkeypatch.setattr(sys, "argv", base)
    assert cli.parse_args().window_weeks is None
    monkeypatch.setattr(sys, "argv", base + ["--window-weeks", "10"])
    assert cli.parse_args().window_weeks == 10
