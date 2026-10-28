from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path

import pandas as pd
import pytest

from data.domestic import DataContractViolation
from data.launch_contract import validate_launch_origin_contract, require_verified_training_schedule
from data.launch_universe import build_verified_launch_universe, read_verified_launch_universe, rebind_reviewed_ip_snapshot, validate_observed_cohort_report
from training.checkpoint_replay import align_candidate_cohort_to_schedule, normalize_replay_schedule


CUTOFF = datetime.fromisoformat("2026-09-30T23:59:59+08:00")
PARAMETERS = json.loads((Path(__file__).resolve().parents[1] / "config/algorithm_v3_2.json").read_text())


def inputs():
    rows = [
        ("OLD", "2025-01-01", "2024-09-21", "2024-09-12"),
        ("CONFLICT", "2025-01-12", "2025-01-10", "2025-01-10"),
        ("NO_MASTER", "2025-02-01", None, "2025-02-01"),
        ("PREDECISION_OLD", "2025-03-01", "2025-03-01", "2025-03-01"),
        ("EARLY", "2025-05-01", "2025-05-01", "2025-05-01"),
        ("MIDDLE", "2025-09-01", "2025-09-01", "2025-09-01"),
        ("LATEST_MATURE", "2026-04-02", "2026-04-02", "2026-04-02"),
        ("IMMATURE", "2026-04-03", "2026-04-03", "2026-04-03"),
    ]
    cohort = pd.DataFrame([
        dict(sku_code=sku, dataset_version="old-version", first_actual_sale_date=observed,
             sales_26w_qty=100., split="Train", long_tail_weight=999.)
        for sku, observed, _, _ in rows
    ])
    products = pd.DataFrame([
        dict(sku_code=sku, first_sale_date=master, release_dt=release)
        for sku, _, master, release in rows
    ])
    days = tuple(pd.date_range("2025-01-01", "2026-09-30").date)
    return dict(cohort=cohort, products=products,
                source_partitions={key: days for key in ("sales", "inventory", "placement", "store", "social")},
                available_through=CUTOFF, source_start_date=date(2025, 1, 1), parameters=PARAMETERS)


def test_sales_origin_keeps_missing_and_conflicting_master_dates_but_excludes_known_old_history():
    result = build_verified_launch_universe(**inputs())
    audit = result.audit.set_index("sku_code")
    assert audit.loc["OLD", "product_lifecycle_week_at_observed_start"] == 15
    assert audit.loc["OLD", "observed_first_sale_date"] == "2025-01-01"
    assert "known_pre_source_sale_evidence" in audit.loc["OLD", "candidate_exclusion_reasons"]
    assert audit.loc["CONFLICT", "candidate_exclusion_reasons"] == ""
    assert audit.loc["NO_MASTER", "candidate_exclusion_reasons"] == ""
    assert audit.loc["CONFLICT", "first_actual_sale_date"] == "2025-01-12"
    assert audit.loc["CONFLICT", "launch_origin_eligible"]
    assert not audit.loc["CONFLICT", "launch_origin_verified"]
    assert not {"OLD", "IMMATURE"} & set(result.candidate_cohort.sku_code)
    assert {"CONFLICT", "NO_MASTER"} <= set(result.candidate_cohort.sku_code)
    assert set(result.schedule.sku_code) == {"EARLY", "MIDDLE", "LATEST_MATURE"}
    assert set(result.schedule.prediction_checkpoint) == {"M3"}
    assert result.report["training_ready"] is False
    assert result.report["raw_fact_rows_scanned"] == 0
    assert result.report["dataset_version"] != "old-version"
    validate_launch_origin_contract(result.candidate_cohort, name="test")
    normalize_replay_schedule(result.schedule, available_through=CUTOFF)


def test_core_windows_and_optional_windows_are_separate():
    args = inputs()
    args["source_partitions"]["social"] = ()
    result = build_verified_launch_universe(**args)
    early = result.audit.set_index("sku_code").loc["EARLY"]
    assert early.decision_date == "2025-02-01"
    assert early.prelaunch_target_preflight_eligible
    assert not early.decision_category_history_sales_complete
    assert not early.decision_social_history_social_complete
    old_decision = result.audit.set_index("sku_code").loc["PREDECISION_OLD"]
    assert old_decision.candidate_launch_eligible
    assert not old_decision.prelaunch_target_preflight_eligible
    assert "decision_before_source_start" in old_decision.target_exclusion_reasons


@pytest.mark.parametrize("master", [None, "2025-04-15", "2027-01-01"])
def test_missing_or_disagreeing_master_date_does_not_move_sales_origin_or_exclude_target(master):
    args = inputs()
    args["products"].loc[args["products"].sku_code.eq("EARLY"), "first_sale_date"] = master
    result = build_verified_launch_universe(**args)
    target = result.schedule.set_index("sku_code").loc["EARLY"]
    assert target.first_actual_sale_date == "2025-05-01"
    assert target.launch_window_end == "2025-05-28"
    assert target.distribution_window_end == "2025-10-29"
    assert target.decision_date == "2025-02-01"
    assert target.launch_origin_eligible and not target.launch_origin_verified
    assert target.origin_audit_notes
    validate_launch_origin_contract(result.schedule, name="relaxed targets")


@pytest.mark.parametrize("master,observed,accepted", [
    (None, "2025-01-01", False),
    ("2025-01-10", "2025-01-01", False),
    ("2025-01-01", "2025-01-01", True),
    ("2024-12-31", "2025-02-01", False),
    (None, "2025-02-01", True),
])
def test_left_truncation_guards_are_not_disabled_by_relaxed_product_date_policy(master, observed, accepted):
    args = inputs()
    args["products"].loc[args["products"].sku_code.eq("EARLY"), "first_sale_date"] = master
    args["cohort"].loc[args["cohort"].sku_code.eq("EARLY"), "first_actual_sale_date"] = observed
    result = build_verified_launch_universe(**args)
    assert ("EARLY" in set(result.candidate_cohort.sku_code)) is accepted


def test_legacy_exact_match_contract_remains_readable():
    candidates = build_verified_launch_universe(**inputs()).candidate_cohort
    legacy = candidates.loc[candidates.launch_origin_verified].drop(columns=["launch_origin_policy"])
    validate_launch_origin_contract(legacy, name="v1")
    legacy.loc[legacy.index[0], "product_first_sale_date"] = "2025-01-02"
    with pytest.raises(DataContractViolation, match="launch-origin contract"):
        validate_launch_origin_contract(legacy, name="edited v1")


@pytest.mark.parametrize("master", [None, "2025-04-15"])
def test_placement_launch_window_uses_same_sales_origin_as_m2_and_labels(master):
    from types import SimpleNamespace
    from data.decision_adapter import DecisionDataAdapter
    from test_placement_optimization import inputs as placement_inputs
    args = inputs()
    args["products"].loc[args["products"].sku_code.eq("EARLY"), "first_sale_date"] = master
    origins = build_verified_launch_universe(**args).candidate_cohort
    origins = origins.loc[origins.sku_code.eq("EARLY")]
    calls = []
    class Repository:
        max_rows_per_read = 100
        def read_partition_range(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(frame=pd.DataFrame([dict(store_code="S1", first_store_arrival_time="2025-05-02")]), selected_partitions=(date(2025, 5, 2),))
    placement = placement_inputs()
    adapter = DecisionDataAdapter(Repository())
    kwargs = dict(
        cutoff_time=datetime.fromisoformat("2025-06-01T23:59:59+08:00"),
        candidate_products=args["products"].loc[args["products"].sku_code.eq("EARLY")],
        eligible_stores=placement["stores"], restored_store_weekly=placement["restored_store_weekly"],
        launch_window_days=28, candidate_launch_origins=origins,
    )
    result = adapter.placement_launch_evidence(**kwargs)
    assert calls[0]["start_date"] == date(2025, 5, 1)
    assert calls[0]["end_date"] == date(2025, 5, 28)
    assert set(result.evidence.launch_date_source) == {"cohort.sales_observed_first_positive"}
    calls.clear()
    kwargs["candidate_launch_origins"] = origins.iloc[:0]
    with pytest.raises(DataContractViolation, match="missing canonical launch origins"):
        adapter.placement_launch_evidence(**kwargs)
    assert calls == []


def test_known_bad_inventory_is_not_reintroduced_and_other_labels_are_not_reused():
    args = inputs()
    args["label_quality"] = pd.DataFrame([
        dict(sku_code="EARLY", label_start_date="2025-05-01", label_eligible=False, stock_grain_unreliable_rows=1),
        dict(sku_code="MIDDLE", label_start_date="2025-09-01", label_eligible=False, stock_grain_unreliable_rows=0),
        dict(sku_code="LATEST_MATURE", label_start_date="2025-01-01", label_eligible=False, stock_grain_unreliable_rows=100),
    ])
    result = build_verified_launch_universe(**args)
    audit = result.audit.set_index("sku_code")
    assert "EARLY" not in set(result.candidate_cohort.sku_code)
    assert "MIDDLE" in set(result.candidate_cohort.sku_code)
    assert "MIDDLE" not in set(result.schedule.sku_code)
    assert audit.loc["LATEST_MATURE", "prior_label_quality_status"] == "prior_origin_differs_not_reused"
    assert "LATEST_MATURE" in set(result.schedule.sku_code)
    assert "D_first_ref" not in result.schedule


def test_version_is_order_independent_and_changed_origin_evidence_invalidates_it():
    args = inputs()
    result = build_verified_launch_universe(**args)
    args["cohort"] = args["cohort"].iloc[::-1].reset_index(drop=True)
    assert build_verified_launch_universe(**args).report["dataset_version"] == result.report["dataset_version"]
    args["products"].loc[args["products"].sku_code.eq("EARLY"), "first_sale_date"] = "2025-04-30"
    assert build_verified_launch_universe(**args).report["dataset_version"] != result.report["dataset_version"]


def test_zero_eligible_skus_preserve_audit_and_empty_output_headers():
    args = inputs()
    args["source_partitions"]["inventory"] = ()
    result = build_verified_launch_universe(**args)
    assert result.candidate_cohort.empty and result.schedule.empty
    assert "prediction_checkpoint_key" in result.schedule
    assert len(result.audit) == len(args["cohort"])
    assert result.report["status"] == "no_eligible_targets_review_exclusions"


def test_verified_schedule_rejects_legacy_candidates_and_edited_dates():
    args = inputs()
    result = build_verified_launch_universe(**args)
    with pytest.raises(DataContractViolation, match="legacy Round-3"):
        align_candidate_cohort_to_schedule(args["cohort"], result.schedule)
    aligned, _ = align_candidate_cohort_to_schedule(result.candidate_cohort, result.schedule)
    assert set(aligned.launch_universe_version) == {result.report["dataset_version"]}
    bad = result.candidate_cohort.copy()
    bad.loc[bad.index[0], "first_actual_sale_date"] = "2025-01-01"
    with pytest.raises(DataContractViolation, match="launch-origin contract"):
        validate_launch_origin_contract(bad, name="edited")


def test_m3_only_checkpoint_schedule_is_supported():
    result = build_verified_launch_universe(**inputs())
    selected = normalize_replay_schedule(result.schedule, available_through=CUTOFF, limit_skus=1)
    assert len(selected) == 1 and selected.iloc[0].prediction_checkpoint == "M3"


def test_preflight_is_not_a_training_schedule():
    result = build_verified_launch_universe(**inputs())
    with pytest.raises(DataContractViolation, match="only preflight"):
        require_verified_training_schedule(result.schedule)
    require_verified_training_schedule(result.schedule.assign(training_ready_for_feature_replay=True))


def test_adapter_reads_one_fixed_dimension_and_indexes_each_source_once():
    from types import SimpleNamespace
    args = inputs()
    calls = []
    class Repository:
        def available_partitions(self, source_name):
            calls.append(source_name)
            return args["source_partitions"][next(key for key, value in SOURCES.items() if value == source_name)]
    class Adapter:
        repository = Repository()
        reads = 0
        def products(self, cutoff, **kwargs):
            self.reads += 1
            assert cutoff == CUTOFF
            assert kwargs["fallback_to_historical"] is False
            return SimpleNamespace(products=args["products"], source_partition="2026-09-30")
    from data.decision_adapter import SOURCES
    adapter = Adapter()
    result = read_verified_launch_universe(
        adapter=adapter, cohort=args["cohort"], available_through=CUTOFF,
        source_start_date=args["source_start_date"], parameters=PARAMETERS,
    )
    assert adapter.reads == 1
    assert len(calls) == len(set(calls)) == 5
    assert result.report["source_partition_errors"] == {}


def test_rebound_mapping_preserves_reviewed_multi_ip_rows_without_inference():
    from test_standard_ip_mapping import _resolved_snapshot_fixture
    mapping, source, report = _resolved_snapshot_fixture()
    new = source.loc[source.sku_code.isin(["A", "C"])].assign(
        dataset_version="new-verified-version", split=["Train", "Test"],
        first_actual_sale_date=["2025-07-01", "2025-08-01"],
    )
    output, rebound_report = rebind_reviewed_ip_snapshot(
        mapping=mapping, mapping_report=report, source_cohort=source,
        verified_cohort=new, available_through=CUTOFF,
    )
    preserved = [key for key in mapping if key not in {"dataset_version"}]
    pd.testing.assert_frame_equal(
        output[preserved].sort_values(["sku_code", "standard_ip_code"]).reset_index(drop=True),
        mapping.loc[mapping.sku_code.isin(["A", "C"]), preserved].sort_values(["sku_code", "standard_ip_code"]).reset_index(drop=True),
    )
    assert set(output.dataset_version) == {"new-verified-version"}
    assert rebound_report["ip_inference_rerun"] is False
    assert rebound_report["cohort_skus"] == 2


def test_verified_replay_rejects_old_model_registry_before_application(tmp_path):
    import sys
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from replay_prelaunch_features import _attach_restoration_models
    schedule = build_verified_launch_universe(**inputs()).schedule
    registry = schedule[["sku_code"]].assign(
        model_id="old-model", model_file="old.json", model_train_end_time="2025-01-01T23:59:59+08:00", usage_scope="oof",
    )
    path = tmp_path / "old_registry.csv"
    registry.to_csv(path, index=False)
    with pytest.raises(DataContractViolation, match="old model registries"):
        _attach_restoration_models(schedule, path)


def test_source_cohort_cannot_silently_use_a_later_extract():
    cohort = inputs()["cohort"]
    report = dict(dataset_version="old-version", decision_cutoff_time=CUTOFF.isoformat())
    validate_observed_cohort_report(cohort, report, available_through=CUTOFF)
    report["decision_cutoff_time"] = "2026-10-08T23:59:59+08:00"
    with pytest.raises(DataContractViolation, match="source cohort cutoff"):
        validate_observed_cohort_report(cohort, report, available_through=CUTOFF)


def test_cli_writes_versioned_audits_without_fact_replay_and_preserves_outputs(tmp_path, monkeypatch):
    import logging
    from types import SimpleNamespace
    import sys
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import build_verified_sku_universe as cli
    from data.decision_adapter import SOURCES
    args = inputs()
    cohort_path = tmp_path / "cohort.csv"
    args["cohort"].to_csv(cohort_path, index=False)
    (tmp_path / "round3_report.json").write_text(json.dumps(dict(dataset_version="old-version", decision_cutoff_time=CUTOFF.isoformat())))
    class Adapter:
        def products(self, *values, **kwargs):
            return SimpleNamespace(products=args["products"], source_partition="2026-09-30")
        class Repository:
            def available_partitions(self, source_name):
                return args["source_partitions"][next(key for key, value in SOURCES.items() if value == source_name)]
        repository = Repository()
    monkeypatch.setattr(cli.DecisionDataAdapter, "from_project", lambda **kwargs: Adapter())
    monkeypatch.setattr(cli, "setup_run_logger", lambda *args: (logging.getLogger("universe-test"), tmp_path / "run.log"))
    output = tmp_path / "new-output"
    monkeypatch.setattr(sys, "argv", ["build_verified_sku_universe.py", "--data-root", str(tmp_path / "raw"), "--cohort-file", str(cohort_path), "--decision-cutoff-time", CUTOFF.isoformat(), "--output-dir", str(output)])
    assert cli.main() == 0
    report = json.loads((output / "verified_sku_universe_report.json").read_text())
    assert report["input_cohort_cutoff_verified"] is True
    assert report["target_preflight_skus"] == 3
    original = (output / "sku_launch_audit.csv").read_bytes()
    assert cli.main() == 1
    assert (output / "sku_launch_audit.csv").read_bytes() == original
