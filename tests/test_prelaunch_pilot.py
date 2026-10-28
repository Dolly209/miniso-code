from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from data.artifacts import StageArtifactStore
from data.domestic import DataContractViolation
from data.prelaunch_artifacts import restoration_model_health
from training.checkpoint_replay import normalize_replay_schedule
from training.pilot import freeze_development_pilot


def _schedule():
    rows = []
    for index in range(50):
        day = pd.Timestamp("2025-06-01") + timedelta(days=index)
        cutoff = day - pd.DateOffset(months=3)
        rows.append({
            "sku_code": f"S{index:03d}", "split": "Train" if index < 40 else "Test",
            "first_actual_sale_date": day.date().isoformat(), "release_dt": day.date().isoformat(),
            "decision_cutoff_time": cutoff.date().isoformat() + "T23:59:59+08:00",
            "dataset_version": "universe-test", "prediction_checkpoint": "M3",
            "prediction_checkpoint_key": f"S{index:03d}@M3", "lead_months": 3,
            "temporal_split_boundary_date": "2025-07-10", "training_ready_for_feature_replay": True,
            "D_first_ref": float(index * 10), "W_cover": 10, "sales_26w_qty": float(index * 20),
            "long_tail_weight": 1 / 40 if index < 40 else None,
            "label_quality_status": "eligible_restored_demand", "label_source_model_health": "fit_warning",
        })
    return pd.DataFrame(rows)


def test_pilot_freezes_source_split_and_stable_ids_without_test_outcome_selection():
    source = _schedule()
    pilot = freeze_development_pilot(source)
    assert len(pilot.skus) == 25
    assert pilot.skus["split"].value_counts().to_dict() == {"Train": 20, "Test": 5}
    assert pilot.skus.set_index("sku_code")["split"].eq(source.set_index("sku_code").loc[pilot.skus["sku_code"], "split"]).all()
    assert pilot.time_blocks["time_block"].nunique() == 5
    assert pilot.skus.loc[pilot.skus["split"].eq("Train"), "long_tail_weight"].sum() == pytest.approx(1.)
    assert pilot.skus.loc[pilot.skus["split"].eq("Test"), "long_tail_weight"].isna().all()
    shuffled = freeze_development_pilot(source.sample(frac=1, random_state=4))
    assert shuffled.report["pilot_version"] == pilot.report["pilot_version"]
    source.loc[source["split"].eq("Test"), "D_first_ref"] = 100000000.
    changed = freeze_development_pilot(source)
    assert changed.skus["sku_code"].tolist() == pilot.skus["sku_code"].tolist()
    assert changed.skus.loc[changed.skus["split"].eq("Train"), "long_tail_weight"].tolist() == pilot.skus.loc[pilot.skus["split"].eq("Train"), "long_tail_weight"].tolist()
    assert changed.report["pilot_version"] != pilot.report["pilot_version"]
    assert pilot.report["label_source_model_health"] == ["fit_warning"]


def test_pilot_refuses_unqualified_or_inconsistent_labels_and_bad_splits():
    source = _schedule()
    source.loc[0, "label_quality_status"] = "held_pending_inventory_risk_review"
    with pytest.raises(DataContractViolation, match="label-qualified"):
        freeze_development_pilot(source)
    source = _schedule()
    source.loc[0, "split"] = "Test"
    with pytest.raises(DataContractViolation, match="boundary"):
        freeze_development_pilot(source)
    source = _schedule()
    source.loc[0, "D_first_ref"] = float("nan")
    with pytest.raises(DataContractViolation, match="D_first_ref"):
        freeze_development_pilot(source)
    with pytest.raises(DataContractViolation, match="only 10"):
        freeze_development_pilot(_schedule(), test_skus=11)


def test_pilot_does_not_move_same_day_train_to_test_or_split_time_blocks():
    source = _schedule()
    # Keep enough dates for five blocks, but include multi-SKU launch batches.
    source.loc[:39, "first_actual_sale_date"] = [str((pd.Timestamp("2025-06-01") + timedelta(days=i // 2)).date()) for i in range(40)]
    pilot = freeze_development_pilot(source)
    assert pilot.time_blocks.groupby("first_actual_sale_date")["time_block"].nunique().eq(1).all()
    source.loc[:39, "first_actual_sale_date"] = "2025-06-01"
    with pytest.raises(ValueError, match="distinct"):
        freeze_development_pilot(source)


def test_pilot_schedule_remains_compatible_with_existing_replay_normalizer():
    pilot = freeze_development_pilot(_schedule())
    normalized = normalize_replay_schedule(pilot.schedule, available_through=datetime.fromisoformat("2026-09-30T23:59:59+08:00"))
    assert normalized["sku_code"].nunique() == 25
    assert normalized["pilot_version"].nunique() == 1


def test_label_model_health_does_not_hide_nonconvergence_or_invent_validation():
    assert restoration_model_health(None)["status"] == "not_verified"
    report = {"model": {"model_id": "v1", "nonconverged_scopes": ["__GLOBAL__"], "category_models": {"__GLOBAL__": {"converged": False}}}}
    assert restoration_model_health(report)["status"] == "fit_warning"
    assert not restoration_model_health(report)["all_scopes_converged"]


def test_oof_fit_forwards_numerical_budget_and_records_model_health(tmp_path, monkeypatch):
    from scripts import train_restoration_oof as cli
    store = StageArtifactStore()
    root = tmp_path / "models" / "fold_1"
    args = SimpleNamespace(data_root=tmp_path / "data", probe_dir=tmp_path / "probe", decision_cutoff_time=datetime.fromisoformat("2026-09-30T23:59:59+08:00"),
                           v_model_fit_sku_limit=None, batch_skus=25, event_calendar=None, no_resume=False,
                           huber_max_iter=1000, output_dir=tmp_path / "models")
    monkeypatch.setattr(cli, "_fit_population", lambda *a, **kw: pd.DataFrame({"sku_code": ["A"]}))
    def fake_run(command, **kwargs):
        assert command[command.index("--huber-max-iter") + 1] == "1000"
        store.write_json(root / "restoration_model.json", {"report": {
            "model_id": "model1", "train_end_time": "2025-01-01T23:59:59+08:00",
            "nonconverged_scopes": ["C"], "category_models": {"__GLOBAL__": {"converged": True}},
            "solver_controls": {"huber_max_iter": 1000}}})
    monkeypatch.setattr(cli, "_run", fake_run)
    class Logger:
        def warning(self, *args): pass
    result = cli._fit_one(args=args, candidates=pd.DataFrame(), cutoff_time=datetime.fromisoformat("2025-01-01T23:59:59+08:00"),
                          minimum_mature_weeks=4, root=root, shared_cache=tmp_path / "cache", logger=Logger(), stage="fold_1")
    assert result["global_model_converged"]
    assert result["nonconverged_scope_count"] == 1
    assert result["huber_max_iter"] == 1000


def test_pilot_freeze_cli_reuses_identical_files_but_refuses_reselection_and_tampering(tmp_path, monkeypatch):
    import sys
    from scripts import freeze_prelaunch_pilot as cli
    source = tmp_path / "labels" / "formal_training_schedule.csv"
    source.parent.mkdir()
    _schedule().to_csv(source, index=False)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli.AppConfig, "load", lambda *_: SimpleNamespace(log_dir=tmp_path / "logs", timezone="Asia/Shanghai", training=SimpleNamespace(n_time_blocks=5)))
    command = ["freeze_prelaunch_pilot.py", "--schedule-file", "labels/formal_training_schedule.csv", "--output-dir", "pilot",
               "--decision-cutoff-time", "2026-09-30T23:59:59+08:00"]
    monkeypatch.setattr(sys, "argv", command + ["--dry-run"])
    assert cli.main() == 0
    assert not (tmp_path / "pilot").exists()
    monkeypatch.setattr(sys, "argv", command)
    assert cli.main() == 0
    initial = (tmp_path / "pilot" / "pilot_report.json").read_bytes()
    assert cli.main() == 0
    assert (tmp_path / "pilot" / "pilot_report.json").read_bytes() == initial
    monkeypatch.setattr(sys, "argv", command + ["--limit-test-skus", "4"])
    assert cli.main() == 1
    assert (tmp_path / "pilot" / "pilot_report.json").read_bytes() == initial
    monkeypatch.setattr(sys, "argv", command)
    StageArtifactStore.write_frame(tmp_path / "pilot" / "pilot_skus.csv", pd.DataFrame({"sku_code": ["tampered"]}))
    assert cli.main() == 1


def test_modeling_assembly_retains_development_and_label_fit_warning_metadata():
    from data.modeling_dataset import assemble_modeling_dataset
    from training.pilot import evaluation_scope_metadata
    cohort = pd.DataFrame({"sku_code": ["A"], "first_actual_sale_date": ["2025-01-02"], "split": ["Train"],
                           "dataset_version": ["v1"], "pilot_version": ["p1"], "evaluation_scope": ["development_pilot_not_final_acceptance"],
                           "label_source_model_health": ["fit_warning"], "label_source_model_id": ["fallback-v1"]})
    result = assemble_modeling_dataset(cohort=cohort, feature_frames=[], label_frame=pd.DataFrame({"sku_code": ["A"], "D_first_ref": [10.]}))
    assert result.loc[0, "pilot_version"] == "p1"
    assert result.loc[0, "label_source_model_health"] == "fit_warning"
    meta = evaluation_scope_metadata(result)
    assert meta["evaluation_scope"] == "development_pilot_not_final_acceptance"
    assert not meta["final_acceptance_claim_allowed"]
    assert not meta["label_model_validation_complete"]


def test_oof_pilot_dry_run_has_four_folds_and_does_not_start_a_fit(tmp_path, monkeypatch):
    import sys
    from scripts import train_restoration_oof as cli
    store = StageArtifactStore()
    pilot = freeze_development_pilot(_schedule())
    schedule = tmp_path / "pilot.csv"
    candidates = tmp_path / "candidates.csv"
    store.write_frame(schedule, pilot.schedule)
    store.write_frame(candidates, _schedule()[["sku_code", "dataset_version", "first_actual_sale_date", "split", "sales_26w_qty", "long_tail_weight"]])
    monkeypatch.setattr(cli, "_run", lambda *a, **kw: pytest.fail("dry run must not launch a restoration fit"))
    monkeypatch.setattr(sys, "argv", ["train_restoration_oof.py", "--data-root", str(tmp_path / "no_raw_data"),
                                    "--schedule-file", str(schedule), "--candidate-cohort-file", str(candidates),
                                    "--output-dir", str(tmp_path / "models"), "--decision-cutoff-time", "2026-09-30T23:59:59+08:00",
                                    "--prediction-checkpoint", "M3", "--huber-max-iter", "1000", "--dry-run"])
    assert cli.main() == 0
    assert not (tmp_path / "models").exists()
