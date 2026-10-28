"""Fixed hyperparameters must not freeze the parameters being learned."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from first_order.pipeline import FullChainReplay, BASE, PREFIX
from training.comparable_tuning import fit_comparable_parameters
from training.section84 import fix_section84_dataset, train_section84_first_order, replay_section84_locked_test
from test_section84 import _dataset, _train_kwargs
from test_step5_full_chain import context_fixture, PARAMETERS


def test_fixed_preset_retains_theta_initialization_and_tau_training_grid(monkeypatch):
    from data.domain import load_algorithm_parameters
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripts.train import _fixed_hyperparameter_settings
    parameters = load_algorithm_parameters(Path(__file__).resolve().parents[1] / "config/algorithm_v3_2_step5_pilot.json")
    before = json.dumps(parameters, sort_keys=True)
    settings = _fixed_hyperparameter_settings(parameters)
    assert settings["similarity_weights_initialization_only"] == parameters["comparable"]["similarity_weights"]
    assert settings["temperature_training_grid"] == [.1,.15] and settings["top_k"] == 3
    assert settings["similarity_regularization"] == .01
    assert settings["ridge"] == .1 and settings["l1_ratio"] == 0. and settings["half_life_weeks"] == 52
    assert json.dumps(parameters, sort_keys=True) == before


def test_theta_moves_from_initial_weights_when_training_signal_supports_it():
    targets = pd.DataFrame({"prediction_checkpoint_key":["T@M3"], "sku_code":["T"],
                            "first_actual_sale_date":["2025-01-01"], "label":[240.], "base":[100.]})
    detail = pd.DataFrame({"prediction_checkpoint_key":["T@M3","T@M3"], "target_sku_code":["T","T"],
        "candidate_sku_code":["C1","C2"], "selected":[1,1], "alpha_weight":[.5,.5],
        "historical_velocity":[10.,20.], "quality_score":[1.,1.], "path_priority":[1,1],
        "similarity_emotion":[.9,.1], "similarity_ip":[.1,.9]})
    def recompute(rows, model):
        return rows.assign(base=100. * (.5 + 2 * model.component_weights[0]))
    learned = fit_comparable_parameters(targets, detail, key_column="prediction_checkpoint_key",
        base_demand_column="base", initial_component_weights={"emotion":.5,"ip":.5},
        temperature_grid=(.1,.15), top_k_grid=(2,), regularization=.01,
        half_life=52, full_recompute=recompute)
    assert learned.component_weights[0] > .5
    assert sum(learned.component_weights) == pytest.approx(1.)
    assert min(learned.component_weights) >= 0
    assert learned.softmax_temperature in (.1,.15)
    assert learned.optimizer == "projected_coordinate_3_step_sweep_budget_v1"


def test_fixed_outer_preset_fits_comparable_and_demand_in_all_folds_and_final(monkeypatch):
    import training.section84 as module
    frame = _dataset()
    frame[BASE] = frame.base
    column = PREFIX + "delta_factor__form_play_fit"
    frame[column] = .1
    contexts, parts = {}, []
    for row in frame.itertuples(index=False):
        context = context_fixture()
        for name in ("features", "comparable_summary"):
            context["frames"][name]["sku_code"] = context["frames"][name].sku_code.replace({"T":row.sku_code})
        context["frames"]["features"].loc[lambda f: f.sku_code.eq(row.sku_code), "form_play_fit"] = float(int(row.sku_code[1:]))
        context["frames"]["comparables"]["target_sku_code"] = row.sku_code
        context["target_sku"], context["cutoff"] = row.sku_code, row.decision_cutoff_time
        contexts[row.prediction_checkpoint_key] = context
        parts.append(context["frames"]["comparables"].assign(prediction_checkpoint_key=row.prediction_checkpoint_key))
    detail = pd.concat(parts)
    comparable_calls, demand_calls = [], []
    original_comparable, original_demand = module.fit_comparable_parameters, module.fit_demand_residual_model
    def fit_comparable(rows, *args, **kwargs):
        comparable_calls.append(set(rows.prediction_checkpoint_key))
        assert list(kwargs["temperature_grid"]) == [.1,.15]
        return original_comparable(rows, *args, **kwargs)
    def fit_demand(rows, *args, **kwargs):
        demand_calls.append(set(rows.prediction_checkpoint_key))
        return original_demand(rows, *args, **kwargs)
    monkeypatch.setattr(module, "fit_comparable_parameters", fit_comparable)
    monkeypatch.setattr(module, "fit_demand_residual_model", fit_demand)
    oof = FullChainReplay(lambda key:contexts[key], key_column="prediction_checkpoint_key")
    final = FullChainReplay(lambda key:contexts[key], key_column="prediction_checkpoint_key")
    result = train_section84_first_order(fixed=fix_section84_dataset(frame),
        oof_feature_rows=frame.loc[frame.split.eq("Train")], final_feature_rows=frame,
        oof_candidate_detail=detail, final_candidate_detail=detail,
        initial_similarity_weights={"x":1.}, comparable_temperature_grid=(.1,.15), comparable_top_k_grid=(2,),
        similarity_regularization_grid=(.01,), feature_columns=(column,), feature_signs={column:"nonnegative"},
        feature_groups={column:"PRODUCT"}, base_demand_column=BASE, ridge_grid=(.1,), l1_ratio_grid=(0.,),
        half_life_grid=(52,), demand_bounds=(.5,2.), coefficient_bounds=PARAMETERS["factors"]["coefficient_bounds"],
        grade_thresholds=(15000,30000,50000), oof_replay=oof, final_replay=final, fixed_hyperparameters=True)
    assert len(comparable_calls) == len(demand_calls) == 5
    assert [len(keys) for keys in comparable_calls] == [4,8,12,16,20]
    assert comparable_calls == demand_calls
    assert result.report["training_mode"] == "fixed_hyperparameters_learned_parameters"
    assert result.report["parameter_fit_calls"] == {"comparable_theta_tau":5,"first_order_coefficients":5}
    assert len(result.model_selection) == 1 and len(result.fold_models) == 4
    assert len(result.oof_predictions) == 16 and result.report["test_rows_consumed"] == 0
    assert result.comparable_model.optimizer == "projected_coordinate_3_step_sweep_budget_v1"
    test_keys = set(frame.loc[frame.split.eq("Test"), "prediction_checkpoint_key"])
    assert not any(keys & test_keys for keys in comparable_calls + demand_calls)
    for record in result.fold_models.itertuples(index=False):
        transform = json.loads(record.frozen_transform_json)
        assert not test_keys.intersection(transform["fit_checkpoint_keys"])
        assert len(transform["fit_checkpoint_keys"]) == record.fit_rows
    before = json.dumps(result.model.as_dict(), sort_keys=True)
    predictions, report = replay_section84_locked_test(frame, result.model, candidate_detail=detail,
        comparable_model=result.comparable_model, interval_log_quantiles=result.report["oof_log_residual_quantiles"],
        grade_thresholds=(15000,30000,50000), full_replay=final)
    assert len(predictions) == 5 and np.isfinite(predictions.prediction).all()
    assert not report["model_refit_during_test"] and not report["model_selection_reopened"]
    assert len(comparable_calls) == len(demand_calls) == 5
    assert json.dumps(result.model.as_dict(), sort_keys=True) == before


def test_fixed_hyperparameters_reject_multiple_outer_configurations():
    frame = _dataset()
    with pytest.raises(ValueError, match="one predetermined outer"):
        train_section84_first_order(fixed=fix_section84_dataset(frame),
            oof_feature_rows=frame.loc[frame.split.eq("Train")], final_feature_rows=frame,
            **_train_kwargs(frame), feature_columns=("positive",), feature_signs={"positive":"nonnegative"},
            base_demand_column="base", ridge_grid=(.1,1.), l1_ratio_grid=(0.,), half_life_grid=(52,),
            demand_bounds=(.5,2.), grade_thresholds=(15000,30000,50000), fixed_hyperparameters=True)


def test_obsolete_freezing_mode_rejected_and_new_mode_parses(monkeypatch,capsys):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripts.train import parse_args
    args=["train.py","--oof-replay-dir","oof","--final-replay-dir","final", "--oof-modeling-dataset","o.csv",
          "--final-modeling-dataset","f.csv","--oof-comparable-detail","o.csv","--final-comparable-detail","f.csv",
          "--decision-cutoff-time","2026-09-30T23:59:59+08:00"]
    monkeypatch.setattr("sys.argv",args+["--fixed-reference-parameters"])
    with pytest.raises(SystemExit):parse_args()
    assert "freezes trainable theta/tau" in capsys.readouterr().err
    monkeypatch.setattr("sys.argv",args+["--fixed-hyperparameters"])
    assert parse_args().fixed_hyperparameters
