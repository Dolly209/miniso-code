import numpy as np
import pandas as pd
import pytest

from data.forecast_eligibility import forecast_fit_population
from data.modeling_dataset import assess_modeling_readiness, assemble_modeling_dataset
from training.fit_population import prepare_fit_population
from training.section84 import fix_section84_dataset, train_section84_first_order
from test_section84 import _dataset, _train_kwargs


def forecast_rows(frame):
    return frame.assign(
        decision_status="calculated", decision_reason="", training_feature_eligible=True,
        planned_store_count=10, D_demand_base=frame["base"],
        feature_cutoff_time=frame["decision_cutoff_time"],
    )


def block(rows, key):
    result = rows.copy()
    at = result.prediction_checkpoint_key.eq(key)
    result.loc[at, "decision_status"] = "blocked_no_supported_stores"
    result.loc[at, "decision_reason"] = "no_evidence_supported_store"
    result.loc[at, "training_feature_eligible"] = False
    result.loc[at, "planned_store_count"] = 0
    result.loc[at, "D_demand_base"] = np.nan
    return result


def training_frame():
    original = _dataset()
    raw = block(forecast_rows(original), "S00@M3")
    for column in ("decision_status", "decision_reason", "training_feature_eligible", "planned_store_count", "D_demand_base"):
        original["feature__first_order__" + column] = raw[column]
    original.loc[original.sku_code.eq("S00"), "base"] = np.nan
    return original


def test_readiness_preserves_blocked_train_identity_and_label_with_explicit_fit_exclusion():
    cohort = _dataset().assign(lead_months=3, decision_date="2024-01-01", release_dt="2025-01-01",
                              forecast_scope="first_order", checkpoint_weight_policy="independent_checkpoint")
    source = block(forecast_rows(cohort), "S00@M3")
    labels = cohort[["sku_code", "label"]].rename(columns={"label": "D_first_ref"})
    report = assess_modeling_readiness(cohort=cohort, feature_frames=[("first_order", source)], label_frame=labels)
    assert report["status"] == "ready_for_train_oof_and_locked_test"
    assert report["feature_artifacts"]["first_order"]["usable_train_skus"] == 19
    assert report["feature_artifacts"]["first_order"]["fit_excluded_forecasts"][0]["sku_code"] == "S00"
    dataset = assemble_modeling_dataset(cohort=cohort, feature_frames=[("first_order", source)], label_frame=labels)
    assert len(dataset) == 25
    bad = dataset.loc[dataset.sku_code.eq("S00")].iloc[0]
    assert not bad.parameter_fit_eligible and not bad.forecast_feature_available
    assert bad.label == cohort.iloc[0].label  # demand/label never replaced with zero
    assert dataset.loc[dataset.split.eq("Test"), "forecast_feature_available"].all()
    assert not dataset.loc[dataset.split.eq("Test"), "parameter_fit_eligible"].any()


def test_unavailable_test_cannot_be_silently_omitted_and_bad_diagnostics_fail():
    cohort = _dataset()
    with pytest.raises(ValueError, match="locked Test"):
        forecast_fit_population(cohort, block(forecast_rows(cohort), "S20@M3"))
    raw = block(forecast_rows(cohort), "S00@M3")
    raw.loc[0, "D_demand_base"] = 0.
    with pytest.raises(ValueError, match="inconsistent"):
        forecast_fit_population(cohort, raw)
    raw = block(forecast_rows(cohort), "S00@M3").drop(columns="decision_status")
    with pytest.raises(ValueError, match="lack explicit"):
        forecast_fit_population(cohort, raw)


def test_original_blocks_and_registry_population_remain_frozen_and_only_fit_uses_19():
    final = training_frame()
    fixed = fix_section84_dataset(final)
    original_rows = fixed.rows.copy(deep=True)
    original_blocks = fixed.time_blocks.copy(deep=True)
    oof = final.loc[final.split.eq("Train")].copy()
    fit_fixed, fit_oof, fit_final, excluded = prepare_fit_population(fixed, oof, final)
    pd.testing.assert_frame_equal(fixed.rows, original_rows)
    pd.testing.assert_frame_equal(fit_fixed.time_blocks, original_blocks)
    assert set(fit_final.loc[fit_final.split.eq("Test"), "sku_code"]) == set(final.loc[final.split.eq("Test"), "sku_code"])
    assert excluded.sku_code.tolist() == ["S00"]
    assert len(fit_oof) == 19 and len(fit_final) == 24
    assert fit_fixed.folds.fit_skus.tolist() == [3, 7, 11, 15]
    assert fit_fixed.folds.validation_skus.tolist() == [4, 4, 4, 4]
    kwargs = _train_kwargs(final)
    kwargs.update(comparable_temperature_grid=(.15,), comparable_top_k_grid=(3,), similarity_regularization_grid=(.1,))
    trained = train_section84_first_order(
        fixed=fit_fixed, oof_feature_rows=fit_oof, final_feature_rows=fit_final, **kwargs,
        feature_columns=("positive", "price"), feature_signs={"positive":"nonnegative", "price":"nonpositive"},
        base_demand_column="base", ridge_grid=(.1,), l1_ratio_grid=(0.,),
        half_life_grid=("infinity",), demand_bounds=(.5,2.), grade_thresholds=(15000,30000,50000))
    assert trained.report["train_skus"] == trained.report["final_refit_rows"] == 19
    assert trained.report["oof_rows"] == 16 and trained.report["test_skus_locked"] == 5
    assert trained.fold_models.fit_rows.tolist() == [3, 7, 11, 15]
    assert "S00" not in set(trained.oof_predictions.sku_code)


def test_unpaired_blocked_or_modified_label_rejected_before_fitting():
    final = training_frame()
    fixed = fix_section84_dataset(final)
    oof = final.loc[final.split.eq("Train")].copy()
    oof.loc[oof.sku_code.eq("S00"), "feature__first_order__decision_status"] = "calculated"
    oof.loc[oof.sku_code.eq("S00"), "feature__first_order__training_feature_eligible"] = True
    with pytest.raises(ValueError, match="OOF/final unavailable"):
        prepare_fit_population(fixed, oof, final)
    oof = final.loc[final.split.eq("Train")].copy()
    oof.loc[oof.sku_code.eq("S00"), "label"] = 0.
    with pytest.raises(ValueError, match="identity/label"):
        prepare_fit_population(fixed, oof, final)


def test_empty_validation_block_fails_without_repartitioning():
    final = training_frame()
    for i in range(4, 8):
        at = final.sku_code.eq(f"S{i:02d}")
        final.loc[at, "feature__first_order__decision_status"] = "blocked_no_supported_stores"
        final.loc[at, "feature__first_order__decision_reason"] = "no_evidence_supported_store"
        final.loc[at, "feature__first_order__training_feature_eligible"] = False
        final.loc[at, "feature__first_order__planned_store_count"] = 0
        final.loc[at, "feature__first_order__D_demand_base"] = np.nan
    with pytest.raises(ValueError, match="fold 1 empty"):
        prepare_fit_population(fix_section84_dataset(final), final.loc[final.split.eq("Train")], final)
