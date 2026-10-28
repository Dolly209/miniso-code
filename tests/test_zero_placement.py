from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from data.artifacts import StageArtifactStore
from data.modeling_dataset import assess_modeling_readiness
from data.placement_checkpoint import PlacementCheckpointStore
from first_order.engine import calculate_first_order
from first_order.pipeline import full_first_order, FullChainReplay, BASE
from placement.engine import plan_placement, CURVE_COLUMNS
from training.comparable_tuning import ComparableParameterModel, select_comparable_parameters
from training.section84 import fit_demand_residual_model
from test_step5_full_chain import context_fixture, transform_for


def zero_context():
    context = context_fixture()
    f = context["frames"]
    # There are many nominally eligible stores, but no observed launch arrival
    # and no supplied store counterfactual model supports expanding into them.
    f["stores"] = pd.concat([f["stores"].assign(store_code=lambda x: x.store_code + f"_{n}")
                             for n in range(40)], ignore_index=True)
    f["placement_evidence"] = pd.DataFrame([
        dict(sku_code=sku, store_code=row.store_code, store_tier=row.store_tier,
             eligible_at_launch=1, arrived_in_launch_window=0, v_neutral=np.nan,
             stock_pressure=np.nan, evidence_weight=0.)
        for sku in ("C1", "C2") for row in f["stores"].itertuples(index=False)])
    return context


def placement_for(context, **kwargs):
    f = context["frames"]
    return plan_placement(targets=f["features"].loc[f["features"].sku_code.eq("T")],
        comparables=f["comparables"], stores=f["stores"], restored_store_weekly=f["restored_store_weekly"],
        restored_tier_weekly=f["restored_tier_weekly"], config=context["parameters"]["placement"],
        placement_evidence=f.get("placement_evidence"), **kwargs)


def test_zero_store_result_has_schema_diagnostics_and_resumable_checkpoint(tmp_path):
    context = zero_context()
    saved = PlacementCheckpointStore(tmp_path / "cache", "zero-case")
    result = placement_for(context, save_target=saved.save)
    assert result.store_plan.selected.sum() == 0
    assert result.summary.iloc[0].width_target_store_count > 0
    assert result.summary.iloc[0].placement_selection_pool_store_count == 0
    assert result.summary.iloc[0].placement_status == "blocked_no_supported_stores"
    assert result.demand_curve.empty
    assert list(result.demand_curve) == list(CURVE_COLUMNS)
    assert int(result.demand_curve.curve_status.eq("VERSIONED_INITIAL_PARAMETER").sum()) == 0
    assert result.compliance.loc[result.compliance.concept.eq("complete_lifecycle_curve"), "status"].eq("BLOCKED_NO_SUPPORTED_STORES").all()
    path = tmp_path / "demand_curve.csv"
    StageArtifactStore.write_frame(path, result.demand_curve)
    assert StageArtifactStore.read_frame(path).empty  # header, not a corrupt blank CSV
    reused = placement_for(context, load_target=saved.load)
    pd.testing.assert_frame_equal(result.summary, reused.summary)
    pd.testing.assert_frame_equal(result.demand_curve, reused.demand_curve)


def test_zero_store_first_order_abstains_and_keeps_demand_missing():
    context = zero_context()
    model = ComparableParameterModel(("similarity_x",), (1.,), .15, 2, .1)
    chosen = select_comparable_parameters(context["frames"]["comparables"], model)
    result, placement, _ = full_first_order(context, chosen, frozen_transform=transform_for(context))
    row = result.decisions.iloc[0]
    assert row.decision_status == "blocked_no_supported_stores"
    assert not row.training_feature_eligible
    assert pd.isna(row.D_demand_base) and pd.isna(row.D_demand_adjusted)
    assert row.Q_feas == row.Q_plan == 0
    assert pd.isna(row.demand_grade)
    assert result.store_targets.empty and "store_code" in result.store_targets
    assert result.weekly_demand.empty and "sku_code" in result.weekly_demand
    demand = result.predictions.loc[result.predictions.horizon_code.eq("FIRST_ORDER_DEMAND")]
    assert demand.prediction_value.isna().all()
    # An unexplained empty upstream artifact remains a hard error.
    with pytest.raises(ValueError, match="missing explicit zero-placement"):
        calculate_first_order(targets=context["frames"]["features"].iloc[:1], placement_plan=placement.store_plan,
            demand_curve=placement.demand_curve, adjustments=pd.DataFrame(dict(sku_code=["T"], C_demand=[1.], input_completeness=[10.])),
            comparable_summary=context["frames"]["comparable_summary"], config=context["parameters"]["first_order"], factor_config=context["parameters"]["factors"])


def test_blocked_rows_not_ready_for_training_and_not_fitted_as_zero_sales():
    context = zero_context()
    model = ComparableParameterModel(("similarity_x",), (1.,), .15, 2, .1)
    chosen = select_comparable_parameters(context["frames"]["comparables"], model)
    result, _, _ = full_first_order(context, chosen, frozen_transform=transform_for(context))
    frame = result.decisions.assign(feature_cutoff_time="2025-01-01T23:59:59+08:00")
    cohort = pd.DataFrame(dict(sku_code=["T"], first_actual_sale_date=["2025-04-01"], split=["Train"]))
    labels = pd.DataFrame(dict(sku_code=["T"], D_first_ref=[100.]))
    report = assess_modeling_readiness(cohort=cohort, feature_frames=[("first_order", frame)], label_frame=labels, required_splits=("train",))
    assert report["feature_artifacts"]["first_order"]["blocked_forecast_rows"] == 1
    assert any("unavailable forecasts" in item for item in report["blocking_reasons"])
    with pytest.raises(ValueError, match="unavailable"):
        fit_demand_residual_model(cohort.assign(label=100., base=np.nan, x=0.), feature_columns=("x",),
            feature_signs={"x":"nonnegative"}, base_demand_column="base", ridge=.1, l1_ratio=0., half_life="infinity", demand_bounds=(.5,2.))


def test_healthy_target_not_changed_by_explicit_abstention_support():
    context = context_fixture()
    chosen = select_comparable_parameters(context["frames"]["comparables"], ComparableParameterModel(("similarity_x",), (1.,), .15, 2, .1))
    result, _, _ = full_first_order(context, chosen, frozen_transform=transform_for(context))
    assert result.decisions.iloc[0].decision_status == "calculated"
    assert result.decisions.iloc[0].training_feature_eligible
    assert np.isfinite(result.decisions.iloc[0].D_demand_base)


def test_first_order_cli_continues_zero_placement_diagnostic_without_raw_data(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripts.run_first_order import main
    from data.decision_adapter import DecisionDataAdapter
    context = zero_context()
    chosen = select_comparable_parameters(context["frames"]["comparables"], ComparableParameterModel(("similarity_x",), (1.,), .15, 2, .1))
    _, placement, factors = full_first_order(context, chosen, frozen_transform=transform_for(context))
    for name, frame in (("round6/placement_plan.csv", placement.store_plan), ("round6/demand_curve.csv", placement.demand_curve),
                        ("round6/placement_summary.csv", placement.summary), ("round7/demand_adjustments.csv", factors),
                        ("round5/comparable_summary.csv", context["frames"]["comparable_summary"].assign(selected_candidate_count=2))):
        StageArtifactStore.write_frame(tmp_path / name, frame)
    monkeypatch.setattr(DecisionDataAdapter, "from_project", lambda **kwargs: SimpleNamespace(
        products=lambda *a, **k: SimpleNamespace(products=context["frames"]["features"].iloc[:1])))
    monkeypatch.setattr("sys.argv", ["run_first_order.py", "--data-root", "/unused",
        "--placement-dir", str(tmp_path/"round6"), "--factor-dir", str(tmp_path/"round7"),
        "--comparable-dir", str(tmp_path/"round5"), "--output-dir", str(tmp_path/"round8"),
        "--decision-cutoff-time", context["cutoff"]])
    assert main() == 0
    report = StageArtifactStore.read_json(tmp_path / "round8/first_order_report.json")
    assert report["status"] == "completed_with_blocked_targets"
    assert report["blocked_target_count"] == 1
    result = StageArtifactStore.read_frame(tmp_path/"round8/first_order_decisions.csv")
    assert result.D_demand_base.isna().all()
