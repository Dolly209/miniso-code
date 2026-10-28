from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from data.recompute import write_recompute_frames, seal_recompute_bundle, load_recompute_bundle, ReplayBundleStore
from data.artifacts import StageArtifactStore
from factors.frozen import fit_factor_transform, apply_factor_transform
from first_order.pipeline import FullChainReplay, full_first_order, BASE, PREFIX, exact_group_effect
from training.comparable_tuning import ComparableParameterModel, select_comparable_parameters, fit_comparable_parameters
from training.section84 import DemandResidualModel, predict_demand_residual, replay_section84_locked_test
from training.ablation import train_ablation
from test_remaining_rounds import _placement_inputs, PARAMETERS


def context_fixture():
    targets, detail, stores, history, tiers = _placement_inputs()
    detail = pd.concat([detail.assign(candidate_sku_code="C1", historical_velocity=5., quality_score=1., path_priority=1, similarity_x=.2),
                        detail.assign(candidate_sku_code="C2", historical_velocity=20., quality_score=1., path_priority=1, similarity_x=.9)], ignore_index=True)
    detail["alpha_weight"] = .5
    history = pd.concat([history, history.assign(sku_code="C2", v_neutral=20.)], ignore_index=True)
    tiers = pd.concat([tiers, tiers.assign(sku_code="C2", v_neutral=20.)], ignore_index=True)
    features = pd.DataFrame(dict(sku_code=["T", "C1", "C2"], retail_price=[20., 10., 40.], cate_level4_code=["4"] * 3,
                                 ip_social_volume=[2., 1., 8.], ip_social_momentum=[np.nan, 2., 3.],
                                 form_play_fit=[2., 4., 1.]))
    return {"frames": dict(features=features, comparables=detail, stores=stores,
        restored_store_weekly=history, restored_tier_weekly=tiers,
        comparable_summary=pd.DataFrame(dict(sku_code=["T"], path_reliability=[80.], external_direction_coefficient=[1.]))),
        "parameters": deepcopy(PARAMETERS), "target_sku": "T", "cutoff": "2025-01-01T23:59:59+08:00"}


def transform_for(context):
    cfg = context["parameters"]["factors"]
    return fit_factor_transform(context["frames"]["features"], factor_weights=cfg["small_factor_weights"], winsor=(.01, .99))


def test_frozen_cdf_is_batch_independent_missing_and_unseen_category():
    fit = pd.DataFrame(dict(sku_code=["A", "B", "C"], cate_level4_code=["a"]*3, x=[1., 2., 3.]))
    frozen = fit_factor_transform(fit, factor_weights={"IP": {"x": 1., "missing": 1.}}, winsor=(0., 1.))
    single = pd.DataFrame(dict(sku_code=["T"], cate_level4_code=["unseen"], x=[2.]))
    one = apply_factor_transform(single, frozen, {"IP": {"x": 1., "missing": 1.}})
    many = apply_factor_transform(pd.concat([single, single.assign(sku_code="V", x=1e12)]), frozen, {"IP": {"x": 1., "missing": 1.}})
    assert one.iloc[0]["score__x"] == many.iloc[0]["score__x"]
    assert one.iloc[0]["score__x"] == pytest.approx(200/3)
    assert one.iloc[0]["completeness__IP"] == .5
    assert one["score__missing"].isna().all()


def test_weight_changes_recompute_price_curve_stores_not_scalar_velocity_ratio():
    context = context_fixture()
    replay = FullChainReplay(lambda key: context, key_column="key")
    rows = pd.DataFrame(dict(key=["T@M3"], sku_code=["T"], split=["train"], label=[10.], **{BASE: [999.]}))
    transform = transform_for(context)
    k1 = ComparableParameterModel(("similarity_x",), (1.,), .15, 1, .1)
    k2 = ComparableParameterModel(("similarity_x",), (1.,), .15, 2, .1)
    first = replay.recompute(rows, k1, transform)
    second = replay.recompute(rows, k2, transform)
    assert first.iloc[0]["step5_selected_candidates"] == "C2"
    assert second.iloc[0]["step5_selected_candidates"] == "C2|C1"
    assert first.iloc[0][PREFIX + "comparable_price"] != second.iloc[0][PREFIX + "comparable_price"]
    assert first.iloc[0][BASE] != second.iloc[0][BASE]
    changed_input = rows.copy()
    changed_input[BASE] = 1e12  # obsolete scalar base is NOT used
    again = replay.recompute(changed_input, k1, transform)
    assert again.iloc[0][BASE] == first.iloc[0][BASE]
    assert replay.hits == 1


def test_missing_factor_renormalises_target_and_each_peer_separately():
    f = pd.DataFrame({"__target_score__a": [100.], "__target_score__b": [np.nan],
        "__peer0_score__a": [0.], "__peer0_score__b": [100.], "__peer0_alpha": [1.],
        "__peer1_score__a": [np.nan], "__peer1_score__b": [np.nan], "__peer1_alpha": [0.],
        "__peer2_score__a": [np.nan], "__peer2_score__b": [np.nan], "__peer2_alpha": [0.]})
    assert exact_group_effect(f, ["a", "b"], np.array([.25, .25]))[0] == .25


def test_formal_entry_and_backtest_exact_demand_parity_and_safety_independence():
    context = context_fixture()
    transform = transform_for(context)
    comp = ComparableParameterModel(("similarity_x",), (1.,), .15, 2, .1)
    replay = FullChainReplay(lambda key: context, key_column="prediction_checkpoint_key")
    rows = pd.DataFrame(dict(prediction_checkpoint_key=["T@M3"], sku_code=["T"], split=["test"],
        label=[10.], first_actual_sale_date=["2025-04-01"], **{BASE: [999.]}))
    columns = (PREFIX+"delta_factor__ip_social_volume", PREFIX+"delta_factor__ip_social_momentum", PREFIX+"delta_log_price", PREFIX+"competition_trigger_count")
    model = DemandResidualModel(columns, ("nonnegative", "nonnegative", "nonpositive", "nonpositive"),
        ("IP", "IP", "PRICE", "COMPETITION"), BASE, (0.,)*4, (0.,)*4, (1.,)*4,
        (.3, .2, -.2, -.1), 0., .1, 0., "infinity", (.5, 2.),
        PARAMETERS["factors"]["coefficient_bounds"], (.85, 1.15), 1e-6, transform)
    encoded = replay.recompute(rows, comp, transform)
    prediction = predict_demand_residual(model, encoded)
    chosen = select_comparable_parameters(context["frames"]["comparables"], comp)
    formal, _, _ = full_first_order(context, chosen, frozen_transform=transform, learned_model=model)
    assert prediction[0] == pytest.approx(formal.decisions.iloc[0].D_demand_adjusted)
    changed = deepcopy(context)
    changed["parameters"]["first_order"]["safety_stock_quantile_by_tier"] = {k: v*2 for k,v in changed["parameters"]["first_order"]["safety_stock_quantile_by_tier"].items()}
    safe, _, _ = full_first_order(changed, chosen, frozen_transform=transform, learned_model=model)
    assert safe.decisions.iloc[0].D_demand_adjusted == pytest.approx(prediction[0])
    assert safe.decisions.iloc[0].Q_base != formal.decisions.iloc[0].Q_base
    test_predictions, _ = replay_section84_locked_test(rows, model, candidate_detail=context["frames"]["comparables"].assign(prediction_checkpoint_key="T@M3"),
        comparable_model=comp, interval_log_quantiles={"p10": -.1, "p90": .1}, grade_thresholds=(15000,30000,50000), full_replay=replay)
    assert test_predictions.prediction.iloc[0] == pytest.approx(prediction[0])
    assert "trained_order__Q_feas" in test_predictions


def test_bundle_hash_and_config_guard(tmp_path):
    context = context_fixture()
    root = tmp_path / "checkpoint_groups" / "g1"
    StageArtifactStore.write_frame(root / "target_schedule.csv", pd.DataFrame(dict(sku_code=["T"], prediction_checkpoint_key=["T@M3"])))
    bundle = root / "round6" / "_recompute_inputs"
    write_recompute_frames(bundle, context["frames"])
    seal_recompute_bundle(bundle, context["parameters"], context["cutoff"])
    loader = ReplayBundleStore(tmp_path, context["parameters"])
    assert loader("T@M3")["target_sku"] == "T"
    with pytest.raises(ValueError, match="rule/config"):
        ReplayBundleStore(tmp_path, {"bad": 1})("T@M3")
    (bundle / "features.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_recompute_bundle(bundle)


def test_train_cdf_rejects_test_and_records_only_fit_keys():
    context = context_fixture()
    replay = FullChainReplay(lambda key: context, key_column="key")
    rows = pd.DataFrame(dict(key=["train-key"], split=["train"]))
    frozen = replay.fit_transform(rows)
    assert frozen["fit_checkpoint_keys"] == ["train-key"]
    with pytest.raises(ValueError, match="non-Train"):
        replay.fit_transform(rows.assign(split="test"))


def test_ablation_missing_versions_not_fabricated_and_test_rejected():
    oof = pd.DataFrame(dict(key=["a", "b"], label=[100., 200.], base_demand=[300., 300.], prediction=[120., 180.]))
    result = train_ablation(oof=oof, key="key", base_column=BASE, grade_thresholds=(15000,30000,50000), sales_band_edges=(0,500,1000))
    assert result.status.tolist() == ["not_available", "not_available", "evaluated", "evaluated"]
    assert result.iloc[-1].WAPE_change < 0
    with pytest.raises(ValueError, match="Train OOF"):
        train_ablation(oof=oof.assign(split="test"), key="key", base_column=BASE, grade_thresholds=(1,2,3), sales_band_edges=(0,500))


def test_full_section84_keeps_fold_cdf_train_only_and_final_test_locked():
    from test_section84 import _dataset
    from training.section84 import fix_section84_dataset, train_section84_first_order
    frame = _dataset()
    frame[BASE] = frame.base
    column = PREFIX + "delta_factor__form_play_fit"
    frame[column] = .1
    contexts = {}
    candidate_frames = []
    for row in frame.itertuples(index=False):
        context = context_fixture()
        for name in ("features", "comparable_summary"):
            context["frames"][name]["sku_code"] = context["frames"][name].sku_code.replace({"T": row.sku_code})
        context["frames"]["features"].loc[lambda f: f.sku_code.eq(row.sku_code), "form_play_fit"] = float(int(row.sku_code[1:]))
        context["frames"]["comparables"]["target_sku_code"] = row.sku_code
        context["target_sku"] = row.sku_code
        context["cutoff"] = row.decision_cutoff_time
        contexts[row.prediction_checkpoint_key] = context
        candidate_frames.append(context["frames"]["comparables"].assign(prediction_checkpoint_key=row.prediction_checkpoint_key, sku_code=row.sku_code))
    detail = pd.concat(candidate_frames)
    oof_replay = FullChainReplay(lambda key: contexts[key], key_column="prediction_checkpoint_key")
    final_replay = FullChainReplay(lambda key: contexts[key], key_column="prediction_checkpoint_key")
    fixed = fix_section84_dataset(frame, n_time_blocks=5)
    result = train_section84_first_order(fixed=fixed, oof_feature_rows=frame.loc[frame.split.eq("Train")],
        final_feature_rows=frame, oof_candidate_detail=detail, final_candidate_detail=detail,
        initial_similarity_weights={"x": 1.}, comparable_temperature_grid=(.15,), comparable_top_k_grid=(1,),
        similarity_regularization_grid=(.1,), feature_columns=(column,), feature_signs={column: "nonnegative"},
        feature_groups={column: "PRODUCT"}, base_demand_column=BASE, ridge_grid=(.1,), l1_ratio_grid=(0.,),
        half_life_grid=("infinity",), demand_bounds=(.5,2.), coefficient_bounds=PARAMETERS["factors"]["coefficient_bounds"],
        grade_thresholds=(15000,30000,50000), oof_replay=oof_replay, final_replay=final_replay)
    assert result.report["test_rows_consumed"] == 0
    assert result.report["downstream_recompute"] == "full_M3_M4_M5_M6_M7"
    assert len(result.model.frozen_transform["fit_checkpoint_keys"]) == 20
    test_keys = set(frame.loc[frame.split.eq("Test"), "prediction_checkpoint_key"])
    for record in result.fold_models.itertuples(index=False):
        transform = json.loads(record.frozen_transform_json)
        assert not test_keys.intersection(transform["fit_checkpoint_keys"])
        assert len(transform["fit_checkpoint_keys"]) == record.fit_rows
    trained = DemandResidualModel.from_dict(result.model.as_dict())
    assert trained.frozen_transform == result.model.frozen_transform


def test_frozen_social_platform_stats_do_not_fit_on_validation_history():
    from factors.frozen import SOCIAL_METRICS
    contexts = {}
    for key, multiplier in (("fit", 1.), ("validation", 1000.)):
        context = context_fixture()
        context["cutoff"] = "2025-01-30T23:59:59+08:00"
        context["frames"]["features"]["release_dt"] = "2025-04-30"
        history = pd.DataFrame(dict(stat_date=pd.date_range("2025-01-01", periods=30),
            external_ip_key=["a"]*30, platform_code=["p"]*30))
        for metric in SOCIAL_METRICS:
            history[metric] = np.arange(30) * multiplier
        context["frames"]["social_history"] = history
        context["frames"]["social_aliases"] = pd.DataFrame(dict(sku_code=["T", "C1", "C2"], external_ip_key=["a"]*3))
        contexts[key] = context
    replay = FullChainReplay(lambda key: contexts[key], key_column="key")
    frozen = replay.fit_transform(pd.DataFrame(dict(key=["fit"], split=["train"])))
    assert frozen["social_platform_transform"]["p"]["post_count"]["high"] < np.log1p(30)
    contexts["validation"]["frames"]["social_history"]["post_count"] = 1e15
    independent = FullChainReplay(lambda key: contexts[key], key_column="key").fit_transform(pd.DataFrame(dict(key=["fit"], split=["train"])))
    assert independent == frozen


def test_store_week1_sufficient_statistics_preserve_full_placement_output():
    from placement.engine import plan_placement, _normalize_policy
    context = context_fixture()
    f, cfg = context["frames"], context["parameters"]["placement"]
    kwargs = dict(targets=f["features"].loc[f["features"].sku_code.eq("T")], comparables=f["comparables"],
        stores=f["stores"], restored_store_weekly=f["restored_store_weekly"], restored_tier_weekly=f["restored_tier_weekly"], config=cfg)
    complete = plan_placement(**kwargs)
    policy, origin = _normalize_policy(policy=None, store_tiers=sorted(f["stores"].store_tier.unique()), history=f["restored_store_weekly"], config=cfg)
    kwargs["restored_store_weekly"] = f["restored_store_weekly"].loc[f["restored_store_weekly"].lifecycle_week.eq(1)]
    slim = plan_placement(**kwargs, placement_policy=policy.assign(step5_policy_origin=origin))
    pd.testing.assert_frame_equal(complete.store_plan, slim.store_plan)
    pd.testing.assert_frame_equal(complete.demand_curve, slim.demand_curve)
    pd.testing.assert_frame_equal(complete.compliance, slim.compliance)


def test_trained_first_order_cli_uses_bundle_without_raw_source_access(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripts.run_first_order import main
    from data.decision_adapter import DecisionDataAdapter
    context = context_fixture()
    transform = transform_for(context)
    column = PREFIX + "delta_factor__form_play_fit"
    model = DemandResidualModel((column,), ("nonnegative",), ("PRODUCT",), BASE, (0.,), (0.,), (1.,),
        (.3,), 0., .1, 0., "infinity", (.5,2.), PARAMETERS["factors"]["coefficient_bounds"], (.85,1.15), 1e-6, transform)
    comp = ComparableParameterModel(("similarity_x",), (1.,), .15, 2, .1)
    payload = {**model.as_dict(), "model_id": "frozen-test-model", "comparable_model": comp.as_dict(),
               "pipeline_version": "step5-full-chain-v1", "pipeline_parameters": context["parameters"]}
    bundle = tmp_path / "bundle"
    write_recompute_frames(bundle, context["frames"])
    seal_recompute_bundle(bundle, context["parameters"], context["cutoff"])
    StageArtifactStore.write_json(tmp_path / "model.json", payload)
    monkeypatch.setattr(DecisionDataAdapter, "from_project", lambda **kwargs: pytest.fail("trained inference must not read raw data"))
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", ["run_first_order.py", "--data-root", "/server-absent",
        "--model-file", str(tmp_path / "model.json"), "--recompute-input-dir", str(bundle),
        "--sku-code", "T", "--decision-cutoff-time", context["cutoff"], "--output-dir", str(output)])
    assert main() == 0
    chosen = select_comparable_parameters(context["frames"]["comparables"], comp)
    expected, _, _ = full_first_order(context, chosen, frozen_transform=transform, learned_model=model)
    actual = StageArtifactStore.read_frame(output / "first_order_decisions.csv")
    assert actual.D_demand_adjusted.iloc[0] == pytest.approx(expected.decisions.D_demand_adjusted.iloc[0])
    assert actual.Q_feas.iloc[0] == pytest.approx(expected.decisions.Q_feas.iloc[0])


def test_pilot_search_config_keeps_business_rules_and_bounds_unchanged():
    path = Path(__file__).resolve().parents[1] / "config/algorithm_v3_2_step5_pilot.json"
    pilot = json.loads(path.read_text())
    for section, original in PARAMETERS.items():
        if section not in {"training", "parameter_version"}:
            assert pilot[section] == original
    grids = {"comparable_temperature_grid", "comparable_top_k_grid", "similarity_regularization_grid",
             "ridge_grid", "l1_ratio_grid", "half_life_weeks"}
    for name, value in PARAMETERS["training"].items():
        if name in grids:
            assert set(pilot["training"][name]).issubset(value)
        else:
            assert pilot["training"][name] == value
