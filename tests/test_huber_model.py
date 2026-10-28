from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import huber

from config.settings import AppConfig
from data.restoration_features import enrich_weekly_store_attributes
from restore.historical import fit_restoration_model, inventory_risk_review_summary
from restore.huber_model import HierarchicalHuberProblem, fit_hierarchical_huber


ROOT = Path(__file__).resolve().parents[1]


def _config(**kwargs):
    return replace(AppConfig.load(ROOT).restoration, **kwargs)


def _frame():
    rng = np.random.default_rng(61)
    n = 120
    return pd.DataFrame({
        "sku_code": np.tile(["A", "B", "C"], n // 3),
        "store_code": np.tile(["S1", "S2", "S3", "S4"], n // 4),
        "cate_level4_code": "C", "store_tier": np.tile(["7|A", "7|B"], n // 2),
        "city_name": np.tile(["X", "Y", "Z"], n // 3),
        "operate_size": rng.uniform(10, 50, n), "meter_sum": np.nan,
        "lifecycle_week": np.tile([1, 2, 3, 4, 5], n // 5),
        "v_observed_raw": np.expm1(rng.uniform(0, 3, n)),
        "fit_weight": rng.uniform(.2, 3, n),
        "mean_price_log_ratio": rng.uniform(-.3, .3, n),
    })


def test_exact_objective_gradient_matches_central_difference():
    problem = HierarchicalHuberProblem(_frame(), _config(minimum_price_observations=2))
    point = np.random.default_rng(2).normal(0, .15, problem.size)
    _, analytical = problem.value_gradient(point)
    numeric = np.zeros(problem.size)
    for i in range(problem.size):
        step = np.zeros(problem.size)
        step[i] = 1e-6
        numeric[i] = (problem.value_gradient(point + step)[0] - problem.value_gradient(point - step)[0]) / 2e-6
    np.testing.assert_allclose(analytical, numeric, rtol=2e-5, atol=1e-7)


def test_solver_matches_independent_dense_reference_with_literal_ridge():
    frame = _frame()[["sku_code", "store_code", "v_observed_raw", "fit_weight", "mean_price_log_ratio"]]
    config = _config(minimum_price_observations=1000)
    problem = HierarchicalHuberProblem(frame, config)
    # Independent small dense implementation, no use of problem gradients or
    # coding blocks, and literal PDF lambda * u^2 (not lambda/2).
    items = pd.get_dummies(frame["sku_code"]).to_numpy(float)
    stores = pd.get_dummies(frame["store_code"]).to_numpy(float)
    design = np.column_stack([np.ones(len(frame)), items, stores])
    y = np.log1p(frame["v_observed_raw"].to_numpy())
    weights = frame["fit_weight"].to_numpy()
    ridge = np.r_[0., np.repeat(config.item_effect_ridge, 3), np.repeat(config.store_effect_ridge, 4)]
    def reference(p):
        residual = design @ p - y
        return ((weights * huber(config.huber_delta, residual)).sum() + (ridge * p * p).sum()) / weights.sum()
    dense = minimize(reference, np.zeros(design.shape[1]), method="BFGS", options={"gtol": 1e-8})
    report, _, _ = fit_hierarchical_huber(frame, config=config)
    assert report["converged"], report
    assert report["final_objective"] == pytest.approx(dense.fun, abs=1e-9)
    # Objective normalization scales every term equally.
    assert problem.value_gradient(problem.initial)[0] == pytest.approx(reference(problem.initial / problem.sqrt_diagonal))


def test_zero_mad_heavy_zeros_have_fixed_objective_and_finite_solution():
    frame = _frame()
    frame["v_observed_raw"] = np.r_[np.zeros(115), [1., 3., 9., 30., 500.]]
    report, _, _ = fit_hierarchical_huber(frame, config=_config())
    assert report["objective_decreased"]
    assert np.isfinite(report["final_projected_gradient_inf"])
    assert "no iteration_dependent_MAD" in report["residual_scale_policy"]
    values = [row["objective"] for row in report["objective_trace"]]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))
    assert report["feature_input_audit"]["meter_sum"]["status"] == "all_missing"
    assert not report["numeric_covariates"]["meter_sum"]["enabled"]


def test_positive_price_slope_stays_on_nonpositive_constraint():
    frame = _frame().drop(columns=["store_tier", "city_name", "operate_size", "meter_sum", "lifecycle_week"])
    frame["mean_price_log_ratio"] = np.linspace(-1, 1, len(frame))
    frame["v_observed_raw"] = np.expm1(2 + frame["mean_price_log_ratio"])
    report, _, _ = fit_hierarchical_huber(frame, config=_config(minimum_price_observations=2))
    assert report["converged"]
    assert report["price_elasticity"] == pytest.approx(0., abs=1e-12)


def test_failed_global_fit_is_actually_intercept_only():
    fitted = fit_restoration_model(_frame(), config=_config(huber_max_iter=1, huber_convergence_tolerance=1e-12))
    report, items, stores = fitted.models["__GLOBAL__"]
    assert not report["converged"]
    assert report["prediction_policy"] == "intercept_only_nonconverged_global_fallback"
    assert not report["numeric_covariates"] and not report["categorical_covariates"]
    assert not items and not stores
    assert report["unsuccessful_fit_covariates"]["numeric"]


def test_store_enrichment_keeps_grain_and_uses_real_snapshot_values():
    weekly = pd.DataFrame({"store_code": ["S1", "S1", "S2"], "city_name": [None]*3, "clean_sales_qty": [2, 3, 4]})
    stores = pd.DataFrame({"store_code": ["S1"], "city_name": ["Shanghai"], "operate_size": [120.]})
    enriched = enrich_weekly_store_attributes(weekly, stores)
    assert enriched["clean_sales_qty"].tolist() == [2, 3, 4]
    assert enriched["city_name"].iloc[:2].eq("Shanghai").all()
    assert pd.isna(enriched.loc[2, "city_name"])
    with pytest.raises(ValueError, match="unique"):
        enrich_weekly_store_attributes(weekly, pd.concat([stores, stores]))


def test_inventory_signal_is_pending_not_confirmed_or_automatically_accepted():
    rows = [{"sku_code": "A", "stock_grain_unreliable": True}, {"sku_code": "B", "stock_grain_unreliable": False}]
    result = inventory_risk_review_summary(rows)
    assert result["pending_review_skus"] == ["A"]
    assert result["confirmed_unrestorable_sku_count"] == 0
    assert not result["confirmation_evidence_available"]
    assert rows[0]["stock_grain_evidence_status"] == "pending_review"
    assert not rows[0]["stock_grain_confirmed_unrestorable"]
