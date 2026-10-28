"""Convex V3.2 hierarchical Huber regression without a dense dummy matrix."""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config.settings import RestorationConfig


SOLVER_VERSION = "weighted-huber-joint-lbfgsb-v1"
CATEGORICAL_FEATURES = ("store_tier", "city_level", "city_name", "secbiz_area_name", "area_class1_name")
NUMERIC_FEATURES = ("operate_size", "meter_sum", "lifecycle_week")
MODEL_FEATURES = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)


class HierarchicalHuberProblem:
    """Coded categorical/random effects give O(rows * feature_count) evaluations.

    Objective: sum(w * Huber_delta(prediction - log1p(v)))
      + lambda_item * sum(item_effect**2) + lambda_store * sum(store_effect**2).
    All terms share one numerical normalization, so the minimizer is unchanged.
    The Huber threshold is fixed in log-speed units; no residual recentering or
    iteration-dependent MAD changes the objective while it is being solved.
    """

    def __init__(self, frame: pd.DataFrame, config: RestorationConfig):
        self.config = config
        self.y = np.log1p(pd.to_numeric(frame["v_observed_raw"], errors="raise").clip(lower=0).to_numpy(float))
        self.weights = pd.to_numeric(frame["fit_weight"], errors="raise").to_numpy(float)
        if not len(frame) or not np.isfinite(self.y).all() or not np.isfinite(self.weights).all() or (self.weights <= 0).any():
            raise ValueError("Huber training requires finite speeds and positive finite weights")
        self.weight_sum = float(self.weights.sum())
        self.input_audit: dict[str, Any] = {}
        self.numeric_metadata: dict[str, Any] = {}
        self.categorical_metadata: dict[str, Any] = {}
        self.blocks: list[tuple[str, np.ndarray, pd.Index, slice, float]] = []
        self.size = 1
        self.diagonal = [1.0]
        self.bounds: list[tuple[float | None, float | None]] = [(None, None)]
        self.fixed_columns: list[np.ndarray] = []
        self.fixed_names: list[str] = []
        price = pd.to_numeric(frame["mean_price_log_ratio"], errors="coerce").to_numpy(float)
        known_price = np.isfinite(price)
        self.price_active = bool(known_price.sum() >= config.minimum_price_observations and np.std(price[known_price]) > 1e-8)
        if self.price_active:
            self._add_fixed("price", np.where(known_price, price, 0.0), upper=0.0)
        self.input_audit["price"] = {"observed_rows": int(known_price.sum()), "enabled": self.price_active}
        for name in NUMERIC_FEATURES:
            if name not in frame:
                self.input_audit[name] = {"status": "absent_from_fit_frame", "enabled": False}
                continue
            raw = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            if name in ("operate_size", "meter_sum"):
                raw = np.log1p(np.clip(raw, 0.0, None))
                transform = "log1p"
            else:
                transform = "identity"
            known = np.isfinite(raw)
            center = float(np.median(raw[known])) if known.any() else 0.0
            filled = np.where(known, raw, center)
            scale = float(np.std(filled))
            enabled = bool(known.any() and scale > 1e-12)
            metadata = dict(center=center, scale=scale if enabled else 1., transform=transform, coefficient=0., enabled=enabled)
            self.numeric_metadata[name] = metadata
            self.input_audit[name] = dict(observed_rows=int(known.sum()), missing_rows=int((~known).sum()),
                                          enabled=enabled, status="used" if enabled else "all_missing" if not known.any() else "constant")
            if enabled:
                self._add_fixed(name, (filled - center) / scale)
        self.fixed_matrix = np.column_stack(self.fixed_columns) if self.fixed_columns else np.empty((len(frame), 0))
        for name in CATEGORICAL_FEATURES:
            if name not in frame:
                self.input_audit[name] = {"status": "absent_from_fit_frame", "enabled": False}
                continue
            raw = frame[name]
            known = raw.notna() & ~raw.astype(str).isin(("", "__UNKNOWN__", "nan", "None"))
            codes, levels = pd.factorize(raw.where(known, "__UNKNOWN__").astype(str), sort=True)
            self.categorical_metadata[name] = {str(level): 0. for level in levels}
            enabled = bool(len(levels) > 1 and known.any())
            self.input_audit[name] = dict(observed_rows=int(known.sum()), missing_rows=int((~known).sum()), levels=len(levels),
                                          enabled=enabled, status="used" if enabled else "all_missing" if not known.any() else "constant")
            if enabled:
                # Reference-code fixed effects, no undocumented random-effect
                # penalty on city/tier covariates, and no dense one-hot matrix.
                self._add_block(name, codes - 1, levels[1:], ridge=0.)
        for name, column, ridge in (("item", "sku_code", config.item_effect_ridge), ("store", "store_code", config.store_effect_ridge)):
            codes, levels = pd.factorize(frame[column].astype(str), sort=True)
            self._add_block(name, codes, levels, ridge=ridge)
        self.sqrt_diagonal = np.sqrt(np.maximum(np.asarray(self.diagonal), 1e-14))
        self.initial = np.zeros(self.size)
        order = np.argsort(self.y)
        middle = np.searchsorted(np.cumsum(self.weights[order]), self.weight_sum / 2.)
        self.initial[0] = self.y[order[min(middle, len(order)-1)]]

    def _add_fixed(self, name: str, values: np.ndarray, upper: float | None = None) -> None:
        self.fixed_names.append(name)
        self.fixed_columns.append(values)
        self.diagonal.append(float(np.dot(self.weights, values * values) / self.weight_sum))
        self.bounds.append((None, upper))
        self.size += 1

    def _add_block(self, name: str, codes: np.ndarray, levels: pd.Index, ridge: float) -> None:
        block = slice(self.size, self.size + len(levels))
        self.blocks.append((name, codes, levels, block, ridge))
        positive = codes >= 0
        sums = np.bincount(codes[positive], weights=self.weights[positive], minlength=len(levels))
        self.diagonal.extend(((sums + 2. * ridge) / self.weight_sum).tolist())
        self.bounds.extend([(None, None)] * len(levels))
        self.size += len(levels)

    def value_gradient(self, scaled: np.ndarray) -> tuple[float, np.ndarray]:
        parameters = scaled / self.sqrt_diagonal
        prediction = np.full(len(self.y), parameters[0])
        n_fixed = len(self.fixed_names)
        if n_fixed:
            prediction += self.fixed_matrix @ parameters[1:1+n_fixed]
        penalty = 0.
        for _, codes, _, block, ridge in self.blocks:
            values = parameters[block]
            # -1 reference categories index a final zero, without per-row masks.
            prediction += np.append(values, 0.)[codes]
            penalty += ridge * float(np.dot(values, values))
        residual = prediction - self.y
        absolute = np.abs(residual)
        delta = self.config.huber_delta
        loss = np.where(absolute <= delta, .5 * residual * residual, delta * (absolute - .5 * delta))
        influence = self.weights * np.clip(residual, -delta, delta)
        gradient = np.zeros(self.size)
        gradient[0] = influence.sum()
        if n_fixed:
            gradient[1:1+n_fixed] = self.fixed_matrix.T @ influence
        for _, codes, levels, block, ridge in self.blocks:
            # Shift reference code -1 to zero and discard that reference bin.
            gradient[block] = np.bincount(codes + 1, weights=influence, minlength=len(levels)+1)[1:] + 2.*ridge*parameters[block]
        return (float(np.dot(self.weights, loss)) + penalty) / self.weight_sum, gradient / self.weight_sum / self.sqrt_diagonal

    def projected_gradient(self, scaled: np.ndarray, gradient: np.ndarray) -> float:
        projected = gradient.copy()
        for index, (_, upper) in enumerate(self.bounds):
            if upper is not None and scaled[index] >= upper - 1e-12 and projected[index] < 0:
                projected[index] = 0.
        return float(np.max(np.abs(projected)))


def fit_hierarchical_huber(
    frame: pd.DataFrame, *, config: RestorationConfig,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    problem = HierarchicalHuberProblem(frame, config)
    initial_value, _ = problem.value_gradient(problem.initial)
    history: list[dict[str, float]] = []
    previous = problem.initial.copy()
    last_change = 0.
    def callback(parameters: np.ndarray) -> None:
        nonlocal previous, last_change
        last_change = float(np.max(np.abs((parameters - previous) / problem.sqrt_diagonal)))
        previous = parameters.copy()
        if not history or len(history) < 2 or iteration[0] % 10 == 0:
            value, gradient = problem.value_gradient(parameters)
            record = dict(iteration=iteration[0], objective=value, projected_gradient_inf=problem.projected_gradient(parameters, gradient))
            history.append(record)
            if progress: progress("huber_iteration", record)
        iteration[0] += 1
    iteration = [1]
    result = minimize(
        problem.value_gradient, problem.initial, jac=True, method="L-BFGS-B", bounds=problem.bounds,
        callback=callback, options=dict(maxiter=config.huber_max_iter, gtol=config.huber_convergence_tolerance,
                                       ftol=np.finfo(float).eps, maxls=40, maxcor=10),
    )
    value, gradient = problem.value_gradient(result.x)
    kkt = problem.projected_gradient(result.x, gradient)
    # An optimizer's ftol/iteration exit is not itself proof of convergence.
    converged = bool(np.isfinite(value) and kkt <= config.huber_convergence_tolerance)
    parameters = result.x / problem.sqrt_diagonal
    fixed = {name: float(parameters[1+index]) for index, name in enumerate(problem.fixed_names)}
    for name, metadata in problem.numeric_metadata.items():
        metadata["coefficient"] = fixed.get(name, 0.)
    effects: dict[str, dict[str, float]] = {}
    for name, _, levels, block, _ in problem.blocks:
        values = {str(level): float(coefficient) for level, coefficient in zip(levels, parameters[block])}
        if name in CATEGORICAL_FEATURES: problem.categorical_metadata[name].update(values)
        else: effects[name] = values
    report = dict(
        intercept=float(parameters[0]), price_elasticity=fixed.get("price", 0.),
        price_effect_status="estimated" if problem.price_active else "neutral_insufficient_variation",
        observations=len(frame), iterations=int(result.nit), converged=converged,
        convergence_tolerance=config.huber_convergence_tolerance,
        convergence_rule="projected_gradient_inf_after_diagonal_preconditioning <= configured_tolerance",
        final_parameter_change=last_change, final_effect_change=last_change,
        final_projected_gradient_inf=kkt, initial_objective=initial_value, final_objective=value,
        objective_diagnostics_scope="attempted_joint_fit_before_any_prediction_fallback",
        objective_decreased=bool(value <= initial_value + 1e-12), objective_trace=history,
        solver=SOLVER_VERSION, solver_message=str(result.message), function_evaluations=int(result.nfev),
        objective="sum(w*Huber_delta(log1p(v)-Xb-item-store)) + lambda_item*sum(item^2) + lambda_store*sum(store^2)",
        residual_scale_policy="fixed_log_speed_units; no iteration_dependent_MAD_or_residual_recentering",
        numeric_covariates=problem.numeric_metadata, categorical_covariates=problem.categorical_metadata,
        feature_input_audit=problem.input_audit,
        covariate_policy="real available store tier/city/business-area/area and lifecycle-week fixed effects plus item/store random effects",
    )
    return report, effects.get("item", {}), effects.get("store", {})
