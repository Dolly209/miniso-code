from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data.domain import require_columns
from training.modeling import _sample_weights


@dataclass(frozen=True)
class ComparableParameterModel:
    component_columns: tuple[str, ...]
    component_weights: tuple[float, ...]
    softmax_temperature: float
    top_k: int
    regularization: float
    optimizer: str = "legacy_scalar_slsqp"

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "artifact_type": "v3_2_comparable_parameter_model",
            "artifact_version": 1,
            "component_columns": list(self.component_columns),
            "component_weights": list(self.component_weights),
            "softmax_temperature": self.softmax_temperature,
            "top_k": self.top_k,
            "regularization": self.regularization,
            "optimizer": self.optimizer,
        }
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["model_id"] = sha256(rendered.encode("utf-8")).hexdigest()[:20]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComparableParameterModel":
        return cls(
            tuple(str(value) for value in payload["component_columns"]),
            tuple(float(value) for value in payload["component_weights"]),
            float(payload["softmax_temperature"]),
            int(payload["top_k"]),
            float(payload.get("regularization", 0.0)),
            str(payload.get("optimizer", "legacy_scalar_slsqp")),
        )


def _project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto non-negative weights summing to one."""

    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, len(values) + 1)
    valid = ordered - (cumulative - 1.0) / indices > 0
    if not valid.any():
        return np.full(len(values), 1.0 / len(values))
    rho = int(indices[valid][-1])
    threshold = (cumulative[rho - 1] - 1.0) / rho
    return np.maximum(values - threshold, 0.0)


def _selected_candidates(group: pd.DataFrame) -> pd.DataFrame:
    selected = group.loc[
        pd.to_numeric(group["selected"], errors="coerce").fillna(0).eq(1)
        & ~group["candidate_sku_code"].astype(str).str.startswith("__STATIC__")
    ].copy()
    selected["historical_velocity"] = pd.to_numeric(
        selected["historical_velocity"], errors="coerce"
    )
    selected["quality_score"] = pd.to_numeric(
        selected["quality_score"], errors="coerce"
    ).fillna(0.0)
    return selected.loc[
        selected["historical_velocity"].notna()
        & selected["historical_velocity"].ge(0)
    ]


def _prepare_candidate_detail(
    candidate_detail: pd.DataFrame,
    *,
    key_column: str,
    component_columns: Sequence[str],
) -> pd.DataFrame:
    """Validate the checkpoint-keyed Top-K detail without inventing values."""

    required = (
        key_column,
        "candidate_sku_code",
        "selected",
        "alpha_weight",
        "historical_velocity",
        "quality_score",
        "path_priority",
    )
    require_columns(candidate_detail, required, "comparable candidate detail")
    detail = candidate_detail.copy()
    detail[key_column] = detail[key_column].astype(str)
    detail["candidate_sku_code"] = detail["candidate_sku_code"].astype(str)
    for column in component_columns:
        if column not in detail:
            # A missing similarity dimension is unavailable, not zero.  The
            # weighted score below renormalises over the dimensions that are
            # actually present, matching M1's production calculation.
            detail[column] = np.nan
    if detail.duplicated([key_column, "candidate_sku_code"]).any():
        examples = detail.loc[
            detail.duplicated([key_column, "candidate_sku_code"], keep=False),
            [key_column, "candidate_sku_code"],
        ].head(10)
        raise ValueError(
            "duplicate target/comparable rows in candidate detail: "
            f"{examples.to_dict(orient='records')}"
        )
    if "target_sku_code" in detail:
        self_match = detail["target_sku_code"].astype(str).eq(
            detail["candidate_sku_code"]
        )
        if self_match.any():
            raise ValueError("leave-one-SKU-out comparable detail contains self matches")
    return detail


def select_comparable_parameters(
    group: pd.DataFrame,
    model: ComparableParameterModel,
) -> pd.DataFrame:
    if "target_sku_code" in group and (group.target_sku_code.astype(str) == group.candidate_sku_code.astype(str)).any():
        raise ValueError("leave-one-SKU-out candidate envelope contains a self match")
    candidates = _selected_candidates(group)
    if candidates.empty:
        return group.loc[group["selected"].eq(1)].copy()
    component = candidates.loc[:, list(model.component_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    weights = np.asarray(model.component_weights, dtype=float)
    available = np.isfinite(component)
    denominator = available @ weights
    score = np.divide(
        np.nan_to_num(component, nan=0.0) @ weights,
        denominator,
        out=np.zeros(len(candidates), dtype=float),
        where=denominator > 0,
    )
    candidates = candidates.assign(__score=score, similarity_score=score,
                                   similarity_completeness=denominator)
    selected_parts: list[pd.DataFrame] = []
    remaining = model.top_k
    for priority in sorted(
        pd.to_numeric(candidates["path_priority"], errors="coerce")
        .fillna(999)
        .unique()
    ):
        pool = candidates.loc[
            pd.to_numeric(candidates["path_priority"], errors="coerce")
            .fillna(999)
            .eq(priority)
        ].copy()
        pool["__rank_score"] = pool["__score"] * pool["quality_score"]
        pool = pool.sort_values(
            ["__rank_score", "candidate_sku_code"],
            ascending=[False, True],
            kind="stable",
        ).head(remaining)
        selected_parts.append(pool)
        remaining -= len(pool)
        if remaining <= 0:
            break
    chosen = pd.concat(selected_parts, ignore_index=True)
    logits = chosen["__score"].to_numpy(dtype=float) / max(
        model.softmax_temperature, 1e-6
    )
    logits -= logits.max()
    alpha_raw = np.exp(logits) * np.maximum(
        chosen["quality_score"].to_numpy(dtype=float), 1e-6
    )
    alpha = alpha_raw / max(alpha_raw.sum(), 1e-12)
    chosen["alpha_weight"] = alpha
    chosen["selected"] = 1
    return chosen.drop(columns=["__score", "__rank_score"], errors="ignore")


def _anchor(group: pd.DataFrame, model: ComparableParameterModel) -> float | None:
    chosen = select_comparable_parameters(group, model)
    if chosen.empty or chosen["candidate_sku_code"].astype(str).str.startswith("__STATIC__").all():
        return None
    return float(np.dot(chosen["alpha_weight"], chosen["historical_velocity"]))


def apply_comparable_parameters(
    targets: pd.DataFrame,
    candidate_detail: pd.DataFrame,
    model: ComparableParameterModel,
    *,
    key_column: str,
    base_demand_column: str,
) -> np.ndarray:
    """Recalculate D_base from the fold-fitted comparable parameters."""

    require_columns(
        targets, (key_column, base_demand_column), "comparable target rows"
    )
    detail = _prepare_candidate_detail(
        candidate_detail,
        key_column=key_column,
        component_columns=model.component_columns,
    )
    grouped = {
        str(key): group
        for key, group in detail.groupby(key_column, sort=False)
    }
    output: list[float] = []
    for row in targets.itertuples(index=False):
        key = str(getattr(row, key_column))
        base = float(getattr(row, base_demand_column))
        group = grouped.get(key)
        if group is None:
            raise ValueError(f"no candidate detail for target key {key}")
        selected = _selected_candidates(group)
        if selected.empty:
            output.append(base)
            continue
        current_alpha = pd.to_numeric(
            selected["alpha_weight"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        current_velocity = selected["historical_velocity"].to_numpy(dtype=float)
        current_anchor = float(np.dot(current_alpha, current_velocity))
        fitted_anchor = _anchor(group, model)
        if current_anchor <= 1e-12 or fitted_anchor is None:
            output.append(base)
        else:
            output.append(max(base * fitted_anchor / current_anchor, 0.0))
    return np.asarray(output, dtype=float)


def fit_comparable_parameters(
    targets: pd.DataFrame,
    candidate_detail: pd.DataFrame,
    *,
    key_column: str,
    base_demand_column: str,
    initial_component_weights: Mapping[str, float],
    temperature_grid: Sequence[float],
    top_k_grid: Sequence[int],
    regularization: float,
    half_life: int | str,
    full_recompute: Callable[[pd.DataFrame, ComparableParameterModel], pd.DataFrame] | None = None,
) -> ComparableParameterModel:
    """Section 8.3 B fold-local leave-one-SKU-out parameter fitting.

    Candidate identities come from that target's own point-in-time M1 retrieval;
    no global Train candidate list is selected before the rolling fold.
    """

    require_columns(
        targets,
        (key_column, "sku_code", "first_actual_sale_date", "label", base_demand_column),
        "comparable fit targets",
    )
    if not initial_component_weights:
        raise ValueError("initial comparable similarity weights are empty")
    component_columns = tuple(
        f"similarity_{name}" for name in initial_component_weights
    )
    detail = _prepare_candidate_detail(
        candidate_detail,
        key_column=key_column,
        component_columns=component_columns,
    )
    expected = set(targets[key_column].astype(str))
    actual_keys = set(detail[key_column].astype(str))
    missing = sorted(expected - actual_keys)
    if missing:
        raise ValueError(
            "comparable candidate detail does not cover fit targets: "
            f"{missing[:10]}"
        )
    initial = np.asarray(
        [float(initial_component_weights[name]) for name in initial_component_weights],
        dtype=float,
    )
    initial = _project_simplex(initial)
    sample_weights = _sample_weights(targets, half_life)
    actual = pd.to_numeric(targets["label"], errors="raise").to_numpy(dtype=float)
    grouped = {
        str(key): group for key, group in detail.groupby(key_column, sort=False)
    }
    compiled: list[
        tuple[float, float | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for row in targets.itertuples(index=False):
        target_key = str(getattr(row, key_column))
        base = float(getattr(row, base_demand_column))
        selected = _selected_candidates(grouped[target_key])
        if selected.empty:
            compiled.append(
                (
                    base,
                    None,
                    np.empty((0, len(component_columns))),
                    np.empty(0),
                    np.empty(0),
                    np.empty(0),
                )
            )
            continue
        current_alpha = pd.to_numeric(
            selected["alpha_weight"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        velocity = selected["historical_velocity"].to_numpy(dtype=float)
        current_anchor = float(np.dot(current_alpha, velocity))
        compiled.append(
            (
                base,
                current_anchor,
                selected.loc[:, list(component_columns)]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float),
                selected["quality_score"].to_numpy(dtype=float),
                pd.to_numeric(selected["path_priority"], errors="coerce")
                .fillna(999)
                .to_numpy(dtype=float),
                velocity,
            )
        )

    def predictions(theta: np.ndarray, temperature: float, top_k: int) -> np.ndarray:
        if full_recompute is not None:
            fitted = ComparableParameterModel(component_columns, tuple(theta), temperature, top_k, regularization)
            return full_recompute(targets, fitted)[base_demand_column].to_numpy(dtype=float)
        output = np.empty(len(compiled), dtype=float)
        for index, (base, current_anchor, components, quality, priority, velocity) in enumerate(compiled):
            if current_anchor is None or current_anchor <= 1e-12:
                output[index] = base
                continue
            available = np.isfinite(components)
            denominator = available @ theta
            score = np.divide(
                np.nan_to_num(components, nan=0.0) @ theta,
                denominator,
                out=np.zeros(len(components), dtype=float),
                where=denominator > 0,
            )
            # Path priority is a hard rule. Within the same path, the fitted
            # similarity x quality score ranks candidates. Stable index order
            # supplies a deterministic final tie-break.
            order = np.lexsort((np.arange(len(score)), -(score * quality), priority))
            chosen = order[: min(int(top_k), len(order))]
            logits = score[chosen] / max(float(temperature), 1e-6)
            logits -= logits.max()
            alpha_raw = np.exp(logits) * np.maximum(quality[chosen], 1e-6)
            alpha = alpha_raw / max(alpha_raw.sum(), 1e-12)
            fitted_anchor = float(np.dot(alpha, velocity[chosen]))
            output[index] = max(base * fitted_anchor / current_anchor, 0.0)
        return output

    def objective(theta: np.ndarray, temperature: float, top_k: int) -> float:
        prediction = predictions(theta, temperature, top_k)
        if not np.isfinite(prediction).all():
            return float("inf")
        residual = np.log((actual + 1e-6) / (prediction + 1e-6))
        absolute = np.abs(residual)
        huber = np.where(absolute <= 1.0, 0.5 * residual**2, absolute - 0.5)
        return float(
            np.dot(sample_weights, huber)
            + float(regularization) * np.sum((theta - initial) ** 2)
        )

    best: ComparableParameterModel | None = None
    best_objective = np.inf
    for top_k in top_k_grid:
        if int(top_k) <= 0:
            continue
        for temperature in temperature_grid:
            if float(temperature) <= 0:
                continue
            if full_recompute is not None:
                # Projected coordinate descent, not finite differences through
                # integer store selection. All evaluations rebuild M3-M7.
                theta = initial.copy()
                current = objective(theta, float(temperature), int(top_k))
                for step in (0.10, 0.03, 0.01):
                    for index in range(len(theta)):
                        for direction in (-1.0, 1.0):
                            proposal = theta.copy()
                            proposal[index] += direction * step
                            proposal = _project_simplex(proposal)
                            loss = objective(proposal, float(temperature), int(top_k))
                            if loss < current - 1e-10:
                                theta, current = proposal, loss
                model = ComparableParameterModel(component_columns, tuple(map(float, theta)), float(temperature), int(top_k), float(regularization), "projected_coordinate_3_step_sweep_budget_v1")
                if current < best_objective:
                    best, best_objective = model, current
                continue
            result = minimize(
                lambda theta: objective(theta, float(temperature), int(top_k)),
                initial,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * len(initial),
                constraints={"type": "eq", "fun": lambda theta: theta.sum() - 1.0},
                options={"maxiter": 100, "ftol": 1e-9, "disp": False},
            )
            theta = _project_simplex(np.asarray(result.x, dtype=float))
            current = objective(theta, float(temperature), int(top_k))
            model = ComparableParameterModel(
                component_columns,
                tuple(float(value) for value in theta),
                float(temperature),
                int(top_k),
                float(regularization),
            )
            if current < best_objective:
                best, best_objective = model, current
    if best is None:
        raise ValueError("no valid comparable temperature/K configuration")
    return best
