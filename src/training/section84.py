from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from data.domain import require_columns
from training.comparable_tuning import (
    ComparableParameterModel,
    apply_comparable_parameters,
    fit_comparable_parameters,
)
from training.modeling import _evaluation_weights, _sample_weights
from training.split import SkuSample, build_time_blocks, expanding_window_folds
from training.weights import V3_2_SALES_BAND_EDGES


@dataclass(frozen=True)
class FixedDataset:
    rows: pd.DataFrame
    exclusions: pd.DataFrame
    time_blocks: pd.DataFrame
    folds: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class DemandResidualModel:
    feature_columns: tuple[str, ...]
    feature_signs: tuple[str, ...]
    feature_groups: tuple[str, ...]
    base_demand_column: str
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    ridge: float
    l1_ratio: float
    half_life_weeks: str
    demand_bounds: tuple[float, float]
    coefficient_bounds: dict[str, tuple[float, float]]
    price_bounds: tuple[float, float]
    epsilon: float
    frozen_transform: dict[str, Any] = field(default_factory=dict)
    interval_log_quantiles: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "v3_2_first_order_demand_residual_model",
            "artifact_version": 2 if self.frozen_transform else 1,
            "model_type": "bounded_weighted_log_ratio_elastic_net",
            "feature_columns": list(self.feature_columns),
            "feature_signs": list(self.feature_signs),
            "feature_groups": list(self.feature_groups),
            "base_demand_column": self.base_demand_column,
            "medians": list(self.medians),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "ridge": self.ridge,
            "l1_ratio": self.l1_ratio,
            "half_life_weeks": self.half_life_weeks,
            "demand_bounds": list(self.demand_bounds),
            "coefficient_bounds": {
                key: list(value) for key, value in self.coefficient_bounds.items()
            },
            "price_bounds": list(self.price_bounds),
            "epsilon": self.epsilon,
            "frozen_transform": self.frozen_transform,
            "oof_log_residual_quantiles": self.interval_log_quantiles,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DemandResidualModel":
        return cls(
            tuple(str(value) for value in payload["feature_columns"]),
            tuple(str(value) for value in payload["feature_signs"]),
            tuple(
                str(value)
                for value in payload.get(
                    "feature_groups", ["DEMAND"] * len(payload["feature_columns"])
                )
            ),
            str(payload["base_demand_column"]),
            tuple(float(value) for value in payload["medians"]),
            tuple(float(value) for value in payload["means"]),
            tuple(float(value) for value in payload["scales"]),
            tuple(float(value) for value in payload["coefficients"]),
            float(payload["intercept"]),
            float(payload["ridge"]),
            float(payload.get("l1_ratio", 0.0)),
            str(payload["half_life_weeks"]),
            tuple(float(value) for value in payload["demand_bounds"]),
            {
                str(key): tuple(float(item) for item in value)
                for key, value in payload.get(
                    "coefficient_bounds",
                    {"DEMAND": payload["demand_bounds"]},
                ).items()
            },
            tuple(float(value) for value in payload.get("price_bounds", (0.85, 1.15))),
            float(payload.get("epsilon", 1e-6)),
            dict(payload.get("frozen_transform", {})),
            dict(payload.get("oof_log_residual_quantiles", {})),
        )


@dataclass(frozen=True)
class Section84TrainingResult:
    model: DemandResidualModel
    comparable_model: ComparableParameterModel
    oof_predictions: pd.DataFrame
    model_selection: pd.DataFrame
    fold_models: pd.DataFrame
    report: dict[str, Any]


def replay_section84_locked_test(
    frame: pd.DataFrame,
    model: DemandResidualModel,
    *,
    candidate_detail: pd.DataFrame,
    comparable_model: ComparableParameterModel,
    interval_log_quantiles: Mapping[str, float],
    grade_thresholds: Sequence[float],
    sales_band_edges: Sequence[float] = V3_2_SALES_BAND_EDGES,
    full_replay=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the frozen final model on Test without refitting anything."""

    require_columns(
        frame,
        (
            "sku_code",
            "split",
            "label",
            "first_actual_sale_date",
            model.base_demand_column,
        ),
        "locked_test_dataset",
    )
    test = frame.loc[
        frame["split"].astype(str).str.lower().eq("test")
    ].copy()
    if test.empty:
        raise ValueError("locked Test dataset has no Test rows")
    key = (
        "prediction_checkpoint_key"
        if "prediction_checkpoint_key" in test
        else "sku_code"
    )
    if full_replay is not None:
        full_replay.validate_inputs(test, candidate_detail)
        test = full_replay.recompute(test, comparable_model, model.frozen_transform)
    else:
        if model.frozen_transform:
            raise ValueError("full-chain model requires recompute inputs; scalar approximation is forbidden")
        test[model.base_demand_column] = apply_comparable_parameters(
            test, candidate_detail, comparable_model,
            key_column=key, base_demand_column=model.base_demand_column,
        )
    prediction = predict_demand_residual(model, test)
    actual = pd.to_numeric(test["label"], errors="raise").to_numpy(dtype=float)
    weights = np.ones(len(test), dtype=float) / len(test)
    metrics = _metrics(
        actual,
        prediction,
        weights,
        grade_thresholds=grade_thresholds,
        sales_band_edges=sales_band_edges,
    )
    p10 = float(interval_log_quantiles["p10"])
    p90 = float(interval_log_quantiles["p90"])
    result = test.copy()
    result["prediction"] = prediction
    if full_replay is not None:
        orders = full_replay.infer_orders(test, comparable_model, model).set_index(key)
        if not np.allclose(prediction, orders.loc[result[key], "D_demand_adjusted"], rtol=1e-10, atol=1e-8):
            raise AssertionError("formal first-order / backtest pure-demand parity failed")
        for name in ("D_sell", "D_demand_adjusted", "Q_base", "Q_feas", "grade", "supply_action"):
            if name in orders:
                result["trained_order__" + name] = result[key].map(orders[name])
    result["prediction_p10"] = np.maximum(prediction * np.exp(p10), 0.0)
    result["prediction_p90"] = np.maximum(prediction * np.exp(p90), 0.0)
    result["actual_demand_grade"] = _grade(actual, grade_thresholds)
    result["prediction_demand_grade"] = _grade(prediction, grade_thresholds)
    result["actual_grade"] = result["actual_demand_grade"]
    result["prediction_grade"] = result["prediction_demand_grade"]
    supply_grade_column = next(
        (
            column
            for column in (
                "feature__first_order__supply_grade",
                "feature__first_order__grade",
            )
            if column in result
        ),
        None,
    )
    result["input_supply_feasible_grade"] = (
        result[supply_grade_column] if supply_grade_column else pd.NA
    )
    if "trained_order__grade" in result:
        result["input_supply_feasible_grade"] = result["trained_order__grade"]
    result["supply_grade_evaluation_status"] = (
        "reported_separately_not_compared_with_sales_demand_label"
    )
    confusion = (
        result.groupby(
            ["actual_demand_grade", "prediction_demand_grade"], dropna=False
        )
        .size()
        .rename("count")
        .reset_index()
        .to_dict(orient="records")
    )
    report = {
        "spec_version": "3.2",
        "section": "8.4 step 7 / 8.5",
        "status": "locked_test_replayed_once",
        "test_rows": int(len(result)),
        "test_skus": int(result["sku_code"].astype(str).nunique()),
        "model_refit_during_test": False,
        "model_selection_reopened": False,
        "comparable_model_id": comparable_model.as_dict()["model_id"],
        "metrics": metrics,
        "metric_quantity_scope": "pure_sales_demand_piece_quantity",
        "grade_scope": "demand_grade_only; Q_feas supply grade is separate",
        "band_mae_definition": (
            "equal mean of unweighted quantity MAE across non-empty actual-sales bands"
        ),
        "confusion": confusion,
    }
    return result, report


def _stable_frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    normalized = frame.loc[:, list(columns)].copy()
    for column in normalized:
        normalized[column] = normalized[column].astype("string").fillna("<NA>")
    normalized = normalized.sort_values(list(columns), kind="stable")
    content = normalized.to_csv(index=False, lineterminator="\n")
    return sha256(content.encode("utf-8")).hexdigest()


def fix_section84_dataset(
    frame: pd.DataFrame,
    *,
    n_time_blocks: int = 5,
) -> FixedDataset:
    """Freeze the Section 8.4 dataset before any model selection.

    Invalid rows are recorded rather than silently filled.  Ambiguous duplicate
    keys or a non-chronological locked split are hard failures because there is
    no defensible automatic repair for either condition.
    """

    require_columns(
        frame,
        (
            "sku_code",
            "dataset_version",
            "first_actual_sale_date",
            "split",
            "label",
        ),
        "section_8_4_dataset",
    )
    key = (
        "prediction_checkpoint_key"
        if "prediction_checkpoint_key" in frame
        else "sku_code"
    )
    rows = frame.copy()
    rows["sku_code"] = rows["sku_code"].astype("string").str.strip()
    rows[key] = rows[key].astype("string").str.strip()
    parsed_date = pd.to_datetime(rows["first_actual_sale_date"], errors="coerce")
    numeric_label = pd.to_numeric(rows["label"], errors="coerce")
    split = rows["split"].astype("string").str.strip().str.lower()
    version = rows["dataset_version"].astype("string").str.strip()

    reasons = pd.Series("", index=rows.index, dtype="string")

    def add_reason(mask: pd.Series, value: str) -> None:
        existing = reasons.loc[mask]
        reasons.loc[mask] = np.where(existing.eq(""), value, existing + "|" + value)

    add_reason(rows[key].isna() | rows[key].eq(""), "missing_primary_key")
    add_reason(rows["sku_code"].isna() | rows["sku_code"].eq(""), "missing_sku_code")
    add_reason(parsed_date.isna(), "invalid_first_actual_sale_date")
    add_reason(~split.isin({"train", "test"}), "invalid_split")
    add_reason(version.isna() | version.eq(""), "missing_dataset_version")
    add_reason(numeric_label.isna(), "immature_or_missing_label")
    add_reason(numeric_label.lt(0), "negative_label")
    exclusion_mask = reasons.ne("")
    exclusion_columns = [key, "sku_code"] if key != "sku_code" else ["sku_code"]
    exclusions = rows.loc[exclusion_mask, exclusion_columns].copy()
    exclusions["exclusion_reason"] = reasons.loc[exclusion_mask]

    rows = rows.loc[~exclusion_mask].copy()
    rows["first_actual_sale_date"] = parsed_date.loc[rows.index].dt.date
    rows["label"] = numeric_label.loc[rows.index].astype(float)
    rows["split"] = split.loc[rows.index]
    rows["dataset_version"] = version.loc[rows.index]
    if rows.empty:
        raise ValueError("no eligible rows remain after Section 8.4 data checks")
    if rows[key].duplicated().any():
        examples = rows.loc[rows[key].duplicated(keep=False), key].head(10).tolist()
        raise ValueError(f"Section 8.4 primary key is not unique: {examples}")
    versions = sorted(set(rows["dataset_version"]))
    if len(versions) != 1:
        raise ValueError(
            "Section 8.4 requires exactly one fixed dataset_version; got "
            f"{versions}"
        )
    sku_consistency = rows.groupby("sku_code", dropna=False).agg(
        first_sale_values=("first_actual_sale_date", "nunique"),
        split_values=("split", "nunique"),
        label_values=("label", "nunique"),
    )
    if sku_consistency.gt(1).any(axis=1).any():
        examples = sku_consistency.loc[
            sku_consistency.gt(1).any(axis=1)
        ].head(10).index.tolist()
        raise ValueError(
            "one SKU has conflicting date/split/label values: " f"{examples}"
        )

    identities = rows[
        ["sku_code", "first_actual_sale_date", "split"]
    ].drop_duplicates("sku_code")
    train_identities = identities.loc[identities["split"].eq("train")].copy()
    test_identities = identities.loc[identities["split"].eq("test")].copy()
    if train_identities.empty or test_identities.empty:
        raise ValueError("fixed Section 8.4 data requires non-empty Train and Test")
    shared_dates = set(train_identities["first_actual_sale_date"]) & set(
        test_identities["first_actual_sale_date"]
    )
    if shared_dates:
        raise ValueError(
            "same-day first-sale SKUs are split across Train/Test: "
            f"{sorted(shared_dates)[:10]}"
        )
    if train_identities["first_actual_sale_date"].max() >= test_identities[
        "first_actual_sale_date"
    ].min():
        raise ValueError("Train/Test is not a strictly chronological locked split")

    samples = [
        SkuSample(row.first_actual_sale_date, str(row.sku_code))
        for row in train_identities.itertuples(index=False)
    ]
    blocks = build_time_blocks(samples, n_time_blocks=n_time_blocks)
    folds = expanding_window_folds(blocks)
    block_by_sku: dict[str, str] = {}
    block_rows: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks, start=1):
        for sample in block:
            code = f"B{block_index}"
            block_by_sku[sample.sku_code] = code
            block_rows.append(
                {
                    "sku_code": sample.sku_code,
                    "first_actual_sale_date": sample.first_actual_sale_date,
                    "time_block": code,
                }
            )
    rows["time_block"] = rows["sku_code"].map(block_by_sku)
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (fit_samples, validation_samples) in enumerate(folds, start=1):
        fit_blocks = sorted(
            {block_by_sku[sample.sku_code] for sample in fit_samples}
        )
        validation_blocks = {
            block_by_sku[sample.sku_code] for sample in validation_samples
        }
        if len(validation_blocks) != 1:
            raise AssertionError("one rolling fold must validate exactly one block")
        validation_block = next(iter(validation_blocks))
        fold_rows.append(
            {
                "fold": fold_index,
                "fit_blocks": "+".join(fit_blocks),
                "validation_block": validation_block,
                "fit_skus": len({sample.sku_code for sample in fit_samples}),
                "validation_skus": len(
                    {sample.sku_code for sample in validation_samples}
                ),
                "validation_first_sale_min": min(
                    sample.first_actual_sale_date for sample in validation_samples
                ),
                "validation_first_sale_max": max(
                    sample.first_actual_sale_date for sample in validation_samples
                ),
            }
        )
    hash_columns = [
        key,
        "sku_code",
        "dataset_version",
        "first_actual_sale_date",
        "split",
        "label",
    ]
    manifest = {
        "spec_version": "3.2",
        "section": "8.4",
        "dataset_version": versions[0],
        "dataset_hash": _stable_frame_hash(rows, hash_columns),
        "primary_key": key,
        "eligible_rows": int(len(rows)),
        "excluded_rows": int(len(exclusions)),
        "train_skus": int(train_identities["sku_code"].nunique()),
        "test_skus": int(test_identities["sku_code"].nunique()),
        "time_blocks": n_time_blocks,
        "rolling_folds": len(folds),
        "test_rows_consumed_during_fix": 0,
        "fixed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return FixedDataset(
        rows.reset_index(drop=True),
        exclusions.reset_index(drop=True),
        pd.DataFrame(block_rows),
        pd.DataFrame(fold_rows),
        manifest,
    )


def _apply_sign(value: float, sign: str) -> float:
    if sign == "nonnegative":
        return max(value, 0.0)
    if sign == "nonpositive":
        return min(value, 0.0)
    if sign == "free":
        return value
    raise ValueError(f"unknown coefficient sign constraint: {sign}")


def _production_multiplier(
    x: np.ndarray,
    coefficients: np.ndarray,
    feature_groups: Sequence[str],
    *,
    demand_bounds: tuple[float, float],
    coefficient_bounds: Mapping[str, tuple[float, float]],
    price_bounds: tuple[float, float],
    frame: pd.DataFrame | None = None,
    feature_columns: Sequence[str] = (),
    score_arrays: Mapping[str, tuple] | None = None,
) -> np.ndarray:
    """Evaluate the bounded multiplicative M5 production formula."""

    output = np.ones(x.shape[0], dtype=float)
    groups = np.asarray(feature_groups, dtype=object)
    for group in sorted(set(feature_groups) - {"PRICE", "COMPETITION"}):
        indices = np.flatnonzero(groups == group)
        names = [feature_columns[i].split("delta_factor__", 1)[1] for i in indices
                 if feature_columns and "delta_factor__" in feature_columns[i]]
        if names and len(names) == len(indices) and frame is not None and all("__target_score__" + name in frame for name in names):
            from first_order.pipeline import exact_group_effect
            effect = exact_group_effect(frame, names, coefficients[indices], (score_arrays or {}).get(group))
        else:
            effect = x[:, indices] @ coefficients[indices]
        lower, upper = coefficient_bounds[group]
        output *= np.clip(1.0 + effect, lower, upper)
    price_indices = np.flatnonzero(groups == "PRICE")
    if len(price_indices):
        price_effect = x[:, price_indices] @ coefficients[price_indices]
        output *= np.clip(np.exp(price_effect), price_bounds[0], price_bounds[1])
    competition_indices = np.flatnonzero(groups == "COMPETITION")
    if len(competition_indices):
        competition_effect = x[:, competition_indices] @ coefficients[
            competition_indices
        ]
        # Competition parameters are non-positive, so an active rule can only
        # reduce demand.  The global demand protection band remains the final
        # guardrail exactly as in production.
        output *= np.exp(np.minimum(competition_effect, 0.0))
    if frame is not None and "__external_coefficient" in frame:
        output *= frame["__external_coefficient"].to_numpy(float)
    return np.clip(output, demand_bounds[0], demand_bounds[1])


def fit_demand_residual_model(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    feature_signs: Mapping[str, str],
    feature_groups: Mapping[str, str] | None = None,
    base_demand_column: str,
    ridge: float,
    l1_ratio: float,
    half_life: int | str,
    demand_bounds: tuple[float, float],
    coefficient_bounds: Mapping[str, Sequence[float]] | None = None,
    price_bounds: tuple[float, float] = (0.85, 1.15),
    epsilon: float = 1e-6,
) -> DemandResidualModel:
    if frame.empty:
        raise ValueError("cannot train a demand residual model on an empty frame")
    require_columns(
        frame,
        ("sku_code", "first_actual_sale_date", "label", base_demand_column),
        "section_8_4_fit_frame",
    )
    if not 0 <= l1_ratio <= 1:
        raise ValueError("l1_ratio must be between 0 and 1")
    lower, upper = map(float, demand_bounds)
    if not 0 < lower <= upper:
        raise ValueError("demand bounds must satisfy 0 < lower <= upper")
    raw = frame.loc[:, list(feature_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    # M5 relative deltas are neutral at zero.  Standardising them would move
    # the business meaning of zero and would no longer call the production
    # clip/multiplication formula from Sections 5.4 and 6.2.
    medians = np.zeros(raw.shape[1], dtype=float)
    x = np.where(np.isfinite(raw), raw, medians)
    means = np.zeros(raw.shape[1], dtype=float)
    scales = np.ones(raw.shape[1], dtype=float)
    actual = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=float)
    base = pd.to_numeric(
        frame[base_demand_column], errors="raise"
    ).to_numpy(dtype=float)
    if (actual < 0).any() or (base < 0).any():
        raise ValueError("label and base demand must be non-negative")
    if not np.isfinite(actual).all() or not np.isfinite(base).all():
        raise ValueError("label/base demand unavailable; blocked forecasts must not enter parameter fitting")
    target = np.log((actual + epsilon) / (base + epsilon))
    weights = _sample_weights(frame, half_life)
    coefficients = np.zeros(x.shape[1], dtype=float)
    intercept = 0.0
    l1_penalty = float(ridge) * float(l1_ratio)
    l2_penalty = float(ridge) * (1.0 - float(l1_ratio))
    ordered_signs = tuple(feature_signs.get(name, "nonnegative") for name in feature_columns)
    groups = tuple(
        (feature_groups or {}).get(
            name,
            "PRICE"
            if name.endswith("delta_log_price")
            else "COMPETITION"
            if name.endswith("competition_trigger_count")
            else "DEMAND",
        )
        for name in feature_columns
    )
    bounds = {
        str(key): tuple(float(item) for item in value)
        for key, value in (coefficient_bounds or {"DEMAND": demand_bounds}).items()
    }
    for group in set(groups) - {"PRICE", "COMPETITION"}:
        bounds.setdefault(group, (lower, upper))
    price_bounds = tuple(float(value) for value in price_bounds)
    score_arrays = {}
    from first_order.pipeline import compile_group_scores
    for group in set(groups) - {"PRICE", "COMPETITION"}:
        names = [name.split("delta_factor__", 1)[1] for name, g in zip(feature_columns, groups)
                 if g == group and "delta_factor__" in name]
        if names and all("__target_score__" + name in frame for name in names):
            score_arrays[group] = compile_group_scores(frame, names)

    def multiplier(values: np.ndarray) -> np.ndarray:
        return _production_multiplier(
            x,
            values,
            groups,
            demand_bounds=(lower, upper),
            coefficient_bounds=bounds,
            price_bounds=price_bounds,
            frame=frame,
            feature_columns=feature_columns,
            score_arrays=score_arrays,
        )

    def objective(values: np.ndarray) -> float:
        residual = target - np.log(np.maximum(multiplier(values), epsilon))
        absolute = np.abs(residual)
        huber = np.where(absolute <= 1.0, 0.5 * residual**2, absolute - 0.5)
        penalty = l1_penalty * np.abs(values).sum() + l2_penalty * np.dot(values, values)
        return float(np.dot(weights, huber) + penalty)

    # Constrained Elastic Net supplies a deterministic initial point. The
    # subsequent coordinate updates always evaluate the actual M5 clip/product
    # formula, including per-object availability renormalisation.
    linear_target = target - np.log(np.maximum(
        frame.get("__external_coefficient", pd.Series(1., index=frame.index)).to_numpy(float), epsilon))
    for _ in range(50):
        before = coefficients.copy()
        residual = linear_target - x @ coefficients
        for index in range(len(coefficients)):
            partial = residual + x[:, index] * coefficients[index]
            numerator = float(np.dot(weights * x[:, index], partial))
            denominator = float(np.dot(weights, x[:, index] ** 2)) + 2 * l2_penalty
            value = np.sign(numerator) * max(abs(numerator) - l1_penalty, 0.) / max(denominator, epsilon)
            coefficients[index] = _apply_sign(float(value), ordered_signs[index])
            residual = partial - x[:, index] * coefficients[index]
        if np.max(np.abs(coefficients-before), initial=0.) < 1e-8:
            break
    current = objective(coefficients)
    finite_step = 1e-5
    for iteration in range(80):
        before = coefficients.copy()
        for index in range(len(coefficients)):
            plus = coefficients.copy()
            minus = coefficients.copy()
            plus[index] += finite_step
            minus[index] -= finite_step
            gradient = (objective(plus) - objective(minus)) / (2 * finite_step)
            step = .25 / np.sqrt(iteration + 1.)
            for _ in range(12):
                proposal = coefficients.copy()
                proposal[index] = _apply_sign(float(coefficients[index]-step*gradient), ordered_signs[index])
                loss = objective(proposal)
                if loss <= current + 1e-12:
                    coefficients, current = proposal, loss
                    break
                step *= .5
        if np.max(np.abs(coefficients-before), initial=0.) < 1e-8:
            break
    return DemandResidualModel(
        tuple(feature_columns),
        ordered_signs,
        groups,
        base_demand_column,
        tuple(float(value) for value in medians),
        tuple(float(value) for value in means),
        tuple(float(value) for value in scales),
        tuple(float(value) for value in coefficients),
        intercept,
        float(ridge),
        float(l1_ratio),
        str(half_life),
        (lower, upper),
        bounds,
        price_bounds,
        float(epsilon),
    )


def predict_demand_residual(
    model: DemandResidualModel, frame: pd.DataFrame
) -> np.ndarray:
    require_columns(frame, (model.base_demand_column,), "prediction_frame")
    raw = frame.loc[:, list(model.feature_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    x = np.where(np.isfinite(raw), raw, np.asarray(model.medians, dtype=float))
    base = pd.to_numeric(
        frame[model.base_demand_column], errors="raise"
    ).to_numpy(dtype=float)
    if (base < 0).any():
        raise ValueError("base demand must be non-negative")
    if not np.isfinite(base).all():
        raise ValueError("base demand unavailable; inspect blocked placement diagnostics before prediction/evaluation")
    multiplier = _production_multiplier(
        x,
        np.asarray(model.coefficients, dtype=float),
        model.feature_groups,
        demand_bounds=model.demand_bounds,
        coefficient_bounds=model.coefficient_bounds,
        price_bounds=model.price_bounds,
        frame=frame,
        feature_columns=model.feature_columns,
    )
    return np.maximum(base * multiplier, 0.0)


def _grade(values: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    ordered = np.asarray(sorted(float(value) for value in thresholds), dtype=float)
    return np.searchsorted(ordered, np.asarray(values, dtype=float), side="right")


def _metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    *,
    grade_thresholds: Sequence[float],
    sales_band_edges: Sequence[float] = V3_2_SALES_BAND_EDGES,
) -> dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(weights.sum(), 1e-12)
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - actual
    denominator = max(float(np.dot(weights, np.abs(actual))), 1e-12)
    actual_grade = _grade(actual, grade_thresholds)
    prediction_grade = _grade(prediction, grade_thresholds)
    grade_distance = np.abs(prediction_grade - actual_grade)
    band_edges = np.asarray(
        sorted({float(value) for value in sales_band_edges}), dtype=float
    )
    if len(band_edges) < 2 or band_edges[0] != 0:
        raise ValueError(
            "sales_band_edges must contain at least two unique values and start at 0"
        )
    actual_band = np.searchsorted(band_edges[1:], actual, side="right")
    band_quantity_mae = [
        float(np.mean(np.abs(error[actual_band == band])))
        for band in np.unique(actual_band)
    ]
    return {
        "WAPE": float(np.dot(weights, np.abs(error)) / denominator),
        "Bias": float(np.dot(weights, error) / denominator),
        "BandMAE": float(np.mean(band_quantity_mae)),
        "GradeHit": float(np.dot(weights, grade_distance == 0)),
        "GradeHitAdj": float(np.dot(weights, grade_distance <= 1)),
        "GradeCross": float(np.dot(weights, grade_distance >= 2)),
    }


def _model_id(model: DemandResidualModel) -> str:
    payload = json.dumps(
        model.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:20]


def train_section84_first_order(
    *,
    fixed: FixedDataset,
    oof_feature_rows: pd.DataFrame,
    final_feature_rows: pd.DataFrame,
    oof_candidate_detail: pd.DataFrame,
    final_candidate_detail: pd.DataFrame,
    initial_similarity_weights: Mapping[str, float],
    comparable_temperature_grid: Sequence[float],
    comparable_top_k_grid: Sequence[int],
    similarity_regularization_grid: Sequence[float],
    feature_columns: Sequence[str],
    feature_signs: Mapping[str, str],
    feature_groups: Mapping[str, str] | None = None,
    base_demand_column: str,
    ridge_grid: Sequence[float],
    l1_ratio_grid: Sequence[float],
    half_life_grid: Sequence[int | str],
    demand_bounds: tuple[float, float],
    coefficient_bounds: Mapping[str, Sequence[float]] | None = None,
    price_bounds: tuple[float, float] = (0.85, 1.15),
    grade_thresholds: Sequence[float],
    sales_band_edges: Sequence[float] = V3_2_SALES_BAND_EDGES,
    oof_replay=None,
    final_replay=None,
    progress=None,
    fixed_hyperparameters: bool = False,
) -> Section84TrainingResult:
    """Run the first-order portion of V3.2 Section 8.4.

    ``oof_feature_rows`` and ``oof_candidate_detail`` contain point-in-time
    upstream outputs for every fixed Train row. ``final_*`` contains the
    separately replayed all-Train and locked-Test inputs.  Comparable
    identities are the target's own M1 Top-K at that cutoff; each fold refits
    theta/tau/K only on its historical fit blocks before applying them to the
    next block. Test labels are validated for identity but never used for fit,
    selection, or calibration.

    With ``fixed_hyperparameters`` this is a diagnostic run with one preset
    outer configuration. Theta and tau still train under Section 8.3B, demand
    coefficients still train under 8.3C, with four OOF fits and a final refit.
    """

    key = str(fixed.manifest["primary_key"])
    if oof_replay is not None:
        oof_replay.validate_inputs(oof_feature_rows, oof_candidate_detail)
    if final_replay is not None:
        # Validate identities without fitting any transform or predicting Test.
        final_replay.validate_inputs(final_feature_rows, final_candidate_detail)
    identity_columns = [key, "sku_code", "split", "first_actual_sale_date", "label"]
    require_columns(oof_feature_rows, identity_columns, "OOF feature rows")
    require_columns(final_feature_rows, identity_columns, "final feature rows")
    for column in (*feature_columns, base_demand_column):
        if column not in oof_feature_rows or column not in final_feature_rows:
            raise ValueError(f"required training feature is missing: {column}")
    require_columns(
        oof_candidate_detail,
        (key, "candidate_sku_code", "selected", "historical_velocity"),
        "OOF comparable candidate detail",
    )
    require_columns(
        final_candidate_detail,
        (key, "candidate_sku_code", "selected", "historical_velocity"),
        "final comparable candidate detail",
    )

    fixed_rows = fixed.rows.copy()
    train_fixed = fixed_rows.loc[fixed_rows["split"].eq("train")].copy()
    test_fixed = fixed_rows.loc[fixed_rows["split"].eq("test")].copy()
    oof = oof_feature_rows.copy()
    oof["split"] = oof["split"].astype(str).str.lower()
    oof = oof.loc[oof["split"].eq("train")].copy()
    if set(oof[key].astype(str)) != set(train_fixed[key].astype(str)):
        missing = sorted(set(train_fixed[key].astype(str)) - set(oof[key].astype(str)))
        extra = sorted(set(oof[key].astype(str)) - set(train_fixed[key].astype(str)))
        raise ValueError(
            "OOF upstream features do not exactly cover fixed Train: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    if oof[key].astype(str).duplicated().any():
        raise ValueError("OOF upstream features are not unique by fixed primary key")
    oof_candidate_keys = set(oof_candidate_detail[key].astype(str))
    if not set(train_fixed[key].astype(str)).issubset(oof_candidate_keys):
        missing = sorted(
            set(train_fixed[key].astype(str)) - oof_candidate_keys
        )
        raise ValueError(
            "OOF comparable detail does not cover fixed Train: "
            f"{missing[:10]}"
        )
    oof = oof.drop(columns=["time_block"], errors="ignore").merge(
        train_fixed[[key, "time_block"]],
        on=key,
        how="left",
        validate="one_to_one",
    )
    final = final_feature_rows.copy()
    final["split"] = final["split"].astype(str).str.lower()
    if set(final[key].astype(str)) != set(fixed_rows[key].astype(str)):
        missing = sorted(set(fixed_rows[key].astype(str)) - set(final[key].astype(str)))
        extra = sorted(set(final[key].astype(str)) - set(fixed_rows[key].astype(str)))
        raise ValueError(
            "final replay does not exactly cover the fixed Train/Test rows: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    if final[key].astype(str).duplicated().any():
        raise ValueError("final replay rows are not unique by fixed primary key")
    final_candidate_keys = set(final_candidate_detail[key].astype(str))
    if not set(fixed_rows[key].astype(str)).issubset(final_candidate_keys):
        missing = sorted(set(fixed_rows[key].astype(str)) - final_candidate_keys)
        raise ValueError(
            "final comparable detail does not cover fixed Train/Test: "
            f"{missing[:10]}"
        )
    # Identity and labels must agree, but Test label values are deliberately not
    # copied into any training matrix or metric below.
    identity_check = fixed_rows[[key, "sku_code", "split", "label"]].merge(
        final[[key, "sku_code", "split", "label"]],
        on=key,
        how="outer",
        suffixes=("__fixed", "__feature"),
        indicator=True,
        validate="one_to_one",
    )
    mismatch = (
        identity_check["_merge"].ne("both")
        | identity_check["sku_code__fixed"].astype(str).ne(
            identity_check["sku_code__feature"].astype(str)
        )
        | identity_check["split__fixed"].astype(str).str.lower().ne(
            identity_check["split__feature"].astype(str).str.lower()
        )
        | ~np.isclose(
            pd.to_numeric(identity_check["label__fixed"], errors="coerce"),
            pd.to_numeric(identity_check["label__feature"], errors="coerce"),
            equal_nan=True,
        )
    )
    if mismatch.any():
        raise ValueError("final replay identity/label snapshot differs from fixed data")

    configurations: list[tuple[float, float, int | str, float]] = []
    for ridge in ridge_grid:
        for l1_ratio in l1_ratio_grid:
            if float(ridge) == 0.0 and float(l1_ratio) != 0.0:
                continue
            for half_life in half_life_grid:
                for similarity_regularization in similarity_regularization_grid:
                    if float(similarity_regularization) < 0:
                        raise ValueError(
                            "similarity regularization must be non-negative"
                        )
                    configurations.append(
                        (
                            float(ridge),
                            float(l1_ratio),
                            half_life,
                            float(similarity_regularization),
                        )
                    )
    if not configurations:
        raise ValueError("empty Section 8.4 hyperparameter grid")
    if fixed_hyperparameters and (len(configurations) != 1 or len(comparable_top_k_grid) != 1):
        raise ValueError("fixed-hyperparameter diagnostic requires one predetermined outer configuration and K")

    selection_rows: list[dict[str, Any]] = []
    predictions_by_config: dict[tuple[float, float, str, float], pd.DataFrame] = {}
    fold_models_by_config: dict[tuple[float, float, str, float], list[dict[str, Any]]] = {}
    # Comparable theta/tau/K depends on the rolling fit rows and the long-tail /
    # recency weights, but not on first-order ridge/l1. Cache it once per
    # (fold, half-life) instead of refitting for every downstream configuration.
    comparable_fold_cache: dict[
        tuple[int, str, float], ComparableParameterModel
    ] = {}
    parameter_fit_calls = {"comparable_theta_tau": 0, "first_order_coefficients": 0}
    for configuration_index, (ridge, l1_ratio, half_life, similarity_regularization) in enumerate(configurations, 1):
        fold_parts: list[pd.DataFrame] = []
        model_rows: list[dict[str, Any]] = []
        coefficient_history: dict[str, list[float]] = {
            name: [] for name in feature_columns
        }
        for fold in fixed.folds.itertuples(index=False):
            fit_blocks = str(fold.fit_blocks).split("+")
            fit_rows = oof.loc[oof["time_block"].isin(fit_blocks)].copy()
            validation_rows = oof.loc[
                oof["time_block"].eq(str(fold.validation_block))
            ].copy()
            transform = oof_replay.fit_transform(fit_rows) if oof_replay is not None else None
            if progress:
                progress({"phase": "rolling_fold", "configuration": configuration_index,
                    "configurations": len(configurations), "fold": int(fold.fold),
                    "fit_skus": len(fit_rows), "validation_skus": len(validation_rows)})
            comparable_cache_key = (
                int(fold.fold),
                str(half_life),
                similarity_regularization,
            )
            comparable_model = comparable_fold_cache.get(comparable_cache_key)
            if comparable_model is None:
                parameter_fit_calls["comparable_theta_tau"] += 1
                comparable_model = fit_comparable_parameters(
                    fit_rows,
                    oof_candidate_detail,
                    key_column=key,
                    base_demand_column=base_demand_column,
                    initial_component_weights=initial_similarity_weights,
                    temperature_grid=comparable_temperature_grid,
                    top_k_grid=comparable_top_k_grid,
                    regularization=similarity_regularization,
                    half_life=half_life,
                    full_recompute=(None if oof_replay is None else
                        lambda rows, m: oof_replay.recompute(rows, m, transform)),
                )
                comparable_fold_cache[comparable_cache_key] = comparable_model
            if oof_replay is not None:
                fit_rows = oof_replay.recompute(fit_rows, comparable_model, transform)
                validation_rows = oof_replay.recompute(validation_rows, comparable_model, transform)
            else:
                fit_rows[base_demand_column] = apply_comparable_parameters(
                    fit_rows, oof_candidate_detail, comparable_model,
                    key_column=key, base_demand_column=base_demand_column,
                )
                validation_rows[base_demand_column] = apply_comparable_parameters(
                    validation_rows, oof_candidate_detail, comparable_model,
                    key_column=key, base_demand_column=base_demand_column,
                )
            parameter_fit_calls["first_order_coefficients"] += 1
            model = fit_demand_residual_model(
                fit_rows,
                feature_columns=feature_columns,
                feature_signs=feature_signs,
                feature_groups=feature_groups,
                base_demand_column=base_demand_column,
                ridge=ridge,
                l1_ratio=l1_ratio,
                half_life=half_life,
                demand_bounds=demand_bounds,
                coefficient_bounds=coefficient_bounds,
                price_bounds=price_bounds,
            )
            if transform:
                model = DemandResidualModel.from_dict({**model.as_dict(), "frozen_transform": transform})
            prediction = predict_demand_residual(model, validation_rows)
            part_columns = [
                key,
                "sku_code",
                "first_actual_sale_date",
                "label",
                "time_block",
            ]
            for optional in (
                "prediction_checkpoint",
                "decision_cutoff_time",
                "lead_months",
            ):
                if optional in validation_rows and optional not in part_columns:
                    part_columns.append(optional)
            part = validation_rows[part_columns].copy()
            for column in validation_rows:
                if column.startswith("step5_") or column in {
                    "feature__adjustments__comparable_price", "feature__adjustments__delta_log_price",
                    "feature__adjustments__delta_IP", "feature__adjustments__delta_PRODUCT",
                    "feature__adjustments__delta_OPERATION", "feature__first_order__Q_base"}:
                    part[column] = validation_rows[column].to_numpy()
            part["prediction"] = prediction
            part["base_demand"] = pd.to_numeric(
                validation_rows[base_demand_column], errors="raise"
            ).to_numpy(dtype=float)
            part["fold"] = int(fold.fold)
            fold_parts.append(part)
            model_rows.append(
                {
                    "fold": int(fold.fold),
                    "fit_blocks": fold.fit_blocks,
                    "validation_block": fold.validation_block,
                    "ridge": ridge,
                    "l1_ratio": l1_ratio,
                    "half_life_weeks": str(half_life),
                    "similarity_regularization": similarity_regularization,
                    "model_id": _model_id(model),
                    "comparable_model_id": comparable_model.as_dict()["model_id"],
                    "comparable_temperature": comparable_model.softmax_temperature,
                    "comparable_top_k": comparable_model.top_k,
                    "comparable_weights_json": json.dumps(
                        dict(
                            zip(
                                comparable_model.component_columns,
                                comparable_model.component_weights,
                            )
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "fit_rows": len(fit_rows),
                    "validation_rows": len(validation_rows),
                    "test_rows_consumed": 0,
                    "model_json": json.dumps(
                        model.as_dict(), ensure_ascii=False, sort_keys=True
                    ),
                    "frozen_transform_json": json.dumps(transform, ensure_ascii=False, sort_keys=True),
                }
            )
            for name, coefficient in zip(
                feature_columns, model.coefficients
            ):
                coefficient_history[name].append(float(coefficient))
        combined = pd.concat(fold_parts, ignore_index=True)
        weights = _evaluation_weights(combined, oof, half_life)
        metrics = _metrics(
            combined["label"].to_numpy(dtype=float),
            combined["prediction"].to_numpy(dtype=float),
            weights,
            grade_thresholds=grade_thresholds,
            sales_band_edges=sales_band_edges,
        )
        stable_features = 0
        for name, values in coefficient_history.items():
            nonzero = [np.sign(value) for value in values if abs(value) > 1e-10]
            if not nonzero or len(set(nonzero)) == 1:
                stable_features += 1
        direction_stability = stable_features / max(len(feature_columns), 1)
        row = {
            "ridge": ridge,
            "l1_ratio": l1_ratio,
            "half_life_weeks": str(half_life),
            "similarity_regularization": similarity_regularization,
            **metrics,
            "abs_Bias": abs(metrics["Bias"]),
            "direction_stability": direction_stability,
        }
        selection_rows.append(row)
        key_config = (
            ridge,
            l1_ratio,
            str(half_life),
            similarity_regularization,
        )
        predictions_by_config[key_config] = combined
        fold_models_by_config[key_config] = model_rows

    selection = pd.DataFrame(selection_rows).sort_values(
        ["WAPE", "abs_Bias", "BandMAE", "direction_stability", "ridge"],
        ascending=[True, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    best = selection.iloc[0]
    best_key = (
        float(best["ridge"]),
        float(best["l1_ratio"]),
        str(best["half_life_weeks"]),
        float(best["similarity_regularization"]),
    )
    oof_predictions = predictions_by_config[best_key].copy()
    log_residual = np.log(
        (oof_predictions["label"].to_numpy(dtype=float) + 1e-6)
        / (oof_predictions["prediction"].to_numpy(dtype=float) + 1e-6)
    )
    q10, q90 = np.quantile(log_residual, [0.10, 0.90])
    oof_predictions["prediction_p10"] = np.maximum(
        oof_predictions["prediction"] * np.exp(q10), 0.0
    )
    oof_predictions["prediction_p90"] = np.maximum(
        oof_predictions["prediction"] * np.exp(q90), 0.0
    )

    final_train = final.loc[final["split"].eq("train")].copy()
    if progress:
        progress({"phase": "final_all_Train_refit", "train_rows": len(final_train), "test_rows_consumed": 0})
    final_transform = final_replay.fit_transform(final_train) if final_replay is not None else None
    if set(final_train[key].astype(str)) != set(train_fixed[key].astype(str)):
        raise AssertionError("full-Train refit rows changed after identity validation")
    parameter_fit_calls["comparable_theta_tau"] += 1
    final_comparable_model = fit_comparable_parameters(
        final_train,
        final_candidate_detail,
        key_column=key,
        base_demand_column=base_demand_column,
        initial_component_weights=initial_similarity_weights,
        temperature_grid=comparable_temperature_grid,
        top_k_grid=comparable_top_k_grid,
        regularization=float(best["similarity_regularization"]),
        half_life=str(best["half_life_weeks"]),
        full_recompute=(None if final_replay is None else
            lambda rows, m: final_replay.recompute(rows, m, final_transform)),
    )
    if final_replay is not None:
        final_train = final_replay.recompute(final_train, final_comparable_model, final_transform)
    else:
        final_train[base_demand_column] = apply_comparable_parameters(
            final_train, final_candidate_detail, final_comparable_model,
            key_column=key, base_demand_column=base_demand_column,
        )
    parameter_fit_calls["first_order_coefficients"] += 1
    final_model = fit_demand_residual_model(
        final_train,
        feature_columns=feature_columns,
        feature_signs=feature_signs,
        feature_groups=feature_groups,
        base_demand_column=base_demand_column,
        ridge=float(best["ridge"]),
        l1_ratio=float(best["l1_ratio"]),
        half_life=str(best["half_life_weeks"]),
        demand_bounds=demand_bounds,
        coefficient_bounds=coefficient_bounds,
        price_bounds=price_bounds,
    )
    model_payload = final_model.as_dict()
    model_payload["frozen_transform"] = final_transform or {}
    model_payload["model_id"] = _model_id(final_model)
    model_payload["dataset_version"] = fixed.manifest["dataset_version"]
    model_payload["dataset_hash"] = fixed.manifest["dataset_hash"]
    model_payload["oof_log_residual_quantiles"] = {
        "p10": float(q10),
        "p90": float(q90),
    }
    final_model = DemandResidualModel.from_dict(model_payload)
    final_demand_model_id = _model_id(final_model)
    final_comparable_model_id = final_comparable_model.as_dict()["model_id"]
    final_train_input_hash = _stable_frame_hash(final_train, sorted(final_train.columns))
    pipeline_parameters = (final_replay.load(str(final_train.iloc[0][key]))["parameters"]
                           if final_replay is not None else None)
    final_package_id = sha256(
        json.dumps(
            {
                "demand_model": final_model.as_dict(),
                "comparable_model": final_comparable_model.as_dict(),
                "dataset_hash": fixed.manifest["dataset_hash"],
                "final_train_input_hash": final_train_input_hash,
                "pipeline_parameters": pipeline_parameters,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]
    fold_models = pd.DataFrame(fold_models_by_config[best_key])
    report = {
        "spec_version": "3.2",
        "section": "8.4",
        "status": "section_8_4_first_order_train_complete_test_still_locked",
        "dataset_version": fixed.manifest["dataset_version"],
        "dataset_hash": fixed.manifest["dataset_hash"],
        "train_skus": int(train_fixed["sku_code"].nunique()),
        "test_skus_locked": int(test_fixed["sku_code"].nunique()),
        "test_rows_consumed": 0,
        "parameter_fit_calls": parameter_fit_calls,
        "temperature_training_grid": list(comparable_temperature_grid),
        "training_mode": "fixed_hyperparameters_learned_parameters" if fixed_hyperparameters else "rolling_grid_search",
        "configuration_selection_policy": (
            "single outer preset fixed before OOF; theta/tau and demand coefficients still fitted"
            if fixed_hyperparameters else "Train OOF only"
        ),
        "comparable_parameter_policy": (
            "theta fitted by projected coordinate descent; tau fitted on configured positive grid; each historical Train fold and final all-Train"
        ),
        "rolling_policy": "B1->B2; B1+B2->B3; B1+B2+B3->B4; B1+B2+B3+B4->B5",
        "oof_rows": int(len(oof_predictions)),
        "oof_blocks": ["B2", "B3", "B4", "B5"],
        "selected_hyperparameters": {
            "ridge": float(best["ridge"]),
            "l1_ratio": float(best["l1_ratio"]),
            "half_life_weeks": str(best["half_life_weeks"]),
            "similarity_regularization": float(
                best["similarity_regularization"]
            ),
            "comparable_temperature": final_comparable_model.softmax_temperature,
            "comparable_top_k": final_comparable_model.top_k,
            "comparable_component_weights": dict(
                zip(
                    final_comparable_model.component_columns,
                    final_comparable_model.component_weights,
                )
            ),
        },
        "selection_metrics": {
            name: float(best[name])
            for name in (
                "WAPE",
                "Bias",
                "BandMAE",
                "GradeHit",
                "GradeHitAdj",
                "GradeCross",
                "direction_stability",
            )
        },
        "metric_quantity_scope": "pure_sales_demand_piece_quantity",
        "grade_scope": "demand grade only; supply-feasible Q_feas grade is separate",
        "band_mae_definition": (
            "equal mean of unweighted quantity MAE across non-empty actual-sales bands"
        ),
        "oof_log_residual_quantiles": {
            "p10": float(q10),
            "p90": float(q90),
        },
        "final_model_id": final_package_id,
        "final_demand_model_id": final_demand_model_id,
        "final_comparable_model_id": final_comparable_model_id,
        "final_refit_rows": int(len(final_train)),
        "final_refit_scope": "all fixed Train only",
        "final_train_input_hash": final_train_input_hash,
        "locked_test_policy": "not predicted or evaluated until backtest.py",
        "downstream_recompute": "full_M3_M4_M5_M6_M7" if oof_replay is not None else "legacy_scalar_approximation_not_production",
        "factor_transform_scope": "fold_historical_Train; final_all_Train_frozen" if final_transform else "legacy_loaded_pool",
        "demand_optimizer": "constrained_EN_initial_then_actual_production_coordinate_descent_v1",
        "feature_fit_diagnostics": [
            {"feature": name, "coefficient": float(coef),
             "fit_relative_input_nonzero_rows": int(pd.to_numeric(final_train[name], errors="coerce").fillna(0).ne(0).sum()),
             "fit_relative_input_distinct_values": int(pd.to_numeric(final_train[name], errors="coerce").nunique()),
             "group": group}
            for name, coef, group in zip(final_model.feature_columns, final_model.coefficients, final_model.feature_groups)],
        "comparable_search_scope": "within_each_target_restored_preselected_Top3; not_whole_universe_retrieval",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return Section84TrainingResult(
        final_model,
        final_comparable_model,
        oof_predictions,
        selection,
        fold_models,
        report,
    )
