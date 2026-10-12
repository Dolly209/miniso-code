from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from data.domain import require_columns
from training.split import SkuSample, build_time_blocks, expanding_window_folds
from training.weights import long_tail_weights


@dataclass(frozen=True)
class FittedModel:
    feature_columns: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    ridge: float
    l1_ratio: float
    half_life_weeks: str
    nonnegative: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_type": "weighted_log1p_elastic_net",
            "feature_columns": list(self.feature_columns),
            "medians": list(self.medians),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "ridge": self.ridge,
            "l1_ratio": self.l1_ratio,
            "half_life_weeks": self.half_life_weeks,
            "nonnegative": self.nonnegative,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FittedModel":
        return cls(
            tuple(payload["feature_columns"]),
            tuple(float(value) for value in payload["medians"]),
            tuple(float(value) for value in payload["means"]),
            tuple(float(value) for value in payload["scales"]),
            tuple(float(value) for value in payload["coefficients"]),
            float(payload["intercept"]),
            float(payload["ridge"]),
            float(payload.get("l1_ratio", 0.0)),
            str(payload["half_life_weeks"]),
            bool(payload.get("nonnegative", True)),
        )


@dataclass(frozen=True)
class TrainingResult:
    model: FittedModel
    oof_predictions: pd.DataFrame
    selection: pd.DataFrame
    report: dict[str, Any]


def _matrix(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    medians: np.ndarray | None = None,
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    fitted_medians = np.nanmedian(raw, axis=0) if medians is None else medians
    fitted_medians = np.where(np.isfinite(fitted_medians), fitted_medians, 0.0)
    filled = np.where(np.isnan(raw), fitted_medians, raw)
    fitted_means = filled.mean(axis=0) if means is None else means
    fitted_scales = filled.std(axis=0) if scales is None else scales
    fitted_scales = np.where(fitted_scales > 1e-12, fitted_scales, 1.0)
    return (filled - fitted_means) / fitted_scales, fitted_medians, fitted_means, fitted_scales


def _time_weights(dates: pd.Series, half_life: int | str) -> np.ndarray:
    parsed = pd.to_datetime(dates, errors="raise")
    age_weeks = (parsed.max() - parsed).dt.days.to_numpy(dtype=float) / 7.0
    if str(half_life).lower() == "infinity":
        return np.ones(len(parsed), dtype=float)
    return np.power(2.0, -age_weeks / float(half_life))


def _sample_weights(frame: pd.DataFrame, half_life: int | str) -> np.ndarray:
    labels = frame.groupby("sku_code")["label"].first().to_dict()
    sku_weight = {row.sku_code: row.weight for row in long_tail_weights(labels)}
    node_count = frame.groupby("sku_code").size().to_dict()
    long_tail = np.array(
        [sku_weight[str(sku)] / node_count[str(sku)] for sku in frame["sku_code"].astype(str)],
        dtype=float,
    )
    combined = long_tail * _time_weights(frame["first_actual_sale_date"], half_life)
    return combined / max(combined.sum(), 1e-12)


def _evaluation_weights(
    evaluation: pd.DataFrame,
    full_train: pd.DataFrame,
    half_life: int | str,
) -> np.ndarray:
    labels = full_train.groupby("sku_code")["label"].first().to_dict()
    sku_weight = {row.sku_code: row.weight for row in long_tail_weights(labels)}
    node_count = evaluation.groupby("sku_code").size().to_dict()
    long_tail = np.array(
        [
            sku_weight[str(sku)] / node_count[str(sku)]
            for sku in evaluation["sku_code"].astype(str)
        ],
        dtype=float,
    )
    combined = long_tail * _time_weights(
        evaluation["first_actual_sale_date"], half_life
    )
    return combined / max(combined.sum(), 1e-12)


def fit_model(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    ridge: float,
    half_life: int | str,
    l1_ratio: float = 0.0,
    nonnegative: bool = True,
) -> FittedModel:
    if frame.empty:
        raise ValueError("cannot fit an empty modeling frame")
    x, medians, means, scales = _matrix(frame, feature_columns)
    y = np.log1p(pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=float))
    if (y < 0).any():
        raise ValueError("training labels must be non-negative")
    weights = _sample_weights(frame, half_life)
    if not 0 <= l1_ratio <= 1:
        raise ValueError("l1_ratio must be between 0 and 1")
    coefficients = np.zeros(x.shape[1], dtype=float)
    intercept = float(np.average(y, weights=weights))
    l1_penalty = float(ridge) * float(l1_ratio)
    l2_penalty = float(ridge) * (1.0 - float(l1_ratio))
    for _ in range(500):
        previous = coefficients.copy()
        intercept = float(np.average(y - x @ coefficients, weights=weights))
        for index in range(x.shape[1]):
            partial = y - intercept - x @ coefficients + x[:, index] * coefficients[index]
            rho = float(np.dot(weights * x[:, index], partial))
            denominator = float(np.dot(weights, x[:, index] ** 2) + l2_penalty)
            if rho > l1_penalty:
                updated = (rho - l1_penalty) / max(denominator, 1e-12)
            elif rho < -l1_penalty:
                updated = (rho + l1_penalty) / max(denominator, 1e-12)
            else:
                updated = 0.0
            coefficients[index] = max(updated, 0.0) if nonnegative else updated
        if np.max(np.abs(coefficients - previous)) < 1e-8:
            break
    intercept = float(np.average(y - x @ coefficients, weights=weights))
    return FittedModel(
        tuple(feature_columns), tuple(medians), tuple(means), tuple(scales),
        tuple(coefficients), intercept, float(ridge), float(l1_ratio),
        str(half_life), nonnegative,
    )


def predict(model: FittedModel, frame: pd.DataFrame) -> np.ndarray:
    x, _, _, _ = _matrix(
        frame,
        model.feature_columns,
        medians=np.array(model.medians),
        means=np.array(model.means),
        scales=np.array(model.scales),
    )
    log_prediction = model.intercept + x @ np.array(model.coefficients)
    return np.maximum(np.expm1(log_prediction), 0.0)


def weighted_metrics(actual: np.ndarray, prediction: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    weights = np.ones(len(actual), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / max(weights.sum(), 1e-12)
    error = prediction - actual
    denominator = max(float(np.dot(weights, np.abs(actual))), 1e-12)
    return {
        "WAPE": float(np.dot(weights, np.abs(error)) / denominator),
        "Bias": float(np.dot(weights, error) / denominator),
        "MAE": float(np.dot(weights, np.abs(error))),
        "LogMAE": float(np.dot(weights, np.abs(np.log1p(prediction) - np.log1p(actual)))),
    }


def _prepare_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, ("sku_code", "first_actual_sale_date", "split", "label"), "modeling_dataset")
    result = frame.copy()
    result["sku_code"] = result["sku_code"].astype(str)
    result["first_actual_sale_date"] = pd.to_datetime(result["first_actual_sale_date"], errors="raise").dt.date
    result["label"] = pd.to_numeric(result["label"], errors="raise")
    if (result["label"] < 0).any():
        raise ValueError("modeling_dataset contains negative labels")
    return result


def rolling_oof_train(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    n_time_blocks: int,
    ridge_grid: Sequence[float],
    half_life_grid: Sequence[int | str],
    l1_ratio_grid: Sequence[float] = (0.0,),
    nonnegative: bool = True,
) -> TrainingResult:
    data = _prepare_training_frame(frame)
    train = data.loc[data["split"].astype(str).str.lower() == "train"].copy()
    if train.empty:
        raise ValueError("modeling_dataset has no Train rows")
    sku_dates = train.groupby("sku_code")["first_actual_sale_date"].first()
    samples = [SkuSample(value, sku) for sku, value in sku_dates.items()]
    blocks = build_time_blocks(samples, n_time_blocks=n_time_blocks)
    folds = expanding_window_folds(blocks)
    selection_rows: list[dict[str, Any]] = []
    predictions_by_config: dict[tuple[float, float, str], list[pd.DataFrame]] = {}
    coefficient_signs: dict[str, list[float]] = {name: [] for name in feature_columns}
    for ridge in ridge_grid:
        for l1_ratio in l1_ratio_grid:
            for half_life in half_life_grid:
                if float(ridge) == 0.0 and float(l1_ratio) != 0.0:
                    continue
                parts: list[pd.DataFrame] = []
                for fold_index, (fit_samples, validation_samples) in enumerate(folds, start=1):
                    fit_skus = {sample.sku_code for sample in fit_samples}
                    validation_skus = {sample.sku_code for sample in validation_samples}
                    fit_frame = train.loc[train["sku_code"].isin(fit_skus)]
                    validation_frame = train.loc[
                        train["sku_code"].isin(validation_skus)
                    ].copy()
                    model = fit_model(
                        fit_frame,
                        feature_columns=feature_columns,
                        ridge=float(ridge),
                        l1_ratio=float(l1_ratio),
                        half_life=half_life,
                        nonnegative=nonnegative,
                    )
                    validation_frame["prediction"] = predict(model, validation_frame)
                    validation_frame["fold"] = fold_index
                    validation_frame["ridge"] = float(ridge)
                    validation_frame["l1_ratio"] = float(l1_ratio)
                    validation_frame["half_life_weeks"] = str(half_life)
                    parts.append(validation_frame[["sku_code", "first_actual_sale_date", "label", "prediction", "fold", "ridge", "l1_ratio", "half_life_weeks"]])
                    if float(ridge) == float(ridge_grid[0]) and float(l1_ratio) == 0.0 and str(half_life) == str(half_life_grid[0]):
                        for name, coefficient in zip(feature_columns, model.coefficients):
                            coefficient_signs[name].append(float(np.sign(coefficient)))
                combined = pd.concat(parts, ignore_index=True)
                metric_weights = _evaluation_weights(combined, train, half_life)
                metrics = weighted_metrics(
                    combined["label"].to_numpy(),
                    combined["prediction"].to_numpy(),
                    metric_weights,
                )
                selection_rows.append({"ridge": float(ridge), "l1_ratio": float(l1_ratio), "half_life_weeks": str(half_life), **metrics})
                predictions_by_config[(float(ridge), float(l1_ratio), str(half_life))] = parts
    selection = pd.DataFrame(selection_rows).sort_values(["LogMAE", "ridge", "l1_ratio", "half_life_weeks"])
    best = selection.iloc[0]
    key = (float(best["ridge"]), float(best["l1_ratio"]), str(best["half_life_weeks"]))
    oof = pd.concat(predictions_by_config[key], ignore_index=True)
    residual = np.log1p(oof["label"]) - np.log1p(oof["prediction"])
    quantiles = residual.quantile([0.10, 0.90])
    oof["prediction_p10"] = np.maximum(np.expm1(np.log1p(oof["prediction"]) + float(quantiles.loc[0.10])), 0.0)
    oof["prediction_p90"] = np.maximum(np.expm1(np.log1p(oof["prediction"]) + float(quantiles.loc[0.90])), 0.0)
    full_model = fit_model(
        train,
        feature_columns=feature_columns,
        ridge=float(best["ridge"]),
        l1_ratio=float(best["l1_ratio"]),
        half_life=str(best["half_life_weeks"]),
        nonnegative=nonnegative,
    )
    oof_metrics = weighted_metrics(oof["label"].to_numpy(), oof["prediction"].to_numpy())
    factor_consistency = {
        name: {
            "nonnegative_fold_count": int(sum(value >= 0 for value in signs)),
            "fold_count": len(signs),
            "enabled": int(sum(value >= 0 for value in signs)) >= min(3, len(signs)),
        }
        for name, signs in coefficient_signs.items()
    }
    report = {
        "spec_version": "3.2",
        "status": "trained_locally",
        "train_rows": len(train),
        "train_skus": train["sku_code"].nunique(),
        "test_rows_consumed": 0,
        "oof_rows": len(oof),
        "oof_policy": "B2..Bk predicted only by earlier blocks",
        "selected_ridge": float(best["ridge"]),
        "selected_l1_ratio": float(best["l1_ratio"]),
        "selected_half_life_weeks": str(best["half_life_weeks"]),
        "oof_metrics": oof_metrics,
        "oof_log_residual_quantiles": {"p10": float(quantiles.loc[0.10]), "p90": float(quantiles.loc[0.90])},
        "factor_direction_consistency": factor_consistency,
        "model": full_model.as_dict(),
    }
    return TrainingResult(full_model, oof, selection, report)


def replay_test(
    frame: pd.DataFrame,
    model: FittedModel,
    *,
    interval_log_quantiles: Mapping[str, float] | None = None,
    grade_thresholds: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = _prepare_training_frame(frame)
    test = data.loc[data["split"].astype(str).str.lower() == "test"].copy()
    if test.empty:
        raise ValueError("modeling_dataset has no Test rows")
    test["prediction"] = predict(model, test)
    if interval_log_quantiles:
        test["prediction_p10"] = np.maximum(
            np.expm1(np.log1p(test["prediction"]) + float(interval_log_quantiles["p10"])), 0.0
        )
        test["prediction_p90"] = np.maximum(
            np.expm1(np.log1p(test["prediction"]) + float(interval_log_quantiles["p90"])), 0.0
        )
        coverage = float(((test["label"] >= test["prediction_p10"]) & (test["label"] <= test["prediction_p90"])).mean())
    else:
        test["prediction_p10"] = np.nan
        test["prediction_p90"] = np.nan
        coverage = np.nan
    metrics = weighted_metrics(test["label"].to_numpy(), test["prediction"].to_numpy())
    grade_metrics: dict[str, Any] = {}
    if grade_thresholds is not None:
        def grade(value: float) -> str:
            if value <= float(grade_thresholds["C_max"]):
                return "C"
            if value < float(grade_thresholds["B_max"]):
                return "B"
            if value < float(grade_thresholds["A_max"]):
                return "A"
            return "S"

        order = {"C": 0, "B": 1, "A": 2, "S": 3}
        test["actual_grade"] = test["label"].map(grade)
        test["predicted_grade"] = test["prediction"].map(grade)
        distance = np.array(
            [
                abs(order[actual] - order[predicted])
                for actual, predicted in zip(
                    test["actual_grade"], test["predicted_grade"]
                )
            ],
            dtype=float,
        )
        confusion = (
            test.groupby(["actual_grade", "predicted_grade"])
            .size()
            .rename("count")
            .reset_index()
            .to_dict(orient="records")
        )
        grade_metrics = {
            "GradeHit": float((distance == 0).mean()),
            "AdjacentOrExactHit": float((distance <= 1).mean()),
            "CrossGradeRate": float((distance > 1).mean()),
            "BandMAE": float(distance.mean()),
            "confusion": confusion,
        }
    return test, {
        "spec_version": "3.2",
        "status": "test_replayed_once",
        "test_rows": len(test),
        "test_skus": test["sku_code"].nunique(),
        "metrics": metrics,
        "grade_metrics": grade_metrics,
        "interval_coverage": coverage,
        "test_did_not_select_model": True,
    }
