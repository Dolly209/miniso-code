from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.domain import require_columns


@dataclass(frozen=True)
class FactorResult:
    detail: pd.DataFrame
    adjustments: pd.DataFrame
    transformed_features: pd.DataFrame
    warnings: tuple[str, ...]


COUNT_FEATURES = {"ip_breadth", "member_count", "pack_size", "historical_velocity"}
RATIO_FEATURES = {
    "member_female_rate", "member_young_rate", "supplier_fulfillment",
    "multi_supplier", "exposure_quality", "lifecycle_persistence",
}


def _robust_z(values: pd.Series, *, winsor: tuple[float, float]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    low, high = numeric.quantile(list(winsor))
    clipped = numeric.clip(lower=low, upper=high)
    median = float(clipped.median())
    mad = float((clipped - median).abs().median())
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return pd.Series(np.where(clipped.notna(), 0.0, np.nan), index=values.index, dtype=float)
    return (clipped - median) / scale


def transform_factor_features(
    features: pd.DataFrame,
    *,
    factor_weights: Mapping[str, Mapping[str, float]],
    winsor: tuple[float, float],
) -> pd.DataFrame:
    require_columns(features, ("sku_code",), "factor_features")
    result = features.copy()
    all_features = sorted({name for group in factor_weights.values() for name in group})
    for name in all_features:
        if name not in result:
            result[name] = np.nan
        values = pd.to_numeric(result[name], errors="coerce")
        if name in COUNT_FEATURES:
            values = np.log1p(values.clip(lower=0))
        result[f"z__{name}"] = _robust_z(values, winsor=winsor)
    for group, weights in factor_weights.items():
        group_values: list[float] = []
        group_completeness: list[float] = []
        for row in result.itertuples(index=False):
            available = {
                name: getattr(row, f"z__{name}")
                for name in weights
                if pd.notna(getattr(row, f"z__{name}"))
            }
            denominator = sum(float(weights[name]) for name in available)
            group_values.append(
                sum(float(weights[name]) * float(value) for name, value in available.items()) / denominator
                if denominator else np.nan
            )
            group_completeness.append(
                denominator / max(sum(float(value) for value in weights.values()), 1e-12)
            )
        result[f"u__{group}"] = group_values
        result[f"completeness__{group}"] = group_completeness
        valid = result[f"u__{group}"].dropna().sort_values()
        result[f"score__{group}"] = result[f"u__{group}"].map(
            lambda value: np.nan if pd.isna(value) else 100.0 * float((valid <= value).mean())
        )
    return result


def calculate_demand_adjustments(
    *,
    targets: pd.DataFrame,
    features: pd.DataFrame,
    comparables: pd.DataFrame,
    comparable_summary: pd.DataFrame,
    config: Mapping[str, Any],
    competition_rules: pd.DataFrame | None = None,
) -> FactorResult:
    """M5 robust factor scoring and bounded multiplicative demand adjustment."""

    require_columns(targets, ("sku_code",), "targets")
    require_columns(features, ("sku_code", "retail_price"), "features")
    require_columns(
        comparables,
        ("target_sku_code", "candidate_sku_code", "selected", "alpha_weight"),
        "comparables",
    )
    require_columns(
        comparable_summary,
        ("sku_code", "external_direction_coefficient"),
        "comparable_summary",
    )
    weights = config["small_factor_weights"]
    transformed = transform_factor_features(
        features,
        factor_weights=weights,
        winsor=tuple(config["winsor_quantiles"]),
    )
    by_sku = transformed.drop_duplicates("sku_code").set_index("sku_code", drop=False)
    external = comparable_summary.set_index("sku_code")["external_direction_coefficient"].to_dict()
    detail_rows: list[dict[str, Any]] = []
    adjustment_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for target_sku in targets["sku_code"].astype(str).drop_duplicates():
        if target_sku not in by_sku.index:
            raise ValueError(f"factor features missing target SKU: {target_sku}")
        target = by_sku.loc[target_sku]
        chosen = comparables.loc[
            (comparables["target_sku_code"].astype(str) == target_sku)
            & (pd.to_numeric(comparables["selected"], errors="coerce").fillna(0) == 1)
            & (~comparables["candidate_sku_code"].astype(str).str.startswith("__STATIC__"))
        ].copy()
        coefficients: dict[str, float] = {}
        completeness_values: list[tuple[float, float]] = []
        for group in weights:
            new_score = target.get(f"score__{group}")
            valid_comparables = chosen.loc[chosen["candidate_sku_code"].astype(str).isin(by_sku.index)].copy()
            comparable_values: list[float] = []
            comparable_weights: list[float] = []
            for row in valid_comparables.itertuples(index=False):
                value = by_sku.loc[str(row.candidate_sku_code)].get(f"score__{group}")
                if pd.notna(value):
                    comparable_values.append(float(value))
                    comparable_weights.append(float(row.alpha_weight))
            comparable_score = (
                float(np.average(comparable_values, weights=comparable_weights))
                if comparable_values and sum(comparable_weights) > 0 else np.nan
            )
            group_complete = float(target.get(f"completeness__{group}", 0.0))
            if pd.isna(new_score) or pd.isna(comparable_score):
                delta = 0.0
                coefficient = 1.0
                status = "neutral_missing_group"
                warnings.append(f"{target_sku}: {group} factor group is neutral because inputs are missing")
            else:
                delta = (float(new_score) - float(comparable_score)) / 100.0
                beta = float(config["group_betas"][group])
                lower, upper = map(float, config["coefficient_bounds"][group])
                coefficient = float(np.clip(1.0 + beta * delta, lower, upper))
                status = "estimated"
            coefficients[group] = coefficient
            group_weight = float(config["group_weights"][group])
            completeness_values.append((group_weight, group_complete))
            detail_rows.append(
                {
                    "sku_code": target_sku,
                    "factor_group": group,
                    "new_score": new_score,
                    "comparable_score": comparable_score,
                    "relative_score_delta": delta,
                    "coefficient": coefficient,
                    "group_completeness": group_complete,
                    "status": status,
                }
            )
        candidate_prices: list[float] = []
        candidate_weights: list[float] = []
        for row in chosen.itertuples(index=False):
            if str(row.candidate_sku_code) not in by_sku.index:
                continue
            price = by_sku.loc[str(row.candidate_sku_code)].get("retail_price")
            if pd.notna(price) and float(price) > 0:
                candidate_prices.append(float(price))
                candidate_weights.append(float(row.alpha_weight))
        target_price = pd.to_numeric(pd.Series([target.get("retail_price")]), errors="coerce").iloc[0]
        if candidate_prices and sum(candidate_weights) > 0 and pd.notna(target_price) and target_price > 0:
            normalized = np.array(candidate_weights) / sum(candidate_weights)
            comp_price = float(np.exp(np.dot(normalized, np.log(np.array(candidate_prices) + 1e-9))))
            delta_log_price = log((float(target_price) + 1e-9) / (comp_price + 1e-9))
            price_lower, price_upper = map(float, config["price_bounds"])
            c_price = float(np.clip(exp(float(config["price_elasticity"]) * delta_log_price), price_lower, price_upper))
            price_status = "estimated"
        else:
            comp_price, delta_log_price, c_price = np.nan, 0.0, 1.0
            price_status = "neutral_missing_price"
        c_competition = 1.0
        triggered_rules: list[str] = []
        if competition_rules is not None and not competition_rules.empty:
            require_columns(competition_rules, ("sku_code", "rule_code", "triggered", "coefficient"), "competition_rules")
            active = competition_rules.loc[
                (competition_rules["sku_code"].astype(str) == target_sku)
                & (pd.to_numeric(competition_rules["triggered"], errors="coerce").fillna(0) == 1)
            ]
            for row in active.itertuples(index=False):
                c_competition *= min(float(row.coefficient), 1.0)
                triggered_rules.append(str(row.rule_code))
        c_ext = float(external.get(target_sku, 1.0))
        raw = (
            coefficients.get("IP", 1.0)
            * c_ext
            * coefficients.get("PRODUCT", 1.0)
            * c_price
            * coefficients.get("OPERATION", 1.0)
            * c_competition
        )
        lower, upper = map(float, config["demand_bounds"])
        c_demand = float(np.clip(raw, lower, upper))
        total_group_weight = sum(weight for weight, _ in completeness_values)
        input_completeness = 100.0 * sum(
            weight * value for weight, value in completeness_values
        ) / max(total_group_weight, 1e-12)
        adjustment_rows.append(
            {
                "sku_code": target_sku,
                "C_IP": coefficients.get("IP", 1.0),
                "C_ext": c_ext,
                "C_product": coefficients.get("PRODUCT", 1.0),
                "C_price": c_price,
                "C_operation": coefficients.get("OPERATION", 1.0),
                "C_competition": c_competition,
                "C_raw": raw,
                "C_demand": c_demand,
                "target_price": target_price,
                "comparable_price": comp_price,
                "delta_log_price": delta_log_price,
                "price_status": price_status,
                "triggered_competition_rules": "|".join(triggered_rules),
                "input_completeness": input_completeness,
            }
        )
    return FactorResult(
        pd.DataFrame(detail_rows),
        pd.DataFrame(adjustment_rows),
        transformed,
        tuple(dict.fromkeys(warnings)),
    )
