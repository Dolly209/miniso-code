from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.domain import require_columns


class PlacementComplianceError(ValueError):
    """Raised when a formal V3.2 placement run lacks a required input."""


@dataclass(frozen=True)
class PlacementResult:
    store_plan: pd.DataFrame
    summary: pd.DataFrame
    demand_curve: pd.DataFrame
    diagnostics: pd.DataFrame
    compliance: pd.DataFrame
    warnings: tuple[str, ...]


def _weighted_log_average(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    valid = (
        numeric_values.notna()
        & numeric_weights.notna()
        & (numeric_weights > 0)
        & (numeric_values >= 0)
    )
    if not valid.any():
        return float("nan")
    normalized = numeric_weights.loc[valid] / numeric_weights.loc[valid].sum()
    return float(
        np.expm1(np.dot(normalized, np.log1p(numeric_values.loc[valid])))
    )


def _tier_rank(level: object) -> tuple[int, str]:
    normalized = str(level).upper()
    return ({"S": 0, "A": 1, "B": 2, "C": 3}.get(normalized, 9), normalized)


def _compliance_row(
    *, target_sku: str, concept: str, status: str, source: str, detail: str
) -> dict[str, str]:
    return {
        "target_sku_code": target_sku,
        "module": "M3/M4",
        "concept": concept,
        "status": status,
        "source": source,
        "detail": detail,
    }


def _normalize_policy(
    *,
    policy: pd.DataFrame | None,
    store_tiers: list[str],
    history: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    required = ("store_tier", "v_min", "inv_max", "tau_tier")
    if policy is not None and not policy.empty:
        require_columns(policy, required, "placement policy")
        normalized = policy.copy()
        normalized["store_tier"] = normalized["store_tier"].astype(str)
        if normalized["store_tier"].duplicated().any():
            raise PlacementComplianceError(
                "placement policy must be unique by store_tier"
            )
        for column in required[1:]:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        invalid = (
            normalized[list(required[1:])].isna().any(axis=1)
            | (normalized["v_min"] < 0)
            | ~normalized["inv_max"].between(0, 1)
            | ~normalized["tau_tier"].between(0, 1)
        )
        if invalid.any():
            examples = normalized.loc[invalid, list(required)].head(5)
            raise PlacementComplianceError(
                "placement policy contains invalid V3.2 thresholds: "
                f"{examples.to_dict(orient='records')}"
            )
        missing = sorted(set(store_tiers) - set(normalized["store_tier"]))
        if not missing:
            return normalized, "business_versioned_policy"
    else:
        normalized = pd.DataFrame(columns=required)

    defaults = config.get("initial_defaults", {})
    if not defaults:
        raise PlacementComplianceError(
            "placement.initial_defaults is required for uncalibrated parameters"
        )
    existing = set(normalized.get("store_tier", pd.Series(dtype=str)).astype(str))
    rows: list[dict[str, float | str]] = []
    for tier in store_tiers:
        if tier in existing:
            continue
        values = pd.to_numeric(
            history.loc[history["store_tier"].astype(str) == tier, "v_neutral"],
            errors="coerce",
        ).dropna()
        rows.append(
            {
                "store_tier": tier,
                "v_min": (
                    float(values.median())
                    if not values.empty
                    else float(defaults["v_min_when_no_history"])
                ),
                "inv_max": float(defaults["inv_max"]),
                "tau_tier": float(defaults["tau_tier"]),
            }
        )
    normalized = pd.concat([normalized, pd.DataFrame(rows)], ignore_index=True)
    source = (
        "mixed_business_policy_and_versioned_defaults"
        if policy is not None and not policy.empty
        else "versioned_initial_defaults"
    )
    return normalized, source


def _derive_restore_evidence(
    *,
    candidate_skus: list[str],
    eligible_stores: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    universe = (
        pd.DataFrame({"sku_code": candidate_skus})
        .assign(_join=1)
        .merge(
            eligible_stores[["store_code", "store_tier"]].assign(_join=1),
            on="_join",
            how="inner",
        )
        .drop(columns="_join")
    )
    aggregates = (
        history.loc[history["sku_code"].astype(str).isin(candidate_skus)]
        .groupby(["sku_code", "store_code", "store_tier"], as_index=False)
        .agg(
            v_neutral=("v_neutral", "mean"),
            stockout_days=("stockout_days", "sum"),
            source_day_rows=("source_day_rows", "sum"),
            restored_exposure_days=("restored_exposure_days", "sum"),
        )
    )
    evidence = universe.merge(
        aggregates,
        on=["sku_code", "store_code", "store_tier"],
        how="left",
        validate="one_to_one",
    )
    evidence["eligible_at_launch"] = 1
    evidence["arrived_in_launch_window"] = (
        pd.to_numeric(evidence["restored_exposure_days"], errors="coerce")
        .fillna(0)
        .gt(0)
        .astype(int)
    )
    evidence["stock_pressure"] = pd.to_numeric(
        evidence["stockout_days"], errors="coerce"
    ) / pd.to_numeric(evidence["source_day_rows"], errors="coerce").replace(
        0, np.nan
    )
    evidence["evidence_weight"] = 1.0
    return evidence[
        [
            "sku_code",
            "store_code",
            "store_tier",
            "eligible_at_launch",
            "arrived_in_launch_window",
            "v_neutral",
            "stock_pressure",
            "evidence_weight",
        ]
    ]


def _prepare_evidence(
    *,
    evidence: pd.DataFrame | None,
    candidate_skus: list[str],
    eligible_stores: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    required = (
        "sku_code",
        "store_code",
        "store_tier",
        "eligible_at_launch",
        "arrived_in_launch_window",
        "v_neutral",
        "stock_pressure",
    )
    if evidence is not None and not evidence.empty:
        require_columns(evidence, required, "placement evidence")
        result = evidence.copy()
        for column in ("sku_code", "store_code", "store_tier"):
            result[column] = result[column].astype(str)
        result = result.loc[result["sku_code"].isin(candidate_skus)].copy()
        if result.empty:
            raise PlacementComplianceError(
                "placement evidence has no selected comparable SKU rows"
            )
        if "evidence_weight" not in result:
            result["evidence_weight"] = 1.0
        result["evidence_weight"] = pd.to_numeric(
            result["evidence_weight"], errors="coerce"
        ).fillna(0.0)
        if result.duplicated(["sku_code", "store_code"]).any():
            raise PlacementComplianceError(
                "placement evidence must be unique by sku_code,store_code"
            )
        probability_columns = (
            "prior_probability",
            "performance_probability",
            "reliability_r",
        )
        has_probabilities = all(column in result for column in probability_columns)
        if has_probabilities:
            for column in probability_columns:
                result[column] = pd.to_numeric(result[column], errors="coerce")
            invalid = result[list(probability_columns)].isna().any(axis=1)
            invalid |= ~result[list(probability_columns)].apply(
                lambda column: column.between(0, 1)
            ).all(axis=1)
            if invalid.any():
                raise PlacementComplianceError(
                    "placement evidence contains invalid store-level probability rows"
                )
            return result, "modeled_store_probability_evidence"
        return result, "raw_launch_window_empirical_probabilities"
    return (
        _derive_restore_evidence(
            candidate_skus=candidate_skus,
            eligible_stores=eligible_stores,
            history=history,
        ),
        "restore_pool_empirical_proxy",
    )


def _candidate_reasonableness(
    *, evidence: pd.DataFrame, policy: pd.DataFrame, shrinkage_k: float
) -> pd.DataFrame:
    merged = evidence.merge(
        policy[["store_tier", "v_min", "inv_max", "tau_tier"]],
        on="store_tier",
        how="left",
        validate="many_to_one",
    )
    if merged[["v_min", "inv_max", "tau_tier"]].isna().any(axis=None):
        missing = sorted(
            merged.loc[
                merged[["v_min", "inv_max", "tau_tier"]].isna().any(axis=1),
                "store_tier",
            ]
            .astype(str)
            .unique()
        )
        raise PlacementComplianceError(
            f"no complete V3.2 placement policy for tiers: {missing[:10]}"
        )
    eligible = pd.to_numeric(
        merged["eligible_at_launch"], errors="coerce"
    ).fillna(0).eq(1)
    arrived = pd.to_numeric(
        merged["arrived_in_launch_window"], errors="coerce"
    ).fillna(0).eq(1)
    velocity = pd.to_numeric(merged["v_neutral"], errors="coerce")
    pressure = pd.to_numeric(merged["stock_pressure"], errors="coerce")
    merged["eligible_at_launch_bool"] = eligible
    merged["arrived_bool"] = arrived
    if all(
        column in merged
        for column in (
            "prior_probability",
            "performance_probability",
            "reliability_r",
        )
    ):
        merged["prior_probability"] = pd.to_numeric(
            merged["prior_probability"], errors="raise"
        )
        merged["performance_probability"] = pd.to_numeric(
            merged["performance_probability"], errors="raise"
        )
        merged["reliability_r"] = pd.to_numeric(
            merged["reliability_r"], errors="raise"
        )
        merged["p_star"] = (
            (1.0 - merged["reliability_r"]) * merged["prior_probability"]
            + merged["reliability_r"] * merged["performance_probability"]
        ).clip(0, 1)
        merged["reasonable"] = (
            merged["eligible_at_launch_bool"]
            & (merged["p_star"] >= merged["tau_tier"])
        ).astype(int)
        merged["performance_evidence_weight"] = pd.to_numeric(
            merged["evidence_weight"], errors="coerce"
        ).fillna(0.0)
        return merged[
            [
                "sku_code",
                "store_code",
                "store_tier",
                "eligible_at_launch_bool",
                "arrived_bool",
                "prior_probability",
                "performance_probability",
                "performance_evidence_weight",
                "reliability_r",
                "p_star",
                "tau_tier",
                "reasonable",
            ]
        ].copy()

    merged["performance_observed"] = arrived & velocity.notna() & pressure.notna()
    merged["productive"] = np.where(
        merged["performance_observed"],
        ((velocity >= merged["v_min"]) & (pressure <= merged["inv_max"])).astype(float),
        np.nan,
    )
    rows: list[dict[str, Any]] = []
    for (sku, tier), group in merged.groupby(
        ["sku_code", "store_tier"], dropna=False
    ):
        launch = group.loc[group["eligible_at_launch_bool"]]
        prior = (
            float(
                pd.to_numeric(
                    launch["arrived_in_launch_window"], errors="coerce"
                ).mean()
            )
            if not launch.empty
            else np.nan
        )
        performance = group.loc[group["performance_observed"]].copy()
        weights = pd.to_numeric(
            performance["evidence_weight"], errors="coerce"
        ).fillna(0.0)
        if performance.empty or weights.sum() <= 0:
            p_perf = np.nan
            evidence_count = 0.0
        else:
            p_perf = float(np.average(performance["productive"], weights=weights))
            evidence_count = float(weights.sum())
        reliability = evidence_count / (evidence_count + shrinkage_k)
        p_star = (
            (1.0 - reliability) * prior + reliability * p_perf
            if pd.notna(prior) and pd.notna(p_perf)
            else prior
            if pd.notna(prior)
            else p_perf
        )
        tau = float(group["tau_tier"].iloc[0])
        for store_row in launch.itertuples(index=False):
            rows.append(
                {
                    "sku_code": str(sku),
                    "store_code": str(store_row.store_code),
                    "store_tier": str(tier),
                    "eligible_at_launch_bool": True,
                    "arrived_bool": bool(store_row.arrived_bool),
                    "prior_probability": prior,
                    "performance_probability": p_perf,
                    "performance_evidence_weight": evidence_count,
                    "reliability_r": reliability,
                    "p_star": p_star,
                    "tau_tier": tau,
                    "reasonable": int(pd.notna(p_star) and p_star >= tau),
                }
            )
    return pd.DataFrame(rows)


def _target_feature_frame(
    *,
    target_sku: str,
    stores: pd.DataFrame,
    target_store_features: pd.DataFrame | None,
) -> pd.DataFrame:
    result = stores.copy()
    result["category_fit"] = np.nan
    result["execution"] = np.nan
    result["category_fit_status"] = "missing_renormalized"
    result["execution_status"] = "missing_renormalized"
    if target_store_features is None or target_store_features.empty:
        return result
    require_columns(
        target_store_features,
        ("target_sku_code", "store_code", "category_fit", "execution"),
        "target store features",
    )
    feature = target_store_features.loc[
        target_store_features["target_sku_code"].astype(str) == target_sku
    ].copy()
    if feature.duplicated("store_code").any():
        raise PlacementComplianceError(
            f"target store features are not unique by store_code for {target_sku}"
        )
    keep = ["store_code", "category_fit", "execution"]
    for status_column in ("category_fit_status", "execution_status"):
        if status_column in feature:
            keep.append(status_column)
    result = result.drop(
        columns=[
            "category_fit",
            "execution",
            "category_fit_status",
            "execution_status",
        ]
    ).merge(feature[keep], on="store_code", how="left", validate="one_to_one")
    for column in ("category_fit", "execution"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        invalid = result[column].notna() & ~result[column].between(0, 1)
        if invalid.any():
            raise PlacementComplianceError(f"{column} must be within [0,1]")
    for status_column in ("category_fit_status", "execution_status"):
        if status_column not in result:
            value_column = (
                "category_fit" if status_column == "category_fit_status" else "execution"
            )
            result[status_column] = np.where(
                result[value_column].notna(),
                "provided_unverified",
                "missing_renormalized",
            )
        else:
            value_column = (
                "category_fit" if status_column == "category_fit_status" else "execution"
            )
            result[status_column] = result[status_column].where(
                result[status_column].notna(),
                np.where(
                    result[value_column].notna(),
                    "provided_unverified",
                    "missing_renormalized",
                ),
            )
    return result


def _hierarchical_history_slices(
    *,
    restored_tier_weekly: pd.DataFrame,
    target_attributes: Mapping[str, Any],
    store_tier: str,
) -> list[tuple[str, pd.DataFrame]]:
    tier_history = restored_tier_weekly.loc[
        restored_tier_weekly["store_tier"].astype(str) == str(store_tier)
    ].copy()
    slices: list[tuple[str, pd.DataFrame]] = []
    for column, label in (
        ("cate_level4_code", "category_l4"),
        ("cate_level3_code", "category_l3"),
        ("cate_level2_code", "category_l2"),
        ("cate_level1_code", "category_l1"),
    ):
        if column not in tier_history:
            continue
        value = target_attributes.get(column)
        if pd.isna(value) or str(value).strip() in {"", "nan", "None"}:
            continue
        matched = tier_history.loc[
            tier_history[column].astype(str) == str(value)
        ].copy()
        if not matched.empty:
            slices.append((label, matched))
    if not tier_history.empty:
        slices.append(("global_tier", tier_history))
    return slices


def _normalize_curve_profile(
    *,
    velocity_by_week: pd.Series,
    cv_by_week: Mapping[int, float],
    sample_count_by_week: pd.Series,
    lifecycle_weeks: int,
    min_week1_velocity: float,
    max_factor: float,
    min_observed_weeks: int,
    min_week1_samples: int,
) -> tuple[dict[int, float], dict[int, float], dict[int, str], str]:
    raw_velocity: dict[int, float] = {}
    for week, value in velocity_by_week.items():
        week_number = int(week)
        numeric = float(value)
        if 1 <= week_number <= lifecycle_weeks and np.isfinite(numeric) and numeric >= 0:
            raw_velocity[week_number] = numeric
    required_weeks = min(min_observed_weeks, lifecycle_weeks)
    if len(raw_velocity) < required_weeks:
        return {}, {}, {}, f"observed_weeks={len(raw_velocity)}<{required_weeks}"
    if 1 not in raw_velocity:
        return {}, {}, {}, "week1_missing"
    week1 = raw_velocity[1]
    if week1 <= min_week1_velocity:
        return {}, {}, {}, (
            f"week1_velocity={week1:.12g}<=minimum={min_week1_velocity:.12g}"
        )
    week1_samples = int(
        pd.to_numeric(
            pd.Series([sample_count_by_week.get(1, 0)]), errors="coerce"
        ).fillna(0).iloc[0]
    )
    if week1_samples < min_week1_samples:
        return {}, {}, {}, (
            f"week1_samples={week1_samples}<minimum={min_week1_samples}"
        )
    raw_factors = {
        week: float(value / week1) for week, value in raw_velocity.items()
    }
    factor_values = np.asarray(list(raw_factors.values()), dtype=float)
    if not np.isfinite(factor_values).all():
        return {}, {}, {}, "non_finite_factor"
    if (factor_values < 0).any():
        return {}, {}, {}, "negative_factor"
    observed_max = float(factor_values.max())
    if observed_max > max_factor:
        return {}, {}, {}, (
            f"max_factor={observed_max:.6g}>maximum={max_factor:.6g}"
        )

    observed_weeks = set(raw_factors)
    complete = pd.Series(raw_factors, dtype=float).reindex(
        range(1, lifecycle_weeks + 1)
    )
    complete = complete.interpolate(method="linear", limit_direction="both")
    complete.iloc[0] = 1.0
    factors = {int(week): float(value) for week, value in complete.items()}
    first_observed = min(observed_weeks)
    last_observed = max(observed_weeks)
    point_status: dict[int, str] = {}
    for week in range(1, lifecycle_weeks + 1):
        if week in observed_weeks:
            point_status[week] = "observed"
        elif first_observed < week < last_observed:
            point_status[week] = "interpolated"
        else:
            point_status[week] = "edge_carried"
    cvs = {
        int(week): float(value)
        for week, value in cv_by_week.items()
        if int(week) in observed_weeks and pd.notna(value)
    }
    return factors, cvs, point_status, "valid"


def _hierarchical_curve(
    *,
    restored_tier_weekly: pd.DataFrame,
    target_attributes: Mapping[str, Any],
    store_tier: str,
    lifecycle_weeks: int,
    min_week1_velocity: float,
    max_factor: float,
    min_observed_weeks: int,
    min_week1_samples: int,
) -> tuple[
    dict[int, float],
    dict[int, float],
    dict[int, str],
    str,
    str,
]:
    rejected: list[str] = []
    for source, history in _hierarchical_history_slices(
        restored_tier_weekly=restored_tier_weekly,
        target_attributes=target_attributes,
        store_tier=store_tier,
    ):
        history["v_neutral"] = pd.to_numeric(
            history["v_neutral"], errors="coerce"
        )
        grouped = history.groupby("lifecycle_week")["v_neutral"]
        median = grouped.median().dropna()
        sample_counts = grouped.count()
        cvs: dict[int, float] = {}
        for week, values in grouped:
            numeric = pd.to_numeric(
                values, errors="coerce"
            ).dropna()
            mean = float(numeric.mean()) if not numeric.empty else np.nan
            cvs[int(week)] = (
                float(numeric.std(ddof=1) / mean)
                if len(numeric) >= 2 and mean > 0
                else np.nan
            )
        factors, valid_cvs, point_status, reason = _normalize_curve_profile(
            velocity_by_week=median,
            cv_by_week=cvs,
            sample_count_by_week=sample_counts,
            lifecycle_weeks=lifecycle_weeks,
            min_week1_velocity=min_week1_velocity,
            max_factor=max_factor,
            min_observed_weeks=min_observed_weeks,
            min_week1_samples=min_week1_samples,
        )
        if factors:
            return factors, valid_cvs, point_status, source, "valid"
        rejected.append(f"{source}:{reason}")
    flat = {week: 1.0 for week in range(1, lifecycle_weeks + 1)}
    point_status = {week: "versioned_flat_default" for week in flat}
    reason = ";".join(rejected) if rejected else "no_historical_profile"
    return flat, {}, point_status, "versioned_flat_default", reason


def _hierarchical_path4_anchor(
    *,
    restored_tier_weekly: pd.DataFrame,
    target_attributes: Mapping[str, Any],
    store_tiers: list[str],
    default_value: float,
    min_week1_velocity: float,
    min_week1_samples: int,
) -> tuple[float, str]:
    for source in ("category_l4", "category_l3", "category_l2", "category_l1", "global"):
        frames: list[pd.DataFrame] = []
        for tier in store_tiers:
            for slice_source, history in _hierarchical_history_slices(
                restored_tier_weekly=restored_tier_weekly,
                target_attributes=target_attributes,
                store_tier=tier,
            ):
                normalized = "global" if slice_source == "global_tier" else slice_source
                if normalized == source:
                    frames.append(history)
                    break
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        week_one = pd.to_numeric(
            combined.loc[
                pd.to_numeric(combined["lifecycle_week"], errors="coerce").eq(1),
                "v_neutral",
            ],
            errors="coerce",
        ).dropna()
        if len(week_one) < min_week1_samples:
            continue
        median = float(week_one.median())
        if np.isfinite(median) and median > min_week1_velocity:
            return median, source
    return float(default_value), "versioned_static_base"


def plan_placement(
    *,
    targets: pd.DataFrame,
    comparables: pd.DataFrame,
    stores: pd.DataFrame,
    restored_store_weekly: pd.DataFrame,
    restored_tier_weekly: pd.DataFrame,
    config: Mapping[str, Any],
    placement_policy: pd.DataFrame | None = None,
    placement_evidence: pd.DataFrame | None = None,
    target_store_features: pd.DataFrame | None = None,
) -> PlacementResult:
    """Calculate V3.2 M3/M4 placement from standardized domain frames.

    Missing calibrated inputs use versioned, auditable initial parameters and
    empirical evidence.  Supplied business parameter/evidence snapshots override
    those defaults.  Missing score components are never fabricated; V3.2 weight
    renormalization is used instead.
    """

    require_columns(targets, ("sku_code",), "targets")
    require_columns(
        comparables,
        ("target_sku_code", "candidate_sku_code", "selected", "alpha_weight"),
        "comparables",
    )
    require_columns(
        stores,
        (
            "store_code",
            "store_tier",
            "store_level",
            "store_business_state",
            "eligible",
        ),
        "stores",
    )
    require_columns(
        restored_store_weekly,
        (
            "sku_code",
            "store_code",
            "store_tier",
            "lifecycle_week",
            "v_neutral",
            "stockout_days",
            "source_day_rows",
            "restored_exposure_days",
        ),
        "restored_store_weekly",
    )
    require_columns(
        restored_tier_weekly,
        ("sku_code", "store_tier", "lifecycle_week", "v_neutral"),
        "restored_tier_weekly",
    )
    eligible_stores = stores.loc[
        pd.to_numeric(stores["eligible"], errors="coerce").fillna(0).eq(1)
    ].copy()
    if eligible_stores.empty:
        raise ValueError("no eligible stores at the decision cutoff")
    eligible_stores["store_code"] = eligible_stores["store_code"].astype(str)
    eligible_stores["store_tier"] = eligible_stores["store_tier"].astype(str)
    history = restored_store_weekly.copy()
    for column in ("sku_code", "store_code", "store_tier"):
        history[column] = history[column].astype(str)
    history["v_neutral"] = pd.to_numeric(history["v_neutral"], errors="coerce")

    store_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    compliance_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    rate_low, rate_high = map(float, config["placement_rate_bounds"])
    ratio_low, ratio_high = map(float, config["placement_ratio_bounds"])
    epsilon = float(config["epsilon"])
    score_weights = config["store_score_weights"]
    if abs(sum(float(value) for value in score_weights.values()) - 0.95) > 1e-9:
        raise PlacementComplianceError("V3.2 store_score_weights must sum to 0.95")
    lifecycle_weeks = int(config["lifecycle_curve_weeks"])
    if lifecycle_weeks <= 0:
        raise PlacementComplianceError("lifecycle_curve_weeks must be positive")
    initial_defaults = config.get("initial_defaults", {})
    curve_min_week1_velocity = float(
        initial_defaults["curve_min_week1_velocity"]
    )
    curve_max_factor = float(initial_defaults["curve_max_factor"])
    curve_min_observed_weeks = int(
        initial_defaults["curve_min_observed_weeks"]
    )
    category_curve_min_week1_samples = int(
        initial_defaults["category_curve_min_week1_samples"]
    )
    path4_min_week1_samples = int(
        initial_defaults["path4_min_week1_samples"]
    )
    if curve_min_week1_velocity < 0 or curve_max_factor < 1:
        raise PlacementComplianceError(
            "curve_min_week1_velocity must be non-negative and curve_max_factor >= 1"
        )
    if min(
        curve_min_observed_weeks,
        category_curve_min_week1_samples,
        path4_min_week1_samples,
    ) <= 0:
        raise PlacementComplianceError("curve sample thresholds must be positive")
    target_attributes = (
        targets.assign(sku_code=targets["sku_code"].astype(str))
        .drop_duplicates("sku_code")
        .set_index("sku_code")
    )

    for target_sku in targets["sku_code"].astype(str).drop_duplicates():
        chosen_all = comparables.loc[
            (comparables["target_sku_code"].astype(str) == target_sku)
            & pd.to_numeric(comparables["selected"], errors="coerce")
            .fillna(0)
            .eq(1)
        ].copy()
        if chosen_all.empty:
            raise ValueError(f"no selected comparables for {target_sku}")
        chosen_all["alpha_weight"] = pd.to_numeric(
            chosen_all["alpha_weight"], errors="coerce"
        ).fillna(0.0)
        chosen = chosen_all.loc[
            ~chosen_all["candidate_sku_code"].astype(str).str.startswith("__STATIC__")
        ].copy()
        candidate_skus = chosen["candidate_sku_code"].astype(str).tolist()
        store_tiers = sorted(eligible_stores["store_tier"].unique().tolist())
        target_attribute_map = (
            target_attributes.loc[target_sku].to_dict()
            if target_sku in target_attributes.index
            else {}
        )
        policy, policy_source = _normalize_policy(
            policy=placement_policy,
            store_tiers=store_tiers,
            history=history,
            config=config,
        )
        compliance_rows.append(
            _compliance_row(
                target_sku=target_sku,
                concept="v_min_g_inv_max_g_tau_tier",
                status=(
                    "IMPLEMENTED_V3_2"
                    if policy_source == "business_versioned_policy"
                    else "VERSIONED_INITIAL_PARAMETER"
                ),
                source=policy_source,
                detail=(
                    "Versioned thresholds supplied by the run."
                    if policy_source == "business_versioned_policy"
                    else "v_min is the restore-pool store-tier median; inv_max and "
                    "tau_tier are versioned initial defaults pending Train calibration."
                ),
            )
        )

        if not candidate_skus:
            reasonableness = pd.DataFrame()
            evidence_source = "path4_no_comparable_evidence"
            rho_new = float(
                np.clip(
                    initial_defaults.get("path4_placement_rate", rate_low),
                    rate_low,
                    rate_high,
                )
            )
            anchor_velocity, path4_anchor_source = _hierarchical_path4_anchor(
                restored_tier_weekly=restored_tier_weekly,
                target_attributes=target_attribute_map,
                store_tiers=store_tiers,
                default_value=float(initial_defaults["path4_anchor_velocity"]),
                min_week1_velocity=curve_min_week1_velocity,
                min_week1_samples=path4_min_week1_samples,
            )
            warnings.append(
                f"{target_sku}: Path4 anchor source={path4_anchor_source}; placement "
                "rate uses the versioned initial Path4 parameter"
            )
            compliance_rows.append(
                _compliance_row(
                    target_sku=target_sku,
                    concept="path4_static_base",
                    status=(
                        "V3_2_CATEGORY_PRIOR_FALLBACK"
                        if path4_anchor_source != "versioned_static_base"
                        else "VERSIONED_INITIAL_PARAMETER"
                    ),
                    source=path4_anchor_source,
                    detail=(
                        f"anchor_velocity={anchor_velocity:.6f}; hierarchy is L4, "
                        "L3, L2, L1, global, then the versioned static base."
                    ),
                )
            )
        else:
            evidence, evidence_source = _prepare_evidence(
                evidence=placement_evidence,
                candidate_skus=candidate_skus,
                eligible_stores=eligible_stores,
                history=history,
            )
            reasonableness = _candidate_reasonableness(
                evidence=evidence,
                policy=policy,
                shrinkage_k=float(config["shrinkage_k"]),
            )
            compliance_rows.append(
                _compliance_row(
                    target_sku=target_sku,
                    concept="launch_window_prior_and_performance_probability",
                    status=(
                        "IMPLEMENTED_V3_2"
                        if evidence_source == "modeled_store_probability_evidence"
                        else "EMPIRICAL_INITIAL_MODEL"
                    ),
                    source=evidence_source,
                    detail=(
                        "Store-level conditional p_prior/p_perf/reliability are supplied."
                        if evidence_source == "modeled_store_probability_evidence"
                        else "Group empirical p_prior/p_perf with V3.2 shrinkage is used; "
                        "the source and limits are recorded pending Train calibration."
                    ),
                )
            )
            candidate_rates: list[tuple[float, float]] = []
            ratio_terms: list[float] = []
            for row in chosen.itertuples(index=False):
                candidate = reasonableness.loc[
                    (reasonableness["sku_code"] == str(row.candidate_sku_code))
                    & reasonableness["eligible_at_launch_bool"]
                ]
                denominator = len(candidate)
                if denominator <= 0:
                    rho_observed = 0.0
                    rho_star = 0.0
                else:
                    rho_observed = float(
                        candidate["arrived_bool"].sum() / denominator
                    )
                    rho_star = float(candidate["reasonable"].mean())
                alpha = float(row.alpha_weight)
                candidate_rates.append((alpha, rho_observed))
                ratio_terms.append(
                    alpha
                    * log(
                        float(
                            np.clip(
                                (rho_star + epsilon) / (rho_observed + epsilon),
                                ratio_low,
                                ratio_high,
                            )
                        )
                    )
                )
            rho_hist = sum(alpha * rate for alpha, rate in candidate_rates)
            rho_new = float(
                np.clip(rho_hist * exp(sum(ratio_terms)), rate_low, rate_high)
            )
            historical_velocity = pd.to_numeric(
                chosen.get("historical_velocity"), errors="coerce"
            )
            valid_anchor = historical_velocity.notna() & chosen["alpha_weight"].gt(0)
            anchor_velocity = (
                float(
                    np.average(
                        historical_velocity.loc[valid_anchor],
                        weights=chosen.loc[valid_anchor, "alpha_weight"],
                    )
                )
                if valid_anchor.any()
                else np.nan
            )
            if not np.isfinite(anchor_velocity):
                raise PlacementComplianceError(
                    f"no finite comparable anchor velocity for {target_sku}"
                )

        total_eligible = len(eligible_stores)
        k_total = int(np.clip(round(rho_new * total_eligible), 1, total_eligible))
        target_store = _target_feature_frame(
            target_sku=target_sku,
            stores=eligible_stores,
            target_store_features=target_store_features,
        )
        target_store["v_anchor_week1"] = np.nan
        target_store["v_anchor_source"] = "missing"
        target_store["productive_probability"] = np.nan
        selected_candidates = chosen[["candidate_sku_code", "alpha_weight"]].copy()
        week_one_history = history.loc[
            history["sku_code"].isin(candidate_skus)
            & pd.to_numeric(history["lifecycle_week"], errors="coerce").eq(1)
        ].copy()
        store_velocity_map = (
            week_one_history.groupby(["sku_code", "store_code"])["v_neutral"]
            .mean()
            .to_dict()
        )
        tier_velocity_map = (
            week_one_history.groupby(["sku_code", "store_tier"])["v_neutral"]
            .mean()
            .to_dict()
        )
        probability_store_map = (
            reasonableness.groupby(["sku_code", "store_code"])["p_star"]
            .mean()
            .to_dict()
            if not reasonableness.empty
            else {}
        )
        probability_tier_map = (
            reasonableness.groupby(["sku_code", "store_tier"])["p_star"]
            .mean()
            .to_dict()
            if not reasonableness.empty
            else {}
        )
        for index, store in target_store.iterrows():
            velocity_values: list[float] = []
            velocity_weights: list[float] = []
            productive_values: list[float] = []
            productive_weights: list[float] = []
            used_store_specific = False
            for candidate_row in selected_candidates.itertuples(index=False):
                candidate_code = str(candidate_row.candidate_sku_code)
                store_key = (candidate_code, str(store["store_code"]))
                tier_key = (candidate_code, str(store["store_tier"]))
                candidate_velocity = store_velocity_map.get(store_key)
                if candidate_velocity is not None and pd.notna(candidate_velocity):
                    used_store_specific = True
                else:
                    candidate_velocity = tier_velocity_map.get(tier_key)
                if candidate_velocity is not None and pd.notna(candidate_velocity):
                    velocity_values.append(float(candidate_velocity))
                    velocity_weights.append(float(candidate_row.alpha_weight))
                probability = probability_store_map.get(store_key)
                if probability is None or pd.isna(probability):
                    probability = probability_tier_map.get(tier_key)
                if probability is not None and pd.notna(probability):
                    productive_values.append(float(probability))
                    productive_weights.append(float(candidate_row.alpha_weight))
            if velocity_values and sum(velocity_weights) > 0:
                target_store.at[index, "v_anchor_week1"] = float(
                    np.expm1(
                        np.average(
                            np.log1p(np.maximum(velocity_values, 0)),
                            weights=velocity_weights,
                        )
                    )
                )
                target_store.at[index, "v_anchor_source"] = (
                    "comparable_store" if used_store_specific else "comparable_tier"
                )
            else:
                target_store.at[index, "v_anchor_week1"] = anchor_velocity
                target_store.at[index, "v_anchor_source"] = (
                    path4_anchor_source if chosen.empty else "comparable_global"
                )
            if productive_values and sum(productive_weights) > 0:
                target_store.at[index, "productive_probability"] = float(
                    np.average(productive_values, weights=productive_weights)
                )

        velocity = pd.to_numeric(
            target_store["v_anchor_week1"], errors="coerce"
        ).clip(lower=0)
        target_store["velocity_norm"] = velocity.rank(method="average", pct=True)
        scores: list[float] = []
        completeness: list[float] = []
        for row in target_store.itertuples(index=False):
            components = {
                "productive": getattr(row, "productive_probability"),
                "velocity": getattr(row, "velocity_norm"),
                "category_fit": getattr(row, "category_fit"),
                "execution": getattr(row, "execution"),
            }
            available = {
                key: float(value)
                for key, value in components.items()
                if pd.notna(value)
            }
            denominator = sum(float(score_weights[key]) for key in available)
            scores.append(
                sum(
                    float(score_weights[key]) * value
                    for key, value in available.items()
                )
                / denominator
                if denominator
                else np.nan
            )
            completeness.append(denominator / 0.95)
        target_store["store_score"] = scores
        target_store["score_completeness"] = completeness
        target_store["selected"] = 0

        # First allocate K by store level; within the same level sort all
        # business-state groups together by store_score.
        level_values = sorted(
            target_store["store_level"].astype(str).unique(), key=_tier_rank
        )
        quota_by_level: dict[str, int] = {}
        remaining = k_total
        for level in level_values:
            count = int(target_store["store_level"].astype(str).eq(level).sum())
            quota_by_level[level] = min(count, remaining)
            remaining -= quota_by_level[level]
            if remaining <= 0:
                break
        for level, count in quota_by_level.items():
            indices = (
                target_store.loc[target_store["store_level"].astype(str) == level]
                .sort_values(
                    ["store_score", "store_code"],
                    ascending=[False, True],
                    na_position="last",
                )
                .head(count)
                .index
            )
            target_store.loc[indices, "selected"] = 1
        target_store["sku_code"] = target_sku
        store_rows.extend(target_store.to_dict(orient="records"))
        category_coverage = float(target_store["category_fit"].notna().mean())
        execution_coverage = float(target_store["execution"].notna().mean())
        category_formal = bool(
            category_coverage == 1.0
            and target_store.loc[
                target_store["category_fit"].notna(), "category_fit_status"
            ].eq("v3_2_complete_90d").all()
        )
        execution_formal = bool(
            execution_coverage == 1.0
            and target_store.loc[
                target_store["execution"].notna(), "execution_status"
            ].eq("v3_2_complete_execution_universe").all()
        )
        compliance_rows.extend(
            [
                _compliance_row(
                    target_sku=target_sku,
                    concept="category_fit_90d_restored_neutral_velocity_percentile",
                    status=(
                        "IMPLEMENTED_V3_2" if category_formal else
                        "PROVIDED_UNVERIFIED" if category_coverage > 0 else
                        "MISSING_RENORMALIZED"
                    ),
                    source=(
                        "target_store_features"
                        if category_coverage > 0
                        else "not_supplied"
                    ),
                    detail=(
                        f"coverage={category_coverage:.6f}; missing values are omitted "
                        "and score weights are renormalized."
                    ),
                ),
                _compliance_row(
                    target_sku=target_sku,
                    concept="execution_instock_nondisplay_zero_ratio",
                    status=(
                        "IMPLEMENTED_V3_2" if execution_formal else
                        "PROVIDED_UNVERIFIED" if execution_coverage > 0 else
                        "MISSING_RENORMALIZED"
                    ),
                    source=(
                        "target_store_features"
                        if execution_coverage > 0
                        else "not_supplied"
                    ),
                    detail=(
                        f"coverage={execution_coverage:.6f}; no 0.5 or full-score "
                        "placeholder is inserted."
                    ),
                ),
                _compliance_row(
                    target_sku=target_sku,
                    concept="hard_eligibility",
                    status="PARTIAL_SOURCE_FIELDS",
                    source="store snapshot",
                    detail=(
                        "Open/closed status is applied. Distribution reachability, "
                        "store bans and theme/category qualification are not delivered."
                    ),
                ),
            ]
        )
        summary_rows.append(
            {
                "sku_code": target_sku,
                "eligible_store_count": total_eligible,
                "placement_rate": rho_new,
                "planned_store_count": int(target_store["selected"].sum()),
                "anchor_velocity_week1": float(
                    target_store.loc[
                        target_store["selected"] == 1, "v_anchor_week1"
                    ].mean()
                ),
                "category_fit_coverage": category_coverage,
                "execution_coverage": execution_coverage,
                "shelf_capacity_status": "missing_not_hard_filtered",
                "formal_v3_2_rate": int(
                    policy_source == "business_versioned_policy"
                    and evidence_source == "modeled_store_probability_evidence"
                ),
            }
        )
        for level, count in quota_by_level.items():
            diagnostic_rows.append(
                {
                    "sku_code": target_sku,
                    "store_level": level,
                    "eligible_count": int(
                        target_store["store_level"].astype(str).eq(level).sum()
                    ),
                    "quota": count,
                    "selection_rule": "global_store_score_within_level",
                }
            )
        if not reasonableness.empty:
            grouped_reason = (
                reasonableness.groupby(["sku_code", "store_tier"], as_index=False)
                .agg(
                    eligible_count=("eligible_at_launch_bool", "sum"),
                    arrived_count=("arrived_bool", "sum"),
                    prior_probability=("prior_probability", "mean"),
                    performance_probability=("performance_probability", "mean"),
                    reliability_r=("reliability_r", "mean"),
                    p_star=("p_star", "mean"),
                    tau_tier=("tau_tier", "first"),
                    reasonable_rate=("reasonable", "mean"),
                )
            )
            for row in grouped_reason.itertuples(index=False):
                diagnostic_rows.append(
                    {
                        "sku_code": target_sku,
                        "store_level": "",
                        "eligible_count": row.eligible_count,
                        "quota": np.nan,
                        "selection_rule": "comparable_reasonableness",
                        "candidate_sku_code": row.sku_code,
                        "store_tier": row.store_tier,
                        "prior_probability": row.prior_probability,
                        "performance_probability": row.performance_probability,
                        "reliability_r": row.reliability_r,
                        "p_star": row.p_star,
                        "tau_tier": row.tau_tier,
                        "reasonable": row.reasonable_rate,
                    }
                )

        selected_stores = target_store.loc[target_store["selected"] == 1]
        target_curve_fallbacks = 0
        for tier, tier_stores in selected_stores.groupby("store_tier"):
            comparable_curve = restored_tier_weekly.loc[
                restored_tier_weekly["store_tier"].astype(str) == str(tier)
            ].merge(
                selected_candidates,
                left_on="sku_code",
                right_on="candidate_sku_code",
                how="inner",
            )
            comparable_factors: dict[int, float] = {}
            comparable_cvs: dict[int, float] = {}
            comparable_point_status: dict[int, str] = {}
            comparable_reason = "no_selected_comparable_history"
            if not comparable_curve.empty:
                grouped = comparable_curve.groupby("lifecycle_week")
                velocity_by_week = grouped.apply(
                    lambda group: _weighted_log_average(
                        group["v_neutral"], group["alpha_weight"]
                    )
                )
                sample_counts = grouped["v_neutral"].count()
                for week, group in grouped:
                    values = pd.to_numeric(group["v_neutral"], errors="coerce").dropna()
                    mean = float(values.mean()) if not values.empty else np.nan
                    comparable_cvs[int(week)] = (
                        float(values.std(ddof=1) / mean)
                        if len(values) >= 2 and mean > 0
                        else np.nan
                    )
                (
                    comparable_factors,
                    comparable_cvs,
                    comparable_point_status,
                    comparable_reason,
                ) = _normalize_curve_profile(
                    velocity_by_week=velocity_by_week,
                    cv_by_week=comparable_cvs,
                    sample_count_by_week=sample_counts,
                    lifecycle_weeks=lifecycle_weeks,
                    min_week1_velocity=curve_min_week1_velocity,
                    max_factor=curve_max_factor,
                    min_observed_weeks=curve_min_observed_weeks,
                    min_week1_samples=1,
                )

            if comparable_factors:
                profile_factors = comparable_factors
                profile_cvs = comparable_cvs
                profile_point_status = comparable_point_status
                profile_source = "selected_comparables"
                profile_status = "IMPLEMENTED_V3_2"
                profile_reason = "valid"
            else:
                (
                    profile_factors,
                    profile_cvs,
                    profile_point_status,
                    hierarchy_source,
                    hierarchy_reason,
                ) = _hierarchical_curve(
                    restored_tier_weekly=restored_tier_weekly,
                    target_attributes=target_attribute_map,
                    store_tier=str(tier),
                    lifecycle_weeks=lifecycle_weeks,
                    min_week1_velocity=curve_min_week1_velocity,
                    max_factor=curve_max_factor,
                    min_observed_weeks=curve_min_observed_weeks,
                    min_week1_samples=category_curve_min_week1_samples,
                )
                if hierarchy_source == "versioned_flat_default":
                    profile_source = hierarchy_source
                    profile_status = "VERSIONED_INITIAL_PARAMETER"
                else:
                    profile_source = hierarchy_source + "_lifecycle_prior"
                    profile_status = "V3_2_CATEGORY_PRIOR_FALLBACK"
                profile_reason = (
                    f"selected_comparables:{comparable_reason};"
                    f"fallback:{hierarchy_reason}"
                )
                target_curve_fallbacks += lifecycle_weeks
                warnings.append(
                    f"{target_sku} tier={tier}: lifecycle curve fallback; "
                    f"{profile_reason}; source={profile_source}"
                )
            diagnostic_rows.append(
                {
                    "sku_code": target_sku,
                    "store_level": "",
                    "eligible_count": len(tier_stores),
                    "quota": np.nan,
                    "selection_rule": "lifecycle_curve_profile",
                    "store_tier": str(tier),
                    "curve_source": profile_source,
                    "curve_validation": profile_reason,
                }
            )
            for week in range(1, lifecycle_weeks + 1):
                factor = profile_factors[week]
                cv = profile_cvs.get(week, np.nan)
                curve_rows.append(
                    {
                        "sku_code": target_sku,
                        "store_tier": str(tier),
                        "lifecycle_week": week,
                        "selected_store_count": len(tier_stores),
                        "v_anchor_week1": float(
                            tier_stores["v_anchor_week1"].mean()
                        ),
                        "lifecycle_factor": factor,
                        "demand_cv": cv,
                        "uncertainty_status": (
                            "estimated_from_history"
                            if pd.notna(cv)
                            else "missing_not_fabricated"
                        ),
                        "period_factor_h": 1.0,
                        "period_factor_status": "neutral_no_confirmed_schedule",
                        "curve_source": profile_source,
                        "curve_status": profile_status,
                        "curve_point_status": profile_point_status[week],
                        "curve_profile_validation": profile_reason,
                    }
                )
        compliance_rows.append(
            _compliance_row(
                target_sku=target_sku,
                concept="complete_lifecycle_curve",
                status=(
                    "IMPLEMENTED_V3_2"
                    if target_curve_fallbacks == 0
                    else "V3_2_CATEGORY_PRIOR_OR_VERSIONED_DEFAULT"
                ),
                source="selected comparables/category lifecycle prior",
                detail=(
                    f"weeks=1..{lifecycle_weeks}; fallback_profile_rows="
                    f"{target_curve_fallbacks}. Each tier uses one validated, "
                    "week-1-normalized source; no cross-source weekly splicing."
                ),
            )
        )

    return PlacementResult(
        pd.DataFrame(store_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(curve_rows),
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(compliance_rows),
        tuple(warnings),
    )
