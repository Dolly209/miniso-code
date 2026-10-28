from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any, Mapping, Callable
from time import monotonic

import numpy as np
import pandas as pd

from data.domain import require_columns
from placement.lifecycle import all_store_lifecycle, mix_comparable_curves


class PlacementComplianceError(ValueError):
    """Raised when a formal V3.2 placement run lacks a required input."""


CURVE_COLUMNS = (
    "sku_code", "store_tier", "lifecycle_week", "selected_store_count", "v_anchor_week1",
    "lifecycle_factor", "demand_cv", "uncertainty_status", "period_factor_h", "period_factor_status",
    "curve_source", "curve_status", "curve_point_status", "curve_profile_validation", "curve_formula",
    "all_store_aggregation_basis", "lifecycle_component_observed", "lifecycle_component_interpolated",
    "lifecycle_component_edge_carried", "lifecycle_component_flat_default",
)


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
            | (normalized["inv_max"] < 0)
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
            origin = (str(normalized["step5_policy_origin"].iloc[0])
                      if "step5_policy_origin" in normalized else "business_versioned_policy")
            return normalized, origin
    else:
        normalized = pd.DataFrame(columns=required)

    defaults = config.get("initial_defaults", {})
    if not defaults:
        raise PlacementComplianceError(
            "placement.initial_defaults is required for uncalibrated parameters"
        )
    existing = set(normalized.get("store_tier", pd.Series(dtype=str)).astype(str))
    rows: list[dict[str, float | str]] = []
    medians = history.assign(v_neutral=pd.to_numeric(history["v_neutral"], errors="coerce")).groupby("store_tier")["v_neutral"].median()
    for tier in store_tiers:
        if tier in existing:
            continue
        median = medians.get(tier, np.nan)
        rows.append(
            {
                "store_tier": tier,
                "v_min": (
                    float(median)
                    if pd.notna(median)
                    else float(defaults["v_min_when_no_history"])
                ),
                "inv_max": float(defaults["inv_max_days"]),
                "tau_tier": float(defaults["tau_tier"]),
            }
        )
    if rows:
        normalized = pd.DataFrame(rows) if normalized.empty else pd.concat([normalized, pd.DataFrame(rows)], ignore_index=True)
    source = (
        "mixed_business_policy_and_versioned_defaults"
        if policy is not None and not policy.empty
        else "versioned_initial_defaults"
    )
    return normalized, source


def _prepare_evidence(
    *,
    evidence: pd.DataFrame | None,
    candidate_skus: list[str],
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
    # V3.2 launch placement evidence must be measured inside the comparable's
    # fixed launch window. Later Round-4 exposure is useful for velocity and
    # lifecycle restoration, but it cannot prove launch arrival.
    return (
        pd.DataFrame(columns=[*required, "evidence_weight"]),
        "missing_launch_window_evidence",
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
        merged["evidence_scope"] = "modeled_store_probability"
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
                "evidence_scope",
            ]
        ].copy()

    merged["performance_observed"] = (
        eligible & arrived & velocity.notna() & pressure.notna()
    )
    merged["productive"] = np.where(
        merged["performance_observed"],
        ((velocity >= merged["v_min"]) & (pressure <= merged["inv_max"])).astype(float),
        np.nan,
    )
    evidence_weight = pd.to_numeric(
        merged["evidence_weight"], errors="coerce"
    ).fillna(0.0)
    # V3.2 p_prior is conditional on business state, store level, region and
    # business area.  Use every delivered dimension at that grain; category
    # movement is unavailable and is disclosed in the status rather than
    # fabricated.  This replaces the old SKU x store-tier probability that
    # made an entire level pass/fail together.
    segment_dimensions = [
        column
        for column in (
            "store_business_state",
            "store_level",
            "province_name",
            "secbiz_area_name",
        )
        if column in merged
    ]
    for column in segment_dimensions:
        merged[column] = merged[column].fillna("__UNKNOWN__").astype(str)
    segment_keys = ["sku_code", *segment_dimensions]
    if len(segment_keys) == 1:
        # Legacy evidence snapshots only carry store_tier.  Retain this as an
        # explicitly coarse fallback so old checkpoints remain readable.
        segment_keys.append("store_tier")
        segment_status = "coarse_store_tier_fallback_missing_condition_fields"
    else:
        segment_status = (
            "business_state_level_region_business_area;"
            "category_movement_missing"
        )
    merged["_eligible_weight"] = evidence_weight.where(eligible, 0.0)
    merged["_arrival_weight"] = merged["_eligible_weight"] * arrived.astype(float)
    merged["_performance_weight"] = evidence_weight.where(
        merged["performance_observed"], 0.0
    )
    merged["_productive_weight"] = (
        merged["_performance_weight"] * pd.to_numeric(
            merged["productive"], errors="coerce"
        ).fillna(0.0)
    )
    grouped = merged.groupby(segment_keys, dropna=False, sort=False)
    eligible_group_weight = grouped["_eligible_weight"].transform("sum")
    arrival_group_weight = grouped["_arrival_weight"].transform("sum")
    performance_group_weight = grouped["_performance_weight"].transform("sum")
    productive_group_weight = grouped["_productive_weight"].transform("sum")
    prior = np.divide(
        arrival_group_weight,
        eligible_group_weight,
        out=np.full(len(merged), np.nan),
        where=eligible_group_weight.to_numpy(dtype=float) > 0,
    )
    performance = np.divide(
        productive_group_weight,
        performance_group_weight,
        out=np.full(len(merged), np.nan),
        where=performance_group_weight.to_numpy(dtype=float) > 0,
    )
    prior = pd.Series(prior, index=merged.index)
    performance = pd.Series(performance, index=merged.index)
    reliability = performance_group_weight / (
        performance_group_weight + shrinkage_k
    )
    p_star = prior.copy()
    both = prior.notna() & performance.notna()
    p_star.loc[both] = (
        (1.0 - reliability.loc[both]) * prior.loc[both]
        + reliability.loc[both] * performance.loc[both]
    )
    merged["prior_probability"] = prior
    merged["performance_probability"] = performance
    merged["performance_evidence_weight"] = performance_group_weight
    merged["reliability_r"] = reliability
    merged["p_star"] = p_star
    merged["prior_group_effective_weight"] = eligible_group_weight
    merged["performance_group_effective_weight"] = performance_group_weight
    merged["prior_group_definition"] = segment_status
    merged["evidence_scope"] = np.where(
        ~arrived,
        "group_probability_but_nonarrival_insufficient_no_auto_expand",
        np.where(
            merged["performance_observed"],
            "conditional_group_arrival_and_performance",
            "conditional_group_arrival_only",
        ),
    )
    merged["reasonable"] = (
        eligible & p_star.notna() & p_star.ge(merged["tau_tier"])
    ).astype(int)
    return merged.loc[eligible, [
        "sku_code", "store_code", "store_tier", "eligible_at_launch_bool", "arrived_bool",
        "prior_probability", "performance_probability", "performance_evidence_weight",
        "reliability_r", "p_star", "tau_tier", "reasonable", "evidence_scope",
        "prior_group_effective_weight", "performance_group_effective_weight",
        "prior_group_definition",
    ]].copy()


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
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    load_target: Callable[[str], PlacementResult | None] | None = None,
    save_target: Callable[[str, PlacementResult], None] | None = None,
    prepared_cache: dict | None = None,
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
    if "usable_for_launch_curve" in history:
        history.loc[~history.usable_for_launch_curve.fillna(False), "v_neutral"] = np.nan
    restored_tier_weekly = restored_tier_weekly.copy()
    if "usable_for_launch_curve" in restored_tier_weekly:
        restored_tier_weekly.loc[~restored_tier_weekly.usable_for_launch_curve.fillna(False), "v_neutral"] = np.nan

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

    started = monotonic()
    def emit(event: str, **details: Any) -> None:
        if progress:
            progress(event, {**details, "elapsed_seconds": round(monotonic() - started, 3)})

    emit("prepare_start", history_rows=len(history))
    # A store can be eligible in a comparable's historical launch window and
    # closed by the target decision cutoff.  Its historical tier still belongs
    # in p_prior/p_perf, although it must never enter the target's current
    # selection pool.  Build policy coverage for both universes.
    policy_tiers = set(eligible_stores["store_tier"].astype(str))
    if placement_evidence is not None and not placement_evidence.empty:
        policy_tiers.update(placement_evidence["store_tier"].astype(str))
    store_tiers = sorted(policy_tiers)
    policy, policy_source = _normalize_policy(
        policy=placement_policy,
        store_tiers=store_tiers,
        history=history,
        config=config,
    )
    selected = comparables.loc[pd.to_numeric(comparables["selected"], errors="coerce").fillna(0).eq(1)].copy()
    selected["target_sku_code"] = selected["target_sku_code"].astype(str)
    selected_by_target = {str(k): v for k, v in selected.groupby("target_sku_code", sort=False)}
    candidate_codes = set(selected["candidate_sku_code"].astype(str))
    candidate_codes = {k for k in candidate_codes if not k.startswith("__STATIC__")}
    evidence_groups = {} if placement_evidence is None else placement_evidence.assign(sku_code=placement_evidence["sku_code"].astype(str)).groupby("sku_code", sort=False).indices
    prepared_cache = prepared_cache if prepared_cache is not None else {}
    reason_cache = prepared_cache.setdefault("reason", {})
    hierarchy_cache = prepared_cache.setdefault("hierarchy", {})
    def cached_hierarchy(function, **kwargs):
        attributes = tuple(sorted((k, str(v)) for k, v in kwargs["target_attributes"].items()))
        key = (function.__name__, attributes, kwargs.get("store_tier"), tuple(kwargs.get("store_tiers", ())))
        if key not in hierarchy_cache:
            hierarchy_cache[key] = function(**kwargs)
        return hierarchy_cache[key]
    if "fixed_statistics" not in prepared_cache:
        week_one_history = history.loc[pd.to_numeric(history["lifecycle_week"], errors="coerce").eq(1)]
        store_velocity_series = week_one_history.groupby(["sku_code", "store_code"])["v_neutral"].mean()
        tier_velocity_series = restored_tier_weekly.loc[pd.to_numeric(restored_tier_weekly.lifecycle_week, errors="coerce").eq(1)].groupby(["sku_code", "store_tier"])["v_neutral"].mean()
        lifecycle_all = all_store_lifecycle(restored_tier_weekly)
        prepared_cache["fixed_statistics"] = (store_velocity_series, tier_velocity_series, lifecycle_all)
    store_velocity_series, tier_velocity_series, lifecycle_all = prepared_cache["fixed_statistics"]
    all_store_basis = ";".join(sorted(lifecycle_all.aggregation_basis.unique())) if not lifecycle_all.empty else "missing"
    emit("prepare_complete", candidates=len(candidate_codes))
    target_codes = targets["sku_code"].astype(str).drop_duplicates().tolist()
    output_lists = [store_rows, summary_rows, curve_rows, diagnostic_rows, compliance_rows, warnings]
    for target_index, target_sku in enumerate(target_codes, 1):
        emit("target_start", target_index=target_index, target_count=len(target_codes), sku_code=target_sku)
        cached = load_target(target_sku) if load_target else None
        if cached is not None:
            for destination, frame in zip(output_lists[:5], [cached.store_plan, cached.summary, cached.demand_curve, cached.diagnostics, cached.compliance]):
                destination.extend(frame.to_dict(orient="records"))
            warnings.extend(cached.warnings)
            emit("target_checkpoint_reused", target_index=target_index, target_count=len(target_codes), sku_code=target_sku)
            continue
        offsets = [len(rows) for rows in output_lists]
        chosen_all = selected_by_target.get(target_sku, selected.iloc[:0]).copy()
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

        placement_fallback_no_evidence = False
        missing_launch_evidence_candidates: list[str] = []
        all_valid_evidence_modeled = False
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
            anchor_velocity, path4_anchor_source = cached_hierarchy(_hierarchical_path4_anchor,
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
            for code in candidate_skus:
                if code not in reason_cache:
                    candidate_evidence = placement_evidence.iloc[evidence_groups[code]] if code in evidence_groups else None
                    evidence, source = _prepare_evidence(
                        evidence=candidate_evidence,
                        candidate_skus=[code],
                    )
                    reason_cache[code] = (
                        _candidate_reasonableness(
                            evidence=evidence,
                            policy=policy,
                            shrinkage_k=float(config["shrinkage_k"]),
                        )
                        if not evidence.empty
                        else pd.DataFrame(),
                        source,
                    )
            reason_frames = [
                reason_cache[code][0]
                for code in candidate_skus
                if not reason_cache[code][0].empty
            ]
            reasonableness = (
                pd.concat(reason_frames, ignore_index=True)
                if reason_frames
                else pd.DataFrame()
            )
            missing_launch_evidence_candidates = [
                code for code in candidate_skus if reason_cache[code][0].empty
            ]
            valid_sources = {
                reason_cache[code][1]
                for code in candidate_skus
                if not reason_cache[code][0].empty
            }
            placement_fallback_no_evidence = not reason_frames
            if placement_fallback_no_evidence:
                evidence_source = "missing_launch_window_evidence_versioned_prior"
            elif missing_launch_evidence_candidates:
                evidence_source = "partial_launch_window_evidence_missing_candidates"
            else:
                evidence_source = (
                    next(iter(valid_sources))
                    if len(valid_sources) == 1
                    else "mixed_valid_launch_window_evidence"
                )
            all_valid_evidence_modeled = bool(valid_sources) and all(
                source == "modeled_store_probability_evidence"
                for source in valid_sources
            )
            compliance_rows.append(
                _compliance_row(
                    target_sku=target_sku,
                    concept="launch_window_prior_and_performance_probability",
                    status=(
                        "IMPLEMENTED_V3_2"
                        if all_valid_evidence_modeled and not missing_launch_evidence_candidates
                        else "VERSIONED_INITIAL_PARAMETER"
                        if placement_fallback_no_evidence
                        else "EMPIRICAL_INITIAL_MODEL"
                    ),
                    source=evidence_source,
                    detail=(
                        "Store-level conditional p_prior/p_perf/reliability are supplied."
                        if all_valid_evidence_modeled and not missing_launch_evidence_candidates
                        else (
                            "No selected comparable has source coverage in its true launch "
                            "window. Placement width uses the versioned conservative prior; "
                            "later Round-4 exposure is not treated as launch arrival."
                            if placement_fallback_no_evidence
                            else "Only selected comparables with true launch-window evidence "
                            "are used for placement width and their alpha weights are "
                            "renormalized. Missing candidates are skipped, never replaced "
                            "with later Round-4 exposure."
                        )
                    ),
                )
            )
            candidate_rates: list[tuple[float, float]] = []
            ratio_terms: list[float] = []
            valid_rate_rows: list[tuple[float, float, float]] = []
            for row in chosen.itertuples(index=False):
                candidate = (
                    reasonableness.loc[
                        (reasonableness["sku_code"] == str(row.candidate_sku_code))
                        & reasonableness["eligible_at_launch_bool"]
                    ]
                    if not reasonableness.empty
                    else pd.DataFrame()
                )
                denominator = len(candidate)
                alpha = max(float(row.alpha_weight), 0.0)
                if denominator > 0 and alpha > 0:
                    valid_rate_rows.append(
                        (
                            alpha,
                            float(candidate["arrived_bool"].sum() / denominator),
                            float(candidate["reasonable"].mean()),
                        )
                    )
            if valid_rate_rows:
                valid_alpha_sum = sum(alpha for alpha, _, _ in valid_rate_rows)
                for alpha, rho_observed, rho_star in valid_rate_rows:
                    normalized_alpha = alpha / valid_alpha_sum
                    candidate_rates.append((normalized_alpha, rho_observed))
                    ratio_terms.append(
                        normalized_alpha
                        * log(
                            float(
                                np.clip(
                                    (rho_star + epsilon)
                                    / (rho_observed + epsilon),
                                    ratio_low,
                                    ratio_high,
                                )
                            )
                        )
                    )
                rho_hist = sum(
                    alpha * rate for alpha, rate in candidate_rates
                )
                rho_new = float(
                    np.clip(
                        rho_hist * exp(sum(ratio_terms)), rate_low, rate_high
                    )
                )
            else:
                rho_new = float(
                    np.clip(
                        initial_defaults.get(
                            "missing_launch_evidence_placement_rate", rate_low
                        ),
                        rate_low,
                        rate_high,
                    )
                )
                warnings.append(
                    f"{target_sku}: selected comparables have no true launch-window "
                    "placement evidence; using versioned conservative placement rate "
                    f"{rho_new:.6f}; missing_candidates="
                    f"{','.join(missing_launch_evidence_candidates)}"
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
        k_total = int(
            np.clip(
                round(rho_new * total_eligible),
                1 if (not candidate_skus or placement_fallback_no_evidence) else 0,
                total_eligible,
            )
        )
        target_store = _target_feature_frame(
            target_sku=target_sku,
            stores=eligible_stores,
            target_store_features=target_store_features,
        )
        target_store["v_anchor_week1"] = np.nan
        target_store["v_anchor_source"] = "missing"
        target_store["productive_probability"] = np.nan
        target_store["historical_arrival_probability"] = np.nan
        selected_candidates = chosen[["candidate_sku_code", "alpha_weight"]].copy()
        probability_store_map = (
            reasonableness.groupby(["sku_code", "store_code"])["p_star"].mean()
            if not reasonableness.empty else pd.Series(dtype=float)
        )
        prior_store_map = (
            reasonableness.groupby(["sku_code", "store_code"])[
                "prior_probability"
            ].mean()
            if not reasonableness.empty else pd.Series(dtype=float)
        )
        performance_store_map = (
            reasonableness.groupby(["sku_code", "store_code"])[
                "performance_probability"
            ].mean()
            if not reasonableness.empty else pd.Series(dtype=float)
        )
        reliability_store_map = (
            reasonableness.groupby(["sku_code", "store_code"])[
                "reliability_r"
            ].mean()
            if not reasonableness.empty else pd.Series(dtype=float)
        )
        arrival_store_map = (
            reasonableness.groupby(["sku_code", "store_code"])["arrived_bool"].mean()
            if not reasonableness.empty else pd.Series(dtype=float)
        )
        n = len(target_store)
        velocity_sum, velocity_weight = np.zeros(n), np.zeros(n)
        probability_sum, probability_weight = np.zeros(n), np.zeros(n)
        prior_sum, prior_weight = np.zeros(n), np.zeros(n)
        performance_sum, performance_weight = np.zeros(n), np.zeros(n)
        reliability_sum, reliability_weight = np.zeros(n), np.zeros(n)
        arrival_sum, arrival_weight = np.zeros(n), np.zeros(n)
        used_store_specific = np.zeros(n, dtype=bool)
        # O(candidates * stores) array operations, without per-store Python calls.
        for candidate_row in selected_candidates.itertuples(index=False):
            code, alpha = str(candidate_row.candidate_sku_code), float(candidate_row.alpha_weight)
            store_keys = pd.MultiIndex.from_arrays([[code] * n, target_store["store_code"].astype(str)])
            tier_keys = pd.MultiIndex.from_arrays([[code] * n, target_store["store_tier"].astype(str)])
            v = store_velocity_series.reindex(store_keys).to_numpy(dtype=float)
            used_store_specific |= ~np.isnan(v)
            v = np.where(np.isnan(v), tier_velocity_series.reindex(tier_keys).to_numpy(dtype=float), v)
            valid = ~np.isnan(v)
            velocity_sum += np.where(valid, np.maximum(v, 0) * alpha, 0)
            velocity_weight += valid * alpha
            if not probability_store_map.empty:
                prob = probability_store_map.reindex(store_keys).to_numpy(dtype=float)
                valid = ~np.isnan(prob)
                probability_sum += np.where(valid, prob * alpha, 0)
                probability_weight += valid * alpha
                prior_probability = prior_store_map.reindex(store_keys).to_numpy(
                    dtype=float
                )
                prior_valid = ~np.isnan(prior_probability)
                prior_sum += np.where(
                    prior_valid, prior_probability * alpha, 0
                )
                prior_weight += prior_valid * alpha
                performance_probability = performance_store_map.reindex(
                    store_keys
                ).to_numpy(dtype=float)
                performance_valid = ~np.isnan(performance_probability)
                performance_sum += np.where(
                    performance_valid, performance_probability * alpha, 0
                )
                performance_weight += performance_valid * alpha
                reliability_value = reliability_store_map.reindex(
                    store_keys
                ).to_numpy(dtype=float)
                reliability_valid = ~np.isnan(reliability_value)
                reliability_sum += np.where(
                    reliability_valid, reliability_value * alpha, 0
                )
                reliability_weight += reliability_valid * alpha
                arrived_probability = arrival_store_map.reindex(store_keys).to_numpy(
                    dtype=float
                )
                arrived_valid = ~np.isnan(arrived_probability)
                arrival_sum += np.where(
                    arrived_valid, arrived_probability * alpha, 0
                )
                arrival_weight += arrived_valid * alpha
        valid_velocity = velocity_weight > 0
        result_velocity = np.full(n, anchor_velocity, dtype=float)
        result_velocity[valid_velocity] = velocity_sum[valid_velocity] / velocity_weight[valid_velocity]
        target_store["v_anchor_week1"] = result_velocity
        target_store["v_anchor_source"] = np.where(
            valid_velocity, np.where(used_store_specific, "comparable_store", "comparable_tier"),
            path4_anchor_source if chosen.empty else "comparable_global",
        )
        target_store["productive_probability"] = np.divide(
            probability_sum, probability_weight, out=np.full(n, np.nan), where=probability_weight > 0
        )
        target_store["placement_prior_probability"] = np.divide(
            prior_sum,
            prior_weight,
            out=np.full(n, np.nan),
            where=prior_weight > 0,
        )
        target_store["placement_performance_probability"] = np.divide(
            performance_sum,
            performance_weight,
            out=np.full(n, np.nan),
            where=performance_weight > 0,
        )
        target_store["placement_reliability_r"] = np.divide(
            reliability_sum,
            reliability_weight,
            out=np.full(n, np.nan),
            where=reliability_weight > 0,
        )
        target_store["historical_arrival_probability"] = np.divide(
            arrival_sum,
            arrival_weight,
            out=np.full(n, np.nan),
            where=arrival_weight > 0,
        )

        velocity = pd.to_numeric(
            target_store["v_anchor_week1"], errors="coerce"
        ).clip(lower=0)
        target_store["velocity_norm"] = velocity.rank(method="average", pct=True)
        components = target_store[["productive_probability", "velocity_norm", "category_fit", "execution"]]
        weights = np.array([score_weights[k] for k in ("productive", "velocity", "category_fit", "execution")], dtype=float)
        denominator = components.notna().to_numpy() @ weights
        numerator = components.fillna(0).to_numpy(dtype=float) @ weights
        target_store["store_score"] = np.divide(numerator, denominator, out=np.full(len(target_store), np.nan), where=denominator != 0)
        target_store["score_completeness"] = denominator / 0.95
        target_store["selected"] = 0
        tier_tau = policy.set_index("store_tier")["tau_tier"].to_dict()
        target_store["tau_tier"] = (
            target_store["store_tier"].map(tier_tau).astype(float)
        )
        historically_supported = target_store[
            "historical_arrival_probability"
        ].fillna(0).gt(0)
        modeled_counterfactual = (
            all_valid_evidence_modeled
            if candidate_skus
            else False
        ) & target_store["productive_probability"].ge(target_store["tau_tier"])
        if not candidate_skus or placement_fallback_no_evidence:
            target_store["selection_eligible_by_evidence"] = False
            target_store["selection_pool_eligible"] = True
        else:
            target_store["selection_eligible_by_evidence"] = (
                historically_supported | modeled_counterfactual
            )
            target_store["selection_pool_eligible"] = target_store[
                "selection_eligible_by_evidence"
            ]
        selection_pool = target_store.loc[
            target_store["selection_pool_eligible"]
        ]
        width_target_store_count = k_total
        k_total = min(k_total, len(selection_pool))

        # First allocate K by store level; within the same level sort all
        # business-state groups together by store_score.
        level_values = sorted(
            selection_pool["store_level"].astype(str).unique(), key=_tier_rank
        )
        quota_by_level: dict[str, int] = {}
        remaining = k_total
        for level in level_values:
            count = int(selection_pool["store_level"].astype(str).eq(level).sum())
            quota_by_level[level] = min(count, remaining)
            remaining -= quota_by_level[level]
            if remaining <= 0:
                break
        for level, count in quota_by_level.items():
            indices = (
                selection_pool.loc[selection_pool["store_level"].astype(str) == level]
                .sort_values(
                    ["store_score", "store_code"],
                    ascending=[False, True],
                    na_position="last",
                )
                .head(count)
                .index
            )
            target_store.loc[indices, "selected"] = 1
        target_store["placement_origin"] = np.select(
            [
                target_store["selected"].eq(1) & historically_supported,
                target_store["selected"].eq(1) & modeled_counterfactual,
                target_store["selected"].eq(1)
                & np.full(len(target_store), not candidate_skus, dtype=bool),
                target_store["selected"].eq(1)
                & np.full(
                    len(target_store), placement_fallback_no_evidence, dtype=bool
                ),
            ],
            [
                "historical_launch_coverage",
                "modeled_counterfactual_trial",
                "path4_versioned_prior",
                "missing_launch_evidence_versioned_prior",
            ],
            default="not_selected",
        )
        # Store potential only redistributes a tier's demand. It must not
        # change the V3.2 arithmetic, alpha-weighted tier week-one anchor.
        for tier, positions in target_store.loc[target_store.selected.eq(1)].groupby("store_tier").groups.items():
            tier_values = np.asarray([tier_velocity_series.get((str(r.candidate_sku_code), str(tier)), np.nan) for r in selected_candidates.itertuples(index=False)], dtype=float)
            alphas = selected_candidates.alpha_weight.to_numpy(float)
            valid = np.isfinite(tier_values) & (tier_values >= 0) & (alphas > 0)
            if valid.any():
                tier_anchor = float(np.average(tier_values[valid], weights=alphas[valid]))
                current_mean = float(target_store.loc[positions, "v_anchor_week1"].mean())
                if current_mean > 0:
                    target_store.loc[positions, "v_anchor_week1"] *= tier_anchor / current_mean
                else:
                    target_store.loc[positions, "v_anchor_week1"] = tier_anchor
                target_store.loc[positions, "tier_week1_anchor"] = tier_anchor
                target_store.loc[positions, "tier_anchor_status"] = "alpha_weighted_arithmetic_tier_speed"
            elif candidate_skus:
                fallback_anchor, fallback_source = cached_hierarchy(_hierarchical_path4_anchor,
                    restored_tier_weekly=restored_tier_weekly,
                    target_attributes=target_attribute_map, store_tiers=[str(tier)],
                    default_value=float(initial_defaults["path4_anchor_velocity"]),
                    min_week1_velocity=curve_min_week1_velocity,
                    min_week1_samples=path4_min_week1_samples,
                )
                target_store.loc[positions, "v_anchor_week1"] = fallback_anchor
                target_store.loc[positions, "tier_week1_anchor"] = fallback_anchor
                target_store.loc[positions, "tier_anchor_status"] = "missing_week1_" + fallback_source
                target_store.loc[positions, "v_anchor_source"] = "missing_week1_" + fallback_source
                warnings.append(f"{target_sku} tier={tier}: no usable comparable first-week exposure; anchor={fallback_anchor} source={fallback_source}, not a later-life average relabeled as week1")
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
                "width_target_store_count": width_target_store_count,
                "unfilled_store_count": width_target_store_count - int(target_store["selected"].sum()),
                "placement_status": "blocked_no_supported_stores" if not target_store["selected"].any() else "planned",
                "placement_block_reason": ("no_evidence_supported_store" if selection_pool.empty else "rounded_width_zero") if not target_store["selected"].any() else "",
                "historical_coverage_store_count": int(
                    (
                        target_store["selected"].eq(1)
                        & target_store["placement_origin"].eq(
                            "historical_launch_coverage"
                        )
                    ).sum()
                ),
                "counterfactual_expansion_store_count": int(
                    (
                        target_store["selected"].eq(1)
                        & target_store["placement_origin"].eq(
                            "modeled_counterfactual_trial"
                        )
                    ).sum()
                ),
                "evidence_supported_store_count": int(
                    target_store["selection_eligible_by_evidence"].sum()
                ),
                "placement_selection_pool_store_count": int(
                    target_store["selection_pool_eligible"].sum()
                ),
                "placement_evidence_source": evidence_source,
                "missing_launch_evidence_candidate_count": len(
                    missing_launch_evidence_candidates
                ),
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
                    and all_valid_evidence_modeled
                    and not missing_launch_evidence_candidates
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
        if selected_stores.empty:
            warnings.append(f"{target_sku}: no stores selected; width_target={width_target_store_count}, evidence_supported_pool={len(selection_pool)}; demand unavailable, not zero sales")
        target_curve_fallbacks = 0
        shared_factors, shared_components, shared_statuses, shared_reason = mix_comparable_curves(
            selected_candidates, lifecycle_all, _normalize_curve_profile,
            lifecycle_weeks=lifecycle_weeks,
            min_week1_velocity=curve_min_week1_velocity,
            max_factor=curve_max_factor,
            min_observed_weeks=curve_min_observed_weeks,
        )
        spread = lifecycle_all.loc[lifecycle_all.sku_code.isin(selected_candidates.candidate_sku_code.astype(str))].groupby("lifecycle_week").v_all.agg(["mean", "std", "count"])
        shared_cvs = (spread["std"] / spread["mean"].where(spread["mean"].gt(0) & spread["count"].ge(2))).to_dict()
        for tier, tier_stores in selected_stores.groupby("store_tier"):
            comparable_factors = shared_factors
            comparable_cvs = shared_cvs
            comparable_point_status = shared_statuses
            comparable_reason = shared_reason

            if comparable_factors:
                profile_factors = comparable_factors
                profile_cvs = comparable_cvs
                profile_point_status = comparable_point_status
                profile_source = "selected_comparables"
                profile_status = "IMPLEMENTED_V3_2"
                profile_reason = comparable_reason
            else:
                (
                    profile_factors,
                    profile_cvs,
                    profile_point_status,
                    hierarchy_source,
                    hierarchy_reason,
                ) = cached_hierarchy(_hierarchical_curve,
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
                        "curve_formula": ("sum_alpha_of_individually_week1_normalized_all_store_velocity" if comparable_factors else "category_prior_or_disclosed_flat_default"),
                        "all_store_aggregation_basis": all_store_basis,
                        **{
                            "lifecycle_component_" + name: (
                                shared_components[week][name]
                                if comparable_factors else
                                factor if (
                                    name == "flat_default" and profile_source == "versioned_flat_default"
                                    or name == profile_point_status[week] and profile_source != "versioned_flat_default"
                                ) else 0.0
                            ) for name in ("observed", "interpolated", "edge_carried", "flat_default")
                        },
                    }
                )
        compliance_rows.append(
            _compliance_row(
                target_sku=target_sku,
                concept="complete_lifecycle_curve",
                status=(
                    "BLOCKED_NO_SUPPORTED_STORES"
                    if selected_stores.empty else
                    "IMPLEMENTED_V3_2"
                    if target_curve_fallbacks == 0
                    else "V3_2_CATEGORY_PRIOR_OR_VERSIONED_DEFAULT"
                ),
                source="no_selected_stores_no_curve" if selected_stores.empty else "selected comparables/category lifecycle prior",
                detail=(
                    f"weeks=1..{lifecycle_weeks}; fallback_profile_rows="
                    f"{target_curve_fallbacks}. Each tier uses one validated, "
                    "week-1-normalized source; no cross-source weekly splicing."
                ),
            )
        )

        if save_target:
            save_target(target_sku, PlacementResult(
                *(pd.DataFrame(rows[offset:], columns=CURVE_COLUMNS if n == 2 else None)
                  for n, (rows, offset) in enumerate(zip(output_lists[:5], offsets[:5]))),
                tuple(warnings[offsets[5]:]),
            ))
        emit("target_complete", target_index=target_index, target_count=len(target_codes), sku_code=target_sku)

    return PlacementResult(
        pd.DataFrame(store_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(curve_rows, columns=CURVE_COLUMNS),
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(compliance_rows),
        tuple(warnings),
    )
