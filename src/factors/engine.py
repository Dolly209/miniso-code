from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
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


COUNT_FEATURES: set[str] = set()
RATIO_FEATURES = {
    "member_female_rate", "member_young_rate", "supplier_fulfillment",
    "multi_supplier", "exposure_quality", "lifecycle_persistence",
}


def _theil_sen_slope(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) < 2:
        return np.nan
    slopes = [
        (numeric[right] - numeric[left]) / (right - left)
        for left in range(len(numeric) - 1)
        for right in range(left + 1, len(numeric))
    ]
    return float(np.median(slopes)) if slopes else np.nan


def build_social_small_factors(
    *,
    features: pd.DataFrame,
    media_history: pd.DataFrame,
    sku_social_aliases: Mapping[str, set[str]],
    prelaunch_days: int,
    momentum_days: int,
    minimum_days: int,
    decision_cutoff_date: date | None = None,
    platform_transform: Mapping | None = None,
) -> pd.DataFrame:
    """Build V3.2 IP-01/02/03 raw small-factor inputs.

    ``media_history`` is already a canonical data-domain frame.  IP names are
    matched outside this algorithm through the reviewed alias CSV.  Multiple
    IP memberships are averaged after platform standardisation rather than
    added, because the aggregate source cannot deduplicate posts/authors across
    IP buckets (V3.2 section 3.4).
    """

    require_columns(features, ("sku_code", "release_dt"), "factor features")
    require_columns(
        media_history,
        (
            "stat_date",
            "external_ip_key",
            "platform_code",
            "post_count",
            "unique_author_count",
            "ordinary_author_count",
            "koc_author_count",
            "kol_author_count",
            "avg_author_follower_count",
            "max_author_follower_count",
        ),
        "canonical social history",
    )
    result = features.copy()
    for column in (
        "ip_social_volume",
        "ip_social_momentum",
        "ip_fan_asset",
        "social_feature_days",
        "social_feature_platforms",
    ):
        result[column] = np.nan
    result["social_factor_status"] = "missing_no_reviewed_social_match"
    history = media_history.copy()
    history["stat_date"] = pd.to_datetime(history["stat_date"], errors="coerce").dt.date
    history = history.loc[
        history["stat_date"].notna() & history["external_ip_key"].astype(str).ne("")
    ].copy()
    numeric_columns = (
        "post_count",
        "unique_author_count",
        "ordinary_author_count",
        "koc_author_count",
        "kol_author_count",
        "avg_author_follower_count",
        "max_author_follower_count",
    )
    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history["sm01_log_post"] = np.log1p(history["post_count"].clip(lower=0))
    history["sm01_platform_z"] = history.groupby(
        "platform_code", sort=False
    )["sm01_log_post"].transform(
        lambda values: _robust_z(values, winsor=(0.01, 0.99))
    )
    if platform_transform is not None:
        from factors.frozen import frozen_social_z
        history["sm01_platform_z"] = np.nan
        for platform, indices in history.groupby("platform_code", sort=False).groups.items():
            history.loc[indices, "sm01_platform_z"] = frozen_social_z(
                history.loc[indices, "post_count"], platform_transform.get(str(platform), {}).get("post_count"))
    fan_components = []
    for column in (
        "unique_author_count",
        "ordinary_author_count",
        "koc_author_count",
        "kol_author_count",
        "avg_author_follower_count",
        "max_author_follower_count",
    ):
        component = f"__fan_{column}"
        history[component] = history.groupby("platform_code", sort=False)[column].transform(
            lambda values: _robust_z(
                np.log1p(pd.to_numeric(values, errors="coerce").clip(lower=0)),
                winsor=(0.01, 0.99),
            )
        )
        if platform_transform is not None:
            history[component] = np.nan
            for platform, indices in history.groupby("platform_code", sort=False).groups.items():
                history.loc[indices, component] = frozen_social_z(
                    history.loc[indices, column], platform_transform.get(str(platform), {}).get(column))
        fan_components.append(component)
    history["sm_fan_asset"] = history[fan_components].mean(axis=1, skipna=True)
    by_alias = {
        str(key): group.copy()
        for key, group in history.groupby("external_ip_key", sort=False)
    }
    for index, row in result.iterrows():
        sku = str(row["sku_code"])
        aliases = sorted(sku_social_aliases.get(sku, set()))
        release = pd.to_datetime(row.get("release_dt"), errors="coerce")
        matched = [by_alias[key] for key in aliases if key in by_alias]
        if pd.isna(release):
            result.at[index, "social_factor_status"] = "missing_release_dt"
            continue
        if not matched:
            continue
        launch_window_end = release.date() - timedelta(days=1)
        end_date = (
            min(launch_window_end, decision_cutoff_date)
            if decision_cutoff_date is not None
            else launch_window_end
        )
        start_date = end_date - timedelta(days=max(prelaunch_days - 1, 0))
        window = pd.concat(matched, ignore_index=True)
        window = window.loc[window["stat_date"].between(start_date, end_date)].copy()
        if window.empty:
            result.at[index, "social_factor_status"] = "missing_prelaunch_window"
            continue
        # Equal-weight IP fusion at IP x platform x day prevents a multi-IP SKU
        # from receiving an automatic volume bonus merely for having more rows.
        daily = window.groupby(["stat_date", "platform_code"], as_index=False).agg(
            sm01_platform_z=("sm01_platform_z", "mean"),
            post_count=("post_count", "mean"),
            sm_fan_asset=("sm_fan_asset", "mean"),
        )
        observed_days = int(daily["stat_date"].nunique())
        result.at[index, "social_feature_days"] = observed_days
        result.at[index, "social_feature_platforms"] = int(daily["platform_code"].nunique())
        if observed_days < minimum_days:
            result.at[index, "social_factor_status"] = "insufficient_prelaunch_days"
            continue
        result.at[index, "ip_social_volume"] = float(daily["sm01_platform_z"].mean())
        result.at[index, "ip_fan_asset"] = float(daily["sm_fan_asset"].mean())
        momentum_start = end_date - timedelta(days=max(momentum_days - 1, 0))
        momentum = daily.loc[daily["stat_date"].between(momentum_start, end_date)]
        platform_slopes = momentum.sort_values("stat_date").groupby(
            "platform_code", sort=False
        )["post_count"].apply(_theil_sen_slope)
        platform_slopes = platform_slopes.dropna()
        if not platform_slopes.empty:
            recent_start = end_date - timedelta(days=6)
            prior_end = recent_start - timedelta(days=1)
            prior_start = prior_end - timedelta(days=6)
            recent_slopes = daily.loc[daily["stat_date"].between(recent_start, end_date)].sort_values(
                "stat_date"
            ).groupby("platform_code", sort=False)["post_count"].apply(_theil_sen_slope)
            prior_slopes = daily.loc[daily["stat_date"].between(prior_start, prior_end)].sort_values(
                "stat_date"
            ).groupby("platform_code", sort=False)["post_count"].apply(_theil_sen_slope)
            common = recent_slopes.dropna().index.intersection(prior_slopes.dropna().index)
            acceleration = (
                float((recent_slopes.loc[common] - prior_slopes.loc[common]).mean())
                if len(common)
                else 0.0
            )
            result.at[index, "ip_social_momentum"] = float(platform_slopes.mean()) + acceleration
        result.at[index, "social_factor_status"] = (
            "v3_2_ip01_ip02_ip03_point_in_time_available"
            if decision_cutoff_date is not None
            and decision_cutoff_date < launch_window_end
            else "v3_2_ip01_ip02_ip03_available"
        )
    return result


def build_internal_small_factors(
    *,
    features: pd.DataFrame,
    restored_weekly: pd.DataFrame,
    standard_ip_membership: pd.DataFrame,
    cutoff_date: object,
    competition_window_days: int,
) -> pd.DataFrame:
    """Build currently supportable PR/PF/IP-04/CO-01 inputs from canonical data."""

    require_columns(
        features,
        (
            "sku_code",
            "cate_level4_code",
            "manage_cate_name",
            "sales_unit_type",
            "release_dt",
        ),
        "factor features",
    )
    require_columns(
        standard_ip_membership,
        ("sku_code", "ip_code"),
        "standard IP membership",
    )
    result = features.copy()
    result["sku_code"] = result["sku_code"].astype(str)
    membership = standard_ip_membership[["sku_code", "ip_code"]].copy()
    membership["sku_code"] = membership["sku_code"].astype(str)
    membership["ip_code"] = membership["ip_code"].fillna("").astype(str).str.strip()
    membership = membership.loc[membership["ip_code"].ne("")].drop_duplicates()

    for column in (
        "category_prosperity_l4",
        "category_prosperity_manage",
        "category_prosperity",
        "form_play_fit",
        "design_innovation",
        "marketing_resource",
        "same_ip_launch_count",
    ):
        if column not in result:
            result[column] = np.nan
    result["category_prosperity_status"] = "missing_insufficient_12_week_history"
    result["form_play_fit_status"] = "missing_no_standard_ip_form_history"
    result["design_innovation_status"] = "missing_no_form_category_combination"
    result["marketing_resource_status"] = "missing_op04_neutral"

    restored = restored_weekly.copy()
    required_restore = {
        "sku_code",
        "week_end",
        "restored_natural_sales",
        "restored_exposure_days",
    }
    if required_restore.issubset(restored.columns) and not restored.empty:
        restored["sku_code"] = restored["sku_code"].astype(str)
        restored["week_end"] = pd.to_datetime(restored["week_end"], errors="coerce")
        restored = restored.loc[
            restored["week_end"].notna()
            & restored["week_end"].dt.date.le(pd.to_datetime(cutoff_date).date())
        ].copy()
        restored["calendar_week"] = restored["week_end"].dt.to_period("W-SUN").dt.end_time.dt.date
        for column in ("restored_natural_sales", "restored_exposure_days"):
            restored[column] = pd.to_numeric(restored[column], errors="coerce")
        attributes = result[
            ["sku_code", "cate_level4_code", "manage_cate_name"]
        ].drop_duplicates("sku_code")
        # ROUND 4 already carries cate_level4_code in current artifacts, while
        # older artifacts may omit one or both category attributes. Merging
        # the full attribute frame unconditionally would turn an existing
        # column into ``cate_level4_code_x/_y`` and make the canonical name
        # disappear. Enrich both documented product-master attributes in one
        # keyed merge, preserving ROUND 4 values and filling only missing ones
        # from the product snapshot used by ROUND 7.
        category_attributes = ("cate_level4_code", "manage_cate_name")
        existing_attributes = {
            attribute for attribute in category_attributes
            if attribute in restored.columns
        }
        lookup = attributes.rename(
            columns={
                attribute: f"__product_{attribute}"
                for attribute in existing_attributes
            }
        )
        restored = restored.merge(
            lookup,
            on="sku_code",
            how="left",
            validate="many_to_one",
        )
        for attribute in existing_attributes:
            lookup_column = f"__product_{attribute}"
            current = restored[attribute].replace(r"^\s*$", np.nan, regex=True)
            restored[attribute] = current.combine_first(restored[lookup_column])
            restored = restored.drop(columns=lookup_column)

        def prosperity_by(column: str) -> dict[str, float]:
            grouped = restored.groupby([column, "calendar_week"], dropna=False, as_index=False).agg(
                demand=("restored_natural_sales", "sum"),
                exposure_days=("restored_exposure_days", "sum"),
            )
            grouped["velocity"] = np.where(
                grouped["exposure_days"] > 0,
                7.0 * grouped["demand"] / grouped["exposure_days"],
                np.nan,
            )
            values: dict[str, float] = {}
            for key, group in grouped.groupby(column, dropna=False, sort=False):
                ordered = group.sort_values("calendar_week").loc[group["velocity"].notna()].tail(12)
                if len(ordered) < 12:
                    continue
                recent = float(ordered.tail(4)["velocity"].mean())
                baseline = float(ordered.head(8)["velocity"].mean())
                values[str(key)] = float(np.log((recent + 1e-6) / (baseline + 1e-6)))
            return values

        l4_values = prosperity_by("cate_level4_code")
        manage_values = prosperity_by("manage_cate_name")
        result["category_prosperity_l4"] = result["cate_level4_code"].astype(str).map(l4_values)
        result["category_prosperity_manage"] = result["manage_cate_name"].astype(str).map(manage_values)
        available_l4 = result["category_prosperity_l4"].notna()
        available_manage = result["category_prosperity_manage"].notna()
        denominator = 0.70 * available_l4.astype(float) + 0.30 * available_manage.astype(float)
        numerator = (
            0.70 * result["category_prosperity_l4"].fillna(0.0)
            + 0.30 * result["category_prosperity_manage"].fillna(0.0)
        )
        result["category_prosperity"] = numerator.where(denominator.gt(0)) / denominator.where(
            denominator.gt(0)
        )
        result["category_prosperity_status"] = np.select(
            [available_l4 & available_manage, available_l4, available_manage],
            ["v3_2_pr01_pr02", "v3_2_pr01_only", "v3_2_pr02_only"],
            default="missing_insufficient_12_week_history",
        )

    # PF-01 objective component: PR-07, same-IP x delivered product form
    # velocity relative to that IP's overall historical velocity.
    if "historical_velocity" in result:
        velocity = result[["sku_code", "sales_unit_type", "historical_velocity"]].copy()
        velocity["historical_velocity"] = pd.to_numeric(velocity["historical_velocity"], errors="coerce")
        expanded = membership.merge(velocity, on="sku_code", how="left", validate="many_to_one")
        valid = expanded.loc[expanded["historical_velocity"].notna() & expanded["historical_velocity"].ge(0)].copy()
        if not valid.empty:
            ip_median = valid.groupby("ip_code")["historical_velocity"].median().rename("ip_velocity")
            form_median = valid.groupby(["ip_code", "sales_unit_type"])["historical_velocity"].median().rename("form_velocity")
            scored = membership.merge(
                result[["sku_code", "sales_unit_type"]], on="sku_code", how="left", validate="many_to_one"
            ).merge(ip_median, on="ip_code", how="left").merge(
                form_median, on=["ip_code", "sales_unit_type"], how="left"
            )
            scored["form_fit"] = np.log(
                (scored["form_velocity"] + 1e-6) / (scored["ip_velocity"] + 1e-6)
            )
            form_fit = scored.groupby("sku_code")["form_fit"].mean()
            result["form_play_fit"] = result["sku_code"].map(form_fit)
            result["form_play_fit_status"] = np.where(
                result["form_play_fit"].notna(),
                "v3_2_pr07_ip_x_form_history",
                "missing_no_standard_ip_form_history",
            )

    # PF-03 partial PR-13 objective proxy: rarity of the reviewed IP x form x
    # management-category combination. No claim is made about unavailable
    # design/material/play fields.
    expanded_product = membership.merge(
        result[["sku_code", "sales_unit_type", "manage_cate_name"]],
        on="sku_code",
        how="left",
        validate="many_to_one",
    )
    if not expanded_product.empty:
        combo = ["ip_code", "sales_unit_type", "manage_cate_name"]
        counts = expanded_product.groupby(combo, dropna=False)["sku_code"].nunique().rename("combo_count")
        ip_counts = expanded_product.groupby("ip_code")["sku_code"].nunique().rename("ip_count")
        novelty = expanded_product.merge(counts, on=combo, how="left").merge(ip_counts, on="ip_code", how="left")
        novelty["novelty"] = -np.log(novelty["combo_count"] / novelty["ip_count"].clip(lower=1))
        novelty_by_sku = novelty.groupby("sku_code")["novelty"].mean()
        result["design_innovation"] = result["sku_code"].map(novelty_by_sku)
        result["design_innovation_status"] = np.where(
            result["design_innovation"].notna(),
            "partial_v3_2_pr13_ip_form_category_rarity",
            "missing_no_form_category_combination",
        )

    # CO-01: unique same-IP effective new SKUs in the symmetric launch window.
    releases = result[["sku_code", "release_dt"]].copy()
    releases["release_dt"] = pd.to_datetime(releases["release_dt"], errors="coerce").dt.date
    expanded_release = membership.merge(releases, on="sku_code", how="left", validate="many_to_one")
    launch_counts: dict[str, int] = {}
    by_ip = {key: group for key, group in expanded_release.groupby("ip_code", sort=False)}
    for row in expanded_release.itertuples(index=False):
        if pd.isna(row.release_dt):
            continue
        group = by_ip[row.ip_code]
        lower = row.release_dt - timedelta(days=competition_window_days)
        upper = row.release_dt + timedelta(days=competition_window_days)
        peers = group.loc[
            group["release_dt"].notna()
            & group["release_dt"].between(lower, upper)
            & group["sku_code"].ne(row.sku_code),
            "sku_code",
        ].nunique()
        launch_counts[row.sku_code] = max(launch_counts.get(row.sku_code, 0), int(peers))
    result["same_ip_launch_count"] = result["sku_code"].map(launch_counts)
    return result


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
    category = (
        result["cate_level4_code"].fillna("__UNKNOWN__").astype(str)
        if "cate_level4_code" in result
        else pd.Series("__UNKNOWN__", index=result.index)
    )
    for name in all_features:
        z_column = f"z__{name}"
        result[f"score__{name}"] = result.groupby(category, sort=False)[z_column].rank(
            method="average", pct=True
        ) * 100.0
        # A category represented by one row has no usable empirical CDF.  Fall
        # back to the loaded reference pool, while retaining the scope for audit.
        category_count = result.groupby(category, sort=False)[z_column].transform("count")
        global_rank = result[z_column].rank(method="average", pct=True) * 100.0
        fallback = category_count.lt(2) & result[z_column].notna()
        result.loc[fallback, f"score__{name}"] = global_rank.loc[fallback]
        result[f"cdf_scope__{name}"] = np.where(
            result[z_column].isna(),
            "missing",
            np.where(fallback, "loaded_reference_pool_fallback", "cate_level4"),
        )
    for group, weights in factor_weights.items():
        group_values: list[float] = []
        group_completeness: list[float] = []
        for row in result.itertuples(index=False):
            available = {
                name: getattr(row, f"score__{name}")
                for name in weights
                if pd.notna(getattr(row, f"score__{name}"))
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
        result[f"score__{group}"] = result[f"u__{group}"]
    return result


def calculate_demand_adjustments(
    *,
    targets: pd.DataFrame,
    features: pd.DataFrame,
    comparables: pd.DataFrame,
    comparable_summary: pd.DataFrame,
    config: Mapping[str, Any],
    competition_rules: pd.DataFrame | None = None,
    frozen_transform: Mapping[str, Any] | None = None,
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
    if frozen_transform is not None:
        from factors.frozen import apply_factor_transform
        transformed = apply_factor_transform(features, frozen_transform, weights)
    else:
        transformed = transform_factor_features(
            features, factor_weights=weights,
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
        relative_deltas: dict[str, float] = {}
        small_factor_deltas: dict[str, float] = {}
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
            relative_deltas[group] = float(delta)
            # Preserve the exact target-vs-comparable percentile differences
            # needed by Section 8.3 C to estimate theta_(G,f)=beta_G*omega_(G,f).
            # Missing factors remain neutral and are not represented by a
            # fabricated score.
            for factor_name in weights[group]:
                target_factor_score = target.get(f"score__{factor_name}")
                factor_values: list[float] = []
                factor_alpha: list[float] = []
                for row in valid_comparables.itertuples(index=False):
                    value = by_sku.loc[str(row.candidate_sku_code)].get(
                        f"score__{factor_name}"
                    )
                    if pd.notna(value):
                        factor_values.append(float(value))
                        factor_alpha.append(float(row.alpha_weight))
                comparable_factor_score = (
                    float(np.average(factor_values, weights=factor_alpha))
                    if factor_values and sum(factor_alpha) > 0
                    else np.nan
                )
                small_factor_deltas[factor_name] = (
                    (float(target_factor_score) - comparable_factor_score) / 100.0
                    if pd.notna(target_factor_score)
                    and pd.notna(comparable_factor_score)
                    else 0.0
                )
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
        c_ext_observed = float(external.get(target_sku, 1.0))
        social_used_in_ip = any(
            pd.notna(target.get(column))
            for column in (
                "ip_social_volume",
                "ip_social_momentum",
                "ip_fan_asset",
            )
        )
        if bool(config.get("same_social_source_disables_external_coefficient", True)) and social_used_in_ip:
            c_ext = 1.0
            external_status = "neutral_same_social_facts_already_in_C_IP"
        else:
            c_ext = c_ext_observed
            external_status = "applied_independent_or_no_C_IP_social"
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
                # These are the exact pre-coefficient relative inputs used by
                # the production M5 formula.  Section 8.3/8.4 training must fit
                # beta on the deltas rather than regress a second time on the
                # already-bounded C_* outputs.
                "delta_IP": relative_deltas.get("IP", 0.0),
                "delta_PRODUCT": relative_deltas.get("PRODUCT", 0.0),
                "delta_OPERATION": relative_deltas.get("OPERATION", 0.0),
                **{
                    f"delta_factor__{name}": value
                    for name, value in small_factor_deltas.items()
                },
                "C_IP": coefficients.get("IP", 1.0),
                "C_ext": c_ext,
                "C_ext_observed": c_ext_observed,
                "C_ext_status": external_status,
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
                "competition_trigger_count": len(triggered_rules),
                "input_completeness": input_completeness,
            }
        )
    return FactorResult(
        pd.DataFrame(detail_rows),
        pd.DataFrame(adjustment_rows),
        transformed,
        tuple(dict.fromkeys(warnings)),
    )
