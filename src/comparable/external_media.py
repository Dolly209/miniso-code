from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.domain import require_columns
from data.social_ip_aliases import (
    build_sku_social_aliases,
    normalize_social_ip_name,
)


@dataclass(frozen=True)
class ExternalMediaResult:
    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    warnings: tuple[str, ...]


COUNT_FEATURES = {
    "post_count",
    "interaction_count",
    "unique_author_count",
    "koc_author_count",
    "kol_author_count",
    "view_count",
}
INVERTED_FEATURES = {"negative_post_share"}


def _match_sources(
    sku_code: str,
    matches: list[str],
    provenance: Mapping[str, Mapping[str, set[str]]],
) -> str:
    sources = {
        source
        for key in matches
        for source in provenance.get(sku_code, {}).get(key, set())
    }
    return "|".join(sorted(sources))


def _daily_window_aggregate(window: pd.DataFrame, minimum_days: int) -> pd.DataFrame:
    """Create one pre-launch-window row per external IP/platform.

    The delivered table is daily-grain (``date == ds``), not a cumulative
    snapshot: counts may legitimately fall from one day to the next.  Volume
    and author-count features are therefore summed across the window.  Rates
    are weighted across daily rows, while trend and stability retain their
    daily time-series meaning.
    """

    ordered = window.sort_values(
        ["external_ip_key", "platform_code", "stat_date"], kind="stable"
    )
    keys = ["external_ip_key", "platform_code"]
    history = ordered.groupby(keys, sort=False, dropna=False)
    first = history.first()
    last = history.last()
    days = (
        pd.to_datetime(last["stat_date"]) - pd.to_datetime(first["stat_date"])
    ).dt.days
    snapshots = history["stat_date"].nunique()
    enough = snapshots.ge(minimum_days) & days.gt(0)
    trend = pd.DataFrame(index=first.index)
    for column, output in (
        ("post_count", "post_growth_30d"),
        ("interaction_count", "interaction_growth_30d"),
    ):
        first_values = pd.to_numeric(first[column], errors="coerce").clip(lower=0)
        last_values = pd.to_numeric(last[column], errors="coerce").clip(lower=0)
        trend[output] = np.where(
            enough,
            (np.log1p(last_values) - np.log1p(first_values)) * 30.0 / days,
            np.nan,
        )
    stability_source = ordered.assign(
        __log_interaction=np.log1p(
            pd.to_numeric(ordered["interaction_count"], errors="coerce").clip(lower=0)
        )
    )
    stability = stability_source.groupby(keys)["__log_interaction"].agg(
        ["count", "mean", "std"]
    )
    stability["interaction_stability"] = np.where(
        stability["count"].ge(minimum_days),
        1.0 / (1.0 + stability["std"].fillna(0.0)),
        np.nan,
    )
    aggregate = ordered[keys].drop_duplicates().set_index(keys)
    additive = [column for column in COUNT_FEATURES if column in ordered]
    if additive:
        aggregate = aggregate.join(history[additive].sum(min_count=1))

    weighted_rates = {
        "text_coverage_rate": "tracked_post_count",
        "view_coverage_rate": "tracked_post_count",
        "positive_post_share": "post_count",
        "negative_post_share": "post_count",
    }
    for value_column, preferred_weight in weighted_rates.items():
        if value_column not in ordered:
            continue
        weight_column = preferred_weight if preferred_weight in ordered else "post_count"
        values = pd.to_numeric(ordered[value_column], errors="coerce")
        weights = pd.to_numeric(ordered[weight_column], errors="coerce").clip(lower=0)
        valid = values.notna() & weights.notna() & weights.gt(0)
        weighted_source = ordered[keys].assign(
            __numerator=np.where(valid, values * weights, 0.0),
            __denominator=np.where(valid, weights, 0.0),
        )
        weighted = weighted_source.groupby(keys, sort=False, dropna=False)[
            ["__numerator", "__denominator"]
        ].sum()
        weighted_value = weighted["__numerator"].div(
            weighted["__denominator"].replace(0, np.nan)
        )
        fallback = history[value_column].mean()
        weighted_value = weighted_value.fillna(fallback)
        weighted_value.name = value_column
        aggregate = aggregate.join(weighted_value)

    derived = trend.join(stability[["interaction_stability"]])
    return aggregate.join(derived).reset_index()


def _platform_rank_score(
    window: pd.DataFrame,
    *,
    feature_weights: Mapping[str, float],
    sentiment_coverage_min: float,
    minimum_days: int,
) -> pd.DataFrame:
    latest = _daily_window_aggregate(window, minimum_days)
    coverage = latest[["text_coverage_rate", "view_coverage_rate"]].max(
        axis=1, skipna=True
    )
    insufficient_sentiment = coverage.isna() | coverage.lt(sentiment_coverage_min)
    for column in ("positive_post_share", "negative_post_share"):
        latest.loc[insufficient_sentiment, column] = np.nan

    score_columns: list[str] = []
    weight_columns: list[str] = []
    for feature, raw_weight in feature_weights.items():
        if feature not in latest:
            continue
        values = pd.to_numeric(latest[feature], errors="coerce")
        if feature in COUNT_FEATURES:
            values = np.log1p(values.clip(lower=0))
        grouped = values.groupby(latest["platform_code"])
        ranks = grouped.rank(method="average")
        counts = grouped.transform("count")
        # Exact centred percentile: the minimum is -1 and maximum is +1.
        # A platform with only one observed IP carries no directional signal.
        score = pd.Series(
            np.where(counts.gt(1), 2.0 * (ranks - 1.0) / (counts - 1.0) - 1.0, 0.0),
            index=latest.index,
            dtype=float,
        ).where(values.notna())
        if feature in INVERTED_FEATURES:
            score = -score
        score_column = f"__score_{feature}"
        weight_column = f"__weight_{feature}"
        latest[score_column] = score
        latest[weight_column] = np.where(score.notna(), float(raw_weight), 0.0)
        score_columns.append(score_column)
        weight_columns.append(weight_column)
    if not score_columns:
        latest["platform_heat_score"] = np.nan
        latest["platform_feature_completeness"] = 0.0
        return latest
    weighted = sum(
        latest[score] * latest[weight]
        for score, weight in zip(score_columns, weight_columns)
    )
    denominator = sum(latest[column] for column in weight_columns)
    total_weight = max(sum(float(value) for value in feature_weights.values()), 1e-12)
    latest["platform_heat_score"] = weighted / denominator.replace(0, np.nan)
    latest["platform_feature_completeness"] = denominator / total_weight
    return latest


def apply_external_media_correction(
    *,
    comparable_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    standard_ip_mapping: pd.DataFrame,
    social_ip_alias_mapping: pd.DataFrame | None = None,
    media_history: pd.DataFrame,
    config: Mapping[str, Any],
    decision_cutoff_date: date | None = None,
) -> ExternalMediaResult:
    """Apply V3.2's bounded, directional external-media correction.

    External media never produces a sales, placement or inventory proxy.  It
    only compares the target IP with the already-selected internal benchmark
    IPs in the target's 30-day pre-launch window and updates ``C_ext``.
    """

    require_columns(comparable_summary, ("sku_code",), "comparable summary")
    require_columns(
        candidates,
        ("target_sku_code", "candidate_sku_code", "selected", "alpha_weight"),
        "comparable candidates",
    )
    require_columns(targets, ("sku_code", "first_actual_sale_date"), "targets")
    require_columns(
        media_history,
        (
            "stat_date",
            "ip_name",
            "platform_code",
            "post_count",
            "interaction_count",
            "unique_author_count",
            "koc_author_count",
            "kol_author_count",
            "view_count",
            "positive_post_share",
            "negative_post_share",
            "text_coverage_rate",
            "view_coverage_rate",
        ),
        "external media history",
    )
    settings = config["external_media"]
    bounds = tuple(float(value) for value in config["external_coefficient_bounds"])
    score_bounds = tuple(float(value) for value in settings["score_difference_bounds"])
    aliases, alias_provenance = build_sku_social_aliases(
        standard_ip_mapping, social_ip_alias_mapping
    )
    history = media_history.copy()
    history["stat_date"] = pd.to_datetime(history["stat_date"], errors="coerce").dt.date
    history["external_ip_key"] = history["ip_name"].map(
        normalize_social_ip_name
    )
    history = history.loc[history["external_ip_key"].ne("")].copy()
    target_dates = targets[["sku_code", "first_actual_sale_date"]].copy()
    target_dates["sku_code"] = target_dates["sku_code"].astype(str)
    target_dates["first_actual_sale_date"] = pd.to_datetime(
        target_dates["first_actual_sale_date"], errors="coerce"
    ).dt.date
    first_sale_by_sku = target_dates.drop_duplicates("sku_code").set_index("sku_code")[
        "first_actual_sale_date"
    ].to_dict()
    selected = candidates.loc[
        pd.to_numeric(candidates["selected"], errors="coerce").fillna(0).eq(1)
        & ~candidates["candidate_sku_code"].astype(str).str.startswith("__STATIC__")
    ].copy()
    selected["target_sku_code"] = selected["target_sku_code"].astype(str)
    selected["candidate_sku_code"] = selected["candidate_sku_code"].astype(str)
    selected["alpha_weight"] = pd.to_numeric(selected["alpha_weight"], errors="coerce")

    summary = comparable_summary.copy()
    summary["sku_code"] = summary["sku_code"].astype(str)
    diagnostic_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    coefficients: dict[str, float] = {}
    statuses: dict[str, str] = {}
    prelaunch_days = int(settings["prelaunch_days"])
    for target_sku in summary["sku_code"]:
        launch_day = first_sale_by_sku.get(target_sku)
        target_aliases = aliases.get(target_sku, set())
        chosen = selected.loc[selected["target_sku_code"].eq(target_sku)]
        if launch_day is None or pd.isna(launch_day):
            status = "neutral_missing_target_launch_date"
        elif not target_aliases:
            status = "neutral_missing_target_standard_ip"
        elif chosen.empty:
            status = "neutral_missing_external_benchmark_comparable"
        else:
            launch_window_end = launch_day - timedelta(days=1)
            end_day = (
                min(launch_window_end, decision_cutoff_date)
                if decision_cutoff_date is not None
                else launch_window_end
            )
            start_day = end_day - timedelta(days=max(prelaunch_days - 1, 0))
            window = history.loc[
                history["stat_date"].between(start_day, end_day)
            ].copy()
            if window.empty:
                status = "neutral_no_prelaunch_social_snapshots"
            else:
                scored = _platform_rank_score(
                    window,
                    feature_weights=settings["feature_weights"],
                    sentiment_coverage_min=float(settings["sentiment_coverage_min"]),
                    minimum_days=int(settings["minimum_snapshot_days_for_trend"]),
                )
                ip_scores = scored.groupby("external_ip_key", as_index=True).agg(
                    ip_heat_score=("platform_heat_score", "mean"),
                    ip_feature_completeness=("platform_feature_completeness", "mean"),
                    platform_count=("platform_code", "nunique"),
                )
                auditable_features = [
                    feature
                    for feature in settings["feature_weights"]
                    if feature in scored.columns
                ]
                ip_feature_values = (
                    scored.groupby("external_ip_key", as_index=True)[auditable_features].mean()
                    if auditable_features
                    else pd.DataFrame(index=ip_scores.index)
                )
                target_matches = sorted(target_aliases & set(ip_scores.index))
                target_score = (
                    float(ip_scores.loc[target_matches, "ip_heat_score"].mean())
                    if target_matches
                    else np.nan
                )
                benchmark_parts: list[tuple[float, float, str, list[str]]] = []
                for row in chosen.itertuples(index=False):
                    candidate_sku = str(row.candidate_sku_code)
                    matches = sorted(aliases.get(candidate_sku, set()) & set(ip_scores.index))
                    if not matches:
                        continue
                    score = float(ip_scores.loc[matches, "ip_heat_score"].mean())
                    if np.isfinite(score) and float(row.alpha_weight) > 0:
                        benchmark_parts.append(
                            (score, float(row.alpha_weight), candidate_sku, matches)
                        )
                if not np.isfinite(target_score):
                    status = "neutral_target_ip_not_matched_to_social_ip_name"
                elif not benchmark_parts:
                    status = "neutral_comparable_ips_not_matched_to_social_ip_name"
                else:
                    benchmark_weight = sum(part[1] for part in benchmark_parts)
                    benchmark_score = sum(part[0] * part[1] for part in benchmark_parts) / benchmark_weight
                    score_difference = float(target_score - benchmark_score)
                    clipped_difference = float(np.clip(score_difference, *score_bounds))
                    coefficient = float(
                        np.clip(
                            1.0 + float(settings["coefficient_slope"]) * clipped_difference,
                            *bounds,
                        )
                    )
                    status = (
                        "applied_point_in_time_prelaunch_relative_ip_platform_quantile"
                        if decision_cutoff_date is not None
                        and decision_cutoff_date < launch_window_end
                        else "applied_prelaunch_relative_ip_platform_quantile"
                    )
                    coefficients[target_sku] = coefficient
                    statuses[target_sku] = status
                    diagnostic = {
                            "sku_code": target_sku,
                            "launch_date": launch_day,
                            "window_start": start_day,
                            "window_end": end_day,
                            "target_external_ip_names": "|".join(target_matches),
                            "target_external_match_sources": _match_sources(
                                target_sku, target_matches, alias_provenance
                            ),
                            "benchmark_skus": "|".join(part[2] for part in benchmark_parts),
                            "benchmark_external_ip_names": "|".join(
                                sorted({name for part in benchmark_parts for name in part[3]})
                            ),
                            "benchmark_external_match_sources": "|".join(
                                sorted(
                                    {
                                        source
                                        for part in benchmark_parts
                                        for source in _match_sources(
                                            part[2], part[3], alias_provenance
                                        ).split("|")
                                        if source
                                    }
                                )
                            ),
                            "target_external_score": target_score,
                            "benchmark_external_score": benchmark_score,
                            "external_score_difference": score_difference,
                            "external_direction_coefficient": coefficient,
                            "external_status": status,
                            "social_snapshot_rows": len(window),
                            "social_snapshot_days": int(window["stat_date"].nunique()),
                    }
                    for feature in auditable_features:
                        diagnostic[f"target_media__{feature}"] = float(
                            ip_feature_values.loc[target_matches, feature].mean()
                        )
                        comparable_feature_parts: list[tuple[float, float]] = []
                        for _, weight, _, matches in benchmark_parts:
                            value = ip_feature_values.loc[matches, feature].mean()
                            if pd.notna(value):
                                comparable_feature_parts.append((float(value), weight))
                        diagnostic[f"benchmark_media__{feature}"] = (
                            sum(value * weight for value, weight in comparable_feature_parts)
                            / sum(weight for _, weight in comparable_feature_parts)
                            if comparable_feature_parts
                            else np.nan
                        )
                    diagnostic_rows.append(diagnostic)
                    continue
        coefficients[target_sku] = 1.0
        statuses[target_sku] = status
        diagnostic_rows.append(
            {
                "sku_code": target_sku,
                "launch_date": launch_day,
                "window_start": None if launch_day is None or pd.isna(launch_day) else launch_day - timedelta(days=prelaunch_days),
                "window_end": None if launch_day is None or pd.isna(launch_day) else launch_day - timedelta(days=1),
                "target_external_ip_names": "",
                "target_external_match_sources": "",
                "benchmark_skus": "|".join(chosen["candidate_sku_code"].astype(str)),
                "benchmark_external_ip_names": "",
                "benchmark_external_match_sources": "",
                "target_external_score": np.nan,
                "benchmark_external_score": np.nan,
                "external_score_difference": np.nan,
                "external_direction_coefficient": 1.0,
                "external_status": status,
                "social_snapshot_rows": 0,
                "social_snapshot_days": 0,
            }
        )
        warnings.append(f"{target_sku}: {status}; C_ext=1")
    summary["external_direction_coefficient"] = summary["sku_code"].map(coefficients)
    summary["external_status"] = summary["sku_code"].map(statuses)
    return ExternalMediaResult(
        summary,
        pd.DataFrame(diagnostic_rows),
        tuple(warnings),
    )
