from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.domain import require_columns


@dataclass(frozen=True)
class ComparableResult:
    candidates: pd.DataFrame
    summary: pd.DataFrame
    warnings: tuple[str, ...]


def _tokens(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {part.strip() for part in str(value).replace("|", ",").split(",") if part.strip()}


def _jaccard(left: set[str], right: set[str]) -> float | None:
    if not left or not right:
        return None
    return len(left & right) / len(left | right)


def _same(left: object, right: object) -> float | None:
    if pd.isna(left) or pd.isna(right) or str(left).strip() == "" or str(right).strip() == "":
        return None
    return float(str(left) == str(right))


def _price_similarity(left: object, right: object) -> float | None:
    try:
        left_value, right_value = float(left), float(right)
    except (TypeError, ValueError):
        return None
    if left_value <= 0 or right_value <= 0:
        return None
    return float(exp(-abs(log(left_value / right_value))))


def _recent_similarity(left: object, right: object) -> float | None:
    values = pd.to_datetime(pd.Series([left, right]), errors="coerce")
    if values.isna().any():
        return None
    return float(exp(-abs((values.iloc[0] - values.iloc[1]).days) / 365.0))


def _weighted_score(
    values: Mapping[str, float | None], weights: Mapping[str, float]
) -> tuple[float, float]:
    available = {key: value for key, value in values.items() if value is not None and np.isfinite(value)}
    denominator = sum(float(weights.get(key, 0.0)) for key in available)
    if denominator <= 0:
        return 0.0, 0.0
    score = sum(float(weights[key]) * float(value) for key, value in available.items()) / denominator
    completeness = denominator / max(sum(float(value) for value in weights.values()), 1e-12)
    return float(np.clip(score, 0.0, 1.0)), float(np.clip(completeness, 0.0, 1.0))


def _path_for(
    target: pd.Series,
    candidate: pd.Series,
    target_ips: set[str],
    candidate_ips: set[str],
    specified: set[tuple[str, str]],
) -> str:
    pair = (str(target["sku_code"]), str(candidate["sku_code"]))
    if pair in specified:
        return "PATH0"
    if target_ips and candidate_ips and target_ips & candidate_ips:
        return "PATH1"
    target_emotion = _tokens(target.get("emotion_tags"))
    candidate_emotion = _tokens(candidate.get("emotion_tags"))
    if target_emotion and candidate_emotion and target_emotion & candidate_emotion:
        return "PATH2"
    category_fields = ["cate_level4_code", "cate_level3_code", "cate_level2_code"]
    if any(_same(target.get(field), candidate.get(field)) == 1.0 for field in category_fields):
        return "PATH3"
    return "PATH4"


def select_comparables(
    *,
    targets: pd.DataFrame,
    products: pd.DataFrame,
    ip_membership: pd.DataFrame,
    restored_summary: pd.DataFrame,
    config: Mapping[str, Any],
    specified_comparables: pd.DataFrame | None = None,
) -> ComparableResult:
    """M1 Path0-Path4 selection with one weight per historical SKU."""

    require_columns(targets, ("sku_code",), "targets")
    require_columns(
        products,
        ("sku_code", "market_code", "sales_unit_type", "cate_level4_code", "retail_price"),
        "products",
    )
    require_columns(ip_membership, ("sku_code", "ip_code"), "ip_membership")
    require_columns(
        restored_summary,
        ("sku_code", "history_quality", "historical_velocity", "mature_week_count"),
        "restored_summary",
    )
    if products["sku_code"].duplicated().any():
        raise ValueError("products must be one row per SKU")
    specified: set[tuple[str, str]] = set()
    if specified_comparables is not None and not specified_comparables.empty:
        require_columns(
            specified_comparables,
            ("target_sku_code", "candidate_sku_code"),
            "specified_comparables",
        )
        specified = set(
            zip(
                specified_comparables["target_sku_code"].astype(str),
                specified_comparables["candidate_sku_code"].astype(str),
            )
        )
    product_by_sku = products.set_index("sku_code", drop=False)
    history = restored_summary.drop_duplicates("sku_code").set_index("sku_code", drop=False)
    ips = {
        str(sku): {str(value) for value in group["ip_code"] if pd.notna(value) and str(value).strip()}
        for sku, group in ip_membership.groupby("sku_code", dropna=False)
    }
    priority = {path: index for index, path in enumerate(config["path_priority"])}
    weights = config["similarity_weights"]
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for target_sku in targets["sku_code"].astype(str).drop_duplicates():
        if target_sku not in product_by_sku.index:
            raise ValueError(f"target SKU has no product domain row: {target_sku}")
        target = product_by_sku.loc[target_sku]
        candidate_rows: list[dict[str, Any]] = []
        for candidate_sku in history.index.astype(str):
            if candidate_sku == target_sku or candidate_sku not in product_by_sku.index:
                continue
            candidate = product_by_sku.loc[candidate_sku]
            hard_reasons: list[str] = []
            if str(candidate.get("market_code")) != str(target.get("market_code")):
                hard_reasons.append("market_mismatch")
            if str(candidate.get("sales_unit_type")) != str(target.get("sales_unit_type")):
                hard_reasons.append("sales_unit_type_mismatch")
            quality = float(history.loc[candidate_sku].get("history_quality", 0.0))
            velocity = float(history.loc[candidate_sku].get("historical_velocity", np.nan))
            if not np.isfinite(velocity) or velocity < 0:
                hard_reasons.append("immature_or_invalid_history")
            if float(history.loc[candidate_sku].get("mature_week_count", 0.0)) < 26:
                hard_reasons.append("fewer_than_26_lifecycle_weeks")
            values = {
                "emotion": _jaccard(_tokens(target.get("emotion_tags")), _tokens(candidate.get("emotion_tags"))),
                "ip": _jaccard(ips.get(target_sku, set()), ips.get(candidate_sku, set())),
                "form": _same(target.get("sales_unit_type"), candidate.get("sales_unit_type")),
                "category": _same(target.get("cate_level4_code"), candidate.get("cate_level4_code")),
                "price": _price_similarity(target.get("retail_price"), candidate.get("retail_price")),
                "pack": _price_similarity(target.get("pkg_unit"), candidate.get("pkg_unit")),
                "recent": _recent_similarity(target.get("release_dt"), candidate.get("release_dt")),
            }
            score, completeness = _weighted_score(values, weights)
            path = _path_for(
                target, candidate, ips.get(target_sku, set()), ips.get(candidate_sku, set()), specified
            )
            candidate_rows.append(
                {
                    "target_sku_code": target_sku,
                    "candidate_sku_code": candidate_sku,
                    "path_code": path,
                    "path_priority": priority[path],
                    "hard_filter_pass": int(not hard_reasons),
                    "exclusion_reason": "|".join(hard_reasons),
                    "similarity_score": score,
                    "similarity_completeness": completeness,
                    "quality_score": float(np.clip(quality, 0.0, 1.0)),
                    "historical_velocity": velocity,
                    **{f"similarity_{key}": value for key, value in values.items()},
                }
            )
        eligible = [row for row in candidate_rows if row["hard_filter_pass"]]
        selected_path = min((row["path_priority"] for row in eligible), default=priority["PATH4"])
        selected = [row for row in eligible if row["path_priority"] == selected_path]
        selected.sort(
            key=lambda row: (row["similarity_score"] * row["quality_score"], row["candidate_sku_code"]),
            reverse=True,
        )
        selected = selected[: int(config["top_k_internal"])]
        if selected:
            tau = max(float(config["softmax_temperature"]), 1e-6)
            logits = np.array([row["similarity_score"] / tau for row in selected], dtype=float)
            logits -= logits.max()
            raw_weights = np.exp(logits) * np.array(
                [max(row["quality_score"], 1e-6) for row in selected], dtype=float
            )
            alphas = raw_weights / raw_weights.sum()
            selected_codes = {row["candidate_sku_code"] for row in selected}
            for row in candidate_rows:
                row["selected"] = int(row["candidate_sku_code"] in selected_codes)
                row["alpha_weight"] = 0.0
            for row, alpha in zip(selected, alphas):
                next(item for item in candidate_rows if item["candidate_sku_code"] == row["candidate_sku_code"])["alpha_weight"] = float(alpha)
            anchor_velocity = float(sum(float(alpha) * row["historical_velocity"] for row, alpha in zip(selected, alphas)))
            selected_path_code = selected[0]["path_code"]
            path_reliability = float(config["base_reliability"][selected_path_code])
        else:
            warnings.append(f"{target_sku}: no eligible internal comparator; Path4 static base used")
            fallback = {
                "target_sku_code": target_sku,
                "candidate_sku_code": f"__STATIC__:{target.get('cate_level4_code', '__UNKNOWN__')}",
                "path_code": "PATH4",
                "path_priority": priority["PATH4"],
                "hard_filter_pass": 1,
                "exclusion_reason": "",
                "similarity_score": 0.0,
                "similarity_completeness": 0.0,
                "quality_score": 0.2,
                "historical_velocity": float(config["static_base_velocity"]),
                "selected": 1,
                "alpha_weight": 1.0,
            }
            candidate_rows.append(fallback)
            anchor_velocity = float(config["static_base_velocity"])
            selected_path_code = "PATH4"
            path_reliability = float(config["base_reliability"]["PATH4"])
        rows.extend(candidate_rows)
        summary_rows.append(
            {
                "sku_code": target_sku,
                "selected_path": selected_path_code,
                "selected_candidate_count": sum(row.get("selected", 0) for row in candidate_rows),
                "anchor_velocity": anchor_velocity,
                "path_reliability": path_reliability,
                "external_direction_coefficient": 1.0,
                "external_status": "neutral_no_external_domain_input",
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        selected_weights = result.loc[result["selected"] == 1].groupby("target_sku_code")["alpha_weight"].sum()
        if not np.allclose(selected_weights.to_numpy(dtype=float), 1.0):
            raise AssertionError("selected comparable alpha weights do not sum to one")
        duplicate = result.loc[result["selected"] == 1].duplicated(
            ["target_sku_code", "candidate_sku_code"]
        )
        if duplicate.any():
            raise AssertionError("multi-IP expansion produced duplicate candidate weights")
    return ComparableResult(result, pd.DataFrame(summary_rows), tuple(warnings))


def summarize_restoration(restored_weekly: pd.DataFrame) -> pd.DataFrame:
    require_columns(restored_weekly, ("sku_code", "v_neutral", "n_eff"), "restored_weekly")
    frame = restored_weekly.copy()
    frame["v_neutral"] = pd.to_numeric(frame["v_neutral"], errors="coerce")
    frame["n_eff"] = pd.to_numeric(frame["n_eff"], errors="coerce").fillna(0.0)
    return frame.groupby("sku_code", as_index=False).agg(
        historical_velocity=("v_neutral", "mean"),
        history_quality=("n_eff", lambda value: float(np.clip(value.sum() / max(len(value) * 7.0, 1.0), 0.0, 1.0))),
        mature_week_count=("lifecycle_week", "nunique"),
    )
