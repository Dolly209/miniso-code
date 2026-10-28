from __future__ import annotations

from dataclasses import dataclass
import json
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


def _weighted_jaccard(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float | None:
    """V3.2 section 3.4 set similarity, with equal weights as the default.

    The reviewed standard-IP snapshot currently supplies memberships but no
    business-confirmed primary-IP weights.  In that case every membership gets
    equal mass within its SKU, exactly as allowed by the document's default.
    """

    if not left or not right:
        return None
    keys = set(left) | set(right)
    denominator = sum(max(float(left.get(key, 0.0)), float(right.get(key, 0.0))) for key in keys)
    if denominator <= 0:
        return None
    numerator = sum(min(float(left.get(key, 0.0)), float(right.get(key, 0.0))) for key in keys)
    return float(numerator / denominator)


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


def _emotion_vector(row: pd.Series) -> np.ndarray | None:
    """Return the documented 48-dimensional non-negative emotion vector.

    Free-text tags are deliberately not converted into a vector.  V3.2 makes
    the 48-dimensional weighted taxonomy a required Path-2 input; inventing a
    one-hot representation from names would silently change that contract.
    """

    value = row.get("emotion_vector")
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, (list, tuple, np.ndarray)):
        vector = np.asarray(value, dtype=float)
    else:
        columns = [f"emotion_{index:02d}" for index in range(1, 49)]
        if not all(column in row.index for column in columns):
            return None
        vector = pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float)
    if vector.shape != (48,) or not np.isfinite(vector).all() or (vector < 0).any():
        return None
    total = float(vector.sum())
    return vector / total if total > 0 else None


def _emotion_similarity(
    target: pd.Series, candidate: pd.Series
) -> tuple[float | None, float | None]:
    left, right = _emotion_vector(target), _emotion_vector(candidate)
    if left is None or right is None:
        return None, None
    denominator = float(np.maximum(left, right).sum())
    similarity = float(np.minimum(left, right).sum() / denominator) if denominator > 0 else None

    def top_labels(vector: np.ndarray) -> set[int]:
        order = np.argsort(-vector, kind="stable")
        selected: set[int] = set()
        cumulative = 0.0
        for index in order[:5]:
            selected.add(int(index))
            cumulative += float(vector[index])
            if cumulative >= 0.80:
                break
        return selected

    overlap = top_labels(left) & top_labels(right)
    top_overlap = float(sum(min(left[index], right[index]) for index in overlap))
    return similarity, top_overlap


def _ip_similarity(
    target: pd.Series,
    candidate: pd.Series,
    target_ips: Mapping[str, float],
    candidate_ips: Mapping[str, float],
) -> float | None:
    if not target_ips or not candidate_ips:
        return None
    overlap = set(target_ips) & set(candidate_ips)
    if overlap:
        return _weighted_jaccard(target_ips, candidate_ips)
    left_copyright = target.get("copyright_category")
    right_copyright = candidate.get("copyright_category")
    if _same(left_copyright, right_copyright) == 1.0:
        return 0.60
    # Both SKUs have reviewed standard-IP memberships and therefore belong to
    # the IP/trendy-toy universe, but neither shares an IP or copyright type.
    return 0.20


def _category_similarity(target: pd.Series, candidate: pd.Series) -> float | None:
    pairs = (
        ("cate_level4_code", 1.0),
        ("cate_level3_code", 0.7),
        ("manage_cate_name", 0.4),
    )
    comparable = False
    for column, score in pairs:
        left, right = target.get(column), candidate.get(column)
        if pd.notna(left) and pd.notna(right) and str(left).strip() and str(right).strip():
            comparable = True
            if str(left) == str(right):
                return score
    return 0.0 if comparable else None


def _form_similarity(target: pd.Series, candidate: pd.Series) -> float | None:
    """Score only documented material/form/play attributes when delivered."""

    fields = ("material", "product_form", "play_mode")
    scores = [
        _same(target.get(field), candidate.get(field))
        for field in fields
    ]
    available = [value for value in scores if value is not None]
    return float(np.mean(available)) if available else None


def _pack_similarity(target: pd.Series, candidate: pd.Series) -> float | None:
    if str(target.get("sales_unit_type")) != "blind_box":
        return 1.0
    return _price_similarity(target.get("pkg_unit"), candidate.get("pkg_unit"))


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
    target_ips: Mapping[str, float],
    candidate_ips: Mapping[str, float],
    specified: set[tuple[str, str]],
    *,
    emotion_min: float,
    top_overlap_min: float,
) -> str:
    pair = (str(target["sku_code"]), str(candidate["sku_code"]))
    if pair in specified:
        return "PATH0"
    if target_ips and candidate_ips and set(target_ips) & set(candidate_ips):
        return "PATH1"
    emotion, top_overlap = _emotion_similarity(target, candidate)
    if (
        emotion is not None
        and top_overlap is not None
        and emotion >= emotion_min
        and top_overlap >= top_overlap_min
    ):
        return "PATH2"
    if (_category_similarity(target, candidate) or 0.0) > 0:
        return "PATH3"
    return "PATH4"


def _equal_ip_weights(values: set[str]) -> dict[str, float]:
    if not values:
        return {}
    weight = 1.0 / len(values)
    return {value: weight for value in values}


def _path4_anchor(
    target: pd.Series,
    *,
    product_by_sku: pd.DataFrame,
    history: pd.DataFrame,
    default_velocity: float,
) -> tuple[float, str]:
    """Use the documented category hierarchy before the global static base."""

    candidate = product_by_sku.loc[
        product_by_sku.index.astype(str).isin(history.index.astype(str))
    ].copy()
    candidate["historical_velocity"] = pd.to_numeric(
        candidate.index.to_series().map(history["historical_velocity"]), errors="coerce"
    ).to_numpy()
    levels = (
        ("cate_level4_code", "category_l4"),
        ("cate_level3_code", "category_l3"),
        ("manage_cate_name", "manage_category"),
    )
    for column, source in levels:
        value = target.get(column)
        if column not in candidate or pd.isna(value) or not str(value).strip():
            continue
        values = candidate.loc[
            candidate[column].astype(str).eq(str(value)), "historical_velocity"
        ].dropna()
        values = values.loc[values.ge(0)]
        if not values.empty:
            return float(values.median()), source
    global_values = candidate["historical_velocity"].dropna()
    global_values = global_values.loc[global_values.ge(0)]
    if not global_values.empty:
        return float(global_values.median()), "global_history"
    return float(default_velocity), "versioned_static_default"


def select_comparables(
    *,
    targets: pd.DataFrame,
    products: pd.DataFrame,
    ip_membership: pd.DataFrame,
    restored_summary: pd.DataFrame,
    config: Mapping[str, Any],
    specified_comparables: pd.DataFrame | None = None,
    candidate_allowlist: pd.DataFrame | None = None,
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
    allowed_pairs: set[tuple[str, str]] | None = None
    if candidate_allowlist is not None:
        require_columns(
            candidate_allowlist,
            ("target_sku_code", "candidate_sku_code"),
            "candidate_allowlist",
        )
        allowed_pairs = set(
            zip(
                candidate_allowlist["target_sku_code"].astype(str),
                candidate_allowlist["candidate_sku_code"].astype(str),
            )
        )
    product_by_sku = products.set_index("sku_code", drop=False)
    history = restored_summary.drop_duplicates("sku_code").set_index("sku_code", drop=False)
    ips = {
        str(sku): _equal_ip_weights(
            {
                str(value)
                for value in group["ip_code"]
                if pd.notna(value) and str(value).strip()
            }
        )
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
            if (
                allowed_pairs is not None
                and (target_sku, candidate_sku) not in allowed_pairs
            ):
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
            minimum_mature_weeks = int(config.get("minimum_mature_weeks", 4))
            if float(history.loc[candidate_sku].get("mature_week_count", 0.0)) < minimum_mature_weeks:
                hard_reasons.append(f"fewer_than_{minimum_mature_weeks}_lifecycle_weeks")
            emotion_similarity, emotion_top_overlap = _emotion_similarity(target, candidate)
            values = {
                "emotion": emotion_similarity,
                "ip": _ip_similarity(
                    target,
                    candidate,
                    ips.get(target_sku, {}),
                    ips.get(candidate_sku, {}),
                ),
                "form": _form_similarity(target, candidate),
                "category": _category_similarity(target, candidate),
                "price": _price_similarity(target.get("retail_price"), candidate.get("retail_price")),
                "pack": _pack_similarity(target, candidate),
                "recent": _recent_similarity(target.get("release_dt"), candidate.get("release_dt")),
            }
            score, completeness = _weighted_score(values, weights)
            path = _path_for(
                target,
                candidate,
                ips.get(target_sku, {}),
                ips.get(candidate_sku, {}),
                specified,
                emotion_min=float(config.get("emotion_similarity_threshold", 0.50)),
                top_overlap_min=float(config.get("emotion_top_overlap_threshold", 0.20)),
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
                    "emotion_top_overlap": emotion_top_overlap,
                    **{f"similarity_{key}": value for key, value in values.items()},
                }
            )
        eligible = [row for row in candidate_rows if row["hard_filter_pass"]]
        top_k = int(config["top_k_internal"])
        path0 = [row for row in eligible if row["path_code"] == "PATH0"]
        if path0:
            pools = [path0]
        else:
            # V3.2 hard priority: keep every eligible higher-path candidate and
            # supplement, never replace, from the next path only until K is met.
            pools = [
                [row for row in eligible if row["path_code"] == path]
                for path in ("PATH1", "PATH2", "PATH3")
            ]
        selected: list[dict[str, Any]] = []
        for pool in pools:
            pool.sort(
                key=lambda row: (
                    row["similarity_score"] * row["quality_score"],
                    row["candidate_sku_code"],
                ),
                reverse=True,
            )
            selected.extend(pool[: max(top_k - len(selected), 0)])
            if len(selected) >= top_k:
                break
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
            selected_paths = list(dict.fromkeys(row["path_code"] for row in selected))
            selected_path_code = "+".join(selected_paths)
            path_reliability = float(
                sum(
                    float(alpha) * float(config["base_reliability"][row["path_code"]])
                    for row, alpha in zip(selected, alphas)
                )
            )
            path4_source = None
        else:
            warnings.append(
                f"{target_sku}: no eligible internal comparator; "
                "Path4 category hierarchy/static fallback used"
            )
            anchor_velocity, path4_source = _path4_anchor(
                target,
                product_by_sku=product_by_sku,
                history=history,
                default_velocity=float(config["static_base_velocity"]),
            )
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
                "historical_velocity": anchor_velocity,
                "selected": 1,
                "alpha_weight": 1.0,
            }
            candidate_rows.append(fallback)
            selected_path_code = "PATH4"
            path_reliability = float(config["base_reliability"]["PATH4"])
        rows.extend(candidate_rows)
        summary_rows.append(
            {
                "sku_code": target_sku,
                "selected_path": selected_path_code,
                "selected_candidate_count": sum(row.get("selected", 0) for row in candidate_rows),
                "anchor_velocity": anchor_velocity,
                "path4_anchor_source": path4_source,
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


def preselect_comparables(
    *,
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    products: pd.DataFrame,
    ip_membership: pd.DataFrame,
    config: Mapping[str, Any],
    specified_comparables: pd.DataFrame | None = None,
) -> ComparableResult:
    """Run the V3.2 Path/hard-filter Top-K retrieval before M2 restoration.

    This stage deliberately does not invent a restoration-quality score or a
    historical velocity.  All mature candidates receive the same provisional
    quality and velocity, so Path priority and the documented static
    similarity terms alone determine which at-most-K SKUs are sent to M2.
    After M2, :func:`select_comparables` is called again on only this allowlist
    to calculate the real ``Q_i``, ``alpha_i`` and quantity anchor.
    """

    require_columns(candidates, ("sku_code",), "pre-restoration candidates")
    candidate_skus = candidates["sku_code"].astype(str).drop_duplicates()
    minimum_mature_weeks = int(config.get("minimum_mature_weeks", 4))
    provisional_history = pd.DataFrame(
        {
            "sku_code": candidate_skus,
            "history_quality": 1.0,
            "historical_velocity": 1.0,
            "mature_week_count": minimum_mature_weeks,
        }
    )
    result = select_comparables(
        targets=targets,
        products=products,
        ip_membership=ip_membership,
        restored_summary=provisional_history,
        config=config,
        specified_comparables=specified_comparables,
    )
    detail = result.candidates.copy()
    if not detail.empty:
        detail["selection_phase"] = "pre_restore_top_k"
        detail["quality_score_status"] = "deferred_until_m2"
        detail["quality_score"] = np.nan
        detail["alpha_weight"] = np.nan
        detail["historical_velocity"] = np.nan
    summary = result.summary.copy()
    if not summary.empty:
        summary["selection_phase"] = "pre_restore_top_k"
        summary["anchor_velocity"] = np.nan
        summary["external_status"] = "not_evaluated_before_m2"
    return ComparableResult(detail, summary, result.warnings)


def summarize_restoration(
    restored_weekly: pd.DataFrame,
    *,
    minimum_mature_weeks: int = 4,
) -> pd.DataFrame:
    require_columns(restored_weekly, ("sku_code", "v_neutral", "n_eff"), "restored_weekly")
    frame = restored_weekly.copy()
    if "stock_grain_unreliable" in frame:
        unreliable = (
            frame["stock_grain_unreliable"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin({"1", "true", "t", "yes", "y"})
        )
        frame = frame.loc[~unreliable].copy()
    frame["v_neutral"] = pd.to_numeric(frame["v_neutral"], errors="coerce")
    frame["n_eff"] = pd.to_numeric(frame["n_eff"], errors="coerce").fillna(0.0)
    for column, default in (
        ("source_day_rows", 7.0),
        ("unknown_exposure_days", 0.0),
        ("price_observation_days", np.nan),
        ("restored_lost_sales", 0.0),
        ("restored_natural_sales", np.nan),
    ):
        if column not in frame:
            frame[column] = default
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    output_columns = (
        "sku_code",
        "historical_velocity",
        "history_quality",
        "quality_effective_days",
        "quality_repair",
        "quality_price_completeness",
        "quality_inventory_display_completeness",
        "quality_lifecycle_coverage",
        "mature_week_count",
    )
    rows: list[dict[str, Any]] = []
    for sku_code, group in frame.groupby("sku_code", sort=False):
        source_days = float(group["source_day_rows"].fillna(0).sum())
        n_eff = float(group["n_eff"].fillna(0).sum())
        effective_day_quality = float(np.clip(n_eff / max(source_days, 1.0), 0.0, 1.0))
        natural = float(group["restored_natural_sales"].fillna(0).sum())
        repair = float(group["restored_lost_sales"].fillna(0).sum())
        repair_quality = float(np.clip(1.0 - repair / max(natural, 1e-12), 0.0, 1.0))
        price_days = float(group["price_observation_days"].fillna(0).sum())
        price_quality = float(np.clip(price_days / max(source_days, 1.0), 0.0, 1.0))
        exposure_quality = float(
            np.clip(
                1.0 - float(group["unknown_exposure_days"].fillna(0).sum()) / max(source_days, 1.0),
                0.0,
                1.0,
            )
        )
        mature_week_count = int(group["lifecycle_week"].nunique())
        lifecycle_quality = float(
            np.clip(mature_week_count / max(minimum_mature_weeks, 1), 0.0, 1.0)
        )
        history_quality = (
            0.30 * effective_day_quality
            + 0.20 * repair_quality
            + 0.15 * price_quality
            + 0.20 * exposure_quality
            + 0.15 * lifecycle_quality
        )
        rows.append(
            {
                "sku_code": sku_code,
                "historical_velocity": float(group["v_neutral"].mean()),
                "history_quality": float(np.clip(history_quality, 0.0, 1.0)),
                "quality_effective_days": effective_day_quality,
                "quality_repair": repair_quality,
                "quality_price_completeness": price_quality,
                "quality_inventory_display_completeness": exposure_quality,
                "quality_lifecycle_coverage": lifecycle_quality,
                "mature_week_count": mature_week_count,
            }
        )
    return pd.DataFrame(rows, columns=output_columns)
