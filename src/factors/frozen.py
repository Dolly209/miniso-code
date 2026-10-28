"""Train-only robust transforms and category empirical CDFs (V3.2 5.4/8.4)."""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Mapping, Any

import numpy as np
import pandas as pd

SOCIAL_METRICS = ("post_count", "unique_author_count", "ordinary_author_count", "koc_author_count",
                  "kol_author_count", "avg_author_follower_count", "max_author_follower_count")


def fit_social_platform_transform(history: pd.DataFrame) -> dict:
    output = {}
    for platform, group in history.groupby("platform_code", sort=False):
        metrics = {}
        for name in SOCIAL_METRICS:
            values = np.log1p(pd.to_numeric(group[name], errors="coerce").clip(lower=0)).replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                metrics[name] = None
                continue
            low, high = values.quantile([.01, .99])
            clipped = values.clip(low, high)
            median = float(clipped.median())
            metrics[name] = dict(low=float(low), high=float(high), median=median,
                                 scale=float(1.4826 * (clipped-median).abs().median()))
        output[str(platform)] = metrics
    return output


def frozen_social_z(values: pd.Series, spec) -> pd.Series:
    if not spec:
        return pd.Series(np.nan, index=values.index)
    clipped = np.log1p(pd.to_numeric(values, errors="coerce").clip(lower=0)).clip(spec["low"], spec["high"])
    return (clipped - spec["median"]) / (spec["scale"] + 1e-6)


def fit_factor_transform(features: pd.DataFrame, *, factor_weights: Mapping,
                         winsor: tuple[float, float]) -> dict[str, Any]:
    if features.empty:
        raise ValueError("factor transform requires historical Train rows")
    category = features.get("cate_level4_code", pd.Series("__UNKNOWN__", index=features.index)).fillna("__UNKNOWN__").astype(str)
    payload = {"artifact_type": "v3_2_frozen_factor_transform", "version": 1,
               "fit_rows": len(features), "features": {},
               "cdf_method": "frozen_midrank_monotone_step_0_100",
               "fit_policy": "historical_fit_targets_only; no validation/Test fitting"}
    for name in sorted({n for group in factor_weights.values() for n in group}):
        values = pd.to_numeric(features.get(name, pd.Series(np.nan, index=features.index)), errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = values.dropna()
        if valid.empty:
            payload["features"][name] = None
            continue
        low, high = valid.quantile(list(winsor))
        clipped = values.clip(low, high)
        median = float(clipped.median())
        scale = float(1.4826 * (clipped - median).abs().median())
        z = (clipped - median) / (scale + 1e-6)
        payload["features"][name] = {
            "low": float(low), "high": float(high), "median": median, "scale": scale,
            "global": sorted(z.dropna().astype(float).tolist()),
            "categories": {str(c): sorted(z.loc[idx].dropna().astype(float).tolist())
                           for c, idx in category.groupby(category).groups.items()
                           if z.loc[idx].notna().sum() >= 2},
        }
    payload["transform_id"] = sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
    return payload


def apply_factor_transform(features: pd.DataFrame, transform: Mapping,
                           factor_weights: Mapping) -> pd.DataFrame:
    result = features.copy()
    categories = result.get("cate_level4_code", pd.Series("__UNKNOWN__", index=result.index)).fillna("__UNKNOWN__").astype(str)
    for name, spec in transform["features"].items():
        values = pd.to_numeric(result.get(name, pd.Series(np.nan, index=result.index)), errors="coerce").replace([np.inf, -np.inf], np.nan)
        score = pd.Series(np.nan, index=result.index, dtype=float)
        z = score.copy()
        scope = pd.Series("missing_train_reference", index=result.index)
        if spec:
            clipped = values.clip(spec["low"], spec["high"])
            z = (clipped - spec["median"]) / (spec["scale"] + 1e-6)
            for category in categories.unique():
                mask = categories.eq(category) & z.notna()
                reference = np.asarray(spec["categories"].get(category, spec["global"]), dtype=float)
                value = z.loc[mask].to_numpy()
                # Midrank: identical values retain the same percentile at fit/inference.
                score.loc[mask] = 100 * (np.searchsorted(reference, value, side="left") + np.searchsorted(reference, value, side="right") + 1) / (2 * len(reference))
                score.loc[mask] = score.loc[mask].clip(0, 100)
                scope.loc[mask] = "frozen_category_train" if category in spec["categories"] else "frozen_global_train_fallback"
        result[f"z__{name}"] = z
        result[f"score__{name}"] = score
        result[f"cdf_scope__{name}"] = scope
    for group, weights in factor_weights.items():
        names = list(weights)
        scores = result[[f"score__{name}" for name in names]].to_numpy(dtype=float)
        w = np.asarray([weights[name] for name in names], dtype=float)
        denominator = np.isfinite(scores) @ w
        group_score = np.divide(np.nan_to_num(scores) @ w, denominator,
                                out=np.full(len(result), np.nan), where=denominator > 0)
        result[f"score__{group}"] = result[f"u__{group}"] = group_score
        result[f"completeness__{group}"] = denominator / max(w.sum(), 1e-12)
    return result
