"""V3.2 6.1: normalize each comparable's all-store curve, then mix alpha."""
from __future__ import annotations

import numpy as np
import pandas as pd


def all_store_lifecycle(history: pd.DataFrame) -> pd.DataFrame:
    work = history.copy()
    if "usable_for_launch_curve" in work:
        work = work.loc[work.usable_for_launch_curve.fillna(False)].copy()
    v = pd.to_numeric(work.v_neutral, errors="coerce")
    if "restored_exposure_days" in work:
        weight = pd.to_numeric(work.restored_exposure_days, errors="coerce")
        basis = "restored_exposure_days"
    elif "store_count" in work:
        weight = pd.to_numeric(work.store_count, errors="coerce")
        basis = "legacy_store_count_proxy"
    else:
        weight = pd.Series(1.0, index=work.index)
        basis = "legacy_equal_tier_proxy"
    valid = v.notna() & v.ge(0) & weight.notna() & weight.gt(0)
    work["_weight"] = weight.where(valid, 0.0)
    work["_weighted_v"] = v.where(valid, 0.0) * work._weight
    result = work.groupby(["sku_code", "lifecycle_week"], as_index=False).agg(
        weight=("_weight", "sum"), weighted_v=("_weighted_v", "sum")
    )
    result["v_all"] = result.weighted_v / result.weight.replace(0, np.nan)
    result["aggregation_basis"] = basis
    return result


def mix_comparable_curves(chosen, all_store, normalize, **settings):
    factors, components, rejected = {}, {}, []
    profiles = []
    groups = {str(sku): group for sku, group in all_store.groupby("sku_code", sort=False)}
    for row in chosen.itertuples(index=False):
        code, alpha = str(row.candidate_sku_code), float(row.alpha_weight)
        if alpha <= 0:
            continue
        group = groups.get(code)
        if group is None:
            rejected.append(code + ":missing_or_unusable_clock")
            continue
        v = group.set_index("lifecycle_week").v_all
        counts = v.notna().astype(int)
        curve, _, statuses, reason = normalize(
            velocity_by_week=v, cv_by_week=pd.Series(dtype=float),
            sample_count_by_week=counts, min_week1_samples=1, **settings,
        )
        if not curve:
            rejected.append(code + ":" + reason)
            continue
        profiles.append((alpha, curve, statuses))
    if not profiles:
        return {}, {}, {}, ";".join(rejected) or "no_selected_comparable_history"
    total = sum(alpha for alpha, _, _ in profiles)
    for week in range(1, settings["lifecycle_weeks"] + 1):
        pieces = {name: 0.0 for name in ("observed", "interpolated", "edge_carried", "flat_default")}
        for alpha, curve, status in profiles:
            pieces[status[week]] += alpha / total * curve[week]
        components[week] = pieces
        factors[week] = sum(pieces.values())
    statuses = {
        w: next(iter(active)) if len(active := [k for k, value in p.items() if value > 0]) == 1 else "mixed_sources"
        for w, p in components.items()
    }
    reason = "valid" if not rejected else "valid_subset_renormalized;excluded=" + ";".join(rejected)
    return factors, components, statuses, reason
