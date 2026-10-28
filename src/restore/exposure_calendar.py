"""Account for absent source days at week grain, never as zero-sale exposure."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calendar_quality_summary(weekly: pd.DataFrame) -> dict:
    columns = ("calendar_missing_days", "missing_unknown_days", "unknown_exposure_days",
               "observed_exposure_days", "known_censored_days", "missing_sales_partition_days",
               "stock_quantity_conflict_days", "missing_display_flag_days", "arrival_conflict_days")
    summary = {c: float(weekly[c].sum()) for c in columns if c in weekly}
    summary["policy"] = "missing rows/partitions unknown, not normal exposure or observed zero; no full daily grid"
    summary["status"] = "calendar_audited" if "calendar_missing_days" in weekly else "legacy_calendar_not_available"
    return summary


def complete_exposure_calendar(panel: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """O(observed days + observed SKU/store pairs x weeks), not a daily Cartesian grid.

    Only already evidenced SKU/store pairs are considered.  We cannot infer
    never-observed stores, closures or planned arrivals from absent facts.
    """
    needed = {"arrival_date", "first_actual_sale_date", "window_end_date"}
    if panel.empty or not needed.issubset(panel):
        return weekly
    pairs = ["sku_code", "store_code"]
    metadata = [c for c in weekly if c in {
        "cate_level4_code", "store_business_state", "store_level", "store_tier",
        "city_level", "city_name", "secbiz_area_name", "area_class1_name",
        "operate_size", "meter_sum",
    }]
    bounds = panel.groupby(pairs, as_index=False, sort=False).agg(
        observation_origin=("first_actual_sale_date", "first"),
        arrival_date=("arrival_date", "min"),
        window_end=("window_end_date", "first"),
        open_date=("open_date", "first"), close_date=("close_date", "first"),
        first_source_day=("business_day", "min"),
        **{c: (c, "first") for c in metadata},
    )
    for c in ("observation_origin", "arrival_date", "window_end", "open_date", "close_date", "first_source_day"):
        bounds[c] = pd.to_datetime(bounds[c], errors="coerce")
    bounds["calendar_start"] = pd.concat([bounds.observation_origin, bounds[["arrival_date", "first_source_day"]].min(axis=1)], axis=1).max(axis=1)
    bounds["first_week"] = ((bounds.calendar_start - bounds.observation_origin).dt.days // 7 + 1)
    bounds["last_week"] = ((bounds.window_end - bounds.observation_origin).dt.days // 7 + 1)
    counts = (bounds.last_week - bounds.first_week + 1).clip(lower=0).to_numpy(int)
    expanded = bounds.loc[bounds.index.repeat(counts)].copy()
    if expanded.empty:
        return weekly
    offsets = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    expanded["lifecycle_week"] = expanded.first_week.to_numpy(int) + offsets
    expanded["week_start"] = expanded.observation_origin + pd.to_timedelta((expanded.lifecycle_week - 1) * 7, unit="D")
    expanded["week_end"] = expanded.week_start + pd.Timedelta(days=6)
    interval_start = expanded[["week_start", "calendar_start"]].max(axis=1)
    interval_end = expanded[["week_end", "window_end"]].min(axis=1)
    expanded["expected_calendar_days"] = (interval_end - interval_start).dt.days.add(1).clip(lower=0)
    expanded["expected_last_date"] = interval_end
    # Open/close dates are the delivered calendar proxy, not fabricated daily
    # opening records. Missing dates do not prove a missing day was open.
    open_start = pd.concat([interval_start, expanded.open_date], axis=1).max(axis=1)
    open_end = pd.concat([interval_end, expanded.close_date], axis=1).min(axis=1)
    expanded["potential_operating_days"] = (open_end - open_start).dt.days.add(1).clip(lower=0)
    expanded["week_start"] = expanded.week_start.dt.date
    expanded["week_end"] = expanded.week_end.dt.date
    cutoff = panel.attrs.get("cutoff_date", max(panel["window_end_date"]))
    expanded["complete_week"] = expanded.week_end.le(cutoff)
    keys = pairs + ["lifecycle_week"]
    keep = keys + metadata + ["week_start", "week_end", "complete_week", "expected_calendar_days", "potential_operating_days", "arrival_date", "expected_last_date"]
    result = expanded[keep].merge(weekly.drop(columns=metadata + ["week_start", "week_end", "complete_week"], errors="ignore"), on=keys, how="outer", validate="one_to_one")
    # Distinguish an accounting placeholder from an observed zero. No missing
    # day enters normal exposure, n_eff, known-censor repair or v_model fit.
    additive = [c for c in weekly if c.endswith(("_qty", "_days", "_count")) or c in {
        "source_day_rows", "n_eff", "weighted_price_log", "price_weight",
    }]
    for c in additive:
        if c != "ending_available_stock_qty":
            result[c] = result[c].fillna(0.0)
    result["source_day_rows"] = result.source_day_rows.fillna(0)
    if "ending_observed_date" in result:
        missing_end = pd.to_datetime(result.ending_observed_date).ne(pd.to_datetime(result.expected_last_date))
        result.loc[missing_end, ["ending_available_stock_qty", "ending_stock_state"]] = np.nan
        result["ending_inventory_status"] = np.where(missing_end, "last_calendar_day_missing_not_carried_forward", "last_calendar_day_observed")
    result["calendar_missing_days"] = (result.expected_calendar_days - result.source_day_rows).clip(lower=0).fillna(0)
    # Count observed days inside the open/close date bounds separately.
    observed = panel[pairs + ["lifecycle_week", "business_day", "open_date", "close_date"]].copy()
    day = pd.to_datetime(observed.business_day)
    valid = (observed.open_date.isna() | day.ge(pd.to_datetime(observed.open_date))) & (observed.close_date.isna() | day.le(pd.to_datetime(observed.close_date)))
    observed["_eligible_observed"] = valid.astype(int)
    observed_counts = observed.groupby(keys, as_index=False)["_eligible_observed"].sum()
    result = result.merge(observed_counts, on=keys, how="left", validate="one_to_one")
    missing_unknown = (result.potential_operating_days - result._eligible_observed.fillna(0)).clip(lower=0).fillna(0)
    result["unknown_exposure_days"] = result.unknown_exposure_days.fillna(0) + missing_unknown
    result["missing_unknown_days"] = missing_unknown
    result["calendar_row_status"] = np.where(result.source_day_rows.eq(0), "missing_whole_week_not_zero_sales", "observed_sources_with_gap_accounting")
    result["calendar_coverage_policy"] = "evidenced_pairs_only; absent_days_unknown; no_daily_grid"
    sellable = panel.loc[panel.e_open.eq(1).fillna(False) & panel.e_stock.eq(1).fillna(False) & panel.nondisplay_flag.eq(0).fillna(False)
                         & ~panel.get("arrival_evidence_conflict", pd.Series(False, index=panel.index)).fillna(False)]
    proxy = sellable.groupby(pairs, as_index=False).agg(sellable_proxy_date=("business_day", "min"))
    # business_day contains Python dates, whereas the arrival bound is a
    # pandas datetime. An object column mixing both can fail Arrow conversion
    # depending on which row happens to come first. Keep the audit date typed
    # even for empty sellable sets, before filling the low-confidence fallback.
    proxy["sellable_proxy_date"] = pd.to_datetime(proxy["sellable_proxy_date"], errors="raise")
    result = result.merge(proxy, on=pairs, how="left", validate="many_to_one")
    result["sellable_proxy_status"] = np.where(result.sellable_proxy_date.notna(), "first_observed_open_stocked_display_flag_zero", "arrival_only_low_confidence")
    result["sellable_proxy_date"] = result.sellable_proxy_date.fillna(
        pd.to_datetime(result["arrival_date"], errors="raise"))
    # Observation/lifecycle clock is deliberately unchanged by store arrival.
    result["lifecycle_origin_date"] = pd.to_datetime(result.week_start) - pd.to_timedelta((result.lifecycle_week - 1) * 7, unit="D")
    return result.drop(columns="_eligible_observed").sort_values(keys, kind="stable").reset_index(drop=True)
