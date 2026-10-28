"""Independent week-grain checks of M2 exposure and M4/M6 demand ledgers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def exposure_audit(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["model_natural_sales", "restored_exposure_days", "observed_exposure_days",
                "known_censored_days", "v_obs", "unknown_exposure_days", "source_day_rows"]
    missing = sorted(set(required) - set(weekly))
    if missing:
        raise ValueError("Step4 requires restored weekly evidence: " + ",".join(missing))
    work = weekly.copy()
    work["expected_v_obs"] = 7 * work.model_natural_sales / work.restored_exposure_days.replace(0, np.nan)
    work["velocity_formula_error"] = (work.v_obs - work.expected_v_obs).abs()
    work["denominator_error"] = (work.restored_exposure_days - work.observed_exposure_days - work.known_censored_days).abs()
    work["repair_numerator_error"] = (work.model_natural_sales - (
        work.price_neutral_exposed_sales + work.price_neutral_censored_sales + work.restored_lost_sales).clip(lower=0)).abs()
    risk = work.get("stock_grain_risk_flag", pd.Series(False, index=work.index)).fillna(False).astype(bool)
    work["unknown_only_velocity_present"] = work.restored_exposure_days.le(0) & work.v_star.notna() & ~risk
    work["inventory_risk_separate_rows"] = risk.astype(int)
    work["velocity_check_rows"] = work.expected_v_obs.notna() & work.v_obs.notna()
    for column in ("calendar_missing_days", "missing_unknown_days"):
        if column not in work:
            work[column] = np.nan
    diagnostic_counts = ("stock_state_conflict_days", "stock_quantity_conflict_days", "missing_display_flag_days", "missing_sales_partition_days", "arrival_conflict_days")
    for column in diagnostic_counts:
        if column not in work:
            work[column] = np.nan
    ledger = work[[c for c in (
        "sku_code", "store_code", "store_tier", "lifecycle_week", "week_start", "week_end",
        "lifecycle_origin_date", "arrival_date", "sellable_proxy_date", "sellable_proxy_status",
        "source_day_rows", "expected_calendar_days", "calendar_missing_days", "missing_unknown_days",
        "clean_sales_qty", "unknown_exposure_clean_sales_qty", "model_natural_sales",
        "restored_natural_sales", "restored_lost_sales", "observed_exposure_days",
        "known_censored_days", "unknown_exposure_days", "restored_exposure_days", "v_obs",
        "expected_v_obs", "v_model", "reliability_r", "v_star", "v_neutral",
        "velocity_formula_error", "denominator_error", "repair_numerator_error", "unknown_only_velocity_present",
        "stock_grain_risk_flag", "restoration_status", "calendar_row_status",
        "stock_state_conflict_days", "stock_quantity_conflict_days", "missing_display_flag_days", "missing_sales_partition_days", "arrival_conflict_days",
    ) if c in work]]
    summary = work.groupby("sku_code", as_index=False).agg(
        store_week_rows=("store_code", "size"), stores=("store_code", "nunique"),
        clean_sales_qty=("clean_sales_qty", "sum"), restored_demand_qty=("restored_natural_sales", "sum"),
        lost_sales_uplift=("restored_lost_sales", "sum"), normal_days=("observed_exposure_days", "sum"),
        repaired_days=("known_censored_days", "sum"), unknown_days=("unknown_exposure_days", "sum"),
        exposure_days=("restored_exposure_days", "sum"), max_velocity_formula_error=("velocity_formula_error", "max"),
        max_denominator_error=("denominator_error", "max"), unknown_only_velocity_rows=("unknown_only_velocity_present", "sum"),
        calendar_missing_days=("calendar_missing_days", lambda v: v.sum(min_count=1)),
        velocity_check_rows=("velocity_check_rows", "sum"),
        inventory_risk_separate_rows=("inventory_risk_separate_rows", "sum"),
        max_repair_numerator_error=("repair_numerator_error", "max"),
        **{column: (column, lambda v: v.sum(min_count=1)) for column in diagnostic_counts},
    )
    summary["unknown_exposure_rate"] = summary.unknown_days / (summary.exposure_days + summary.unknown_days).replace(0, np.nan)
    summary["quantity_unit_status"] = "same_SKU_raw_count_not_business_pack_certification"
    summary["calendar_audit_status"] = np.where(summary.calendar_missing_days.notna(), "gap_accounting_present", "legacy_missing_gap_accounting")
    return ledger, summary


def lifecycle_contributions(weekly_demand: pd.DataFrame) -> pd.DataFrame:
    work = weekly_demand.copy()
    pieces = []
    component_columns = ["lifecycle_component_" + name for name in ("observed", "interpolated", "edge_carried", "flat_default")]
    if not set(component_columns).issubset(work) or not np.allclose(work[component_columns].sum(axis=1, min_count=4), work.lifecycle_factor, atol=1e-9, rtol=1e-9):
        raise ValueError("lifecycle source components do not reconcile to the curve factor")
    for name in ("observed", "interpolated", "edge_carried", "flat_default"):
        column = "lifecycle_component_" + name
        if column not in work:
            raise ValueError("Step4 needs freshly generated curve component metadata: " + column)
        # Additive source components reconcile exactly, including weeks where
        # one comparable is observed and a second comparable is interpolated.
        factor = pd.to_numeric(work.lifecycle_factor, errors="coerce")
        quantity = work.weekly_sales_demand * work[column] / factor.replace(0, np.nan)
        quantity = quantity.mask(factor.eq(0) & work.weekly_sales_demand.eq(0), 0.0)
        part = work[["sku_code", "store_tier", "lifecycle_week", "curve_source", "coverage_status"]].copy()
        part["component"] = name
        part["sales_demand_qty"] = quantity
        pieces.append(part)
    result = pd.concat(pieces, ignore_index=True)
    totals = work.groupby("sku_code").weekly_sales_demand.sum()
    result["share_of_target_demand"] = result.sales_demand_qty / result.sku_code.map(totals).replace(0, np.nan)
    return result
