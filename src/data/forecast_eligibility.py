"""Audit explicit forecast abstentions without changing the fixed population.

Only the documented no-supported-store state may be omitted from Train fitting.
An unavailable Test forecast remains a blocking coverage failure. This adapter
consumes standardized first-order artifacts, never original business tables.
"""
from __future__ import annotations

import pandas as pd

from data.domain import require_columns


def forecast_fit_population(
    cohort: pd.DataFrame, first_order: pd.DataFrame, *, prefix: str = ""
) -> pd.DataFrame:
    key = "prediction_checkpoint_key" if "prediction_checkpoint_key" in cohort else "sku_code"
    require_columns(cohort, (key, "sku_code", "split"), "fixed forecast population")
    identity = list(dict.fromkeys((key, "sku_code", "split")))
    result = cohort[identity].copy()
    result["forecast_available"] = True
    result["fit_exclusion_reason"] = ""
    status = prefix + "decision_status"
    if status not in first_order:
        eligible = prefix + "training_feature_eligible"
        if eligible in first_order and first_order[eligible].astype(str).str.lower().isin(("false", "0")).any():
            raise ValueError("unavailable forecasts lack explicit decision status")
        return result
    require_columns(first_order, (key, status), "first-order forecast eligibility")
    if first_order[key].astype(str).duplicated().any():
        raise ValueError("first-order forecast eligibility keys are not unique")
    source = first_order.set_index(key)
    rows = source.reindex(result[key]).reset_index(drop=True)
    rows.index = result.index
    blocked = rows[status].eq("blocked_no_supported_stores")
    eligible = prefix + "training_feature_eligible"
    if eligible in rows:
        explicit_false = rows[eligible].astype(str).str.lower().isin(("false", "0"))
        if (explicit_false & ~blocked).any():
            raise ValueError("unavailable forecasts have an unrecognized decision status; review before fitting")
    if blocked.any():
        reason = prefix + "decision_reason"
        stores = prefix + "planned_store_count"
        base = prefix + "D_demand_base"
        require_columns(rows, (reason, stores, base, eligible), "blocked forecast diagnostics")
        valid = (
            rows[reason].eq("no_evidence_supported_store")
            & pd.to_numeric(rows[stores], errors="coerce").eq(0)
            & rows[base].isna()
            & rows[eligible].astype(str).str.lower().isin(("false", "0"))
        )
        if (blocked & ~valid).any():
            raise ValueError("unavailable forecasts have inconsistent no-supported-store diagnostics")
        if (blocked & ~result["split"].astype(str).str.lower().eq("train")).any():
            raise ValueError("unavailable forecasts in locked Test must be reported, not excluded from evaluation")
        result.loc[blocked, "forecast_available"] = False
        result.loc[blocked, "fit_exclusion_reason"] = "no_evidence_supported_store"
    return result
