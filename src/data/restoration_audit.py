"""Delivered-source contracts and lifecycle-clock diagnostics. No name allocation."""
from __future__ import annotations

from datetime import date
import numpy as np
import pandas as pd

from data.decision_adapter import SOURCES


def source_quantity_contract() -> list[dict[str, str]]:
    return [
        {"source": SOURCES[k], "sku_field": "sku_code", "quantity_field": q,
         "definition": meaning, "unit_status": "raw_same_SKU_count; business_unit_not_proven_by_schema",
         "conversion_policy": "no_name_based_parent_child_allocation; no_pack_multiplier_without_confirmation"}
        for k, q, meaning in (
            ("sales", "sale_qty", "signed order quantity (source order_prod_cnt)"),
            ("inventory", "available_stock_qty", "opening available inventory (source stock_available_itm_cnt)"),
            ("placement", "stock_available_itm_cnt", "available inventory snapshot; not a planned quantity"),
        )
    ]


def annotate_lifecycle_clock(history: pd.DataFrame, products: pd.DataFrame,
                             *, source_boundary: date | None,
                             verified_origins: pd.DataFrame | None = None) -> pd.DataFrame:
    """Do not rewrite the user-approved observed-first-sale/label clock.

    Exclude a *proven left-boundary* legacy curve, not every dimension-date
    discrepancy. A master date discrepancy elsewhere is diagnostic only.
    """
    result = history.copy()
    result["usable_for_launch_curve"] = True
    result["lifecycle_clock_status"] = "legacy_origin_unverified"
    if "week_start" not in result:
        return result
    origin = pd.to_datetime(result.week_start, errors="coerce") - pd.to_timedelta((pd.to_numeric(result.lifecycle_week, errors="coerce") - 1) * 7, unit="D")
    result["lifecycle_origin_date"] = origin
    product = products.drop_duplicates("sku_code").set_index("sku_code")
    master = pd.to_datetime(result.sku_code.map(product.first_sale_date), errors="coerce") if "first_sale_date" in product else pd.Series(pd.NaT, index=result.index)
    result["product_first_sale_date"] = master
    verified = pd.Series(False, index=result.index)
    if verified_origins is not None and "launch_universe_version" in verified_origins:
        from data.launch_contract import validate_launch_origin_contract
        validate_launch_origin_contract(verified_origins, name="lifecycle clock origins")
        lookup = verified_origins.drop_duplicates("sku_code").set_index("sku_code")
        expected = pd.to_datetime(result.sku_code.map(lookup.first_actual_sale_date), errors="coerce")
        verified = origin.eq(expected) & expected.notna()
        result.loc[verified, "lifecycle_clock_status"] = "verified_sales_observed_origin_preserved"
    left = pd.Series(False, index=result.index)
    boundary_unverified = pd.Series(False, index=result.index)
    if source_boundary is not None:
        boundary = pd.Timestamp(source_boundary)
        left = ~verified & origin.eq(boundary) & master.lt(boundary)
        boundary_unverified = ~verified & origin.eq(boundary) & ~master.eq(origin)
        result.loc[boundary_unverified, "lifecycle_clock_status"] = "source_boundary_first_week_not_corroborated"
    result.loc[left, "lifecycle_clock_status"] = "known_pre_source_sale_left_truncated_not_week1"
    invalid = origin.isna() | left | boundary_unverified
    result.loc[invalid, "usable_for_launch_curve"] = False
    result.loc[origin.isna(), "lifecycle_clock_status"] = "missing_observation_clock"
    return result
