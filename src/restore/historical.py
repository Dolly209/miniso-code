from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from config.settings import RestorationConfig
from data.domestic import DataContractViolation, validate_delivered_sales_flags
from data.normalization import (
    NONDISPLAY_FLAG_VOCABULARY,
    STOCKOUT_FLAG_VOCABULARY,
    normalize_identifier,
    normalize_nullable_flag,
    summarize_flag_normalization,
)


UNKNOWN_TIER = "__UNKNOWN__"


@dataclass(frozen=True)
class RestorationStats:
    input_order_rows: int
    latest_order_lines: int
    exact_duplicate_rows: int
    signed_aggregation_extra_rows: int
    abnormal_order_lines: int
    daily_panel_rows: int
    weekly_rows: int
    tier_weekly_rows: int
    unknown_exposure_days: int
    stockout_days: int
    nondisplay_days: int
    event_adjusted_rows: int


@dataclass(frozen=True)
class RestorationResult:
    daily: pd.DataFrame
    weekly: pd.DataFrame
    tier_weekly: pd.DataFrame
    stats: RestorationStats
    model_report: dict[str, Any]
    quality_report: dict[str, Any]
    dataset_version: str


@dataclass(frozen=True)
class FittedRestorationModel:
    models: dict[
        str,
        tuple[dict[str, Any], dict[str, float], dict[str, float]],
    ]
    category_p99: dict[str, float]
    global_p99: float
    report: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataContractViolation(f"{name} is missing columns: {', '.join(missing)}")


def _parse_timestamp(values: pd.Series, cutoff_time: datetime, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).head(5).tolist()
        raise DataContractViolation(f"unparseable {name} values: {examples}")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(cutoff_time.tzinfo)
    return parsed.dt.tz_convert(cutoff_time.tzinfo)


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & converted.isna()
        if invalid.any():
            examples = frame.loc[invalid, column].astype(str).head(5).tolist()
            raise DataContractViolation(
                f"{name}.{column} contains non-numeric values: {examples}"
            )
        frame[column] = converted.astype(float)


def _normalize_identifiers(
    frame: pd.DataFrame, columns: Iterable[str], name: str
) -> None:
    for column in columns:
        try:
            frame[column] = normalize_identifier(
                frame[column], field_name=f"{name}.{column}"
            )
        except ValueError as exc:
            raise DataContractViolation(str(exc)) from exc


def normalize_order_lines(
    raw: pd.DataFrame,
    *,
    cutoff_time: datetime,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the signed same-ds and latest-version sales contract.

    Exact physical duplicates are collapsed first. Remaining records with the
    same ds/order-line key are signed events and are netted only when SKU,
    store, and business day agree. Across ds partitions, the latest version is
    retained. A later version whose event time is after the cutoff is removed;
    the function never falls back to an older version.
    """

    required = (
        "ds",
        "order_id",
        "order_detail_id",
        "order_time",
        "sku_code",
        "store_code",
        "sale_qty",
        "order_amt",
        "paid_amt",
        "unit_paid_price",
        "achievement_amt",
        "is_return",
        "gift_flag",
    )
    _require_columns(raw, required, "sales")
    if cutoff_time.tzinfo is None or cutoff_time.utcoffset() is None:
        raise ValueError("cutoff_time must be timezone-aware")
    frame = raw.loc[:, list(required)].copy()
    if frame.empty:
        frame["business_day"] = pd.Series(dtype="object")
        return frame, {
            "input_order_rows": 0,
            "latest_order_lines": 0,
            "exact_duplicate_rows": 0,
            "signed_aggregation_extra_rows": 0,
        }
    validate_delivered_sales_flags(frame)
    if frame[["order_id", "order_detail_id", "sku_code", "store_code"]].isna().any(
        axis=None
    ):
        raise DataContractViolation("sales order-line identity contains null values")
    _normalize_identifiers(
        frame,
        ("order_id", "order_detail_id", "sku_code", "store_code"),
        "sales",
    )
    _to_numeric(
        frame,
        (
            "sale_qty",
            "order_amt",
            "paid_amt",
            "unit_paid_price",
            "achievement_amt",
        ),
        "sales",
    )
    order_time = _parse_timestamp(frame["order_time"], cutoff_time, "order_time")
    frame["order_time"] = order_time
    frame["business_day"] = order_time.dt.date
    frame["ds"] = pd.to_datetime(frame["ds"], errors="raise").dt.date

    input_rows = len(frame)
    before_exact = len(frame)
    frame = frame.drop_duplicates().copy()
    exact_duplicates = before_exact - len(frame)
    key = ["ds", "order_id", "order_detail_id"]
    grouped = frame.groupby(key, sort=False, dropna=False, observed=True)
    grouped_identity = grouped.agg(
        sku_count=("sku_code", "nunique"),
        store_count=("store_code", "nunique"),
        day_count=("business_day", "nunique"),
    )
    conflicts = grouped_identity.loc[
        (grouped_identity["sku_count"] != 1)
        | (grouped_identity["store_count"] != 1)
        | (grouped_identity["day_count"] != 1)
    ]
    if not conflicts.empty:
        raise DataContractViolation(
            "same-ds order-line events cannot be netted because SKU, store, or "
            f"business day conflicts; examples={conflicts.head(5).reset_index().to_dict(orient='records')}"
        )
    group_sizes = grouped.size()
    signed_extra = int((group_sizes - 1).clip(lower=0).sum())
    # Keep the signed-netting contract but avoid one Python lambda invocation
    # per order-line group. GroupBy.sum(min_count=1) preserves the old nullable
    # amount semantics in compiled pandas code.
    summed = grouped[
        ["sale_qty", "order_amt", "paid_amt", "achievement_amt"]
    ].sum(min_count=1)
    aggregated = grouped.agg(
        order_time=("order_time", "max"),
        business_day=("business_day", "min"),
        sku_code=("sku_code", "min"),
        store_code=("store_code", "min"),
        unit_paid_price=("unit_paid_price", "median"),
    )
    aggregated[["sale_qty", "order_amt", "paid_amt", "achievement_amt"]] = summed
    aggregated["physical_event_count"] = group_sizes
    aggregated = aggregated.reset_index().sort_values("ds", kind="stable")
    latest = aggregated.drop_duplicates(
        subset=["order_id", "order_detail_id"], keep="last"
    )
    latest = latest.loc[latest["order_time"] <= cutoff_time].reset_index(drop=True)
    return latest, {
        "input_order_rows": input_rows,
        "latest_order_lines": len(latest),
        "exact_duplicate_rows": exact_duplicates,
        "signed_aggregation_extra_rows": signed_extra,
    }


def prepare_sales_daily(
    latest_lines: pd.DataFrame,
    product_attributes: pd.DataFrame,
    *,
    abnormal_order_quantile: float,
    threshold_fit_skus: set[str] | None = None,
    abnormal_thresholds: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_columns(
        product_attributes,
        ("sku_code", "cate_level4_code", "retail_price"),
        "product_attributes",
    )
    attributes = product_attributes[
        ["sku_code", "cate_level4_code", "retail_price"]
    ].copy()
    _normalize_identifiers(attributes, ("sku_code",), "product_attributes")
    if attributes["sku_code"].duplicated().any():
        raise DataContractViolation("product_attributes must be unique by sku_code")
    if latest_lines.empty:
        columns = [
            "sku_code",
            "store_code",
            "business_day",
            "reported_sale_qty",
            "clean_sale_qty",
            "abnormal_sale_qty",
            "abnormal_order_count",
            "order_line_count",
            "anomaly_confidence",
            "paid_unit_price",
            "retail_price",
            "price_ratio",
            "cate_level4_code",
        ]
        return pd.DataFrame(columns=columns), {
            "abnormal_order_lines": 0,
            "abnormal_thresholds": {},
        }
    attributes["cate_level4_code"] = (
        attributes["cate_level4_code"].fillna(UNKNOWN_TIER).astype(str)
    )
    _to_numeric(attributes, ("retail_price",), "product_attributes")
    frame = latest_lines.merge(
        attributes, on="sku_code", how="left", validate="many_to_one"
    )
    if frame["cate_level4_code"].isna().any():
        missing = frame.loc[frame["cate_level4_code"].isna(), "sku_code"].unique()[:5]
        raise DataContractViolation(
            f"sales SKUs have no product category attributes: {missing.tolist()}"
        )
    positive = frame["sale_qty"] > 0
    if abnormal_thresholds is None:
        threshold_fit = positive
        if threshold_fit_skus is not None:
            threshold_fit = threshold_fit & frame["sku_code"].isin(threshold_fit_skus)
        global_values = frame.loc[threshold_fit, "sale_qty"]
        global_threshold = (
            float(global_values.quantile(abnormal_order_quantile))
            if len(global_values) >= 20
            else float("inf")
        )
        category_counts = frame.loc[threshold_fit].groupby("cate_level4_code").size()
        category_thresholds = (
            frame.loc[threshold_fit]
            .groupby("cate_level4_code")["sale_qty"]
            .quantile(abnormal_order_quantile)
            .to_dict()
        )
        thresholds: dict[str, float] = {}
        for category in frame["cate_level4_code"].unique():
            if int(category_counts.get(category, 0)) >= 20:
                thresholds[str(category)] = float(category_thresholds[category])
            else:
                thresholds[str(category)] = global_threshold
    else:
        thresholds = {str(key): float(value) for key, value in abnormal_thresholds.items()}
        missing_thresholds = sorted(
            set(frame["cate_level4_code"].astype(str).unique()) - set(thresholds)
        )
        if missing_thresholds:
            raise DataContractViolation(
                "precomputed abnormal-order thresholds are missing categories: "
                f"{missing_thresholds[:5]}"
            )
    frame["abnormal_threshold"] = frame["cate_level4_code"].map(thresholds)
    frame["is_abnormal_order"] = positive & (
        frame["sale_qty"] > frame["abnormal_threshold"]
    )
    frame["clean_sale_qty"] = frame["sale_qty"].where(
        ~frame["is_abnormal_order"], 0.0
    )
    frame["abnormal_sale_qty"] = frame["sale_qty"].where(
        frame["is_abnormal_order"], 0.0
    )
    frame["effective_paid_amt"] = frame["paid_amt"].where(
        frame["paid_amt"].notna(), frame["unit_paid_price"] * frame["sale_qty"]
    )
    frame["clean_paid_amt"] = frame["effective_paid_amt"].where(
        ~frame["is_abnormal_order"], 0.0
    )
    frame["abs_sale_qty"] = frame["sale_qty"].abs()
    frame["clean_abs_sale_qty"] = frame["sale_qty"].abs().where(
        ~frame["is_abnormal_order"], 0.0
    )
    keys = ["sku_code", "store_code", "business_day"]
    daily = (
        frame.groupby(keys, as_index=False, dropna=False)
        .agg(
            reported_sale_qty=("sale_qty", "sum"),
            clean_sale_qty=("clean_sale_qty", "sum"),
            abnormal_sale_qty=("abnormal_sale_qty", "sum"),
            abnormal_order_count=("is_abnormal_order", "sum"),
            order_line_count=("sale_qty", "size"),
            paid_amt_clean=("clean_paid_amt", "sum"),
            absolute_qty=("abs_sale_qty", "sum"),
            clean_absolute_qty=("clean_abs_sale_qty", "sum"),
            retail_price=("retail_price", "first"),
            cate_level4_code=("cate_level4_code", "first"),
        )
        .reset_index(drop=True)
    )
    daily["anomaly_confidence"] = np.where(
        daily["absolute_qty"] > 0,
        daily["clean_absolute_qty"] / daily["absolute_qty"],
        1.0,
    )
    daily["paid_unit_price"] = np.where(
        daily["clean_sale_qty"] > 0,
        daily["paid_amt_clean"] / daily["clean_sale_qty"],
        np.nan,
    )
    daily["price_ratio"] = np.where(
        (daily["paid_unit_price"] > 0) & (daily["retail_price"] > 0),
        daily["paid_unit_price"] / daily["retail_price"],
        np.nan,
    )
    return daily, {
        "abnormal_order_lines": int(frame["is_abnormal_order"].sum()),
        "abnormal_thresholds": {
            key: (None if not np.isfinite(value) else value)
            for key, value in sorted(thresholds.items())
        },
    }


def _prepare_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    required = (
        "sku_code",
        "store_code",
        "stock_date",
        "available_stock_qty",
        "out_of_stock_flag",
    )
    _require_columns(frame, required, "inventory")
    result = frame.loc[:, list(required)].copy()
    if result.empty:
        result["business_day"] = pd.Series(dtype="object")
        result["inventory_stockout_flag"] = pd.Series(dtype="Float64")
        result["inventory_stockout_flag_token"] = pd.Series(dtype="object")
        result["inventory_stockout_flag_unrecognized"] = pd.Series(dtype="bool")
        return result.drop(columns=["stock_date"])
    _normalize_identifiers(result, ("sku_code", "store_code"), "inventory")
    result["business_day"] = pd.to_datetime(
        result["stock_date"], errors="coerce"
    ).dt.date
    if result["business_day"].isna().any():
        raise DataContractViolation("inventory contains unparseable stock_date")
    _to_numeric(result, ("available_stock_qty",), "inventory")
    (
        result["inventory_stockout_flag"],
        result["inventory_stockout_flag_token"],
        result["inventory_stockout_flag_unrecognized"],
    ) = normalize_nullable_flag(
        result["out_of_stock_flag"], vocabulary=STOCKOUT_FLAG_VOCABULARY
    )
    if result.duplicated(["sku_code", "store_code", "business_day"]).any():
        raise DataContractViolation("inventory is not unique at SKU x store x day")
    return result.drop(columns=["stock_date"])


def _prepare_placement(frame: pd.DataFrame) -> pd.DataFrame:
    required = (
        "ds",
        "sku_code",
        "store_code",
        "first_store_arrival_time",
        "stock_available_itm_cnt",
        "out_of_stock_flag",
        "in_stock_not_displayed_flag",
    )
    _require_columns(frame, required, "store_goods_state")
    result = frame.loc[:, list(required)].copy()
    if result.empty:
        result["business_day"] = pd.Series(dtype="object")
        result["placement_stockout_flag"] = pd.Series(dtype="Float64")
        result["placement_stockout_flag_token"] = pd.Series(dtype="object")
        result["placement_stockout_flag_unrecognized"] = pd.Series(dtype="bool")
        result["nondisplay_flag"] = pd.Series(dtype="Float64")
        result["nondisplay_flag_token"] = pd.Series(dtype="object")
        result["nondisplay_flag_unrecognized"] = pd.Series(dtype="bool")
        return result.drop(columns=["ds"])
    _normalize_identifiers(result, ("sku_code", "store_code"), "store_goods_state")
    result["business_day"] = pd.to_datetime(result["ds"], errors="coerce").dt.date
    result["first_store_arrival_time"] = pd.to_datetime(
        result["first_store_arrival_time"], errors="coerce", format="mixed"
    ).dt.normalize()
    if result["business_day"].isna().any():
        raise DataContractViolation("store_goods_state contains unparseable ds")
    _to_numeric(result, ("stock_available_itm_cnt",), "store_goods_state")
    (
        result["placement_stockout_flag"],
        result["placement_stockout_flag_token"],
        result["placement_stockout_flag_unrecognized"],
    ) = normalize_nullable_flag(
        result["out_of_stock_flag"], vocabulary=STOCKOUT_FLAG_VOCABULARY
    )
    (
        result["nondisplay_flag"],
        result["nondisplay_flag_token"],
        result["nondisplay_flag_unrecognized"],
    ) = normalize_nullable_flag(
        result["in_stock_not_displayed_flag"],
        vocabulary=NONDISPLAY_FLAG_VOCABULARY,
    )
    if result.duplicated(["sku_code", "store_code", "business_day"]).any():
        raise DataContractViolation("store_goods_state is not unique at SKU x store x day")
    return result.drop(columns=["ds"])


def _prepare_stores(frame: pd.DataFrame) -> pd.DataFrame:
    required = (
        "store_code",
        "store_business_state",
        "store_level",
        "open_date",
        "close_date",
        "sts_state",
    )
    _require_columns(frame, required, "stores")
    result = frame.loc[:, list(required)].copy()
    _normalize_identifiers(result, ("store_code",), "stores")
    if result["store_code"].duplicated().any():
        raise DataContractViolation("store snapshot is not unique by store_code")
    result["open_date"] = pd.to_datetime(
        result["open_date"], errors="coerce", format="mixed"
    ).dt.date
    result["close_date"] = pd.to_datetime(
        result["close_date"], errors="coerce", format="mixed"
    ).dt.date
    result["store_business_state"] = (
        result["store_business_state"].fillna(UNKNOWN_TIER).astype(str)
    )
    result["store_level"] = result["store_level"].fillna(UNKNOWN_TIER).astype(str)
    result["store_tier"] = (
        result["store_business_state"] + "|" + result["store_level"]
    )
    return result


def _derive_source_stock_state(
    quantity: pd.Series,
    stockout_flag: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Derive one source's stock state without hiding internal contradictions."""

    positive_quantity = quantity.notna() & (quantity > 0)
    zero_quantity = quantity.notna() & (quantity <= 0)
    explicit_stockout = stockout_flag == 1
    explicit_in_stock = stockout_flag == 0
    conflict = (positive_quantity & explicit_stockout) | (
        zero_quantity & explicit_in_stock
    )
    conflict = conflict.fillna(False).astype(bool)
    state = pd.Series(pd.NA, index=quantity.index, dtype="Float64")
    state.loc[~conflict & (zero_quantity | explicit_stockout)] = 0.0
    state.loc[~conflict & (positive_quantity | explicit_in_stock)] = 1.0
    return state, conflict


def build_daily_panel(
    *,
    sales_daily: pd.DataFrame,
    inventory: pd.DataFrame,
    placement: pd.DataFrame,
    stores: pd.DataFrame,
    first_sale_dates: Mapping[str, date],
    cutoff_date: date,
    window_days: int,
    promotion_price_ratio_threshold: float,
) -> pd.DataFrame:
    """Build nullable V3.2 Eopen/Estock/Edisplay day states.

    Missing state is kept unknown. A positive clean sale may confirm that the
    store was open, stocked, and displayed on that day, but it never turns an
    explicit stockout or non-display flag into exposure.
    """

    if window_days <= 0:
        raise ValueError("window_days must be positive")
    keys = ["sku_code", "store_code", "business_day"]
    sales = sales_daily.copy()
    if not sales.empty:
        _normalize_identifiers(sales, ("sku_code", "store_code"), "sales_daily")
    inv = _prepare_inventory(inventory)
    state = _prepare_placement(placement)
    store_dim = _prepare_stores(stores)
    panel = sales.merge(inv, on=keys, how="outer", validate="one_to_one")
    panel = panel.merge(state, on=keys, how="outer", validate="one_to_one")
    panel = panel.merge(store_dim, on="store_code", how="left", validate="many_to_one")
    if panel.empty:
        return panel
    panel["first_actual_sale_date"] = panel["sku_code"].map(first_sale_dates)
    if panel["first_actual_sale_date"].isna().any():
        missing = panel.loc[panel["first_actual_sale_date"].isna(), "sku_code"].unique()
        raise DataContractViolation(
            f"daily facts contain SKUs absent from cohort: {missing[:5].tolist()}"
        )
    panel["window_end_date"] = panel["first_actual_sale_date"].map(
        lambda value: min(value + timedelta(days=window_days - 1), cutoff_date)
    )
    in_window = (panel["business_day"] >= panel["first_actual_sale_date"]) & (
        panel["business_day"] <= panel["window_end_date"]
    )
    panel = panel.loc[in_window].copy()
    if panel.empty:
        return panel

    arrival = panel.groupby(["sku_code", "store_code"], dropna=False)[
        "first_store_arrival_time"
    ].transform("min")
    business_timestamps = pd.to_datetime(panel["business_day"], errors="raise")
    inferred_arrival = business_timestamps.groupby(
        [panel["sku_code"], panel["store_code"]], dropna=False
    ).transform("min")
    panel["arrival_date"] = arrival.where(arrival.notna(), inferred_arrival)
    panel["arrival_inferred"] = arrival.isna()
    panel = panel.loc[
        business_timestamps >= panel["arrival_date"]
    ].copy()

    has_operational_row = panel["available_stock_qty"].notna() | panel[
        "stock_available_itm_cnt"
    ].notna()
    active_status = panel["sts_state"].astype(str).str.strip().str.lower().isin(
        {"1", "1.0", "open", "active", "营业", "正常"}
    )
    closed_status = panel["sts_state"].astype(str).str.strip().str.lower().isin(
        {"0", "0.0", "closed", "inactive", "停业", "闭店", "关店"}
    )
    before_open = panel["open_date"].notna() & (
        panel["business_day"] < panel["open_date"]
    )
    after_close = panel["close_date"].notna() & (
        panel["business_day"] > panel["close_date"]
    )
    panel["e_open"] = pd.Series(pd.NA, index=panel.index, dtype="Float64")
    panel.loc[before_open | after_close | closed_status, "e_open"] = 0.0
    open_evidence = active_status | has_operational_row | (
        panel.get("clean_sale_qty", pd.Series(0.0, index=panel.index)).fillna(0) > 0
    )
    panel.loc[panel["e_open"].isna() & open_evidence, "e_open"] = 1.0

    (
        panel["inventory_stock_state"],
        inventory_internal_conflict,
    ) = _derive_source_stock_state(
        panel["available_stock_qty"], panel["inventory_stockout_flag"]
    )
    (
        panel["placement_stock_state"],
        placement_internal_conflict,
    ) = _derive_source_stock_state(
        panel["stock_available_itm_cnt"], panel["placement_stockout_flag"]
    )
    source_disagreement = (
        panel["inventory_stock_state"].notna()
        & panel["placement_stock_state"].notna()
        & (
            panel["inventory_stock_state"]
            != panel["placement_stock_state"]
        )
    )
    panel["stock_state_conflict"] = (
        inventory_internal_conflict
        | placement_internal_conflict
        | source_disagreement.fillna(False)
    )
    panel["e_stock"] = pd.Series(pd.NA, index=panel.index, dtype="Float64")
    stock_states = panel[["inventory_stock_state", "placement_stock_state"]]
    any_stockout = stock_states.eq(0).fillna(False).any(axis=1)
    any_in_stock = stock_states.eq(1).fillna(False).any(axis=1)
    no_stock_conflict = ~panel["stock_state_conflict"]
    panel.loc[no_stock_conflict & any_stockout, "e_stock"] = 0.0
    panel.loc[
        no_stock_conflict & panel["e_stock"].isna() & any_in_stock,
        "e_stock",
    ] = 1.0
    positive_sale = panel.get("clean_sale_qty", pd.Series(0.0, index=panel.index)).fillna(0) > 0
    panel.loc[
        panel["e_stock"].isna()
        & ~panel["stock_state_conflict"]
        & positive_sale,
        "e_stock",
    ] = 1.0

    panel["e_display"] = pd.Series(pd.NA, index=panel.index, dtype="Float64")
    panel.loc[panel["nondisplay_flag"] == 1, "e_display"] = 0.0
    panel.loc[panel["nondisplay_flag"] == 0, "e_display"] = 1.0
    panel.loc[panel["e_display"].isna() & positive_sale, "e_display"] = 1.0

    panel["e_total"] = pd.Series(pd.NA, index=panel.index, dtype="Float64")
    exposure_components = panel[["e_open", "e_stock", "e_display"]]
    any_zero = exposure_components.eq(0).fillna(False).any(axis=1)
    all_one = exposure_components.notna().all(axis=1) & (
        exposure_components.eq(1).fillna(False).all(axis=1)
    )
    panel.loc[any_zero, "e_total"] = 0.0
    panel.loc[all_one, "e_total"] = 1.0
    panel["clean_sale_qty"] = panel.get(
        "clean_sale_qty", pd.Series(0.0, index=panel.index)
    ).fillna(0.0)
    panel["reported_sale_qty"] = panel.get(
        "reported_sale_qty", pd.Series(0.0, index=panel.index)
    ).fillna(0.0)
    panel["abnormal_sale_qty"] = panel.get(
        "abnormal_sale_qty", pd.Series(0.0, index=panel.index)
    ).fillna(0.0)
    panel["abnormal_order_count"] = panel.get(
        "abnormal_order_count", pd.Series(0.0, index=panel.index)
    ).fillna(0).astype(int)
    panel["anomaly_confidence"] = panel.get(
        "anomaly_confidence", pd.Series(1.0, index=panel.index)
    ).fillna(1.0).clip(0, 1)
    price_ratio = panel.get(
        "price_ratio", pd.Series(np.nan, index=panel.index, dtype=float)
    )
    panel["price_log_ratio"] = np.where(
        price_ratio > 0,
        np.log(price_ratio),
        np.nan,
    )
    panel["promotion_price_proxy"] = price_ratio < promotion_price_ratio_threshold
    panel["promotion_flag_source"] = np.where(
        panel["promotion_price_proxy"], "price_ratio_proxy", "not_observed"
    )
    panel["store_business_state"] = panel["store_business_state"].fillna(UNKNOWN_TIER)
    panel["store_level"] = panel["store_level"].fillna(UNKNOWN_TIER)
    panel["store_tier"] = panel["store_tier"].fillna(
        panel["store_business_state"].astype(str)
        + "|"
        + panel["store_level"].astype(str)
    )
    panel["lifecycle_week"] = (
        (pd.to_datetime(panel["business_day"]) - pd.to_datetime(panel["first_actual_sale_date"]))
        .dt.days.floordiv(7)
        .add(1)
        .astype(int)
    )
    first_sale_ts = pd.to_datetime(panel["first_actual_sale_date"], errors="raise")
    panel["week_start"] = (
        first_sale_ts
        + pd.to_timedelta((panel["lifecycle_week"] - 1) * 7, unit="D")
    ).dt.date
    panel["week_end"] = (
        pd.to_datetime(panel["week_start"], errors="raise")
        + pd.to_timedelta(6, unit="D")
    ).dt.date
    panel["complete_week"] = panel["week_end"] <= cutoff_date
    panel["cate_level4_code"] = panel.get(
        "cate_level4_code", pd.Series(UNKNOWN_TIER, index=panel.index)
    )
    category_by_sku = (
        panel.loc[panel["cate_level4_code"].notna()]
        .groupby("sku_code")["cate_level4_code"]
        .first()
        .to_dict()
    )
    panel["cate_level4_code"] = panel["cate_level4_code"].fillna(
        panel["sku_code"].map(category_by_sku)
    ).fillna(UNKNOWN_TIER)
    return panel.sort_values(keys).reset_index(drop=True)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = sorted_weights.sum() / 2
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)])


def _fit_huber_effects(
    observed: pd.DataFrame,
    *,
    config: RestorationConfig,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    y = np.log1p(observed["v_observed_raw"].clip(lower=0).to_numpy(float))
    x = observed["mean_price_log_ratio"].fillna(0.0).to_numpy(float)
    price_known = observed["mean_price_log_ratio"].notna().to_numpy()
    base_weight = observed["fit_weight"].to_numpy(float)
    skus = observed["sku_code"].astype(str)
    stores = observed["store_code"].astype(str)
    sku_codes, sku_levels = pd.factorize(skus, sort=True)
    store_codes, store_levels = pd.factorize(stores, sort=True)
    intercept = _weighted_median(y, base_weight)
    price_beta = 0.0
    price_active = bool(
        price_known.sum() >= config.minimum_price_observations
        and np.nanstd(x[price_known]) > 1e-8
    )
    item_values = np.zeros(len(sku_levels), dtype=float)
    store_values = np.zeros(len(store_levels), dtype=float)
    converged = False
    iteration = 0
    for iteration in range(1, config.huber_max_iter + 1):
        old = np.array([intercept, price_beta])
        old_item_values = item_values.copy()
        old_store_values = store_values.copy()
        item_vector = item_values[sku_codes]
        store_vector = store_values[store_codes]
        residual = y - intercept - price_beta * x - item_vector - store_vector
        center = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - center)) + 1e-9
        standardized = np.abs(residual - center) / scale
        huber = np.ones_like(standardized)
        outside = standardized > config.huber_delta
        huber[outside] = config.huber_delta / standardized[outside]
        weights = base_weight * huber

        target = y - item_vector - store_vector
        if price_active:
            design = np.column_stack([np.ones(len(y)), x])
            normal = design.T @ (weights[:, None] * design)
            normal[1, 1] += 1e-6
            rhs = design.T @ (weights * target)
            try:
                coefficients = np.linalg.solve(normal, rhs)
                intercept = float(coefficients[0])
                price_beta = min(float(coefficients[1]), 0.0)
            except np.linalg.LinAlgError:
                intercept = _weighted_median(target, weights)
                price_beta = 0.0
                price_active = False
        else:
            intercept = _weighted_median(target, weights)
            price_beta = 0.0

        residual_for_item = y - intercept - price_beta * x - store_vector
        item_numerator = np.bincount(
            sku_codes,
            weights=weights * residual_for_item,
            minlength=len(sku_levels),
        )
        item_denominator = (
            np.bincount(sku_codes, weights=weights, minlength=len(sku_levels))
            + config.item_effect_ridge
        )
        item_values = np.divide(
            item_numerator,
            item_denominator,
            out=np.zeros_like(item_numerator),
            where=item_denominator > 0,
        )
        item_vector = item_values[sku_codes]
        residual_for_store = y - intercept - price_beta * x - item_vector
        store_numerator = np.bincount(
            store_codes,
            weights=weights * residual_for_store,
            minlength=len(store_levels),
        )
        store_denominator = (
            np.bincount(
                store_codes, weights=weights, minlength=len(store_levels)
            )
            + config.store_effect_ridge
        )
        store_values = np.divide(
            store_numerator,
            store_denominator,
            out=np.zeros_like(store_numerator),
            where=store_denominator > 0,
        )
        effect_change = max(
            float(np.max(np.abs(item_values - old_item_values)))
            if len(item_values)
            else 0.0,
            float(np.max(np.abs(store_values - old_store_values)))
            if len(store_values)
            else 0.0,
        )
        if (
            np.max(np.abs(np.array([intercept, price_beta]) - old)) < 1e-7
            and effect_change < 1e-7
        ):
            converged = True
            break
    item_effects = {
        str(key): float(value) for key, value in zip(sku_levels, item_values)
    }
    store_effects = {
        str(key): float(value) for key, value in zip(store_levels, store_values)
    }
    return (
        {
            "intercept": intercept,
            "price_elasticity": price_beta,
            "price_effect_status": "estimated" if price_active else "neutral_insufficient_variation",
            "observations": len(observed),
            "iterations": iteration,
            "converged": converged,
        },
        item_effects,
        store_effects,
    )


def _aggregate_weekly(panel: pd.DataFrame, config: RestorationConfig) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    working = panel.copy()
    open_is_one = working["e_open"].eq(1).fillna(False).astype(bool)
    total_is_one = working["e_total"].eq(1).fillna(False).astype(bool)
    total_is_zero = working["e_total"].eq(0).fillna(False).astype(bool)
    stock_is_zero = working["e_stock"].eq(0).fillna(False).astype(bool)
    display_is_zero = working["e_display"].eq(0).fillna(False).astype(bool)
    # Nullable exposure is intentional: unknown is neither exposed nor censored.
    # Convert only the masks consumed by NumPy to strict booleans so pd.NA is
    # never coerced through an ambiguous truth-value operation.
    working["fully_exposed"] = total_is_one
    working["known_censored"] = (
        open_is_one
        & total_is_zero
        & (stock_is_zero | display_is_zero)
    )
    working["unknown_exposure"] = open_is_one & working["e_total"].isna()
    working["weighted_exposure"] = np.where(
        working["fully_exposed"], working["anomaly_confidence"], 0.0
    )
    working["exposed_clean_sale_qty"] = np.where(
        working["fully_exposed"], working["clean_sale_qty"], 0.0
    )
    working["censored_clean_sale_qty"] = np.where(
        working["known_censored"], working["clean_sale_qty"], 0.0
    )
    working["price_weight"] = np.where(
        working["price_log_ratio"].notna(), working["clean_sale_qty"].abs(), 0.0
    )
    working["weighted_price_log"] = (
        working["price_log_ratio"].fillna(0.0) * working["price_weight"]
    )
    working["stockout_day"] = stock_is_zero.astype("int8")
    working["nondisplay_day"] = display_is_zero.astype("int8")
    working["price_observed_day"] = working["price_log_ratio"].notna().astype("int8")
    keys = [
        "sku_code",
        "cate_level4_code",
        "store_code",
        "store_business_state",
        "store_level",
        "store_tier",
        "lifecycle_week",
        "week_start",
        "week_end",
        "complete_week",
    ]
    weekly = (
        working.groupby(keys, as_index=False, dropna=False)
        .agg(
            reported_sales_qty=("reported_sale_qty", "sum"),
            clean_sales_qty=("clean_sale_qty", "sum"),
            exposed_clean_sales_qty=("exposed_clean_sale_qty", "sum"),
            censored_clean_sales_qty=("censored_clean_sale_qty", "sum"),
            abnormal_sales_qty=("abnormal_sale_qty", "sum"),
            abnormal_order_count=("abnormal_order_count", "sum"),
            observed_exposure_days=("fully_exposed", "sum"),
            known_censored_days=("known_censored", "sum"),
            unknown_exposure_days=("unknown_exposure", "sum"),
            stockout_days=("stockout_day", "sum"),
            nondisplay_days=("nondisplay_day", "sum"),
            n_eff=("weighted_exposure", "sum"),
            price_weight=("price_weight", "sum"),
            weighted_price_log=("weighted_price_log", "sum"),
            price_observation_days=("price_observed_day", "sum"),
            promotion_proxy_days=("promotion_price_proxy", "sum"),
            # The daily panel is unique at SKU x store x business_day.
            source_day_rows=("business_day", "size"),
        )
        .reset_index(drop=True)
    )
    weekly["mean_price_log_ratio"] = np.where(
        weekly["price_weight"] > 0,
        weekly["weighted_price_log"] / weekly["price_weight"],
        np.nan,
    )
    weekly["v_observed_raw"] = np.where(
        weekly["observed_exposure_days"] > 0,
        7.0
        * weekly["exposed_clean_sales_qty"]
        / weekly["observed_exposure_days"],
        np.nan,
    )
    for column in (
        "reported_sales_qty",
        "clean_sales_qty",
        "exposed_clean_sales_qty",
        "censored_clean_sales_qty",
        "abnormal_sales_qty",
        "observed_exposure_days",
        "known_censored_days",
        "unknown_exposure_days",
        "stockout_days",
        "nondisplay_days",
        "n_eff",
        "mean_price_log_ratio",
        "v_observed_raw",
    ):
        weekly[column] = pd.to_numeric(weekly[column], errors="coerce")
    flagged_days = weekly["stockout_days"]
    denominator = weekly["source_day_rows"].clip(lower=1)
    weekly["stockout_week_flag"] = (
        (flagged_days >= config.stockout_min_days_per_week)
        | (flagged_days / denominator >= config.stockout_flag_share_threshold)
    )
    return weekly


def prepare_weekly_for_model(
    weekly: pd.DataFrame,
    *,
    fit_skus: set[str],
    training_weights: Mapping[str, float] | None = None,
    fit_weight_mean: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Add Train-only model columns without fitting or applying a model.

    All rows for one SKU must stay in the same batch. That makes the per-SKU
    row count exact even when this function is called on a batch. Passing the
    global ``fit_weight_mean`` on the second pass reproduces the full-frame
    normalization used by V3.2.
    """

    if weekly.empty:
        return weekly.copy(), 0.0
    weekly = weekly.copy()
    weights = training_weights or {}
    weekly["training_weight"] = weekly["sku_code"].map(weights).fillna(1.0)
    weekly["model_fit_eligible"] = (
        weekly["sku_code"].isin(fit_skus)
        & weekly["v_observed_raw"].notna()
        & (weekly["v_observed_raw"] >= 0)
        & (weekly["n_eff"] > 0)
    )
    fit_row_counts = (
        weekly.loc[weekly["model_fit_eligible"]]
        .groupby("sku_code")
        .size()
        .to_dict()
    )
    weekly["sku_fit_row_count"] = weekly["sku_code"].map(fit_row_counts).fillna(0)
    weekly["fit_weight"] = np.where(
        weekly["model_fit_eligible"],
        weekly["n_eff"]
        * weekly["training_weight"]
        / weekly["sku_fit_row_count"].clip(lower=1),
        0.0,
    )
    local_mean = float(
        weekly.loc[weekly["model_fit_eligible"], "fit_weight"].mean()
    )
    selected_mean = local_mean if fit_weight_mean is None else float(fit_weight_mean)
    if selected_mean > 0:
        weekly.loc[weekly["model_fit_eligible"], "fit_weight"] /= (
            selected_mean
        )
    return weekly, local_mean


def fit_restoration_model(
    fit_rows: pd.DataFrame,
    *,
    config: RestorationConfig,
) -> FittedRestorationModel:
    fit_rows = fit_rows.copy()
    if fit_rows.empty:
        raise DataContractViolation(
            "M2 has no clean Train exposure rows for the Huber fallback model"
        )

    global_report, global_items, global_stores = _fit_huber_effects(
        fit_rows, config=config
    )
    model_reports: dict[str, Any] = {"__GLOBAL__": global_report}
    category_models: dict[str, tuple[dict[str, Any], dict[str, float], dict[str, float]]] = {}
    for category, group in fit_rows.groupby("cate_level4_code", dropna=False):
        if len(group) < max(10, config.minimum_price_observations // 2):
            continue
        category_models[str(category)] = _fit_huber_effects(group, config=config)
        model_reports[str(category)] = category_models[str(category)][0]

    fit_rows["v_observed_raw"] = pd.to_numeric(
        fit_rows["v_observed_raw"], errors="raise"
    )
    p99 = {
        str(key): float(value)
        for key, value in fit_rows.groupby("cate_level4_code")[
            "v_observed_raw"
        ].quantile(0.99).items()
    }
    global_p99 = float(fit_rows["v_observed_raw"].quantile(0.99))
    model_lookup = {"__GLOBAL__": (global_report, global_items, global_stores)}
    model_lookup.update(category_models)
    report = {
        "status": "ok",
        "fit_scope": "Train SKUs only",
        "fit_sku_count": int(fit_rows["sku_code"].nunique()),
        "fit_weekly_rows": len(fit_rows),
        "fit_weight_policy": (
            "Train long-tail SKU weight is divided across that SKU's eligible "
            "store-weeks, multiplied by n_eff, then normalized to mean one"
        ),
        "category_models": model_reports,
        "price_policy": (
            "paid/retail price ratio enters the Huber model separately; elasticity "
            "is constrained non-positive and becomes neutral when evidence is insufficient"
        ),
        "p99_cap_policy": (
            "v_model is capped at the Train category P99; global P99 is fallback"
        ),
    }
    return FittedRestorationModel(
        models=model_lookup,
        category_p99=p99,
        global_p99=global_p99,
        report=report,
    )


def apply_restoration_model(
    weekly: pd.DataFrame,
    *,
    fitted: FittedRestorationModel,
    config: RestorationConfig,
) -> pd.DataFrame:
    if weekly.empty:
        return weekly.copy()
    weekly = weekly.copy().reset_index(drop=True)
    category_models = set(fitted.models) - {"__GLOBAL__"}
    weekly["model_scope"] = weekly["cate_level4_code"].astype(str).where(
        weekly["cate_level4_code"].astype(str).isin(category_models),
        "__GLOBAL__",
    )
    prediction = np.zeros(len(weekly), dtype=float)
    elasticity = np.zeros(len(weekly), dtype=float)
    fallback = np.empty(len(weekly), dtype=object)
    for scope, row_index in weekly.groupby(
        "model_scope", sort=False, observed=True
    ).groups.items():
        report, items, stores = fitted.models[str(scope)]
        index = np.asarray(row_index, dtype=int)
        subset = weekly.loc[index]
        item_value = subset["sku_code"].astype(str).map(items)
        store_value = subset["store_code"].astype(str).map(stores)
        item_known = item_value.notna().to_numpy()
        store_known = store_value.notna().to_numpy()
        linear = (
            float(report["intercept"])
            + item_value.fillna(0.0).to_numpy(float)
            + store_value.fillna(0.0).to_numpy(float)
        )
        prediction[index] = np.maximum(np.expm1(linear), 0.0)
        elasticity[index] = float(report["price_elasticity"])
        fallback[index] = np.select(
            [item_known & store_known, item_known, store_known],
            ["item_store", "item_only", "store_only"],
            default="category_or_global",
        )
    weekly["v_model"] = prediction
    weekly["price_elasticity"] = elasticity
    weekly["model_fallback_level"] = fallback
    caps = (
        weekly["cate_level4_code"].astype(str).map(fitted.category_p99)
        .fillna(fitted.global_p99)
    )
    weekly["v_model"] = np.minimum(weekly["v_model"], caps.clip(lower=0))

    price_effect = np.exp(
        weekly["price_elasticity"] * weekly["mean_price_log_ratio"].fillna(0.0)
    )
    weekly["price_neutral_exposed_sales"] = np.where(
        weekly["mean_price_log_ratio"].notna() & (price_effect > 0),
        weekly["exposed_clean_sales_qty"] / price_effect,
        weekly["exposed_clean_sales_qty"],
    )
    weekly["price_neutral_censored_sales"] = np.where(
        weekly["mean_price_log_ratio"].notna() & (price_effect > 0),
        weekly["censored_clean_sales_qty"] / price_effect,
        weekly["censored_clean_sales_qty"],
    )
    modeled_censored_sales = (
        weekly["v_model"] * weekly["known_censored_days"] / 7.0
    )
    weekly["restored_lost_sales"] = (
        modeled_censored_sales - weekly["price_neutral_censored_sales"]
    ).clip(lower=0)
    weekly["restored_natural_sales"] = (
        weekly["price_neutral_exposed_sales"]
        + weekly["price_neutral_censored_sales"]
        + weekly["restored_lost_sales"]
    )
    weekly["restored_exposure_days"] = (
        weekly["observed_exposure_days"] + weekly["known_censored_days"]
    )
    weekly["v_obs"] = np.where(
        weekly["restored_exposure_days"] > 0,
        7.0
        * weekly["restored_natural_sales"]
        / weekly["restored_exposure_days"],
        np.nan,
    )
    weekly["reliability_r"] = weekly["n_eff"] / (
        weekly["n_eff"] + config.shrinkage_k_exposure_days
    )
    observed_or_model = weekly["v_obs"].where(
        weekly["v_obs"].notna(), weekly["v_model"]
    )
    weekly["v_star"] = (
        weekly["reliability_r"] * observed_or_model
        + (1.0 - weekly["reliability_r"]) * weekly["v_model"]
    )
    negative = weekly["restored_natural_sales"] < -1e-9
    if negative.any():
        examples = weekly.loc[
            negative, ["sku_code", "store_code", "lifecycle_week", "restored_natural_sales"]
        ].head(5)
        raise DataContractViolation(
            "signed sales produce negative restored weekly natural sales; "
            f"examples={examples.to_dict(orient='records')}"
        )
    return weekly


def fit_and_restore_weekly(
    panel: pd.DataFrame,
    *,
    config: RestorationConfig,
    fit_skus: set[str],
    training_weights: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    weekly = _aggregate_weekly(panel, config)
    if weekly.empty:
        return weekly, {"status": "empty", "category_models": {}}
    weekly, _ = prepare_weekly_for_model(
        weekly,
        fit_skus=fit_skus,
        training_weights=training_weights,
    )
    fit_rows = weekly.loc[weekly["model_fit_eligible"]].copy()
    fitted = fit_restoration_model(fit_rows, config=config)
    return (
        apply_restoration_model(weekly, fitted=fitted, config=config),
        dict(fitted.report),
    )


def assign_event_codes(weekly: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    result = weekly.copy()
    result["event_code"] = ""
    if events is None or events.empty or result.empty:
        return result
    required = ("event_code", "start_date", "end_date")
    _require_columns(events, required, "event_calendar")
    calendar = events.copy()
    calendar["start_date"] = pd.to_datetime(calendar["start_date"], errors="coerce").dt.date
    calendar["end_date"] = pd.to_datetime(calendar["end_date"], errors="coerce").dt.date
    if calendar[["start_date", "end_date"]].isna().any(axis=None):
        raise DataContractViolation("event_calendar contains unparseable dates")
    if (calendar["start_date"] > calendar["end_date"]).any():
        raise DataContractViolation("event_calendar start_date is after end_date")
    if "cate_level4_code" not in calendar:
        calendar["cate_level4_code"] = ""
    # Event calendars are normally small. Loop over events and evaluate all
    # weekly rows as vectors, rather than looping over every weekly row and
    # repeatedly filtering the calendar. Sorting first preserves the old
    # tie-break: maximum overlap, then lexicographically smallest event code.
    week_start = pd.to_datetime(result["week_start"], errors="raise")
    week_end = pd.to_datetime(result["week_end"], errors="raise")
    row_categories = result["cate_level4_code"].fillna("").astype(str)
    best_overlap = np.full(len(result), -1, dtype=np.int16)
    best_code = np.full(len(result), "", dtype=object)
    for event in calendar.sort_values("event_code", kind="stable").itertuples(
        index=False
    ):
        event_start = pd.Timestamp(event.start_date)
        event_end = pd.Timestamp(event.end_date)
        event_category = str(event.cate_level4_code or "")
        eligible = (week_start <= event_end) & (week_end >= event_start)
        if event_category:
            eligible &= row_categories.eq(event_category)
        overlap_start = week_start.where(week_start >= event_start, event_start)
        overlap_end = week_end.where(week_end <= event_end, event_end)
        overlap = (overlap_end - overlap_start).dt.days.add(1).to_numpy()
        update = eligible.to_numpy() & (overlap > best_overlap)
        best_overlap[update] = overlap[update]
        best_code[update] = str(event.event_code)
    result["event_code"] = best_code
    return result


def neutralize_period_effects(weekly: pd.DataFrame) -> pd.DataFrame:
    """Apply V3.2 H(c,e) without hard-coding any business event calendar."""

    result = weekly.copy()
    result["period_factor_h"] = 1.0
    result["period_factor_status"] = np.where(
        result.get("event_code", pd.Series("", index=result.index)).astype(str) == "",
        "neutral_no_event",
        "neutral_insufficient_neighbors",
    )
    if result.empty or "event_code" not in result:
        result["v_neutral"] = result.get("v_star", pd.Series(dtype=float))
        return result
    if not result["event_code"].astype(str).ne("").any():
        result["v_neutral"] = result["v_star"]
        return result
    evidence = period_effect_evidence(result)
    factor_series = evidence.groupby(
        ["cate_level4_code", "event_code"]
    )["event_ratio"].median()
    return apply_period_factors(result, factor_series)


def period_effect_evidence(weekly: pd.DataFrame) -> pd.DataFrame:
    """Return per-SKU event ratios whose category medians define H(c,e)."""

    if weekly.empty or not weekly.get(
        "event_code", pd.Series("", index=weekly.index)
    ).astype(str).ne("").any():
        return pd.DataFrame(
            columns=["cate_level4_code", "event_code", "event_ratio"]
        )
    aggregate_source = weekly.copy()
    aggregate_source["aggregate_weight"] = aggregate_source["n_eff"].clip(
        lower=1e-9
    )
    aggregate_source["weighted_v_all"] = (
        aggregate_source["v_star"] * aggregate_source["aggregate_weight"]
    )
    aggregate = (
        aggregate_source.groupby(
            ["sku_code", "cate_level4_code", "lifecycle_week", "event_code"],
            as_index=False,
            dropna=False,
        )
        .agg(
            weighted_v_all=("weighted_v_all", "sum"),
            aggregate_weight=("aggregate_weight", "sum"),
        )
        .reset_index(drop=True)
    )
    aggregate["v_all"] = (
        aggregate["weighted_v_all"] / aggregate["aggregate_weight"]
    )
    aggregate = aggregate.sort_values(["sku_code", "lifecycle_week"])
    aggregate["v_previous"] = aggregate.groupby("sku_code")["v_all"].shift(1)
    aggregate["v_next"] = aggregate.groupby("sku_code")["v_all"].shift(-1)
    aggregate["neighbor_median"] = aggregate[["v_previous", "v_next"]].median(
        axis=1, skipna=False
    )
    valid = (
        aggregate["event_code"].astype(str).ne("")
        & aggregate["neighbor_median"].notna()
        & (aggregate["neighbor_median"] > 0)
    )
    aggregate["event_ratio"] = np.where(
        valid, aggregate["v_all"] / aggregate["neighbor_median"], np.nan
    )
    return aggregate.loc[
        valid, ["cate_level4_code", "event_code", "event_ratio"]
    ].reset_index(drop=True)


def apply_period_factors(
    weekly: pd.DataFrame,
    factors: Mapping[tuple[Any, Any], float] | pd.Series,
) -> pd.DataFrame:
    """Apply already-estimated category/event factors to weekly rows."""

    result = weekly.copy()
    result["period_factor_h"] = 1.0
    result["period_factor_status"] = np.where(
        result.get("event_code", pd.Series("", index=result.index)).astype(str) == "",
        "neutral_no_event",
        "neutral_insufficient_neighbors",
    )
    if isinstance(factors, pd.Series):
        factor_series = factors
    else:
        factor_series = pd.Series(dict(factors), dtype=float)
        if len(factor_series):
            factor_series.index = pd.MultiIndex.from_tuples(factor_series.index)
    keys = pd.MultiIndex.from_frame(result[["cate_level4_code", "event_code"]])
    mapped = factor_series.reindex(keys).to_numpy(float)
    estimated = np.isfinite(mapped) & (mapped > 0)
    result.loc[estimated, "period_factor_h"] = mapped[estimated]
    result.loc[estimated, "period_factor_status"] = "estimated"
    result["v_neutral"] = result["v_star"] / result["period_factor_h"]
    return result


def aggregate_store_tiers(weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty:
        return pd.DataFrame()
    frame = weekly.copy()
    frame["tier_weight"] = frame["restored_exposure_days"].clip(lower=0)
    frame["weighted_v_neutral"] = frame["v_neutral"] * frame["tier_weight"]
    keys = [
        "sku_code",
        "cate_level4_code",
        "store_business_state",
        "store_level",
        "store_tier",
        "lifecycle_week",
        "week_start",
        "week_end",
        "event_code",
    ]
    tier = (
        frame.groupby(keys, as_index=False, dropna=False)
        .agg(
            store_count=("store_code", "nunique"),
            restored_natural_sales=("restored_natural_sales", "sum"),
            restored_exposure_days=("restored_exposure_days", "sum"),
            n_eff=("n_eff", "sum"),
            stockout_days=("stockout_days", "sum"),
            nondisplay_days=("nondisplay_days", "sum"),
            unknown_exposure_days=("unknown_exposure_days", "sum"),
            weighted_v_neutral=("weighted_v_neutral", "sum"),
            tier_weight=("tier_weight", "sum"),
            mean_v_neutral=("v_neutral", "mean"),
            period_factor_h=("period_factor_h", "median"),
        )
        .reset_index(drop=True)
    )
    tier["v_neutral"] = np.where(
        tier["tier_weight"] > 0,
        tier["weighted_v_neutral"] / tier["tier_weight"],
        tier["mean_v_neutral"],
    )
    tier["tier_aggregation_fallback"] = np.where(
        tier["tier_weight"] > 0, "restored_exposure_weighted", "model_equal_weight"
    )
    return tier.drop(
        columns=["weighted_v_neutral", "tier_weight", "mean_v_neutral"]
    )


def build_restoration_quality_report(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    requested_fit_skus: set[str],
    model_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize tolerated source variation without silently defaulting it."""

    flag_specs = {
        "inventory_out_of_stock": (
            "out_of_stock_flag_x",
            "inventory_stockout_flag",
            "inventory_stockout_flag_unrecognized",
        ),
        "placement_out_of_stock": (
            "out_of_stock_flag_y",
            "placement_stockout_flag",
            "placement_stockout_flag_unrecognized",
        ),
        "in_stock_not_displayed": (
            "in_stock_not_displayed_flag",
            "nondisplay_flag",
            "nondisplay_flag_unrecognized",
        ),
    }
    flag_audits: dict[str, dict[str, object]] = {}
    for label, (raw_column, normalized_column, unknown_column) in flag_specs.items():
        flag_audits[label] = summarize_flag_normalization(
            raw_values=daily[raw_column],
            normalized=daily[normalized_column],
            unrecognized=daily[unknown_column].fillna(False).astype(bool),
        )

    component_unknown = daily[["e_open", "e_stock", "e_display"]].isna().any(axis=1)
    open_rows = daily["e_open"].eq(1).fillna(False).astype(bool)
    open_row_count = int(open_rows.sum())
    unknown_exposure = (open_rows & daily["e_total"].isna()).fillna(False)
    positive_sale = daily["clean_sale_qty"].gt(0).fillna(False).astype(bool)
    stockout = daily["e_stock"].eq(0).fillna(False).astype(bool)
    stock_conflict = daily["stock_state_conflict"].fillna(False).astype(bool)

    daily_audit = pd.DataFrame(
        {
            "sku_code": daily["sku_code"].astype(str),
            "daily_rows": 1,
            "open_days": open_rows.astype("int8"),
            "stockout_days": (open_rows & stockout).astype("int8"),
            "normal_exposure_days": daily["e_total"].eq(1).fillna(False).astype("int8"),
            "unknown_exposure_days": unknown_exposure.astype("int8"),
            "positive_sales_on_stockout_days": (positive_sale & stockout).astype("int8"),
            "stock_state_conflict_days": stock_conflict.astype("int8"),
        }
    )
    per_sku_frame = daily_audit.groupby(
        "sku_code", as_index=False, sort=False, observed=True
    ).sum()
    weekly_fit_counts = (
        weekly.assign(
            __fit=weekly["model_fit_eligible"].fillna(False).astype("int8")
        )
        .groupby("sku_code", observed=True)["__fit"]
        .sum()
    )
    per_sku_frame["model_fit_eligible_weeks"] = (
        per_sku_frame["sku_code"].map(weekly_fit_counts).fillna(0).astype(int)
    )
    per_sku_frame["all_open_days_stockout"] = (
        per_sku_frame["open_days"].gt(0)
        & per_sku_frame["stockout_days"].eq(per_sku_frame["open_days"])
    )
    all_stockout_skus = sorted(
        per_sku_frame.loc[
            per_sku_frame["all_open_days_stockout"], "sku_code"
        ].astype(str)
    )
    integer_columns = [
        name
        for name in per_sku_frame.columns
        if name not in {"sku_code", "all_open_days_stockout"}
    ]
    per_sku_frame[integer_columns] = per_sku_frame[integer_columns].astype(int)
    per_sku = per_sku_frame.to_dict(orient="records")

    warnings: list[str] = []
    for label, audit in flag_audits.items():
        if int(audit["unrecognized_rows"]) > 0:
            warnings.append(
                f"{label} has {audit['unrecognized_rows']} non-null unmapped values; "
                "they remain unknown and are listed in data_quality.flag_normalization"
            )
    stock_conflict_days = int(stock_conflict.sum())
    if stock_conflict_days:
        warnings.append(
            f"{stock_conflict_days} rows contain contradictory stock evidence; "
            "Estock remains unknown for those rows"
        )
    unknown_exposure_days = int(unknown_exposure.sum())
    unknown_exposure_rate = (
        unknown_exposure_days / open_row_count if open_row_count else 0.0
    )
    if unknown_exposure_rate > 0.05:
        warnings.append(
            f"unknown exposure rate is {unknown_exposure_rate:.2%}, above the 5% review threshold"
        )
    if all_stockout_skus:
        warnings.append(
            "all open days are stockout for selected SKUs: "
            + ", ".join(sorted(all_stockout_skus))
        )
    fitted_sku_count = int(model_report.get("fit_sku_count", 0))
    if fitted_sku_count < len(requested_fit_skus):
        warnings.append(
            f"only {fitted_sku_count} of {len(requested_fit_skus)} requested Train SKUs "
            "have eligible exposure rows for model fitting"
        )
    if fitted_sku_count < 2:
        warnings.append(
            "fewer than two Train SKUs entered model fitting; treat this as a bounded "
            "pipeline check rather than production model validation"
        )
    near_zero_model_rows = int((weekly["v_model"].abs() <= 1e-8).sum())
    if len(weekly) and near_zero_model_rows == len(weekly):
        warnings.append(
            "all v_model predictions are effectively zero; this run validates plumbing "
            "but not lost-sales restoration"
        )

    return {
        "status": "warning" if warnings else "ok",
        "policy": {
            "known_aliases": "normalized to 1/0 by source-specific vocabularies",
            "unrecognized_non_null_values": "preserved as NA and audited",
            "contradictory_stock_sources": "preserved as NA; no source wins silently",
            "three_state_exposure": "any 0 -> 0; all 1 -> 1; otherwise NA",
        },
        "warnings": warnings,
        "flag_normalization": flag_audits,
        "exposure": {
            "component_unknown_days": int(component_unknown.sum()),
            "open_days": open_row_count,
            "unknown_exposure_days": unknown_exposure_days,
            "unknown_exposure_rate": unknown_exposure_rate,
            "stockout_days": int(stockout.sum()),
            "stockout_rate": float(stockout.sum() / len(daily)) if len(daily) else 0.0,
            "positive_sales_on_stockout_days": int((positive_sale & stockout).sum()),
            "positive_sales_qty_on_stockout_days": float(
                daily.loc[positive_sale & stockout, "clean_sale_qty"].sum()
            ),
            "stock_state_conflict_days": stock_conflict_days,
            "all_stockout_skus": sorted(all_stockout_skus),
        },
        "model_gate": {
            "requested_train_sku_count": int(len(requested_fit_skus)),
            "fitted_train_sku_count": fitted_sku_count,
            "minimum_fit_sku_review_threshold": 2,
            "near_zero_model_rows": near_zero_model_rows,
            "weekly_rows": int(len(weekly)),
            "restored_lost_sales_total": float(weekly["restored_lost_sales"].sum()),
        },
        "per_sku": per_sku,
    }


def build_restoration_result(
    *,
    raw_sales: pd.DataFrame,
    inventory: pd.DataFrame,
    placement: pd.DataFrame,
    stores: pd.DataFrame,
    product_attributes: pd.DataFrame,
    first_sale_dates: Mapping[str, date],
    fit_skus: set[str],
    training_weights: Mapping[str, float],
    cutoff_time: datetime,
    window_days: int,
    config: RestorationConfig,
    events: pd.DataFrame | None = None,
) -> RestorationResult:
    latest_lines, order_stats = normalize_order_lines(
        raw_sales, cutoff_time=cutoff_time
    )
    sales_daily, anomaly_report = prepare_sales_daily(
        latest_lines,
        product_attributes,
        abnormal_order_quantile=config.abnormal_order_quantile,
        threshold_fit_skus=fit_skus,
    )
    daily = build_daily_panel(
        sales_daily=sales_daily,
        inventory=inventory,
        placement=placement,
        stores=stores,
        first_sale_dates=first_sale_dates,
        cutoff_date=cutoff_time.date(),
        window_days=window_days,
        promotion_price_ratio_threshold=config.promotion_price_ratio_threshold,
    )
    weekly, model_report = fit_and_restore_weekly(
        daily,
        config=config,
        fit_skus=fit_skus,
        training_weights=training_weights,
    )
    weekly = assign_event_codes(weekly, events)
    weekly = neutralize_period_effects(weekly)
    tier_weekly = aggregate_store_tiers(weekly)
    quality_report = build_restoration_quality_report(
        daily,
        weekly,
        requested_fit_skus=fit_skus,
        model_report=model_report,
    )
    version_payload = {
        "spec_version": "3.2",
        "normalization_version": "source-flags-v2",
        "cutoff_time": cutoff_time.isoformat(),
        "window_days": window_days,
        "config": asdict(config),
        "skus": sorted(first_sale_dates),
        "weekly_rows": len(weekly),
        "weekly_total": float(weekly["restored_natural_sales"].sum()),
    }
    digest = hashlib.sha256(
        json.dumps(version_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    stats = RestorationStats(
        input_order_rows=int(order_stats["input_order_rows"]),
        latest_order_lines=int(order_stats["latest_order_lines"]),
        exact_duplicate_rows=int(order_stats["exact_duplicate_rows"]),
        signed_aggregation_extra_rows=int(
            order_stats["signed_aggregation_extra_rows"]
        ),
        abnormal_order_lines=int(anomaly_report["abnormal_order_lines"]),
        daily_panel_rows=len(daily),
        weekly_rows=len(weekly),
        tier_weekly_rows=len(tier_weekly),
        unknown_exposure_days=int(weekly["unknown_exposure_days"].sum()),
        stockout_days=int(weekly["stockout_days"].sum()),
        nondisplay_days=int(weekly["nondisplay_days"].sum()),
        event_adjusted_rows=int((weekly["period_factor_status"] == "estimated").sum()),
    )
    model_report["abnormal_order_policy"] = {
        "quantile": config.abnormal_order_quantile,
        "strictly_above_threshold": True,
        "thresholds": anomaly_report["abnormal_thresholds"],
    }
    return RestorationResult(
        daily=daily,
        weekly=weekly,
        tier_weekly=tier_weekly,
        stats=stats,
        model_report=model_report,
        quality_report=quality_report,
        dataset_version=f"m2-v3.2-{cutoff_time.date().isoformat()}-{digest[:16]}",
    )


def write_restoration_artifacts(
    *,
    output_dir: Path,
    result: RestorationResult,
    report: dict[str, Any],
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_dir / "round4_report.json",
        "daily": output_dir / "daily_exposure_audit.csv",
        "weekly": output_dir / "restored_store_weekly.csv",
        "tier_weekly": output_dir / "restored_tier_weekly.csv",
    }
    payload = dict(report)
    payload["dataset_version"] = result.dataset_version
    payload["stats"] = asdict(result.stats)
    payload["model"] = result.model_report
    payload["data_quality"] = result.quality_report
    for key, frame in (
        ("daily", result.daily),
        ("weekly", result.weekly),
        ("tier_weekly", result.tier_weekly),
    ):
        target = paths[key]
        temporary = target.with_suffix(target.suffix + ".tmp")
        if progress is not None:
            progress(
                "artifact_write_start",
                {"artifact": key, "rows": len(frame), "path": str(target)},
            )
        started = monotonic()
        frame.to_csv(temporary, index=False)
        temporary.replace(target)
        if progress is not None:
            progress(
                "artifact_write_complete",
                {
                    "artifact": key,
                    "rows": len(frame),
                    "elapsed_seconds": round(monotonic() - started, 3),
                },
            )
    # The report is the success marker and is therefore committed only after
    # all data artifacts have been atomically replaced.
    if progress is not None:
        progress(
            "artifact_write_start",
            {"artifact": "report", "path": str(paths["report"])},
        )
    started = monotonic()
    report_tmp = paths["report"].with_suffix(".json.tmp")
    report_tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_tmp.replace(paths["report"])
    if progress is not None:
        progress(
            "artifact_write_complete",
            {
                "artifact": "report",
                "elapsed_seconds": round(monotonic() - started, 3),
            },
        )
    return paths
