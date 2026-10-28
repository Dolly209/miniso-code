from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from data.domain import require_columns


def build_target_store_features(
    *,
    targets: pd.DataFrame,
    stores: pd.DataFrame,
    restored_store_weekly: pd.DataFrame,
    cutoff_date: object,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Compute V3.2 M4 CategoryFit and execution-prior store features."""

    require_columns(targets, ("sku_code", "cate_level4_code"), "placement targets")
    require_columns(stores, ("store_code",), "placement stores")
    require_columns(
        restored_store_weekly,
        (
            "sku_code",
            "cate_level4_code",
            "store_code",
            "week_end",
            "stockout_days",
            "nondisplay_days",
            "source_day_rows",
        ),
        "restored store weekly",
    )
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    cutoff = pd.to_datetime(cutoff_date).date()
    start = cutoff - timedelta(days=lookback_days - 1)
    history = restored_store_weekly.copy()
    history["week_end"] = pd.to_datetime(history["week_end"], errors="coerce").dt.date
    history = history.loc[history["week_end"].between(start, cutoff)].copy()
    for column in (
        "restored_exposure_days",
        "v_neutral",
        "stockout_days",
        "nondisplay_days",
        "source_day_rows",
    ):
        if column not in history:
            history[column] = np.nan
        history[column] = pd.to_numeric(history[column], errors="coerce")

    target_rows = targets[["sku_code", "cate_level4_code"]].drop_duplicates("sku_code")
    target_rows = target_rows.rename(columns={"sku_code": "target_sku_code"}).assign(_key=1)
    universe = stores[["store_code"]].drop_duplicates("store_code").assign(_key=1)
    output = target_rows.merge(universe, on="_key").drop(columns="_key")
    if history.empty:
        return output.drop(columns="cate_level4_code").assign(
            category_fit=np.nan,
            execution=np.nan,
            category_fit_status="missing_no_90d_restore_history",
            execution_status="missing_no_90d_restore_history",
        )

    history["weighted_velocity"] = (
        history["v_neutral"] * history["restored_exposure_days"].clip(lower=0)
    )
    category_store = history.groupby(
        ["cate_level4_code", "store_code"], as_index=False, dropna=False
    ).agg(
        weighted_velocity=("weighted_velocity", "sum"),
        exposure_days=("restored_exposure_days", "sum"),
        mean_velocity=("v_neutral", "mean"),
    )
    category_store["category_velocity"] = np.where(
        category_store["exposure_days"] > 0,
        category_store["weighted_velocity"] / category_store["exposure_days"],
        category_store["mean_velocity"],
    )
    category_store["category_fit"] = category_store.groupby(
        "cate_level4_code", dropna=False
    )["category_velocity"].rank(method="average", pct=True)

    execution = history.groupby("store_code", as_index=False).agg(
        stockout_days=("stockout_days", "sum"),
        nondisplay_days=("nondisplay_days", "sum"),
        source_day_rows=("source_day_rows", "sum"),
        earliest_week=("week_end", "min"),
        latest_week=("week_end", "max"),
    )
    execution["instock_days"] = (
        execution["source_day_rows"] - execution["stockout_days"]
    ).clip(lower=0)
    execution["execution"] = np.where(
        execution["instock_days"] > 0,
        1.0 - execution["nondisplay_days"] / execution["instock_days"],
        np.nan,
    )
    execution["execution"] = execution["execution"].clip(0.0, 1.0)
    execution["execution_status"] = np.where(
        execution["earliest_week"].le(start + timedelta(days=6))
        & execution["latest_week"].ge(cutoff - timedelta(days=6)),
        "v3_2_complete_execution_universe",
        "partial_90d_execution_history",
    )

    output = output.merge(
        category_store[["cate_level4_code", "store_code", "category_fit"]],
        on=["cate_level4_code", "store_code"],
        how="left",
        validate="many_to_one",
    ).merge(
        execution[["store_code", "execution", "execution_status"]],
        on="store_code",
        how="left",
        validate="many_to_one",
    )
    category_coverage = history.groupby("cate_level4_code")["week_end"].agg(["min", "max"])
    complete_categories = set(
        category_coverage.loc[
            category_coverage["min"].le(start + timedelta(days=6))
            & category_coverage["max"].ge(cutoff - timedelta(days=6))
        ].index.astype(str)
    )
    output["category_fit_status"] = np.where(
        output["category_fit"].isna(),
        "missing_store_category_90d_history",
        np.where(
            output["cate_level4_code"].astype(str).isin(complete_categories),
            "v3_2_complete_90d",
            "partial_90d_category_history",
        ),
    )
    output["execution_status"] = output["execution_status"].fillna(
        "missing_store_execution_history"
    )
    return output.drop(columns="cate_level4_code")
