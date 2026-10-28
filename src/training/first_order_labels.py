from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.domain import require_columns
from data.launch_contract import validate_launch_origin_contract
from data.prelaunch_artifacts import normalize_prelaunch_audit_metadata
from training.split import SkuSample, chronological_ratio_split
from training.weights import long_tail_weights, sales_band
from restore.historical import normalize_order_lines


LABEL_COLUMN = "D_first_ref"
LABEL_DEFINITION = (
    "sum(restored_natural_sales) over lifecycle weeks 1..W_cover after "
    "V3.2 historical operating-condition restoration"
)


@dataclass(frozen=True)
class FirstOrderLabelResult:
    labels: pd.DataFrame
    formal_schedule: pd.DataFrame
    excluded_skus: pd.DataFrame
    report: dict[str, Any]


def build_wcover_windows(
    *,
    schedule: pd.DataFrame,
    first_order_config: Mapping[str, Any],
    constraints: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one SKU-level label window before any sales rows are read."""

    skus = _one_row_per_sku(schedule)
    cover = _cover_parameters(skus, constraints, first_order_config)
    windows = skus.merge(cover, on="sku_code", how="left", validate="one_to_one")
    windows["label_start_date"] = pd.to_datetime(
        windows["first_actual_sale_date"], errors="raise"
    ).dt.date
    windows["label_end_date"] = [
        start + timedelta(days=int(weeks) * 7 - 1)
        for start, weeks in zip(windows["label_start_date"], windows["W_cover"])
    ]
    return windows


def actual_sales_weekly_from_raw(
    *,
    raw_sales: pd.DataFrame,
    windows: pd.DataFrame,
    cutoff_time: datetime,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize signed order lines and make an explicit zero-filled week grid."""

    require_columns(
        windows,
        ("sku_code", "label_start_date", "label_end_date", "W_cover"),
        "W_cover label windows",
    )
    latest, audit = normalize_order_lines(raw_sales, cutoff_time=cutoff_time)
    window = windows[
        ["sku_code", "label_start_date", "label_end_date", "W_cover"]
    ].copy()
    window["sku_code"] = window["sku_code"].astype(str)
    window["label_start_date"] = pd.to_datetime(
        window["label_start_date"], errors="raise"
    ).dt.date
    window["label_end_date"] = pd.to_datetime(
        window["label_end_date"], errors="raise"
    ).dt.date
    if latest.empty:
        observed = pd.DataFrame(
            columns=["sku_code", "lifecycle_week", "clean_sales_qty"]
        )
    else:
        lines = latest.merge(
            window,
            on="sku_code",
            how="inner",
            validate="many_to_one",
        )
        in_window = (
            lines["business_day"].ge(lines["label_start_date"])
            & lines["business_day"].le(lines["label_end_date"])
        )
        lines = lines.loc[in_window].copy()
        lines["lifecycle_week"] = [
            (day - start).days // 7 + 1
            for day, start in zip(
                lines["business_day"], lines["label_start_date"]
            )
        ]
        observed = (
            lines.groupby(["sku_code", "lifecycle_week"], as_index=False)
            .agg(clean_sales_qty=("sale_qty", "sum"))
        )
    grid_rows = [
        {
            "sku_code": str(row.sku_code),
            "lifecycle_week": week,
            "week_end": row.label_start_date + timedelta(days=week * 7 - 1),
        }
        for row in window.itertuples(index=False)
        for week in range(1, int(row.W_cover) + 1)
    ]
    grid = pd.DataFrame(grid_rows)
    weekly = grid.merge(
        observed,
        on=["sku_code", "lifecycle_week"],
        how="left",
        validate="one_to_one",
    )
    weekly["clean_sales_qty"] = weekly["clean_sales_qty"].fillna(0.0)
    weekly["restored_natural_sales"] = weekly["clean_sales_qty"]
    weekly["restored_lost_sales"] = 0.0
    weekly["stock_grain_unreliable_rows"] = 0
    weekly["unknown_exposure_days"] = 0.0
    weekly["complete_week"] = True
    return weekly, audit


def aggregate_weekly_label_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce store-week restoration output to one auditable SKU-week row.

    ``restored_natural_sales`` is the formal label source.  The signed
    ``clean_sales_qty`` total is retained as the observed-sales lower-bound
    audit.  This reduction is deliberately associative so callers can process
    large partitioned Round 4 outputs one file/chunk at a time.
    """

    required = (
        "sku_code",
        "lifecycle_week",
        "week_end",
        "clean_sales_qty",
        "restored_natural_sales",
    )
    require_columns(frame, required, "restored_store_weekly")
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "sku_code",
                "lifecycle_week",
                "week_end",
                "week_end_min",
                "clean_sales_qty",
                "restored_natural_sales",
                "restored_lost_sales",
                "stock_grain_unreliable_rows",
                "unknown_exposure_days",
                "source_day_rows",
                "restored_exposure_days",
                "long_stockout_rows",
                "complete_week",
            ]
        )
    work = frame.copy()
    work["sku_code"] = work["sku_code"].astype(str).str.strip()
    work["lifecycle_week"] = pd.to_numeric(
        work["lifecycle_week"], errors="coerce"
    )
    work["week_end"] = pd.to_datetime(work["week_end"], errors="coerce")
    work["clean_sales_qty"] = pd.to_numeric(
        work["clean_sales_qty"], errors="coerce"
    )
    work["restored_natural_sales"] = pd.to_numeric(work["restored_natural_sales"], errors="coerce")
    if (
        work["sku_code"].eq("").any()
        or work["lifecycle_week"].isna().any()
        or work["week_end"].isna().any()
        or work["clean_sales_qty"].isna().any()
        or not np.isfinite(work["restored_natural_sales"]).all()
        or not np.isfinite(work["clean_sales_qty"]).all()
    ):
        raise ValueError(
            "restored_store_weekly contains blank SKU or invalid lifecycle/week/sales values"
        )
    for column in (
        "restored_natural_sales",
        "restored_lost_sales",
        "unknown_exposure_days",
        "source_day_rows",
        "restored_exposure_days",
    ):
        if column not in work:
            work[column] = 0.0
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
    if "long_stockout_quality_weight" not in work:
        work["long_stockout_quality_weight"] = 1.0
    long_stockout_weight = pd.to_numeric(
        work["long_stockout_quality_weight"], errors="coerce"
    ).fillna(1.0)
    work["long_stockout_rows"] = long_stockout_weight.lt(1.0).astype(int)
    if "stock_grain_unreliable" not in work:
        work["stock_grain_unreliable"] = False
    unreliable = work["stock_grain_unreliable"]
    if unreliable.dtype != bool:
        unreliable = unreliable.astype(str).str.lower().isin({"true", "1"})
    work["stock_grain_unreliable_rows"] = unreliable.astype(int)
    if "complete_week" not in work:
        work["complete_week"] = True
    complete = work["complete_week"]
    if complete.dtype != bool:
        complete = complete.astype(str).str.lower().isin({"true", "1"})
    work["complete_week"] = complete
    return (
        work.groupby(["sku_code", "lifecycle_week"], as_index=False)
        .agg(
            week_end=("week_end", "max"),
            week_end_min=("week_end", "min"),
            clean_sales_qty=("clean_sales_qty", "sum"),
            restored_natural_sales=("restored_natural_sales", "sum"),
            restored_lost_sales=("restored_lost_sales", "sum"),
            stock_grain_unreliable_rows=("stock_grain_unreliable_rows", "sum"),
            unknown_exposure_days=("unknown_exposure_days", "sum"),
            source_day_rows=("source_day_rows", "sum"),
            restored_exposure_days=("restored_exposure_days", "sum"),
            long_stockout_rows=("long_stockout_rows", "sum"),
            complete_week=("complete_week", "all"),
        )
        .reset_index(drop=True)
    )


def combine_weekly_label_inputs(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(
            columns=[
                "sku_code",
                "lifecycle_week",
                "week_end",
                "week_end_min",
                "clean_sales_qty",
                "restored_natural_sales",
                "restored_lost_sales",
                "stock_grain_unreliable_rows",
                "unknown_exposure_days",
                "source_day_rows",
                "restored_exposure_days",
                "long_stockout_rows",
                "complete_week",
            ]
        )
    combined = pd.concat(parts, ignore_index=True)
    if "week_end_min" not in combined:
        combined["week_end_min"] = combined["week_end"]
    require_columns(
        combined,
        (
            "sku_code",
            "lifecycle_week",
            "week_end",
            "clean_sales_qty",
            "restored_natural_sales",
            "restored_lost_sales",
            "stock_grain_unreliable_rows",
            "unknown_exposure_days",
            "source_day_rows",
            "restored_exposure_days",
            "long_stockout_rows",
            "complete_week",
        ),
        "aggregated restored_store_weekly parts",
    )
    return (
        combined.groupby(["sku_code", "lifecycle_week"], as_index=False)
        .agg(
            week_end=("week_end", "max"),
            week_end_min=("week_end_min", "min"),
            clean_sales_qty=("clean_sales_qty", "sum"),
            restored_natural_sales=("restored_natural_sales", "sum"),
            restored_lost_sales=("restored_lost_sales", "sum"),
            stock_grain_unreliable_rows=("stock_grain_unreliable_rows", "sum"),
            unknown_exposure_days=("unknown_exposure_days", "sum"),
            source_day_rows=("source_day_rows", "sum"),
            restored_exposure_days=("restored_exposure_days", "sum"),
            long_stockout_rows=("long_stockout_rows", "sum"),
            complete_week=("complete_week", "all"),
        )
        .reset_index(drop=True)
    )


def _one_row_per_sku(schedule: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        schedule,
        ("sku_code", "first_actual_sale_date", "sales_26w_qty"),
        "formal prelaunch schedule",
    )
    work = schedule.copy()
    work["sku_code"] = work["sku_code"].astype(str).str.strip()
    work["first_actual_sale_date"] = pd.to_datetime(
        work["first_actual_sale_date"], errors="coerce", format="mixed"
    )
    work["sales_26w_qty"] = pd.to_numeric(
        work["sales_26w_qty"], errors="coerce"
    )
    invalid = (
        work["sku_code"].eq("")
        | work["first_actual_sale_date"].isna()
        | work["sales_26w_qty"].isna()
    )
    if invalid.any():
        raise ValueError(
            "formal prelaunch schedule contains invalid SKU, first-sale date, or 26-week audit quantity"
        )
    conflicts = work.groupby("sku_code", dropna=False).agg(
        first_sale_values=("first_actual_sale_date", "nunique"),
        sales_26w_values=("sales_26w_qty", "nunique"),
    )
    if conflicts.gt(1).any(axis=None):
        examples = conflicts.loc[conflicts.gt(1).any(axis=1)].head(5).index.tolist()
        raise ValueError(f"schedule has conflicting SKU attributes: {examples}")
    preferred = [
        column
        for column in (
            "sku_code",
            "first_actual_sale_date",
            "sales_26w_qty",
            "dataset_version",
            "label_start_date",
        )
        if column in work
    ]
    return work[preferred].drop_duplicates("sku_code").reset_index(drop=True)


def _constraint_value(
    row: Mapping[str, Any],
    key: str,
    default: int,
) -> tuple[int, str]:
    value = row.get(key)
    if value is None or pd.isna(value):
        return int(default), "versioned_default"
    numeric = int(float(value))
    if numeric <= 0 and key != "buffer_weeks":
        raise ValueError(f"{key} must be positive")
    if numeric < 0:
        raise ValueError(f"{key} must be non-negative")
    return numeric, "business_constraint"


def _hard_weeks(
    row: Mapping[str, Any],
    *,
    launch_date: date,
    lifecycle_weeks: int,
) -> tuple[int, str]:
    explicit = row.get("hard_life_weeks")
    if explicit is not None and not pd.isna(explicit):
        value = int(float(explicit))
        if value <= 0:
            raise ValueError("hard_life_weeks must be positive")
        return value, "business_constraint"
    dates: list[pd.Timestamp] = []
    for column in ("sell_end_date", "license_end_date", "copyright_exp_date"):
        raw = row.get(column)
        if raw is None or pd.isna(raw):
            continue
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.notna(parsed):
            dates.append(parsed.normalize())
    if dates:
        hard_end = min(dates).date()
        days = (hard_end - launch_date).days
        return max(int(np.floor(days / 7.0)) + 1, 0), "derived_hard_end_date"
    return lifecycle_weeks, "not_available_bounded_by_lifecycle"


def _cover_parameters(
    sku_rows: pd.DataFrame,
    constraints: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    constraint_rows: dict[str, dict[str, Any]] = {}
    if constraints is not None and not constraints.empty:
        require_columns(constraints, ("sku_code",), "first-order constraints")
        source = constraints.copy()
        source["sku_code"] = source["sku_code"].astype(str).str.strip()
        if source["sku_code"].duplicated().any():
            raise ValueError("first-order constraints must contain one row per SKU")
        constraint_rows = source.set_index("sku_code").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for sku in sku_rows.itertuples(index=False):
        values = constraint_rows.get(str(sku.sku_code), {})
        decision, decision_source = _constraint_value(
            values, "decision_weeks", int(config["decision_weeks"])
        )
        buffer, buffer_source = _constraint_value(
            values, "buffer_weeks", int(config["buffer_weeks"])
        )
        lead, lead_source = _constraint_value(
            values, "lead_weeks", int(config["lead_weeks"])
        )
        lifecycle, lifecycle_source = _constraint_value(
            values, "lifecycle_weeks", int(config["lifecycle_weeks"])
        )
        launch = pd.Timestamp(sku.first_actual_sale_date).date()
        hard, hard_source = _hard_weeks(
            values, launch_date=launch, lifecycle_weeks=lifecycle
        )
        supply = decision + buffer + lead
        cover = min(supply, lifecycle, hard)
        rows.append(
            {
                "sku_code": str(sku.sku_code),
                "W_decision": decision,
                "W_buffer": buffer,
                "W_lead": lead,
                "W_supply": supply,
                "W_life": lifecycle,
                "W_hard": hard,
                "W_cover": cover,
                "decision_weeks_source": decision_source,
                "buffer_weeks_source": buffer_source,
                "lead_weeks_source": lead_source,
                "lifecycle_weeks_source": lifecycle_source,
                "hard_weeks_source": hard_source,
            }
        )
    return pd.DataFrame(rows)


def build_first_order_wcover_labels(
    *,
    schedule: pd.DataFrame,
    weekly: pd.DataFrame,
    first_order_config: Mapping[str, Any],
    available_through: datetime,
    constraints: pd.DataFrame | None = None,
    train_ratio: float = 0.8,
) -> FirstOrderLabelResult:
    """Create V3.2 section 8.1 labels and re-split the feasible SKU set."""

    if available_through.tzinfo is None or available_through.utcoffset() is None:
        raise ValueError("available_through must be timezone-aware")
    schedule = normalize_prelaunch_audit_metadata(schedule)
    validate_launch_origin_contract(schedule, name="W_cover label schedule")
    labels = build_wcover_windows(
        schedule=schedule,
        first_order_config=first_order_config,
        constraints=constraints,
    )
    labels["label_mature_at_available_through"] = labels["label_end_date"].map(
        lambda value: value <= available_through.date()
    )

    require_columns(
        weekly,
        (
            "sku_code",
            "lifecycle_week",
            "week_end",
            "clean_sales_qty",
            "restored_natural_sales",
            "restored_lost_sales",
            "stock_grain_unreliable_rows",
            "unknown_exposure_days",
            "restored_exposure_days",
            "long_stockout_rows",
            "complete_week",
        ),
        "aggregated restored_store_weekly",
    )
    source = weekly.copy()
    source["sku_code"] = source["sku_code"].astype(str).str.strip()
    source["lifecycle_week"] = pd.to_numeric(
        source["lifecycle_week"], errors="coerce"
    )
    source = source.merge(
        labels[["sku_code", "W_cover", "label_start_date"]],
        on="sku_code",
        how="inner",
        validate="many_to_one",
    )
    source = source.loc[
        source["lifecycle_week"].between(1, source["W_cover"])
    ].copy()
    if source.duplicated(["sku_code", "lifecycle_week"]).any():
        raise ValueError("label inputs must have one aggregated row per SKU/lifecycle_week")
    expected_end = pd.to_datetime(source["label_start_date"]) + pd.to_timedelta(source["lifecycle_week"] * 7 - 1, unit="D")
    earliest_end = pd.to_datetime(source.get("week_end_min", source["week_end"]), errors="coerce")
    source["origin_mismatch_weeks"] = (
        ~pd.to_datetime(source["week_end"], errors="coerce").eq(expected_end)
        | ~earliest_end.eq(expected_end)
    ).astype(int)
    if source.empty:
        aggregates = pd.DataFrame(columns=["sku_code"])
    else:
        aggregates = (
            source.groupby("sku_code", as_index=False)
            .agg(
                observed_lifecycle_weeks=("lifecycle_week", "nunique"),
                max_observed_week=("lifecycle_week", "max"),
                max_observed_week_end=("week_end", "max"),
                observed_clean_sales_wcover=("clean_sales_qty", "sum"),
                D_first_ref=("restored_natural_sales", "sum"),
                restored_demand_wcover=("restored_natural_sales", "sum"),
                restored_lost_sales_wcover=("restored_lost_sales", "sum"),
                stock_grain_unreliable_rows=(
                    "stock_grain_unreliable_rows",
                    "sum",
                ),
                unknown_exposure_days=("unknown_exposure_days", "sum"),
                source_day_rows=("source_day_rows", "sum"),
                restored_exposure_days=("restored_exposure_days", "sum"),
                long_stockout_rows=("long_stockout_rows", "sum"),
                complete_weeks=("complete_week", "sum"),
                origin_mismatch_weeks=("origin_mismatch_weeks", "sum"),
            )
        )
    labels = labels.merge(aggregates, on="sku_code", how="left", validate="one_to_one")
    numeric_fill_zero = (
        "observed_lifecycle_weeks",
        "max_observed_week",
        "observed_clean_sales_wcover",
        LABEL_COLUMN,
        "restored_demand_wcover",
        "restored_lost_sales_wcover",
        "stock_grain_unreliable_rows",
        "unknown_exposure_days",
        "source_day_rows",
        "restored_exposure_days",
        "long_stockout_rows",
        "complete_weeks",
        "origin_mismatch_weeks",
    )
    for column in numeric_fill_zero:
        if column not in labels:
            labels[column] = 0.0
        labels[column] = pd.to_numeric(labels[column], errors="coerce").fillna(0.0)
    labels["full_wcover_week_coverage"] = (
        labels["observed_lifecycle_weeks"].eq(labels["W_cover"])
        & labels["max_observed_week"].eq(labels["W_cover"])
        & labels["complete_weeks"].eq(labels["W_cover"])
    )
    labels["label_nonnegative"] = labels[LABEL_COLUMN].ge(0)
    labels["restoration_window_aligned"] = labels["origin_mismatch_weeks"].eq(0)
    labels["restoration_uplift_wcover"] = (
        labels[LABEL_COLUMN] - labels["observed_clean_sales_wcover"]
    )
    labels["open_exposure_audit_days"] = (
        labels["restored_exposure_days"] + labels["unknown_exposure_days"]
    )
    labels["unknown_exposure_rate"] = np.where(
        labels["open_exposure_audit_days"].gt(0),
        labels["unknown_exposure_days"] / labels["open_exposure_audit_days"],
        np.where(labels["unknown_exposure_days"].gt(0), 1.0, 0.0),
    )
    max_unknown_exposure_rate = float(
        first_order_config.get("max_unknown_exposure_rate_for_training", 0.05)
    )
    if not 0 <= max_unknown_exposure_rate <= 1:
        raise ValueError(
            "max_unknown_exposure_rate_for_training must be between 0 and 1"
        )
    labels["restoration_available"] = labels[
        "stock_grain_unreliable_rows"
    ].eq(0)
    # Compatibility flag is heuristic only; preserve the current training hold
    # without claiming that opening-stock-zero plus sales proves a grain error.
    labels["inventory_risk_pending_review"] = ~labels["restoration_available"]
    labels["inventory_confirmed_unrestorable"] = False
    labels["missingness_within_training_limit"] = labels[
        "unknown_exposure_rate"
    ].le(max_unknown_exposure_rate)
    labels["label_eligible"] = (
        labels["label_mature_at_available_through"]
        & labels["full_wcover_week_coverage"]
        & labels["restoration_window_aligned"]
        & labels["label_nonnegative"]
        & labels["restoration_available"]
        & labels["missingness_within_training_limit"]
    )
    labels["label_definition"] = LABEL_DEFINITION
    labels["label_source_column"] = "restored_natural_sales"
    labels["label_quantity_scope"] = "pure_sales_demand"
    labels["label_quantity_unit"] = (
        "source_SKU_sales_piece_unit; no unconfirmed case/box conversion"
    )
    labels["label_window_rule"] = (
        "inclusive first_actual_sale_date through first_actual_sale_date + "
        "7*W_cover - 1 day"
    )
    labels["observed_sales_audit_column"] = "clean_sales_qty"
    labels["restoration_is_formal_label_source"] = True
    labels["label_quality_status"] = np.select(
        [
            ~labels["label_mature_at_available_through"],
            ~labels["full_wcover_week_coverage"],
            ~labels["restoration_window_aligned"],
            ~labels["label_nonnegative"],
            ~labels["restoration_available"],
            ~labels["missingness_within_training_limit"],
            labels["long_stockout_rows"].gt(0),
            labels["unknown_exposure_days"].gt(0),
        ],
        [
            "immature_wcover",
            "incomplete_wcover_week_coverage",
            "restoration_lifecycle_origin_mismatch",
            "negative_restored_wcover_demand",
            "held_pending_inventory_risk_review",
            "excluded_large_missing_exposure",
            "eligible_restored_demand_with_long_stockout_downweight",
            "eligible_restored_demand_with_minor_unknown_exposure",
        ],
        default="eligible_restored_demand",
    )

    eligible = labels.loc[labels["label_eligible"]].copy()
    samples = [
        SkuSample(pd.Timestamp(row.first_actual_sale_date).date(), str(row.sku_code))
        for row in eligible.itertuples(index=False)
    ]
    split_ready = len({sample.first_actual_sale_date for sample in samples}) >= 2
    train, test, boundary = chronological_ratio_split(samples, train_ratio=train_ratio) if split_ready else ((), (), None)
    split_by_sku = {row.sku_code: "Train" for row in train}
    split_by_sku.update({row.sku_code: "Test" for row in test})
    labels["split"] = labels["sku_code"].map(split_by_sku)
    labels["temporal_split_boundary_date"] = boundary.isoformat() if boundary else None
    labels["sales_band_26w"] = labels["sales_26w_qty"].map(sales_band)
    train_26w = {
        str(row.sku_code): float(row.sales_26w_qty)
        for row in labels.loc[labels["split"].eq("Train")].itertuples(index=False)
    }
    weights = {row.sku_code: row.weight for row in long_tail_weights(train_26w)} if train_26w else {}
    labels["long_tail_weight"] = labels["sku_code"].map(weights)
    labels["weight_source"] = np.where(
        labels["split"].eq("Train"),
        "Train-only 26-week sales-band weighting",
        "locked Test; no fitting weight",
    )

    eligible_columns = [
        "sku_code",
        "split",
        "long_tail_weight",
        "temporal_split_boundary_date",
        LABEL_COLUMN,
        "W_cover",
        "label_start_date",
        "label_end_date",
        "label_definition",
        "label_quality_status",
    ]
    formal_schedule = schedule.drop(
        columns=[
            column
            for column in (
                "split",
                "long_tail_weight",
                LABEL_COLUMN,
                "temporal_split_boundary_date",
                "label_start_date",
                "W_cover",
                "label_end_date",
                "label_definition",
                "label_quality_status",
            )
            if column in schedule
        ]
    ).merge(
        labels.loc[labels["label_eligible"] & labels["split"].notna(), eligible_columns],
        on="sku_code",
        how="inner",
        validate="many_to_one",
    )
    formal_schedule["training_label_column"] = LABEL_COLUMN
    formal_schedule["training_ready_for_feature_replay"] = True

    excluded = labels.loc[~labels["label_eligible"]].copy()
    report = {
        "spec_version": "3.2",
        "status": "wcover_labels_built" if split_ready else "label_audit_only_insufficient_distinct_eligible_dates",
        "split_ready": split_ready,
        "source_dataset_versions": sorted(schedule["dataset_version"].astype(str).unique()) if "dataset_version" in schedule else [],
        "training_started": False,
        "test_consumed": False,
        "label_column": LABEL_COLUMN,
        "label_definition": LABEL_DEFINITION,
        "restoration_policy": (
            "D_first_ref is restored_natural_sales summed inside W_cover; "
            "clean_sales_qty is retained only as the observed lower-bound audit"
        ),
        "max_unknown_exposure_rate_for_training": max_unknown_exposure_rate,
        "input_skus": int(labels["sku_code"].nunique()),
        "eligible_skus": int(eligible["sku_code"].nunique()),
        "excluded_skus": int(len(excluded)),
        "train_skus": int(labels["split"].eq("Train").sum()),
        "test_skus": int(labels["split"].eq("Test").sum()),
        "checkpoint_rows": int(len(formal_schedule)),
        "temporal_split_boundary_date": boundary.isoformat() if boundary else None,
        "excluded_origin_mismatch_skus": int((~labels["restoration_window_aligned"]).sum()),
        "train_ratio_requested": float(train_ratio),
        "train_ratio_realized": float(
            labels["split"].eq("Train").sum() / max(eligible.shape[0], 1)
        ),
        "weighting": "V3.2 Train-only 26-week sales-band weighting",
        "default_constraint_skus": int(
            labels[
                [
                    "decision_weeks_source",
                    "buffer_weeks_source",
                    "lead_weeks_source",
                    "lifecycle_weeks_source",
                ]
            ]
            .eq("versioned_default")
            .any(axis=1)
            .sum()
        ),
        "stock_grain_warning_skus": int(
            labels["stock_grain_unreliable_rows"].gt(0).sum()
        ),
        "excluded_unrestorable_stock_grain_skus": int(
            (~labels["restoration_available"]).sum()
        ),
        "legacy_stock_grain_count_interpretation": "excluded_unrestorable_stock_grain_skus is a legacy counter for pending review holds, NOT confirmed inventory errors",
        "inventory_risk_pending_review_skus": int(labels["inventory_risk_pending_review"].sum()),
        "inventory_confirmed_unrestorable_skus": 0,
        "excluded_large_missing_exposure_skus": int(
            (~labels["missingness_within_training_limit"]).sum()
        ),
        "unknown_exposure_warning_skus": int(
            labels["unknown_exposure_days"].gt(0).sum()
        ),
    }
    labels["first_actual_sale_date"] = pd.to_datetime(
        labels["first_actual_sale_date"]
    ).dt.date.astype(str)
    labels["label_start_date"] = labels["label_start_date"].astype(str)
    labels["label_end_date"] = labels["label_end_date"].astype(str)
    return FirstOrderLabelResult(
        labels=labels.sort_values(
            ["first_actual_sale_date", "sku_code"], kind="stable"
        ).reset_index(drop=True),
        formal_schedule=formal_schedule.sort_values(
            ["first_actual_sale_date", "sku_code", "decision_cutoff_time"],
            kind="stable",
        ).reset_index(drop=True),
        excluded_skus=excluded.reset_index(drop=True),
        report=report,
    )
