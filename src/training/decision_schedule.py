from __future__ import annotations

from datetime import datetime
from typing import Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from data.domain import require_columns


DEFAULT_PRELAUNCH_LEAD_MONTHS = (3,)


def normalize_lead_months(values: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted({int(value) for value in values}))
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("lead months must contain positive integers")
    return normalized


def build_prelaunch_decision_schedule(
    cohort: pd.DataFrame,
    *,
    lead_months: Sequence[int] = DEFAULT_PRELAUNCH_LEAD_MONTHS,
    timezone: str = "Asia/Shanghai",
    available_through: datetime | None = None,
) -> pd.DataFrame:
    """Expand one historical SKU into independent prelaunch forecast checkpoints.

    The result is a point-in-time schedule only.  It does not read raw business
    tables or build features.  Every checkpoint for one SKU retains the cohort's
    original Train/Test assignment.  Each checkpoint later produces a complete
    first-order demand forecast; this function does not divide training weights,
    labels, or quantities across checkpoints.
    """

    require_columns(
        cohort,
        ("sku_code", "release_dt", "first_actual_sale_date", "split"),
        "historical cohort with fixed product snapshot",
    )
    leads = normalize_lead_months(lead_months)
    base = cohort.copy()
    base["sku_code"] = base["sku_code"].astype(str).str.strip()
    if base["sku_code"].eq("").any():
        raise ValueError("historical cohort contains blank sku_code")
    if base["sku_code"].duplicated().any():
        examples = base.loc[
            base["sku_code"].duplicated(keep=False), "sku_code"
        ].drop_duplicates().head(5).tolist()
        raise ValueError(
            "historical cohort must contain one row per SKU; "
            f"duplicates={examples}"
        )
    first_sale = pd.to_datetime(
        base["first_actual_sale_date"], errors="coerce", format="mixed"
    )
    if first_sale.isna().any():
        examples = base.loc[first_sale.isna(), "sku_code"].head(5).tolist()
        raise ValueError(
            "historical cohort contains invalid first_actual_sale_date; "
            f"examples={examples}"
        )
    base["first_actual_sale_date"] = first_sale.dt.normalize()
    release_date = pd.to_datetime(
        base["release_dt"], errors="coerce", format="mixed"
    )
    if release_date.isna().any():
        examples = base.loc[release_date.isna(), "sku_code"].head(5).tolist()
        raise ValueError(
            "historical cohort contains invalid release_dt; "
            f"examples={examples}"
        )
    base["release_dt"] = release_date.dt.normalize()

    zone = ZoneInfo(timezone)
    if available_through is not None and (
        available_through.tzinfo is None or available_through.utcoffset() is None
    ):
        raise ValueError("available_through must be timezone-aware")
    available_in_zone = (
        None if available_through is None else available_through.astimezone(zone)
    )

    pieces: list[pd.DataFrame] = []
    for lead in leads:
        piece = base.copy()
        decision_day = (
            piece["release_dt"] - pd.DateOffset(months=lead)
        ).dt.normalize()
        # Business dates are day-grain.  Use end of the corresponding calendar
        # day so every source value available on that date can be selected.
        decision_time = decision_day.map(
            lambda value: pd.Timestamp(value).to_pydatetime().replace(
                hour=23,
                minute=59,
                second=59,
                microsecond=0,
                tzinfo=zone,
            )
        )
        piece["lead_months"] = lead
        piece["prediction_checkpoint"] = f"M{lead}"
        piece["decision_cutoff_time"] = decision_time.map(datetime.isoformat)
        piece["decision_date"] = decision_day.dt.date.astype(str)
        piece["days_before_release"] = (
            piece["release_dt"] - decision_day
        ).dt.days.astype(int)
        piece["days_before_first_sale"] = (
            piece["first_actual_sale_date"] - decision_day
        ).dt.days.astype(int)
        piece["decision_time_source"] = (
            "release_dt_from_fixed_product_snapshot_minus_calendar_months"
        )
        piece["point_in_time_rule"] = (
            "event_time<=decision_cutoff_time and "
            "available_time<=decision_cutoff_time"
        )
        piece["label_start_date"] = piece["first_actual_sale_date"].dt.date.astype(
            str
        )
        piece["forecast_scope"] = "complete_first_order_demand"
        piece["checkpoint_weight_policy"] = (
            "no_w_i_div_m_i; evaluate_each_prelaunch_checkpoint_separately"
        )
        piece["within_available_through"] = (
            True
            if available_in_zone is None
            else decision_time.map(lambda value: value <= available_in_zone)
        )
        piece["prediction_checkpoint_key"] = (
            piece["sku_code"] + "@" + piece["prediction_checkpoint"]
        )
        pieces.append(piece)

    schedule = pd.concat(pieces, ignore_index=True)
    schedule["first_actual_sale_date"] = schedule[
        "first_actual_sale_date"
    ].dt.date.astype(str)
    schedule["release_dt"] = schedule["release_dt"].dt.date.astype(str)
    return schedule.sort_values(
        ["first_actual_sale_date", "sku_code", "decision_date"],
        kind="stable",
    ).reset_index(drop=True)
