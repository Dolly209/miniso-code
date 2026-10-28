"""Invariant checks for explicitly verified launch-universe artifacts."""
from __future__ import annotations

import pandas as pd

from data.domain import require_columns
from data.domestic import DataContractViolation


SALES_OBSERVED_ORIGIN_POLICY = "sales_observed_first_positive_v2"


def validate_launch_origin_contract(frame: pd.DataFrame, *, name: str) -> None:
    # Preserve v1 exact-match artifacts; v2 permits audited missing/conflicting
    # product dates but never changes the actual observation/label clock.
    if "launch_universe_version" not in frame:
        return
    require_columns(
        frame,
        ("sku_code", "dataset_version", "launch_origin_verified",
         "product_first_sale_date", "observed_first_sale_date",
         "first_actual_sale_date", "source_start_date"),
        name,
    )
    if frame.empty:
        return
    dates = {
        key: pd.to_datetime(frame[key], errors="coerce", format="mixed").dt.normalize()
        for key in ("product_first_sale_date", "observed_first_sale_date",
                    "first_actual_sale_date", "source_start_date")
    }
    common = (
        dates["first_actual_sale_date"].notna()
        & dates["observed_first_sale_date"].eq(dates["first_actual_sale_date"])
        & dates["first_actual_sale_date"].ge(dates["source_start_date"])
        & frame["launch_universe_version"].notna()
        & frame["launch_universe_version"].astype(str).str.strip().ne("")
        & frame["dataset_version"].astype(str).eq(frame["launch_universe_version"].astype(str))
    )
    if "launch_origin_policy" in frame:
        require_columns(
            frame,
            ("launch_origin_eligible", "source_left_boundary_date", "known_pre_source_sale_evidence", "first_sale_origin_source"),
            name,
        )
        boundary = pd.to_datetime(frame["source_left_boundary_date"], errors="coerce", format="mixed").dt.normalize()
        accepted = frame["launch_origin_eligible"].astype(str).str.lower().isin(("true", "1"))
        older_evidence = dates["product_first_sale_date"].lt(boundary)
        at_boundary = dates["observed_first_sale_date"].eq(boundary)
        corroborated = dates["product_first_sale_date"].eq(dates["observed_first_sale_date"])
        valid = (
            common & accepted & boundary.notna()
            & dates["observed_first_sale_date"].ge(boundary)
            & ~older_evidence & (~at_boundary | corroborated)
            & frame["first_sale_origin_source"].eq("sales_order_first_positive_from_cohort")
            & frame["launch_origin_policy"].eq(SALES_OBSERVED_ORIGIN_POLICY)
        )
    else:
        verified = frame["launch_origin_verified"].astype(str).str.lower().isin(("true", "1"))
        valid = common & verified & dates["product_first_sale_date"].eq(dates["first_actual_sale_date"])
    if not valid.all():
        examples = frame.loc[~valid, "sku_code"].astype(str).head(5).tolist()
        raise DataContractViolation(f"{name} violates verified launch-origin contract: {examples}")


def validate_launch_universe_pair(cohort: pd.DataFrame, schedule: pd.DataFrame) -> None:
    if "launch_universe_version" not in schedule:
        return
    validate_launch_origin_contract(schedule, name="prelaunch schedule")
    if "launch_universe_version" not in cohort:
        raise DataContractViolation(
            "verified launch schedule requires its verified candidate cohort; legacy Round-3 cohort is not allowed"
        )
    validate_launch_origin_contract(cohort, name="historical candidate cohort")
    if set(schedule["launch_universe_version"].astype(str)) != set(cohort["launch_universe_version"].astype(str)):
        raise DataContractViolation("schedule and candidate cohort have different launch_universe_version")


def require_verified_training_schedule(schedule: pd.DataFrame) -> None:
    if "launch_universe_version" not in schedule:
        return
    validate_launch_origin_contract(schedule, name="training schedule")
    if "training_ready_for_feature_replay" not in schedule or not schedule["training_ready_for_feature_replay"].astype(str).str.lower().isin(("true", "1")).all():
        raise DataContractViolation(
            "verified universe is only preflight: run row coverage and fresh restored W_cover labels before training/replay"
        )
