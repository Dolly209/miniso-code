from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import ceil, exp
from time import monotonic
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from data.domain import require_columns


@dataclass(frozen=True)
class ReorderResult:
    decisions: pd.DataFrame
    attribution: pd.DataFrame
    demand_forecast: pd.DataFrame
    diagnostics: pd.DataFrame
    warnings: tuple[str, ...]


def _round_up(value: float, unit: float) -> float:
    return float(ceil(max(value, 0.0) / max(unit, 1e-12)) * unit)


def _lambda_time(age_days: int, schedule: list[list[float]]) -> float:
    for maximum_age, value in schedule:
        if age_days <= int(maximum_age):
            return float(value)
    return float(schedule[-1][1])


def _state(ratio: float, slope: float, age_days: int, config: Mapping[str, Any]) -> str:
    if age_days <= 7 and ratio >= float(config["pulse_ratio"]):
        return "Pulse"
    if ratio <= float(config["cliff_ratio"]) and slope < 1.0:
        return "Cliff"
    if age_days >= int(config["sustained_min_days"]) and ratio >= 1.0 and slope >= 0.95:
        return "Sustained"
    return "Normal"


def _configured(row: pd.Series, key: str, default: Any) -> Any:
    value = row.get(key, default)
    return default if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)) else value


def _configured_bool(row: pd.Series, key: str, default: bool = False) -> bool:
    value = _configured(row, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是"}
    return bool(value)


def _lifecycle_week(launch_date: date, value: date) -> int:
    """Return the one-based lifecycle week containing ``value``."""

    return max((value - launch_date).days // 7 + 1, 1)


def _windowed_weekly_demand(
    weekly_forecast: pd.DataFrame,
    *,
    launch_date: date,
    window_start: date,
    window_end: date,
) -> tuple[float, pd.DataFrame]:
    """Prorate weekly forecasts to the exact inclusive V3.2 decision window."""

    result = weekly_forecast.copy()
    if result.empty:
        result["week_start_date"] = pd.Series(dtype=object)
        result["week_end_date"] = pd.Series(dtype=object)
        result["window_overlap_days"] = pd.Series(dtype=int)
        result["window_demand_contribution"] = pd.Series(dtype=float)
        return 0.0, result

    weeks = pd.to_numeric(result["lifecycle_week"], errors="raise").astype(int)
    result["week_start_date"] = [
        launch_date + timedelta(days=(week - 1) * 7) for week in weeks
    ]
    result["week_end_date"] = [value + timedelta(days=6) for value in result["week_start_date"]]
    if window_end < window_start:
        result["window_overlap_days"] = 0
    else:
        result["window_overlap_days"] = [
            max(
                (
                    min(week_end, window_end)
                    - max(week_start, window_start)
                ).days
                + 1,
                0,
            )
            for week_start, week_end in zip(
                result["week_start_date"], result["week_end_date"]
            )
        ]
    result["window_demand_contribution"] = (
        pd.to_numeric(result["forecast_demand"], errors="coerce").fillna(0.0)
        * result["window_overlap_days"]
        / 7.0
    )
    return float(result["window_demand_contribution"].sum()), result


def calculate_reorder(
    *,
    decision_cutoff_time: datetime,
    first_order_decisions: pd.DataFrame,
    first_order_store_targets: pd.DataFrame,
    actual_daily: pd.DataFrame,
    demand_curve: pd.DataFrame,
    config: Mapping[str, Any],
    constraints: pd.DataFrame | None = None,
    decision_cutoff_by_sku: Mapping[str, datetime] | None = None,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> ReorderResult:
    """N1-N6 reorder from a frozen first-order snapshot and visible actuals."""

    require_columns(
        first_order_decisions,
        ("sku_code", "Q_feas", "planned_store_count", "cover_weeks"),
        "first_order_decisions",
    )
    require_columns(
        first_order_store_targets,
        ("sku_code", "store_code", "store_tier", "neutral_demand", "sellable_demand", "safety_stock"),
        "first_order_store_targets",
    )
    require_columns(
        actual_daily,
        ("sku_code", "store_code", "store_tier", "business_day", "actual_sales", "e_stock", "e_display"),
        "actual_daily",
    )
    require_columns(
        demand_curve,
        ("sku_code", "store_tier", "lifecycle_week", "lifecycle_factor", "period_factor_h"),
        "demand_curve",
    )
    constraint_by_sku = pd.DataFrame()
    if constraints is not None and not constraints.empty:
        require_columns(constraints, ("sku_code",), "reorder_constraints")
        constraint_by_sku = (
            constraints.assign(sku_code=constraints["sku_code"].astype(str))
            .drop_duplicates("sku_code")
            .set_index("sku_code", drop=False)
        )
    first_orders = first_order_decisions.copy()
    first_orders["sku_code"] = first_orders["sku_code"].astype(str)
    store_targets = first_order_store_targets.copy()
    store_targets["sku_code"] = store_targets["sku_code"].astype(str)
    store_targets["store_code"] = store_targets["store_code"].astype(str)
    store_targets["store_tier"] = store_targets["store_tier"].astype(str)
    for column in ("neutral_demand", "sellable_demand", "safety_stock"):
        store_targets[column] = pd.to_numeric(store_targets[column], errors="coerce")
    plan_by_tier_all = store_targets.groupby(
        ["sku_code", "store_tier"], as_index=False
    ).agg(
        plan_sellable=("sellable_demand", "sum"),
        plan_neutral=("neutral_demand", "sum"),
        existing_safety_target=("safety_stock", "sum"),
        store_count=("store_code", "nunique"),
    )
    plan_groups = plan_by_tier_all.groupby("sku_code", sort=False).indices
    actual = actual_daily.copy()
    actual["sku_code"] = actual["sku_code"].astype(str)
    actual["store_code"] = actual["store_code"].astype(str)
    actual["store_tier"] = actual["store_tier"].astype(str)
    actual["business_day"] = pd.to_datetime(actual["business_day"], errors="raise").dt.date
    actual["actual_sales"] = pd.to_numeric(actual["actual_sales"], errors="coerce").fillna(0.0)
    actual["_e_stock_numeric"] = pd.to_numeric(actual["e_stock"], errors="coerce")
    actual["_e_display_numeric"] = pd.to_numeric(actual["e_display"], errors="coerce")
    actual_groups = actual.groupby("sku_code", sort=False).indices
    curves = demand_curve.copy()
    curves["sku_code"] = curves["sku_code"].astype(str)
    curves["store_tier"] = curves["store_tier"].astype(str)
    for column in ("lifecycle_week", "lifecycle_factor", "period_factor_h"):
        curves[column] = pd.to_numeric(curves[column], errors="coerce")
    curve_groups = curves.groupby("sku_code", sort=False).indices
    decision_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    diagnostic_frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    target_rows = list(first_orders.drop_duplicates("sku_code").itertuples(index=False))
    started = monotonic()
    if progress:
        progress("prepare_complete", {
            "target_count": len(target_rows),
            "store_target_rows": len(store_targets),
            "actual_rows": len(actual),
            "curve_rows": len(curves),
            "elapsed_seconds": 0.0,
        })
    for target_index, first in enumerate(target_rows, start=1):
        sku = str(first.sku_code)
        sku_decision_time = (
            decision_cutoff_by_sku.get(sku, decision_cutoff_time)
            if decision_cutoff_by_sku is not None
            else decision_cutoff_time
        )
        if progress:
            progress("target_start", {
                "target_index": target_index,
                "target_count": len(target_rows),
                "sku_code": sku,
                "elapsed_seconds": round(monotonic() - started, 3),
            })
        plan_by_tier = plan_by_tier_all.iloc[plan_groups.get(sku, [])].copy()
        plan_by_tier = plan_by_tier.drop(columns="sku_code", errors="ignore")
        visible = actual.iloc[actual_groups.get(sku, [])].copy()
        if visible.empty:
            raise ValueError(f"no visible actual daily observations for reorder SKU {sku}")
        if (visible["business_day"] > sku_decision_time.date()).any():
            raise ValueError(
                f"actual_daily contains observations after decision cutoff for SKU {sku}"
            )
        constraint = constraint_by_sku.loc[sku] if not constraint_by_sku.empty and sku in constraint_by_sku.index else pd.Series(dtype=object)
        positive_sales = visible.loc[visible["actual_sales"] > 0, "business_day"]
        observed_launch_date = (
            min(positive_sales) if not positive_sales.empty else min(visible["business_day"])
        )
        launch_date = pd.to_datetime(
            _configured(constraint, "launch_date", observed_launch_date),
            errors="raise",
        ).date()
        age_days = max((sku_decision_time.date() - launch_date).days, 0)
        elapsed_weeks = max((age_days + 1) / 7.0, 1.0 / 7.0)
        cover_weeks = max(float(first.cover_weeks), 1.0)
        plan_by_tier["planned_weekly_velocity"] = plan_by_tier["plan_sellable"] / cover_weeks
        curve_sku = curves.iloc[curve_groups.get(sku, [])].copy()
        plan_curve = curve_sku.loc[
            curve_sku["lifecycle_week"].between(1, int(np.ceil(cover_weeks)))
        ].copy()
        plan_curve["plan_factor"] = (
            plan_curve["lifecycle_factor"] * plan_curve["period_factor_h"]
        ).clip(lower=0)
        factor_sum = plan_curve.groupby("store_tier")["plan_factor"].transform("sum")
        plan_curve["plan_week_share"] = np.where(
            factor_sum > 0, plan_curve["plan_factor"] / factor_sum, 0.0
        )
        plan_curve = plan_curve.merge(
            plan_by_tier[["store_tier", "plan_sellable"]],
            on="store_tier",
            how="left",
            validate="many_to_one",
        )
        plan_curve["planned_week_demand"] = (
            plan_curve["plan_sellable"].fillna(0.0) * plan_curve["plan_week_share"]
        )
        planned_week_lookup = plan_curve.set_index(
            ["store_tier", "lifecycle_week"]
        )["planned_week_demand"].to_dict()
        visible = visible.merge(
            plan_by_tier[["store_tier", "planned_weekly_velocity", "store_count"]],
            on="store_tier",
            how="left",
            validate="many_to_one",
        )
        visible["lifecycle_week"] = (
            (pd.to_datetime(visible["business_day"]) - pd.Timestamp(launch_date))
            .dt.days.floordiv(7).add(1).clip(lower=1).astype(int)
        )
        visible["planned_week_demand"] = [
            planned_week_lookup.get((str(tier), int(week)), 0.0)
            for tier, week in zip(visible["store_tier"], visible["lifecycle_week"])
        ]
        visible["planned_daily_store"] = (
            visible["planned_week_demand"]
            / visible["store_count"].fillna(0.0).clip(lower=1)
            / 7.0
        )
        stockout_mask = (
            visible["_e_stock_numeric"].eq(0).fillna(False).to_numpy(dtype=bool)
        )
        nondisplay_mask = (
            visible["_e_display_numeric"].eq(0).fillna(False)
            & visible["_e_stock_numeric"].ne(0).fillna(False)
        ).to_numpy(dtype=bool)
        visible["stockout_lost"] = np.where(
            stockout_mask,
            np.maximum(visible["planned_daily_store"] - visible["actual_sales"], 0.0),
            0.0,
        )
        visible["nondisplay_lost"] = np.where(
            nondisplay_mask,
            np.maximum(visible["planned_daily_store"] - visible["actual_sales"], 0.0),
            0.0,
        )
        visible["actual_demand"] = visible["actual_sales"] + visible["stockout_lost"] + visible["nondisplay_lost"]
        if (visible["actual_demand"] + 1e-9 < visible["actual_sales"]).any():
            raise ValueError(f"restored demand fell below actual sales for SKU {sku}")
        # N1 explicitly removes launch-day through T+2 pulse rows from both
        # numerator and the matching lifecycle-plan denominator.
        pulse_end = launch_date + timedelta(days=2)
        deviation_visible = visible.loc[visible["business_day"].gt(pulse_end)].copy()
        actual_by_tier = deviation_visible.groupby("store_tier", as_index=False).agg(
            actual_demand=("actual_demand", "sum"),
            actual_sales=("actual_sales", "sum"),
            stockout_lost=("stockout_lost", "sum"),
            nondisplay_lost=("nondisplay_lost", "sum"),
            observed_days=("business_day", "nunique"),
            planned_to_date=("planned_daily_store", "sum"),
        ).merge(plan_by_tier, on="store_tier", how="outer")
        denominator = float(actual_by_tier["planned_to_date"].sum())
        actual_total = float(actual_by_tier["actual_demand"].fillna(0).sum())
        r_sales = (actual_total + 1e-6) / (denominator + 1e-6)
        tier_weights = actual_by_tier["planned_to_date"].fillna(0)
        tier_weights = tier_weights / max(float(tier_weights.sum()), 1e-12)
        actual_by_tier["tier_ratio"] = (
            actual_by_tier["actual_demand"].fillna(0) + 1e-6
        ) / (actual_by_tier["planned_to_date"].fillna(0) + 1e-6)
        r_weighted = float(np.dot(tier_weights, actual_by_tier["tier_ratio"])) if len(actual_by_tier) else r_sales
        n_eff = float(deviation_visible[["e_stock", "e_display"]].notna().all(axis=1).sum())
        repair_share = float(
            (deviation_visible["stockout_lost"].sum() + deviation_visible["nondisplay_lost"].sum())
            / max(deviation_visible["actual_demand"].sum(), 1e-12)
        )
        store_sales = deviation_visible.groupby("store_code")["actual_demand"].sum().clip(lower=0)
        if len(store_sales) > 1 and float(store_sales.sum()) > 0:
            ordered_sales = np.sort(store_sales.to_numpy(dtype=float))
            count = len(ordered_sales)
            concentration_gini = float(
                (2.0 * np.dot(np.arange(1, count + 1), ordered_sales))
                / (count * ordered_sales.sum())
                - (count + 1.0) / count
            )
            max_store_share = float(ordered_sales.max() / ordered_sales.sum())
        elif len(store_sales) == 1 and float(store_sales.sum()) > 0:
            concentration_gini, max_store_share = 1.0, 1.0
        else:
            concentration_gini, max_store_share = 0.0, 0.0
        concentration_penalty = float(np.clip(concentration_gini, 0.0, 1.0))
        lambda_raw = (
            _lambda_time(age_days, config["lambda_time_by_age_days"])
            * n_eff / (n_eff + float(config["shrinkage_k"]))
            * (1.0 - min(repair_share, 1.0))
            * (1.0 - min(concentration_penalty, 1.0))
        )
        lambda_lower, lambda_upper = map(float, config["lambda_bounds"])
        lambda_value = float(np.clip(lambda_raw, lambda_lower, lambda_upper))
        m_dev = 1.0 + lambda_value * (r_weighted - 1.0)
        deviation_visible["week_index"] = deviation_visible["lifecycle_week"]
        week_actual = deviation_visible.groupby("week_index")["actual_demand"].sum().sort_index()
        actual_wow = week_actual.pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(3)
        curve_week = curve_sku.groupby("lifecycle_week")["lifecycle_factor"].mean().sort_index()
        curve_wow = curve_week.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        current_week = _lifecycle_week(launch_date, sku_decision_time.date())
        matching_curve_wow = curve_wow.loc[curve_wow.index <= current_week].tail(max(len(actual_wow), 1))
        if len(actual_wow) and len(matching_curve_wow) and abs(float(matching_curve_wow.mean())) > 1e-6:
            slope_raw = (1.0 + float(actual_wow.mean())) / (1.0 + float(matching_curve_wow.mean()))
        else:
            slope_raw = 1.0
        slope_lower, slope_upper = map(float, config["slope_bounds"])
        c_slope = float(np.clip(slope_raw, slope_lower, slope_upper))
        state = _state(r_weighted, c_slope, age_days, config)
        lead_days = int(_configured(constraint, "reorder_lead_days", config["default_reorder_lead_days"]))
        reorder_moq = float(_configured(constraint, "reorder_moq", config["default_reorder_moq"]))
        case_pack = float(_configured(constraint, "case_pack", config["default_case_pack"]))
        lead_weeks = max(lead_days / 7.0, 1.0 / 7.0)
        economic_line = reorder_moq / lead_weeks
        future = curve_sku.loc[curve_sku["lifecycle_week"] > current_week].copy()
        velocity_by_tier = plan_by_tier.set_index("store_tier")["planned_weekly_velocity"].to_dict()
        if future.empty:
            weekly_forecast = pd.DataFrame(
                columns=("lifecycle_week", "forecast_demand")
            )
            forecast_source = "curve_horizon_unavailable"
            warnings.append(
                f"{sku}: lifecycle curve has no weeks after decision week {current_week}; "
                "D_window is unavailable rather than fabricated"
            )
        else:
            future["forecast_demand"] = (
                future["store_tier"].map(velocity_by_tier).fillna(0.0)
                * future["lifecycle_factor"]
                * future["period_factor_h"]
                * c_slope
                * m_dev
            )
            weekly_forecast = (
                future.groupby("lifecycle_week", as_index=False)["forecast_demand"]
                .sum()
                .sort_values("lifecycle_week")
            )
            forecast_source = "v3_2_lifecycle_curve"
        below = weekly_forecast.loc[
            weekly_forecast["forecast_demand"] < economic_line,
            "lifecycle_week",
        ]
        if not below.empty:
            t_econ_week = int(below.iloc[0])
        elif not weekly_forecast.empty:
            t_econ_week = int(weekly_forecast["lifecycle_week"].max()) + 1
        else:
            t_econ_week = current_week
        eta = sku_decision_time.date() + timedelta(days=lead_days)
        next_review_days = int(
            _configured(
                constraint,
                "next_review_days",
                config.get("default_next_review_days", 7),
            )
        )
        next_decision = sku_decision_time.date() + timedelta(days=next_review_days)
        next_arrival = next_decision + timedelta(days=lead_days)
        hard_end_value = constraint.get("hard_end_date")
        if hard_end_value is None or pd.isna(hard_end_value):
            hard_end = None
            hard_end_source = "not_provided_not_applied"
            warnings.append(
                f"{sku}: hard_end_date not provided; T_hard is not applied"
            )
        else:
            hard_end = pd.to_datetime(hard_end_value, errors="raise").date()
            hard_end_source = "constraint"
        econ_end = launch_date + timedelta(days=t_econ_week * 7 - 1)
        window_candidates = [next_arrival, econ_end]
        if hard_end is not None:
            window_candidates.append(hard_end)
        window_end = min(window_candidates)
        d_window, weekly_forecast = _windowed_weekly_demand(
            weekly_forecast,
            launch_date=launch_date,
            window_start=eta,
            window_end=window_end,
        )
        lead_demand, _ = _windowed_weekly_demand(
            weekly_forecast,
            launch_date=launch_date,
            window_start=sku_decision_time.date() + timedelta(days=1),
            window_end=eta - timedelta(days=1),
        )
        observed_structural_share = float(
            visible["nondisplay_lost"].sum()
            / max(float(visible["actual_demand"].sum()), 1e-12)
        )
        structural_lost = lead_demand * min(max(observed_structural_share, 0.0), 1.0)
        d_carry = float(config["transfer_carry_rate_reference"]) * structural_lost
        if "available_stock_qty" in visible and visible["available_stock_qty"].notna().any():
            latest_inventory_day = visible.loc[
                visible["available_stock_qty"].notna(), "business_day"
            ].max()
            observed_store_inventory = float(
                pd.to_numeric(
                    visible.loc[
                        visible["business_day"] == latest_inventory_day,
                        "available_stock_qty",
                    ],
                    errors="coerce",
                ).fillna(0.0).sum()
            )
        else:
            observed_store_inventory = 0.0
        inventory_fields = {
            "store": _configured(constraint, "store_inventory", observed_store_inventory),
            "warehouse": _configured(constraint, "warehouse_inventory", 0.0),
            "inbound": _configured(constraint, "inbound_before_eta", 0.0),
            "waiting": _configured(constraint, "waiting_demand", 0.0),
            "protected": _configured(constraint, "protected_inventory", 0.0),
        }
        i_sellable = max(
            0.0,
            float(inventory_fields["store"])
            + float(inventory_fields["warehouse"])
            + float(inventory_fields["inbound"])
            - float(inventory_fields["waiting"])
            - float(inventory_fields["protected"]),
        )
        ss_target = float(plan_by_tier["existing_safety_target"].sum())
        ss_protected = float(_configured(constraint, "protected_safety_stock", 0.0))
        ss_topup = max(ss_target - ss_protected, 0.0)
        opening_new = float(_configured(constraint, "opening_new_store_qty", 0.0))
        qr_base = max(0.0, d_window + d_carry - i_sellable + ss_topup + opening_new)
        quality_verified = _configured_bool(constraint, "quality_return_contract_verified", False)
        if quality_verified:
            return_rate = float(_configured(constraint, "return_rate", 0.0))
            category_return_rate = float(_configured(constraint, "category_return_rate", 0.0))
            excess = max(return_rate - category_return_rate, 0.0)
            quality_lower, _ = map(float, config["quality_return_bounds"])
            c_quality = float(np.clip(exp(-float(config["quality_kappa"]) * excess), quality_lower, 1.0))
            quality_status = "verified_applied"
        else:
            c_quality = 1.0
            quality_status = "neutral_unverified_contract"
            warnings.append(f"{sku}: refund/quality contract unverified; C_R_quality kept at 1")
        qr_plan = _round_up(qr_base * c_quality, case_pack)
        if qr_plan <= 0 or d_window + d_carry < reorder_moq or qr_plan < reorder_moq:
            qr_feas = 0.0
            economic_pruned = True
        else:
            qr_feas = _round_up(qr_plan, case_pack)
            economic_pruned = False
        capacity = pd.to_numeric(pd.Series([constraint.get("capacity_qty")]), errors="coerce").iloc[0]
        if pd.notna(capacity) and qr_feas > capacity:
            qr_feas = float(np.floor(float(capacity) / case_pack) * case_pack)
            if qr_feas < reorder_moq:
                qr_feas = 0.0
        opportunity_capacity = pd.to_numeric(pd.Series([constraint.get("opportunity_capacity_qty")]), errors="coerce").iloc[0]
        opportunity_resolved = _configured_bool(
            constraint, "opportunity_constraints_resolved", False
        )
        if qr_plan <= 0 or (economic_pruned and not opportunity_resolved):
            qr_opp = 0.0
        else:
            qr_opp = min(
                _round_up(qr_plan, case_pack),
                float(opportunity_capacity),
            ) if pd.notna(opportunity_capacity) else (
                _round_up(qr_plan, case_pack) if opportunity_resolved else qr_feas
            )
        if state == "Cliff":
            qr_feas = 0.0
            action = "freeze"
        elif qr_feas > 0 and qr_feas >= float(_configured(constraint, "remold_threshold", float("inf"))):
            action = "remold"
        elif qr_feas > 0:
            action = "reorder"
        else:
            action = "observe"
        required_flags = {
            "actual_sales": visible["actual_sales"].notna().all(),
            "exposure": visible[["e_stock", "e_display"]].notna().all(axis=1).mean() >= 0.8,
            "inventory": any(pd.notna(value) for value in inventory_fields.values()),
            "lead": pd.notna(constraint.get("reorder_lead_days")),
            "moq": pd.notna(constraint.get("reorder_moq")),
            "capacity": pd.notna(capacity),
        }
        input_completeness = 100.0 * sum(required_flags.values()) / len(required_flags)
        decision_rows.append(
            {
                "sku_code": sku,
                "decision_cutoff_time": sku_decision_time.isoformat(),
                "launch_date": launch_date.isoformat(),
                "age_days": age_days,
                "R_sales": r_weighted,
                "lambda": lambda_value,
                "n_eff_aligned": n_eff,
                "repair_share": repair_share,
                "sales_concentration_gini": concentration_gini,
                "max_store_sales_share": max_store_share,
                "launch_pulse_excluded_through": pulse_end.isoformat(),
                "M_dev": m_dev,
                "C_slope": c_slope,
                "state": state,
                "T_econ_week": t_econ_week,
                "window_start_date": eta.isoformat(),
                "arrival_date": eta.isoformat(),
                "next_decision_date": next_decision.isoformat(),
                "next_arrival_date": next_arrival.isoformat(),
                "window_end_date": window_end.isoformat(),
                "hard_end_date": hard_end.isoformat() if hard_end is not None else None,
                "hard_end_source": hard_end_source,
                "forecast_source": forecast_source,
                "D_window": d_window,
                "D_carry": d_carry,
                "I_sellable": i_sellable,
                "SS_topup": ss_topup,
                "Opening_new": opening_new,
                "C_R_quality": c_quality,
                "QR_base": qr_base,
                "QR_plan": qr_plan,
                "QR_feas": qr_feas,
                "QR_opp": qr_opp,
                "action": action,
                "economic_pruned": economic_pruned,
                "quality_status": quality_status,
                "input_completeness": input_completeness,
                "reliability_status": "uncalibrated" if not _configured_bool(constraint, "reliability_model_calibrated", False) else "calibrated",
            }
        )
        contributions = {
            "future_window_demand": d_window,
            "structural_carry": d_carry,
            "sellable_inventory": -i_sellable,
            "safety_stock_topup": ss_topup,
            "opening_new_stores": opening_new,
            "quality_guard_delta": qr_base * (c_quality - 1.0),
        }
        rank = sorted(contributions, key=lambda key: abs(contributions[key]), reverse=True)
        for index, name in enumerate(rank, start=1):
            attribution_rows.append(
                {"sku_code": sku, "rank": index, "factor_code": name, "contribution": contributions[name], "direction": "positive" if contributions[name] >= 0 else "negative"}
            )
        for row in weekly_forecast.itertuples(index=False):
            forecast_rows.append(
                {
                    "sku_code": sku,
                    "lifecycle_week": int(row.lifecycle_week),
                    "week_start_date": row.week_start_date,
                    "week_end_date": row.week_end_date,
                    "forecast_demand": float(row.forecast_demand),
                    "window_overlap_days": int(row.window_overlap_days),
                    "window_demand_contribution": float(row.window_demand_contribution),
                    "economic_line": economic_line,
                    "forecast_source": forecast_source,
                }
            )
        diagnostic_frames.append(actual_by_tier.assign(sku_code=sku))
        if progress:
            progress("target_complete", {
                "target_index": target_index,
                "target_count": len(target_rows),
                "sku_code": sku,
                "actual_rows": len(visible),
                "state": state,
                "action": action,
                "elapsed_seconds": round(monotonic() - started, 3),
            })
    return ReorderResult(
        pd.DataFrame(decision_rows), pd.DataFrame(attribution_rows),
        pd.DataFrame(forecast_rows),
        pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame(),
        tuple(dict.fromkeys(warnings)),
    )
