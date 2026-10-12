from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil, exp
from typing import Any, Mapping

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


def calculate_reorder(
    *,
    decision_cutoff_time: datetime,
    first_order_decisions: pd.DataFrame,
    first_order_store_targets: pd.DataFrame,
    actual_daily: pd.DataFrame,
    demand_curve: pd.DataFrame,
    config: Mapping[str, Any],
    constraints: pd.DataFrame | None = None,
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
        constraint_by_sku = constraints.drop_duplicates("sku_code").set_index("sku_code", drop=False)
    actual = actual_daily.copy()
    actual["business_day"] = pd.to_datetime(actual["business_day"], errors="raise").dt.date
    if (actual["business_day"] > decision_cutoff_time.date()).any():
        raise ValueError("actual_daily contains observations after decision cutoff")
    actual["actual_sales"] = pd.to_numeric(actual["actual_sales"], errors="coerce").fillna(0.0)
    decision_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for first in first_order_decisions.drop_duplicates("sku_code").itertuples(index=False):
        sku = str(first.sku_code)
        store_plan = first_order_store_targets.loc[
            first_order_store_targets["sku_code"].astype(str) == sku
        ].copy()
        visible = actual.loc[actual["sku_code"].astype(str) == sku].copy()
        if visible.empty:
            raise ValueError(f"no visible actual daily observations for reorder SKU {sku}")
        constraint = constraint_by_sku.loc[sku] if not constraint_by_sku.empty and sku in constraint_by_sku.index else pd.Series(dtype=object)
        launch_date = pd.to_datetime(
            _configured(constraint, "launch_date", min(visible["business_day"])),
            errors="raise",
        ).date()
        age_days = max((decision_cutoff_time.date() - launch_date).days, 0)
        elapsed_weeks = max(age_days / 7.0, 1.0 / 7.0)
        plan_by_tier = store_plan.groupby("store_tier", as_index=False).agg(
            plan_sellable=("sellable_demand", "sum"),
            plan_neutral=("neutral_demand", "sum"),
            existing_safety_target=("safety_stock", "sum"),
            store_count=("store_code", "nunique"),
        )
        cover_weeks = max(float(first.cover_weeks), 1.0)
        plan_by_tier["planned_weekly_velocity"] = plan_by_tier["plan_sellable"] / cover_weeks
        visible = visible.merge(
            plan_by_tier[["store_tier", "planned_weekly_velocity", "store_count"]],
            on="store_tier",
            how="left",
            validate="many_to_one",
        )
        visible["planned_daily_store"] = visible["planned_weekly_velocity"].fillna(0.0) / visible["store_count"].clip(lower=1) / 7.0
        visible["stockout_lost"] = np.where(
            pd.to_numeric(visible["e_stock"], errors="coerce") == 0,
            np.maximum(visible["planned_daily_store"] - visible["actual_sales"], 0.0),
            0.0,
        )
        visible["nondisplay_lost"] = np.where(
            (pd.to_numeric(visible["e_display"], errors="coerce") == 0)
            & (pd.to_numeric(visible["e_stock"], errors="coerce") != 0),
            np.maximum(visible["planned_daily_store"] - visible["actual_sales"], 0.0),
            0.0,
        )
        visible["actual_demand"] = visible["actual_sales"] + visible["stockout_lost"] + visible["nondisplay_lost"]
        actual_by_tier = visible.groupby("store_tier", as_index=False).agg(
            actual_demand=("actual_demand", "sum"),
            actual_sales=("actual_sales", "sum"),
            stockout_lost=("stockout_lost", "sum"),
            nondisplay_lost=("nondisplay_lost", "sum"),
            observed_days=("business_day", "nunique"),
        ).merge(plan_by_tier, on="store_tier", how="outer")
        actual_by_tier["planned_to_date"] = actual_by_tier["planned_weekly_velocity"].fillna(0) * elapsed_weeks
        denominator = float(actual_by_tier["planned_to_date"].sum())
        actual_total = float(actual_by_tier["actual_demand"].fillna(0).sum())
        r_sales = (actual_total + 1e-6) / (denominator + 1e-6)
        tier_weights = actual_by_tier["planned_to_date"].fillna(0)
        tier_weights = tier_weights / max(float(tier_weights.sum()), 1e-12)
        actual_by_tier["tier_ratio"] = (
            actual_by_tier["actual_demand"].fillna(0) + 1e-6
        ) / (actual_by_tier["planned_to_date"].fillna(0) + 1e-6)
        r_weighted = float(np.dot(tier_weights, actual_by_tier["tier_ratio"])) if len(actual_by_tier) else r_sales
        n_eff = float(visible[["e_stock", "e_display"]].notna().all(axis=1).sum())
        repair_share = float(
            (visible["stockout_lost"].sum() + visible["nondisplay_lost"].sum())
            / max(visible["actual_demand"].sum(), 1e-12)
        )
        concentration = float(tier_weights.max()) if len(tier_weights) else 1.0
        concentration_penalty = max(concentration - 0.80, 0.0)
        lambda_raw = (
            _lambda_time(age_days, config["lambda_time_by_age_days"])
            * n_eff / (n_eff + float(config["shrinkage_k"]))
            * (1.0 - min(repair_share, 1.0))
            * (1.0 - min(concentration_penalty, 1.0))
        )
        lambda_lower, lambda_upper = map(float, config["lambda_bounds"])
        lambda_value = float(np.clip(lambda_raw, lambda_lower, lambda_upper))
        m_dev = 1.0 + lambda_value * (r_weighted - 1.0)
        visible["week_index"] = ((pd.to_datetime(visible["business_day"]) - pd.Timestamp(launch_date)).dt.days // 7 + 1).astype(int)
        week_actual = visible.groupby("week_index")["actual_demand"].sum().sort_index()
        actual_wow = week_actual.pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(3)
        curve_sku = demand_curve.loc[demand_curve["sku_code"].astype(str) == sku].copy()
        curve_week = curve_sku.groupby("lifecycle_week")["lifecycle_factor"].mean().sort_index()
        curve_wow = curve_week.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        current_week = max(int(np.ceil(elapsed_weeks)), 1)
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
        future = curve_sku.loc[pd.to_numeric(curve_sku["lifecycle_week"], errors="coerce") > current_week].copy()
        if future.empty:
            future = pd.DataFrame(
                [{"store_tier": row.store_tier, "lifecycle_week": current_week + 1, "lifecycle_factor": 0.0, "period_factor_h": 1.0} for row in store_plan.itertuples(index=False)]
            )
        velocity_by_tier = plan_by_tier.set_index("store_tier")["planned_weekly_velocity"].to_dict()
        future["forecast_demand"] = [
            float(velocity_by_tier.get(str(tier), 0.0))
            * float(lifecycle)
            * float(period)
            * c_slope
            * m_dev
            for tier, lifecycle, period in zip(
                future["store_tier"], future["lifecycle_factor"], future["period_factor_h"]
            )
        ]
        weekly_forecast = future.groupby("lifecycle_week", as_index=False)["forecast_demand"].sum().sort_values("lifecycle_week")
        below = weekly_forecast.loc[weekly_forecast["forecast_demand"] < economic_line, "lifecycle_week"]
        t_econ_week = int(below.iloc[0]) if not below.empty else int(weekly_forecast["lifecycle_week"].max()) + 1
        eta = decision_cutoff_time.date() + timedelta(days=lead_days)
        next_review_days = int(_configured(constraint, "next_review_days", 7))
        next_arrival = eta + timedelta(days=next_review_days)
        hard_end = pd.to_datetime(
            _configured(
                constraint,
                "hard_end_date",
                launch_date + timedelta(days=int(first.cover_weeks * 7)),
            ),
            errors="raise",
        ).date()
        econ_end = launch_date + timedelta(days=t_econ_week * 7)
        window_end = min(next_arrival, econ_end, hard_end)
        window_days = max((window_end - eta).days + 1, 0)
        weekly_window = max(window_days / 7.0, 0.0)
        d_window = float(weekly_forecast["forecast_demand"].head(max(int(ceil(weekly_window)), 0)).sum())
        structural_lost = float(visible["nondisplay_lost"].sum())
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
        ss_target = float(store_plan["safety_stock"].sum())
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
        if d_window + d_carry < reorder_moq:
            qr_feas = 0.0
            economic_pruned = True
        else:
            qr_feas = _round_up(max(qr_plan, reorder_moq), case_pack)
            economic_pruned = False
        capacity = pd.to_numeric(pd.Series([constraint.get("capacity_qty")]), errors="coerce").iloc[0]
        if pd.notna(capacity) and qr_feas > capacity:
            qr_feas = float(np.floor(float(capacity) / case_pack) * case_pack)
            if qr_feas < reorder_moq:
                qr_feas = 0.0
        opportunity_capacity = pd.to_numeric(pd.Series([constraint.get("opportunity_capacity_qty")]), errors="coerce").iloc[0]
        qr_opp = min(_round_up(max(qr_plan, reorder_moq), case_pack), float(opportunity_capacity)) if pd.notna(opportunity_capacity) else qr_feas
        if state == "Cliff" and qr_feas == 0:
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
                "decision_cutoff_time": decision_cutoff_time.isoformat(),
                "age_days": age_days,
                "R_sales": r_weighted,
                "lambda": lambda_value,
                "M_dev": m_dev,
                "C_slope": c_slope,
                "state": state,
                "T_econ_week": t_econ_week,
                "arrival_date": eta.isoformat(),
                "window_end_date": window_end.isoformat(),
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
                {"sku_code": sku, "lifecycle_week": int(row.lifecycle_week), "forecast_demand": float(row.forecast_demand), "economic_line": economic_line}
            )
        diagnostic_rows.extend(actual_by_tier.assign(sku_code=sku).to_dict(orient="records"))
    return ReorderResult(
        pd.DataFrame(decision_rows), pd.DataFrame(attribution_rows),
        pd.DataFrame(forecast_rows), pd.DataFrame(diagnostic_rows),
        tuple(dict.fromkeys(warnings)),
    )
