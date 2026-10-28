from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from time import monotonic
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from data.domain import require_columns


@dataclass(frozen=True)
class FirstOrderResult:
    decisions: pd.DataFrame
    store_targets: pd.DataFrame
    predictions: pd.DataFrame
    calculation_detail: pd.DataFrame
    warnings: tuple[str, ...]
    weekly_demand: pd.DataFrame = field(default_factory=pd.DataFrame)


def _round_up(value: float, unit: float) -> float:
    if unit <= 0:
        raise ValueError("rounding unit must be positive")
    return float(ceil(max(value, 0.0) / unit) * unit)


def _grade(quantity: float, thresholds: Mapping[str, float]) -> str:
    if quantity <= float(thresholds["C_max"]):
        return "C"
    if quantity < float(thresholds["B_max"]):
        return "B"
    if quantity < float(thresholds["A_max"]):
        return "A"
    return "S"


def _confidence_bucket(value: float, thresholds: Mapping[str, float]) -> str:
    if value >= float(thresholds["high"]):
        return "high"
    if value >= float(thresholds["medium"]):
        return "medium"
    return "low"


def _configured(row: pd.Series, key: str, default: Any) -> Any:
    value = row.get(key, default)
    return default if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)) else value


def _configured_bool(row: pd.Series, key: str, default: bool = False) -> bool:
    value = _configured(row, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是"}
    return bool(value)


def calculate_first_order(
    *,
    targets: pd.DataFrame,
    placement_plan: pd.DataFrame,
    demand_curve: pd.DataFrame,
    adjustments: pd.DataFrame,
    comparable_summary: pd.DataFrame,
    config: Mapping[str, Any],
    factor_config: Mapping[str, Any],
    constraints: pd.DataFrame | None = None,
    prediction_intervals: pd.DataFrame | None = None,
    arrival_plan: pd.DataFrame | None = None,
    include_details: bool = True,
    placement_summary: pd.DataFrame | None = None,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> FirstOrderResult:
    """M6/M7 first-order quantities from placement, anchors and bounded factors."""

    require_columns(targets, ("sku_code",), "targets")
    require_columns(
        placement_plan,
        ("sku_code", "store_code", "store_tier", "selected", "v_anchor_week1"),
        "placement_plan",
    )
    require_columns(
        demand_curve,
        ("sku_code", "store_tier", "lifecycle_week", "lifecycle_factor", "period_factor_h"),
        "demand_curve",
    )
    require_columns(adjustments, ("sku_code", "C_demand", "input_completeness"), "adjustments")
    require_columns(comparable_summary, ("sku_code", "path_reliability"), "comparable_summary")
    if arrival_plan is not None:
        require_columns(arrival_plan, ("sku_code", "store_code", "arrival_week"), "confirmed arrival plan")
        arrival_plan = arrival_plan.copy()
        for key in ("sku_code", "store_code"):
            arrival_plan[key] = arrival_plan[key].astype(str)
        arrival_plan["arrival_week"] = pd.to_numeric(arrival_plan.arrival_week, errors="coerce")
        if arrival_plan.duplicated(["sku_code", "store_code"]).any() or arrival_plan.arrival_week.isna().any() or (arrival_plan.arrival_week < 1).any() or (arrival_plan.arrival_week % 1 != 0).any():
            raise ValueError("confirmed arrival plan needs unique stores and integer arrival_week >= 1")
    weekly_demand_frames = []
    constraint_by_sku = (
        constraints.assign(sku_code=constraints["sku_code"].astype(str))
        .drop_duplicates("sku_code").set_index("sku_code", drop=False)
        if constraints is not None and not constraints.empty else pd.DataFrame()
    )
    if constraints is not None and not constraints.empty:
        require_columns(constraints, ("sku_code",), "constraints")
    interval_by_sku = (
        prediction_intervals.assign(
            sku_code=prediction_intervals["sku_code"].astype(str)
        ).drop_duplicates("sku_code").set_index("sku_code", drop=False)
        if prediction_intervals is not None and not prediction_intervals.empty else pd.DataFrame()
    )
    adjustments = adjustments.assign(sku_code=adjustments["sku_code"].astype(str))
    adjustments_by_sku = adjustments.set_index("sku_code", drop=False)
    reliability_by_sku = comparable_summary.assign(
        sku_code=comparable_summary["sku_code"].astype(str)
    ).set_index("sku_code")["path_reliability"].to_dict()
    placement = placement_plan.copy()
    placement["sku_code"] = placement["sku_code"].astype(str)
    placement["store_code"] = placement["store_code"].astype(str)
    placement["store_tier"] = placement["store_tier"].astype(str)
    placement = placement.loc[
        pd.to_numeric(placement["selected"], errors="coerce").fillna(0).eq(1)
    ].copy()
    placement["v_anchor_week1"] = pd.to_numeric(
        placement["v_anchor_week1"], errors="coerce"
    )
    placement_groups = placement.groupby("sku_code", sort=False).indices
    curves = demand_curve.copy()
    curves["sku_code"] = curves["sku_code"].astype(str)
    curves["store_tier"] = curves["store_tier"].astype(str)
    for column in ("lifecycle_week", "lifecycle_factor", "period_factor_h"):
        curves[column] = pd.to_numeric(curves[column], errors="coerce")
    curve_groups = curves.groupby("sku_code", sort=False).indices
    decision_rows: list[dict[str, Any]] = []
    store_frames: list[pd.DataFrame] = []
    prediction_rows: list[dict[str, Any]] = []
    calculation_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    strategy_map = factor_config["strategy_coefficients"]
    target_rows = list(targets.drop_duplicates("sku_code").itertuples(index=False))
    started = monotonic()
    if progress:
        progress("prepare_complete", {
            "target_count": len(target_rows), "selected_store_rows": len(placement),
            "curve_rows": len(curves), "elapsed_seconds": 0.0,
        })
    for target_index, target in enumerate(target_rows, start=1):
        sku = str(target.sku_code)
        if progress:
            progress("target_start", {
                "target_index": target_index, "target_count": len(target_rows),
                "sku_code": sku, "elapsed_seconds": round(monotonic() - started, 3),
            })
        if sku not in adjustments_by_sku.index:
            raise ValueError(f"missing M5 demand adjustment for {sku}")
        target_placement = placement.iloc[placement_groups.get(sku, [])].copy()
        adjustment = adjustments_by_sku.loc[sku]
        constraint = constraint_by_sku.loc[sku] if not constraint_by_sku.empty and sku in constraint_by_sku.index else pd.Series(dtype=object)
        decision_weeks = int(_configured(constraint, "decision_weeks", config["decision_weeks"]))
        buffer_weeks = int(_configured(constraint, "buffer_weeks", config["buffer_weeks"]))
        lead_weeks = int(_configured(constraint, "lead_weeks", config["lead_weeks"]))
        lifecycle_weeks = int(_configured(constraint, "lifecycle_weeks", config["lifecycle_weeks"]))
        hard_weeks = int(_configured(constraint, "hard_life_weeks", lifecycle_weeks))
        cover_weeks = min(decision_weeks + buffer_weeks + lead_weeks, lifecycle_weeks, hard_weeks)
        if cover_weeks <= 0:
            raise ValueError(f"non-positive first-order cover horizon for {sku}")
        if target_placement.empty:
            documented = (placement_summary.loc[placement_summary.sku_code.astype(str).eq(sku)]
                          if placement_summary is not None else pd.DataFrame())
            if len(documented) != 1 or documented.iloc[0].get("placement_status") != "blocked_no_supported_stores" or int(documented.iloc[0].get("planned_store_count", -1)) != 0:
                raise ValueError(f"M4 selected no stores for {sku}; missing explicit zero-placement diagnostic")
            reason = str(documented.iloc[0].get("placement_block_reason", "no_supported_stores"))
            decision_rows.append({
                "sku_code": sku, "cover_weeks": cover_weeks, "planned_store_count": 0,
                "decision_status": "blocked_no_supported_stores", "decision_reason": reason,
                "training_feature_eligible": False, "D_sell": np.nan, "D_first": np.nan,
                "D_demand_base": np.nan, "D_demand_adjusted": np.nan,
                "Q_base": np.nan, "Q_nat": np.nan, "Q_plan": 0., "Q_feas": 0., "Q_opp": 0.,
                "SS_first": 0., "Display_first": 0., "C_demand": float(adjustment.C_demand),
                "supply_action": "blocked_no_supported_stores", "demand_grade": pd.NA,
                "supply_grade": pd.NA, "grade": pd.NA, "reliability_score": 0.,
                "input_completeness": float(adjustment.input_completeness),
                "demand_quantity_definition": "unavailable_no_supported_placement; not_zero_sales",
                "prediction_p10": np.nan, "prediction_p90": np.nan,
            })
            for horizon, value in (("FIRST_ORDER_DEMAND", np.nan), ("FIRST_ORDER_NATURAL", np.nan),
                                   ("FIRST_ORDER_FEASIBLE", 0.), ("FIRST_ORDER_OPPORTUNITY", 0.)):
                prediction_rows.append(dict(sku_code=sku, horizon_code=horizon, prediction_value=value, lower_value=np.nan, upper_value=np.nan))
            calculation_rows.append(dict(sku_code=sku, calc_sequence=1, calc_item_code="BLOCKED_NO_SUPPORTED_STORES", calc_value=np.nan))
            warnings.append(f"{sku}: no evidence-supported placement; demand left unavailable and automatic order blocked; not a zero-sales training example")
            continue
        curve = curves.iloc[curve_groups.get(sku, [])].copy()
        curve = curve.loc[curve["lifecycle_week"] <= cover_weeks]
        if curve.empty:
            raise ValueError(f"no demand curve in first-order horizon for {sku}")
        c_demand = float(adjustment["C_demand"])
        strategy = str(_configured(constraint, "strategy", "normal")).lower()
        if strategy not in strategy_map:
            raise ValueError(f"unknown first-order strategy for {sku}: {strategy}")
        c_strategy = float(strategy_map[strategy])
        minimum_display = float(_configured(constraint, "display_units", config["minimum_display_units"]))
        minimum_store = float(_configured(constraint, "minimum_store_units", config["minimum_store_units"]))
        display_requirement_status = (
            "business_confirmed"
            if pd.notna(constraint.get("display_units"))
            else "missing_set_zero_per_v3_2"
        )
        minimum_store_status = (
            "business_confirmed"
            if pd.notna(constraint.get("minimum_store_units"))
            else "missing_set_zero_per_v3_2"
        )
        placement_tiers = set(target_placement["store_tier"])
        missing_tiers = sorted(placement_tiers - set(curve["store_tier"]))
        if missing_tiers:
            raise ValueError(
                f"no V3.2 demand curve for {sku} store_tier={missing_tiers[0]}; "
                "cross-tier averaging is not allowed"
            )
        curve = curve.loc[curve["store_tier"].isin(placement_tiers)].copy()
        if curve[["lifecycle_week", "lifecycle_factor", "period_factor_h"]].isna().any(axis=None):
            raise ValueError(f"demand curve contains missing lifecycle/H values for {sku}")
        curve["lifecycle_week"] = curve["lifecycle_week"].astype(int)
        if curve.duplicated(["store_tier", "lifecycle_week"]).any():
            raise ValueError(f"duplicate demand curve weeks for {sku}")
        expected_weeks = set(range(1, cover_weeks + 1))
        curve_weeks = curve.groupby("store_tier")["lifecycle_week"].agg(set)
        incomplete = {
            tier: sorted(expected_weeks - weeks)
            for tier, weeks in curve_weeks.items()
            if tier in placement_tiers and expected_weeks - weeks
        }
        if incomplete:
            tier, missing_weeks = next(iter(incomplete.items()))
            raise ValueError(
                f"incomplete demand curve for {sku} store_tier={tier}; "
                f"missing weeks={missing_weeks[:10]}"
            )
        curve = curve.loc[curve["lifecycle_week"].between(1, cover_weeks)].copy()
        curve["sell_factor"] = curve["lifecycle_factor"] * curve["period_factor_h"]
        curve_totals = curve.groupby("store_tier", as_index=False).agg(
            neutral_factor=("lifecycle_factor", "sum"),
            sell_factor=("sell_factor", "sum"),
        )
        store_frame = target_placement.merge(
            curve_totals, on="store_tier", how="left", validate="many_to_one",
            sort=False,
        )
        store_frame["store_level"] = (
            store_frame["store_level"].astype(str)
            if "store_level" in store_frame
            else "__UNKNOWN__"
        )
        store_frame["cover_weeks"] = cover_weeks
        store_frame["neutral_demand"] = (
            store_frame["v_anchor_week1"] * store_frame["neutral_factor"]
        )
        store_frame["sellable_demand"] = (
            store_frame["v_anchor_week1"] * store_frame["sell_factor"]
        )
        plan_for_sku = arrival_plan.loc[arrival_plan.sku_code.eq(sku)] if arrival_plan is not None else pd.DataFrame()
        coverage_status = "normal_launch_all_selected_stores_week1"
        if not plan_for_sku.empty:
            store_frame = store_frame.merge(plan_for_sku[["store_code", "arrival_week"]], on="store_code", how="left", validate="one_to_one")
            if store_frame.arrival_week.isna().any():
                raise ValueError(f"{sku}: confirmed plan must cover every selected store; absent plan is not week1")
            coverage_status = "explicit_business_confirmed_store_arrival_plan"
        else:
            store_frame["arrival_week"] = 1
        # Weekly tier ledger is compact. Loop over W_cover, not over stores
        # or days, and allocate each week's tier demand using active potential.
        if not plan_for_sku.empty:
            store_frame["neutral_demand"] = 0.0
            store_frame["sellable_demand"] = 0.0
        tier_stats = store_frame.groupby("store_tier").v_anchor_week1.mean()
        all_active_stats = store_frame.groupby("store_tier").agg(active_store_count=("store_code", "size"), active_potential=("v_anchor_week1", "sum"))
        for week in (range(1, cover_weeks + 1) if include_details or not plan_for_sku.empty else ()):
            if not plan_for_sku.empty:
                active = store_frame.arrival_week.le(week)
                active_frame = store_frame.loc[active]
                stats = active_frame.groupby("store_tier").agg(active_store_count=("store_code", "size"), active_potential=("v_anchor_week1", "sum"))
            else:
                stats = all_active_stats
            week_curve = curve.loc[curve.lifecycle_week.eq(week)].copy().set_index("store_tier")
            week_curve = week_curve.join(stats).fillna({"active_store_count": 0, "active_potential": 0})
            week_curve["weekly_neutral_demand"] = week_curve.active_store_count * tier_stats.reindex(week_curve.index) * week_curve.lifecycle_factor
            week_curve["weekly_sales_demand"] = week_curve.weekly_neutral_demand * week_curve.period_factor_h
            if not plan_for_sku.empty:
                denominator = active_frame.store_tier.map(stats.active_potential)
                omega = active_frame.v_anchor_week1 / denominator.replace(0, np.nan)
                equal = 1.0 / active_frame.store_tier.map(stats.active_store_count)
                omega = omega.fillna(equal)
                store_frame.loc[active, "neutral_demand"] += omega * active_frame.store_tier.map(week_curve.weekly_neutral_demand)
                store_frame.loc[active, "sellable_demand"] += omega * active_frame.store_tier.map(week_curve.weekly_sales_demand)
            week_curve["coverage_status"] = coverage_status
            week_curve["sku_code"] = sku
            if include_details:
                weekly_demand_frames.append(week_curve.reset_index())
        safety_default = float(
            config["safety_stock_quantile_by_tier"]["__UNKNOWN__"]
        )
        safety_rates = (
            store_frame["store_level"]
            .map(config["safety_stock_quantile_by_tier"])
            .fillna(safety_default)
            .astype(float)
        )
        store_frame["safety_stock"] = safety_rates * store_frame["neutral_demand"]
        demand_with_safety = store_frame["sellable_demand"] + store_frame["safety_stock"]
        # V3.2 section 6.3 has two deliberately different quantities:
        # D_sell/D_first are pure sales demand, while T_s(C) adds the safety,
        # minimum-store and display policies needed to form an order quantity.
        # Keep those ledgers separate so a safety-policy change cannot alter
        # the demand-model target or coefficient fit.
        store_frame["pure_demand_base"] = store_frame["sellable_demand"]
        store_frame["pure_demand_adjusted"] = c_demand * store_frame["sellable_demand"]
        store_frame["display_units"] = minimum_display
        store_frame["minimum_store_units"] = minimum_store
        store_frame["display_requirement_status"] = display_requirement_status
        store_frame["minimum_store_status"] = minimum_store_status
        store_frame["base_target"] = np.maximum(demand_with_safety, minimum_store) + minimum_display
        store_frame["natural_target"] = np.maximum(
            c_demand * demand_with_safety, minimum_store
        ) + minimum_display
        store_frame["planned_target"] = np.maximum(
            c_demand * c_strategy * demand_with_safety, minimum_store
        ) + minimum_display
        output_columns = [
            "sku_code", "store_code", "store_tier", "store_level", "cover_weeks",
            "neutral_demand", "sellable_demand", "pure_demand_base",
            "pure_demand_adjusted", "safety_stock", "display_units",
            "minimum_store_units", "display_requirement_status", "minimum_store_status",
            "base_target", "natural_target", "planned_target",
        ]
        output_columns.extend(
            column
            for column in (
                "placement_origin",
                "historical_arrival_probability",
                "productive_probability",
                "selection_eligible_by_evidence",
                "store_score",
            )
            if column in store_frame and column not in output_columns
        )
        if include_details:
            store_frames.append(store_frame[output_columns].copy())
        q_base = float(store_frame["base_target"].sum())
        d_sell = float(store_frame["sellable_demand"].sum())
        d_first = c_demand * d_sell
        safety_stock_first = float(store_frame["safety_stock"].sum())
        display_first = float(store_frame["display_units"].sum())
        base_before_floor_display = float(
            (store_frame["sellable_demand"] + store_frame["safety_stock"]).sum()
        )
        minimum_floor_uplift_base = float(
            (
                np.maximum(
                    store_frame["sellable_demand"] + store_frame["safety_stock"],
                    store_frame["minimum_store_units"],
                )
                - (
                    store_frame["sellable_demand"]
                    + store_frame["safety_stock"]
                )
            ).sum()
        )
        q_nat = float(store_frame["natural_target"].sum())
        q_plan = float(store_frame["planned_target"].sum())
        limited_quantity = pd.to_numeric(pd.Series([constraint.get("limited_quantity")]), errors="coerce").iloc[0]
        if pd.notna(limited_quantity):
            q_plan = float(max(limited_quantity, 0.0))
            strategy = "limited_quantity_override"
        else:
            q_plan = _round_up(q_plan, 1.0)
        mold_redline = float(_configured(constraint, "mold_redline", config["mold_redline_reference"]))
        licensed = str(getattr(target, "copyright_category", "")).strip() not in {"", "自有IP", "自有", "OWN"}
        default_moq = config["moq_reference"]["licensed_ip" if licensed else "own_ip"]
        moq = float(_configured(constraint, "moq", default_moq))
        case_pack = float(_configured(constraint, "case_pack", 1.0))
        new_mold_required = _configured_bool(constraint, "new_mold_required", True)
        mold_waiver = _configured_bool(constraint, "mold_waiver_approved", False)
        moq_waiver = _configured_bool(constraint, "moq_waiver_approved", False)
        merge_approved = _configured_bool(constraint, "cross_channel_merge_approved", False)
        width_depth_minimum = float(
            (store_frame["minimum_store_units"] + store_frame["display_units"]).sum()
        )
        width_depth_conflict = bool(
            pd.notna(limited_quantity) and q_plan < width_depth_minimum
        )
        if new_mold_required and q_plan < mold_redline and not mold_waiver:
            q_feas = 0.0
            supply_action = "below_mold_redline_no_normal_order"
        elif q_plan < moq and not (moq_waiver or merge_approved):
            q_feas = 0.0
            supply_action = "below_moq_hold_or_merge_next_node"
        else:
            q_feas = _round_up(q_plan, case_pack)
            supply_action = "feasible_round"
        capacity = pd.to_numeric(pd.Series([constraint.get("capacity_qty")]), errors="coerce").iloc[0]
        capacity_status = "missing_not_applied"
        if pd.notna(capacity):
            capacity_status = "applied"
            if capacity <= 0:
                q_feas = 0.0
            elif q_feas > capacity:
                capacity_rounded = np.floor(float(capacity) / case_pack) * case_pack
                minimum_allowed = mold_redline if new_mold_required and not mold_waiver else 0.0
                minimum_allowed = max(minimum_allowed, 0.0 if (moq_waiver or merge_approved) else moq)
                q_feas = float(capacity_rounded) if capacity_rounded >= minimum_allowed else 0.0
                supply_action = "capacity_capped" if q_feas > 0 else "capacity_below_feasible_minimum"
        opportunity_capacity = pd.to_numeric(pd.Series([constraint.get("opportunity_capacity_qty")]), errors="coerce").iloc[0]
        opportunity_resolved = _configured_bool(
            constraint, "opportunity_constraints_resolved", False
        )
        if q_feas == 0 and not opportunity_resolved:
            q_opp = 0.0
        elif pd.notna(opportunity_capacity):
            opportunity_capacity_rounded = (
                np.floor(float(opportunity_capacity) / case_pack) * case_pack
            )
            q_opp = min(_round_up(q_plan, case_pack), opportunity_capacity_rounded)
        else:
            q_opp = _round_up(q_plan, case_pack) if opportunity_resolved else q_feas
        unit_conversion_confirmed = _configured_bool(constraint, "unit_conversion_confirmed", False)
        series_pack_count = pd.to_numeric(pd.Series([constraint.get("series_pack_count")]), errors="coerce").iloc[0]
        sales_unit_type = str(getattr(target, "sales_unit_type", "")).strip().lower()
        is_blind_box = sales_unit_type in {"blind_box", "盲盒"}
        order_uom = "piece"
        q_plan_uom = q_plan
        q_feas_uom = q_feas
        q_opp_uom = q_opp
        if (
            is_blind_box
            and unit_conversion_confirmed
            and pd.notna(series_pack_count)
            and series_pack_count > 0
        ):
            q_plan_uom = float(ceil(q_plan / float(series_pack_count)))
            q_feas_uom = float(ceil(q_feas / float(series_pack_count)))
            q_opp_uom = float(ceil(q_opp / float(series_pack_count)))
            order_uom = "confirmed_series_pack"
        elif is_blind_box and pd.notna(series_pack_count):
            warnings.append(
                f"{sku}: blind-box series_pack_count supplied but conversion is "
                "disabled until the unit definition is confirmed"
            )
        key_flags = {
            "intent": pd.notna(constraint.get("strategy")),
            "period": pd.notna(constraint.get("decision_weeks")),
            "emotion": hasattr(target, "emotion_tags") and bool(str(getattr(target, "emotion_tags", "")).strip()),
        }
        q_key = max(
            0.0,
            100.0 - 40.0 * sum(not value for value in key_flags.values()),
        )
        # Q_field is the completeness of enabled formal small factors from M5;
        # MOQ/capacity/case-pack belong to supply feasibility and must not be
        # smuggled into factor completeness.
        q_field = float(adjustment.get("input_completeness", 0.0))
        input_completeness = 0.5 * q_key + 0.5 * q_field
        q_path = float(reliability_by_sku.get(sku, 20.0))
        comparable_count = float(
            comparable_summary.loc[
                comparable_summary["sku_code"].astype(str).eq(sku),
                "selected_candidate_count",
            ].iloc[0]
        ) if "selected_candidate_count" in comparable_summary and comparable_summary["sku_code"].astype(str).eq(sku).any() else 0.0
        q_sample = float(np.clip(comparable_count / 3.0 * 100.0, 0.0, 100.0))
        q_stability = float(_configured(constraint, "comparable_stability_score", 0.0))
        interval = interval_by_sku.loc[sku] if not interval_by_sku.empty and sku in interval_by_sku.index else pd.Series(dtype=object)
        interval_status = "oof_calibrated" if not interval.empty else "uncalibrated_no_oof_error"
        q_backtest = float(interval.get("backtest_reliability", 0.0)) if not interval.empty else 0.0
        reliability_score = (
            0.40 * q_path
            + 0.25 * q_sample
            + 0.20 * q_stability
            + 0.15 * q_backtest
        )
        p10 = float(interval.get("p10", np.nan)) if not interval.empty else np.nan
        p90 = float(interval.get("p90", np.nan)) if not interval.empty else np.nan
        decision_rows.append(
            {
                "sku_code": sku,
                "cover_weeks": cover_weeks,
                "decision_status": "calculated",
                "decision_reason": "",
                "training_feature_eligible": True,
                "planned_store_count": len(store_frame),
                "arrival_plan_status": coverage_status,
                "D_sell": d_sell,
                "D_first": d_first,
                "D_demand_base": d_sell,
                "D_demand_adjusted": d_first,
                "SS_first": safety_stock_first,
                "Display_first": display_first,
                "minimum_floor_uplift_base": minimum_floor_uplift_base,
                "Q_base_before_floor_display": base_before_floor_display,
                "Q_base": q_base,
                "Q_nat": q_nat,
                "Q_plan": q_plan,
                "Q_feas": q_feas,
                "Q_opp": q_opp,
                "Q_plan_order_uom": q_plan_uom,
                "Q_feas_order_uom": q_feas_uom,
                "Q_opp_order_uom": q_opp_uom,
                "order_uom": order_uom,
                "sales_unit_type": sales_unit_type or "unknown",
                "C_demand": c_demand,
                "C_strategy": c_strategy,
                "strategy": strategy,
                "demand_grade": _grade(
                    d_first,
                    config["grade_thresholds"][
                        "licensed_ip" if licensed else "own_ip"
                    ],
                ),
                "supply_grade": _grade(
                    q_feas,
                    config["grade_thresholds"][
                        "licensed_ip" if licensed else "own_ip"
                    ],
                ),
                # Backward compatible alias.  New evaluation must use the
                # explicit demand_grade and supply_grade columns above.
                "grade": _grade(
                    q_feas,
                    config["grade_thresholds"][
                        "licensed_ip" if licensed else "own_ip"
                    ],
                ),
                "demand_quantity_definition": "pure_sales_demand_inside_W_cover",
                "order_quantity_definition": (
                    "V3.2_Ts(C): demand_plus_safety_then_minimum_plus_display"
                ),
                "grade_threshold_version_status": config["grade_thresholds"].get(
                    "parameter_status", "unknown"
                ),
                "Q_key": q_key,
                "Q_field": q_field,
                "Q_path": q_path,
                "Q_sample": q_sample,
                "Q_stability": q_stability,
                "Q_backtest": q_backtest,
                "input_completeness": input_completeness,
                "input_completeness_bucket": _confidence_bucket(input_completeness, config["input_completeness_thresholds"]),
                "reliability_score": reliability_score,
                "reliability_bucket": _confidence_bucket(reliability_score, config["reliability_thresholds"]),
                "capacity_status": capacity_status,
                "supply_action": supply_action,
                "new_mold_required": new_mold_required,
                "mold_waiver_approved": mold_waiver,
                "moq_waiver_approved": moq_waiver,
                "cross_channel_merge_approved": merge_approved,
                "width_depth_minimum": width_depth_minimum,
                "width_depth_conflict": width_depth_conflict,
                "moq_status": "business_confirmed" if pd.notna(constraint.get("moq")) else "reference_requires_confirmation",
                "lead_time_status": "business_confirmed" if pd.notna(constraint.get("lead_weeks")) else "reference_requires_confirmation",
                "display_requirement_status": display_requirement_status,
                "minimum_store_status": minimum_store_status,
                "interval_status": interval_status,
                "prediction_p10": p10,
                "prediction_p90": p90,
            }
        )
        prediction_rows.extend(
            [
                {"sku_code": sku, "horizon_code": "FIRST_ORDER_DEMAND", "prediction_value": d_first, "lower_value": p10, "upper_value": p90},
                {"sku_code": sku, "horizon_code": "FIRST_ORDER_NATURAL", "prediction_value": q_nat, "lower_value": np.nan, "upper_value": np.nan},
                {"sku_code": sku, "horizon_code": "FIRST_ORDER_FEASIBLE", "prediction_value": q_feas, "lower_value": np.nan, "upper_value": np.nan},
                {"sku_code": sku, "horizon_code": "FIRST_ORDER_OPPORTUNITY", "prediction_value": q_opp, "lower_value": np.nan, "upper_value": np.nan},
            ]
        )
        for sequence, (code, value) in enumerate(
            [
                ("D_DEMAND_BASE", d_sell), ("C_DEMAND", c_demand),
                ("D_DEMAND_ADJUSTED", d_first),
                ("SS_FIRST", safety_stock_first),
                ("DISPLAY_FIRST", display_first),
                ("MINIMUM_FLOOR_UPLIFT_BASE", minimum_floor_uplift_base),
                ("Q_BASE", q_base), ("Q_NAT", q_nat), ("C_STRATEGY", c_strategy),
                ("Q_PLAN", q_plan), ("MOLD_REDLINE", mold_redline), ("MOQ", moq),
                ("CASE_PACK", case_pack), ("Q_FEAS", q_feas), ("Q_OPP", q_opp),
            ],
            start=1,
        ):
            calculation_rows.append(
                {"sku_code": sku, "calc_sequence": sequence, "calc_item_code": code, "calc_value": value}
            )
        if progress:
            progress("target_complete", {
                "target_index": target_index, "target_count": len(target_rows),
                "sku_code": sku, "selected_store_count": len(store_frame),
                "elapsed_seconds": round(monotonic() - started, 3),
            })
    return FirstOrderResult(
        pd.DataFrame(decision_rows),
        pd.concat(store_frames, ignore_index=True) if store_frames else pd.DataFrame(columns=[
            "sku_code", "store_code", "store_tier", "store_level", "cover_weeks", "neutral_demand", "sellable_demand",
            "pure_demand_base", "pure_demand_adjusted", "safety_stock", "display_units", "minimum_store_units",
            "base_target", "natural_target", "planned_target"]),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(calculation_rows),
        tuple(dict.fromkeys(warnings)),
        pd.concat(weekly_demand_frames, ignore_index=True) if weekly_demand_frames else pd.DataFrame(columns=[
            "sku_code", "store_tier", "lifecycle_week", "weekly_neutral_demand", "weekly_sales_demand", "coverage_status"]),
    )
