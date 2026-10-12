from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping

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
    constraint_by_sku = (
        constraints.drop_duplicates("sku_code").set_index("sku_code", drop=False)
        if constraints is not None and not constraints.empty else pd.DataFrame()
    )
    if constraints is not None and not constraints.empty:
        require_columns(constraints, ("sku_code",), "constraints")
    interval_by_sku = (
        prediction_intervals.drop_duplicates("sku_code").set_index("sku_code", drop=False)
        if prediction_intervals is not None and not prediction_intervals.empty else pd.DataFrame()
    )
    adjustments_by_sku = adjustments.set_index("sku_code", drop=False)
    reliability_by_sku = comparable_summary.set_index("sku_code")["path_reliability"].to_dict()
    decision_rows: list[dict[str, Any]] = []
    store_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    calculation_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    strategy_map = factor_config["strategy_coefficients"]
    for target in targets.drop_duplicates("sku_code").itertuples(index=False):
        sku = str(target.sku_code)
        if sku not in adjustments_by_sku.index:
            raise ValueError(f"missing M5 demand adjustment for {sku}")
        placement = placement_plan.loc[
            (placement_plan["sku_code"].astype(str) == sku)
            & (pd.to_numeric(placement_plan["selected"], errors="coerce").fillna(0) == 1)
        ].copy()
        if placement.empty:
            raise ValueError(f"M4 selected no stores for {sku}")
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
        curve = demand_curve.loc[
            (demand_curve["sku_code"].astype(str) == sku)
            & (pd.to_numeric(demand_curve["lifecycle_week"], errors="coerce") <= cover_weeks)
        ].copy()
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
        selected_store_rows: list[dict[str, Any]] = []
        for store in placement.itertuples(index=False):
            tier_curve = curve.loc[curve["store_tier"].astype(str) == str(store.store_tier)].copy()
            if tier_curve.empty:
                raise ValueError(
                    f"no V3.2 demand curve for {sku} store_tier={store.store_tier}; "
                    "cross-tier averaging is not allowed"
                )
            tier_curve["lifecycle_week"] = pd.to_numeric(
                tier_curve["lifecycle_week"], errors="raise"
            ).astype(int)
            if tier_curve["lifecycle_week"].duplicated().any():
                raise ValueError(
                    f"duplicate demand curve weeks for {sku} store_tier={store.store_tier}"
                )
            missing_weeks = sorted(
                set(range(1, cover_weeks + 1))
                - set(tier_curve["lifecycle_week"])
            )
            if missing_weeks:
                raise ValueError(
                    f"incomplete demand curve for {sku} store_tier={store.store_tier}; "
                    f"missing weeks={missing_weeks[:10]}"
                )
            tier_curve = tier_curve.loc[
                tier_curve["lifecycle_week"].between(1, cover_weeks)
            ].sort_values("lifecycle_week")
            lifecycle_factor = pd.to_numeric(
                tier_curve["lifecycle_factor"], errors="coerce"
            )
            period_factor = pd.to_numeric(
                tier_curve["period_factor_h"], errors="coerce"
            )
            if lifecycle_factor.isna().any() or period_factor.isna().any():
                raise ValueError(
                    f"demand curve contains missing lifecycle/H values for {sku} "
                    f"store_tier={store.store_tier}"
                )
            neutral = float(
                (lifecycle_factor * float(store.v_anchor_week1)).sum()
            )
            sell = float(
                (
                    lifecycle_factor
                    * period_factor
                    * float(store.v_anchor_week1)
                ).sum()
            )
            level = str(getattr(store, "store_level", "__UNKNOWN__"))
            safety_rate = float(config["safety_stock_quantile_by_tier"].get(level, config["safety_stock_quantile_by_tier"]["__UNKNOWN__"]))
            safety_stock = safety_rate * neutral
            base_target = max(sell + safety_stock, minimum_store) + minimum_display
            natural_target = max(c_demand * (sell + safety_stock), minimum_store) + minimum_display
            planned_target = max(c_demand * c_strategy * (sell + safety_stock), minimum_store) + minimum_display
            selected_store_rows.append(
                {
                    "sku_code": sku,
                    "store_code": str(store.store_code),
                    "store_tier": str(store.store_tier),
                    "store_level": level,
                    "cover_weeks": cover_weeks,
                    "neutral_demand": neutral,
                    "sellable_demand": sell,
                    "safety_stock": safety_stock,
                    "display_units": minimum_display,
                    "minimum_store_units": minimum_store,
                    "display_requirement_status": display_requirement_status,
                    "minimum_store_status": minimum_store_status,
                    "base_target": base_target,
                    "natural_target": natural_target,
                    "planned_target": planned_target,
                }
            )
        store_frame = pd.DataFrame(selected_store_rows)
        q_base = float(store_frame["base_target"].sum())
        d_sell = float(store_frame["sellable_demand"].sum())
        d_first = c_demand * d_sell
        q_nat = float(store_frame["natural_target"].sum())
        q_plan = float(store_frame["planned_target"].sum())
        limited_quantity = pd.to_numeric(pd.Series([constraint.get("limited_quantity")]), errors="coerce").iloc[0]
        if pd.notna(limited_quantity):
            q_plan = float(max(limited_quantity, 0.0))
            strategy = "limited_quantity_override"
        mold_redline = float(_configured(constraint, "mold_redline", config["mold_redline_reference"]))
        licensed = str(getattr(target, "copyright_category", "")).strip() not in {"", "自有IP", "自有", "OWN"}
        default_moq = config["moq_reference"]["licensed_ip" if licensed else "own_ip"]
        moq = float(_configured(constraint, "moq", default_moq))
        case_pack = float(_configured(constraint, "case_pack", 1.0))
        feasible_minimum = max(mold_redline, moq)
        q_feas = _round_up(max(q_plan, feasible_minimum), case_pack)
        capacity = pd.to_numeric(pd.Series([constraint.get("capacity_qty")]), errors="coerce").iloc[0]
        capacity_status = "missing_not_applied"
        if pd.notna(capacity):
            capacity_status = "applied"
            if capacity <= 0:
                q_feas = 0.0
            elif q_feas > capacity:
                capacity_rounded = np.floor(float(capacity) / case_pack) * case_pack
                q_feas = float(capacity_rounded) if capacity_rounded >= feasible_minimum else 0.0
        opportunity_capacity = pd.to_numeric(pd.Series([constraint.get("opportunity_capacity_qty")]), errors="coerce").iloc[0]
        if pd.notna(opportunity_capacity):
            q_opp = min(_round_up(max(q_plan, feasible_minimum), case_pack), float(opportunity_capacity))
        else:
            q_opp = q_feas
        unit_conversion_confirmed = _configured_bool(constraint, "unit_conversion_confirmed", False)
        series_pack_count = pd.to_numeric(pd.Series([constraint.get("series_pack_count")]), errors="coerce").iloc[0]
        order_uom = "piece"
        q_feas_uom = q_feas
        if unit_conversion_confirmed and pd.notna(series_pack_count) and series_pack_count > 0:
            q_feas_uom = float(ceil(q_feas / float(series_pack_count)))
            order_uom = "confirmed_series_pack"
        elif pd.notna(series_pack_count):
            warnings.append(f"{sku}: series_pack_count supplied but conversion is disabled until confirmed")
        field_flags = {
            "intent": pd.notna(constraint.get("strategy")),
            "period": pd.notna(constraint.get("decision_weeks")),
            "emotion": hasattr(target, "emotion_tags") and bool(str(getattr(target, "emotion_tags", "")).strip()),
            "moq": pd.notna(constraint.get("moq")),
            "capacity": pd.notna(capacity),
            "lead": pd.notna(constraint.get("lead_weeks")),
            "unit": unit_conversion_confirmed,
        }
        q_key = 100.0 * sum(field_flags[name] for name in ("intent", "period", "emotion")) / 3.0
        q_field = 100.0 * sum(field_flags.values()) / len(field_flags)
        input_completeness = 0.5 * q_key + 0.5 * q_field
        source_completeness = float(adjustment.get("input_completeness", 0.0))
        path_reliability = float(reliability_by_sku.get(sku, 20.0))
        reliability_score = 0.40 * path_reliability + 0.25 * source_completeness + 0.20 * input_completeness + 0.15 * 0.0
        interval = interval_by_sku.loc[sku] if not interval_by_sku.empty and sku in interval_by_sku.index else pd.Series(dtype=object)
        interval_status = "oof_calibrated" if not interval.empty else "uncalibrated_no_oof_error"
        p10 = float(interval.get("p10", np.nan)) if not interval.empty else np.nan
        p90 = float(interval.get("p90", np.nan)) if not interval.empty else np.nan
        decision_rows.append(
            {
                "sku_code": sku,
                "cover_weeks": cover_weeks,
                "planned_store_count": len(store_frame),
                "D_sell": d_sell,
                "D_first": d_first,
                "Q_base": q_base,
                "Q_nat": q_nat,
                "Q_plan": q_plan,
                "Q_feas": q_feas,
                "Q_opp": q_opp,
                "Q_feas_order_uom": q_feas_uom,
                "order_uom": order_uom,
                "C_demand": c_demand,
                "C_strategy": c_strategy,
                "strategy": strategy,
                "grade": _grade(q_feas, config["grade_thresholds"]),
                "input_completeness": input_completeness,
                "input_completeness_bucket": _confidence_bucket(input_completeness, config["input_completeness_thresholds"]),
                "reliability_score": reliability_score,
                "reliability_bucket": _confidence_bucket(reliability_score, config["reliability_thresholds"]),
                "capacity_status": capacity_status,
                "moq_status": "business_confirmed" if pd.notna(constraint.get("moq")) else "reference_requires_confirmation",
                "lead_time_status": "business_confirmed" if pd.notna(constraint.get("lead_weeks")) else "reference_requires_confirmation",
                "display_requirement_status": display_requirement_status,
                "minimum_store_status": minimum_store_status,
                "interval_status": interval_status,
                "prediction_p10": p10,
                "prediction_p90": p90,
            }
        )
        store_rows.extend(selected_store_rows)
        prediction_rows.extend(
            [
                {"sku_code": sku, "horizon_code": "FIRST_ORDER_NATURAL", "prediction_value": q_nat, "lower_value": p10, "upper_value": p90},
                {"sku_code": sku, "horizon_code": "FIRST_ORDER_FEASIBLE", "prediction_value": q_feas, "lower_value": np.nan, "upper_value": np.nan},
                {"sku_code": sku, "horizon_code": "FIRST_ORDER_OPPORTUNITY", "prediction_value": q_opp, "lower_value": np.nan, "upper_value": np.nan},
            ]
        )
        for sequence, (code, value) in enumerate(
            [
                ("D_SELL", d_sell), ("C_DEMAND", c_demand), ("D_FIRST", d_first),
                ("Q_BASE", q_base), ("Q_NAT", q_nat), ("C_STRATEGY", c_strategy),
                ("Q_PLAN", q_plan), ("MOLD_REDLINE", mold_redline), ("MOQ", moq),
                ("CASE_PACK", case_pack), ("Q_FEAS", q_feas), ("Q_OPP", q_opp),
            ],
            start=1,
        ):
            calculation_rows.append(
                {"sku_code": sku, "calc_sequence": sequence, "calc_item_code": code, "calc_value": value}
            )
    return FirstOrderResult(
        pd.DataFrame(decision_rows),
        pd.DataFrame(store_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(calculation_rows),
        tuple(dict.fromkeys(warnings)),
    )
