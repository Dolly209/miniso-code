#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

try:  # noqa: E402
    from _pipeline_common import add_execution_arguments, positive_int, print_dry_run
except ModuleNotFoundError:  # imported as scripts.run_reorder by tests/tools
    from scripts._pipeline_common import add_execution_arguments, positive_int, print_dry_run
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.decision_adapter import standardize_reorder_actuals  # noqa: E402
from data.domestic import DataContractViolation  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402
from reorder import calculate_reorder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.2 N1-N6 reorder only from a frozen first-order snapshot.")
    parser.add_argument("--first-order-dir", type=Path, default=Path("artifacts/round7_first_order"))
    parser.add_argument("--placement-dir", type=Path, default=Path("artifacts/round6_placement"))
    parser.add_argument("--actual-file", type=Path, default=Path("artifacts/round4_restore/daily_exposure_audit.csv"))
    parser.add_argument("--constraints-file", type=Path, default=None)
    parser.add_argument(
        "--decision-age-days",
        type=positive_int,
        default=None,
        help=(
            "Historical point-in-time replay: make each selected SKU decision N "
            "days after its first positive sale. The global cutoff remains the "
            "latest date available for mature validation labels."
        ),
    )
    add_execution_arguments(parser, default_output="artifacts/round9_reorder")
    return parser.parse_args()


def _derive_historical_cutoffs(
    *,
    actual: pd.DataFrame,
    target_skus: list[str],
    decision_age_days: int,
    available_through: datetime,
) -> tuple[dict[str, datetime], pd.DataFrame]:
    sales = actual.copy()
    sales["sku_code"] = sales["sku_code"].astype(str)
    sales["actual_sales"] = pd.to_numeric(sales["actual_sales"], errors="coerce").fillna(0.0)
    sales["business_day"] = pd.to_datetime(sales["business_day"], errors="raise").dt.date
    positive = sales.loc[
        sales["sku_code"].isin(target_skus) & sales["actual_sales"].gt(0)
    ]
    first_sales = positive.groupby("sku_code")["business_day"].min().to_dict()
    missing = [sku for sku in target_skus if sku not in first_sales]
    if missing:
        raise DataContractViolation(
            "historical replay needs a positive-sale first date for every target; "
            f"missing={len(missing)} examples={missing[:10]}"
        )
    result: dict[str, datetime] = {}
    audit_rows: list[dict[str, object]] = []
    cutoff_clock = available_through.timetz()
    for sku in target_skus:
        first_sale = first_sales[sku]
        decision_day = first_sale + timedelta(days=decision_age_days)
        decision_time = datetime.combine(decision_day, cutoff_clock)
        if decision_time > available_through:
            raise DataContractViolation(
                f"historical decision for SKU {sku} ({decision_time.isoformat()}) "
                f"is after available-through cutoff {available_through.isoformat()}"
            )
        result[sku] = decision_time
        audit_rows.append(
            {
                "sku_code": sku,
                "first_actual_sale_date": first_sale.isoformat(),
                "decision_age_days": decision_age_days,
                "historical_decision_time": decision_time.isoformat(),
            }
        )
    return result, pd.DataFrame(audit_rows)


def _filter_point_in_time_actuals(
    actual: pd.DataFrame,
    decision_cutoff_by_sku: dict[str, datetime],
) -> pd.DataFrame:
    result = actual.copy()
    result["sku_code"] = result["sku_code"].astype(str)
    days = pd.to_datetime(result["business_day"], errors="raise").dt.date
    cutoff_days = result["sku_code"].map(
        {sku: cutoff.date() for sku, cutoff in decision_cutoff_by_sku.items()}
    )
    keep = cutoff_days.notna() & days.le(cutoff_days)
    return result.loc[keep].reset_index(drop=True)


def _build_historical_validation(
    *,
    decisions: pd.DataFrame,
    full_actual: pd.DataFrame,
    decision_actual: pd.DataFrame,
    first_order_decisions: pd.DataFrame,
    store_targets: pd.DataFrame,
    replay_audit: pd.DataFrame,
    available_through: datetime,
) -> pd.DataFrame:
    actual = full_actual.copy()
    actual["sku_code"] = actual["sku_code"].astype(str)
    actual["store_tier"] = actual["store_tier"].astype(str)
    actual["business_day"] = pd.to_datetime(actual["business_day"], errors="raise").dt.date
    actual["actual_sales"] = pd.to_numeric(actual["actual_sales"], errors="coerce").fillna(0.0)
    first = first_order_decisions.copy()
    first["sku_code"] = first["sku_code"].astype(str)
    cover_by_sku = first.drop_duplicates("sku_code").set_index("sku_code")["cover_weeks"].to_dict()
    targets = store_targets.copy()
    targets["sku_code"] = targets["sku_code"].astype(str)
    targets["store_tier"] = targets["store_tier"].astype(str)
    targets["store_code"] = targets["store_code"].astype(str)
    targets["sellable_demand"] = pd.to_numeric(targets["sellable_demand"], errors="coerce")
    plan = targets.groupby(["sku_code", "store_tier"], as_index=False).agg(
        plan_sellable=("sellable_demand", "sum"),
        store_count=("store_code", "nunique"),
    )
    plan["planned_weekly_velocity"] = plan["plan_sellable"] / plan["sku_code"].map(
        {sku: max(float(value), 1.0) for sku, value in cover_by_sku.items()}
    )
    visible_max = (
        decision_actual.assign(
            business_day=pd.to_datetime(decision_actual["business_day"], errors="raise").dt.date
        )
        .groupby(decision_actual["sku_code"].astype(str))["business_day"]
        .max()
        .to_dict()
    )
    replay = replay_audit.set_index("sku_code").to_dict("index")
    rows: list[dict[str, object]] = []
    for decision in decisions.itertuples(index=False):
        sku = str(decision.sku_code)
        window_start = pd.to_datetime(decision.window_start_date, errors="raise").date()
        window_end = pd.to_datetime(decision.window_end_date, errors="raise").date()
        label = actual.loc[
            actual["sku_code"].eq(sku)
            & actual["business_day"].between(window_start, window_end)
        ].copy()
        sku_plan = plan.loc[plan["sku_code"].eq(sku), [
            "store_tier", "planned_weekly_velocity", "store_count"
        ]]
        label = label.merge(sku_plan, on="store_tier", how="left", validate="many_to_one")
        label["planned_daily_store"] = (
            label["planned_weekly_velocity"].fillna(0.0)
            / label["store_count"].fillna(0.0).clip(lower=1)
            / 7.0
        )
        stock = pd.to_numeric(label["e_stock"], errors="coerce")
        display = pd.to_numeric(label["e_display"], errors="coerce")
        stockout = stock.eq(0).fillna(False).to_numpy(dtype=bool)
        nondisplay = (
            display.eq(0).fillna(False) & stock.ne(0).fillna(False)
        ).to_numpy(dtype=bool)
        repair = np.maximum(label["planned_daily_store"] - label["actual_sales"], 0.0)
        label["stockout_lost"] = np.where(stockout, repair, 0.0)
        label["nondisplay_lost"] = np.where(nondisplay, repair, 0.0)
        label["restored_actual_demand"] = (
            label["actual_sales"] + label["stockout_lost"] + label["nondisplay_lost"]
        )
        actual_sales = float(label["actual_sales"].sum())
        actual_demand = float(label["restored_actual_demand"].sum())
        predicted = float(decision.D_window)
        window_valid = window_end >= window_start
        mature = window_valid and window_end <= available_through.date()
        prediction_error = predicted - actual_demand if mature else np.nan
        rows.append(
            {
                "sku_code": sku,
                **replay[sku],
                "feature_max_business_day": (
                    visible_max[sku].isoformat() if sku in visible_max else None
                ),
                "no_future_data_leak": (
                    visible_max.get(sku) is not None
                    and visible_max[sku]
                    <= pd.to_datetime(decision.decision_cutoff_time).date()
                ),
                "arrival_date": decision.arrival_date,
                "window_start_date": decision.window_start_date,
                "window_end_date": decision.window_end_date,
                "window_valid": window_valid,
                "forecast_source": decision.forecast_source,
                "predicted_D_window": predicted,
                "actual_window_sales": actual_sales,
                "restored_actual_window_demand": actual_demand,
                "prediction_error": prediction_error,
                "absolute_error": abs(prediction_error) if mature else np.nan,
                "absolute_percentage_error": (
                    abs(prediction_error) / max(actual_demand, 1e-6)
                    if mature else np.nan
                ),
                "label_row_count": len(label),
                "label_observed_days": label["business_day"].nunique(),
                "label_mature": mature,
            }
        )
    return pd.DataFrame(rows)


def _read_reorder_actuals(
    *,
    artifact: StageArtifactStore,
    path: Path,
    target_skus: list[str],
    logger,
) -> tuple[pd.DataFrame, str]:
    columns = (
        "sku_code", "store_code", "store_tier", "business_day",
        "e_stock", "e_display",
    )
    optional_columns = ("actual_sales", "clean_sale_qty", "available_stock_qty")
    parts_dir = path.parent / f"{path.stem}_parts"
    part_paths = sorted(parts_dir.glob("*.parquet")) if parts_dir.is_dir() else []
    if part_paths:
        frames: list[pd.DataFrame] = []
        for part_index, part_path in enumerate(part_paths, start=1):
            logger.info(
                "reorder_progress event=actual_part_read_start part=%s parts=%s path=%s",
                part_index, len(part_paths), part_path,
            )
            matching_skus = artifact.read_frame(
                part_path,
                columns=("sku_code",),
                filters={"sku_code": target_skus},
            )
            if matching_skus.empty:
                logger.info(
                    "reorder_progress event=actual_part_skipped part=%s parts=%s reason=no_target_sku",
                    part_index, len(part_paths),
                )
                continue
            del matching_skus
            frame = artifact.read_frame(
                part_path,
                columns=columns,
                optional_columns=optional_columns,
                filters={"sku_code": target_skus},
            )
            frames.append(frame)
            logger.info(
                "reorder_progress event=actual_part_read_complete part=%s parts=%s rows=%s",
                part_index, len(part_paths), len(frame),
            )
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=[*columns, "actual_sales"]),
            "partitioned_parquet",
        )
    frame = artifact.read_frame(
        path,
        columns=columns,
        optional_columns=optional_columns,
        filters={"sku_code": target_skus},
        chunksize=250_000 if path.suffix.lower() == ".csv" else None,
    )
    return frame, "single_artifact"


def _validate_target_coverage(
    *,
    target_skus: list[str],
    store_targets: pd.DataFrame,
    curve: pd.DataFrame,
    actual: pd.DataFrame,
) -> dict[str, int]:
    target_set = set(target_skus)
    sources = {
        "first_order_store_targets": store_targets,
        "demand_curve": curve,
        "actual_daily": actual,
    }
    counts: dict[str, int] = {}
    missing_by_source: dict[str, list[str]] = {}
    for name, frame in sources.items():
        available = (
            set(frame["sku_code"].astype(str).dropna())
            if "sku_code" in frame
            else set()
        )
        counts[name] = len(target_set & available)
        missing = sorted(target_set - available)
        if missing:
            missing_by_source[name] = missing
    if missing_by_source:
        rendered = "; ".join(
            f"{name}: missing={len(values)} examples={values[:10]}"
            for name, values in missing_by_source.items()
        )
        raise DataContractViolation(
            "reorder inputs do not cover the frozen first-order target set; "
            f"targets={len(target_set)}; {rendered}. Rebuild/select every input "
            "artifact for the same target SKU cohort before running N1-N6."
        )
    return counts


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("run_reorder", config.log_dir, config.timezone)
    outputs = ["reorder_decisions.csv", "reorder_attribution.csv", "reorder_demand_forecast.csv", "reorder_diagnostics.csv", "reorder_report.json"]
    if args.decision_age_days is not None:
        outputs.append("reorder_historical_validation.csv")
    if args.dry_run:
        print_dry_run("N1-N6 reorder", args, outputs)
        return 0
    try:
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        artifact = StageArtifactStore()
        logger.info("reorder_progress event=input_read_start source=first_order_decisions")
        decisions = artifact.read_frame(
            PROJECT_ROOT / args.first_order_dir / "first_order_decisions.csv",
            sku_codes=args.sku_code, limit_skus=args.limit_skus,
        )
        target_skus = decisions["sku_code"].astype(str).drop_duplicates().tolist()
        logger.info("reorder_progress event=input_read_complete source=first_order_decisions rows=%s", len(decisions))
        logger.info("reorder_progress event=input_read_start source=first_order_store_targets targets=%s", len(target_skus))
        store_targets = artifact.read_frame(
            PROJECT_ROOT / args.first_order_dir / "first_order_store_targets.csv",
            columns=(
                "sku_code", "store_code", "store_tier", "neutral_demand",
                "sellable_demand", "safety_stock",
            ),
            filters={"sku_code": target_skus},
            chunksize=250_000,
        )
        logger.info("reorder_progress event=input_read_complete source=first_order_store_targets rows=%s", len(store_targets))
        logger.info("reorder_progress event=input_read_start source=demand_curve")
        curve = artifact.read_frame(
            PROJECT_ROOT / args.placement_dir / "demand_curve.csv",
            columns=(
                "sku_code", "store_tier", "lifecycle_week",
                "lifecycle_factor", "period_factor_h",
            ),
            filters={"sku_code": target_skus},
            chunksize=250_000,
        )
        logger.info("reorder_progress event=input_read_complete source=demand_curve rows=%s", len(curve))
        logger.info("reorder_progress event=input_read_start source=actual_daily")
        actual_raw, actual_source_mode = _read_reorder_actuals(
            artifact=artifact,
            path=PROJECT_ROOT / args.actual_file,
            target_skus=target_skus,
            logger=logger,
        )
        actual = standardize_reorder_actuals(actual_raw)
        del actual_raw
        logger.info(
            "reorder_progress event=input_read_complete source=actual_daily rows=%s mode=%s",
            len(actual), actual_source_mode,
        )
        decision_cutoff_by_sku = None
        replay_audit = pd.DataFrame()
        decision_actual = actual
        if args.decision_age_days is not None:
            decision_cutoff_by_sku, replay_audit = _derive_historical_cutoffs(
                actual=actual,
                target_skus=target_skus,
                decision_age_days=args.decision_age_days,
                available_through=args.decision_cutoff_time,
            )
            decision_actual = _filter_point_in_time_actuals(
                actual,
                decision_cutoff_by_sku,
            )
            logger.info(
                "reorder_progress event=historical_point_in_time_split_complete "
                "decision_age_days=%s decision_rows=%s label_source_rows=%s",
                args.decision_age_days,
                len(decision_actual),
                len(actual) - len(decision_actual),
            )
        else:
            visible_mask = (
                pd.to_datetime(actual["business_day"], errors="raise").dt.date
                <= args.decision_cutoff_time.date()
            )
            decision_actual = (
                actual
                if bool(visible_mask.all())
                else actual.loc[visible_mask].reset_index(drop=True)
            )
        coverage = _validate_target_coverage(
            target_skus=target_skus,
            store_targets=store_targets,
            curve=curve,
            actual=decision_actual,
        )
        logger.info(
            "reorder_progress event=input_coverage_complete targets=%s coverage=%s",
            len(target_skus), coverage,
        )
        constraints = (
            None
            if args.constraints_file is None
            else artifact.read_frame(
                PROJECT_ROOT / args.constraints_file,
                filters={"sku_code": target_skus},
            )
        )
        result = calculate_reorder(
            decision_cutoff_time=args.decision_cutoff_time,
            first_order_decisions=decisions,
            first_order_store_targets=store_targets,
            actual_daily=decision_actual,
            demand_curve=curve,
            config=parameters["reorder"],
            constraints=constraints,
            decision_cutoff_by_sku=decision_cutoff_by_sku,
            progress=lambda event, details: logger.info(
                "reorder_progress event=%s details=%s", event, details
            ),
        )
        output_dir = PROJECT_ROOT / args.output_dir
        logger.info("reorder_progress event=output_write_start targets=%s", len(result.decisions))
        for name, frame in zip(outputs[:4], [result.decisions, result.attribution, result.demand_forecast, result.diagnostics]):
            artifact.write_frame(output_dir / name, frame)
        historical_validation = pd.DataFrame()
        if args.decision_age_days is not None:
            historical_validation = _build_historical_validation(
                decisions=result.decisions,
                full_actual=actual,
                decision_actual=decision_actual,
                first_order_decisions=decisions,
                store_targets=store_targets,
                replay_audit=replay_audit,
                available_through=args.decision_cutoff_time,
            )
            artifact.write_frame(
                output_dir / "reorder_historical_validation.csv",
                historical_validation,
            )
        artifact.write_json(output_dir / outputs[4], {
            "spec_version": "3.2", "round": 9, "stage": "N1-N6 reorder", "status": (
                "historical_point_in_time_validation"
                if args.decision_age_days is not None
                else "partial_rule_scaffold_requires_training_and_business_inputs"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "target_count": len(result.decisions), "warnings": list(result.warnings),
            "first_order_snapshot_frozen": True, "manual_parameter_status": parameters["manual_parameter_status"],
            "actual_source_mode": actual_source_mode,
            "input_sku_coverage": coverage,
            "historical_replay": (
                {
                    "enabled": True,
                    "decision_age_days": args.decision_age_days,
                    "future_rows_excluded_from_decision": True,
                    "validation_file": "reorder_historical_validation.csv",
                    "mature_label_count": int(historical_validation["label_mature"].sum()),
                }
                if args.decision_age_days is not None
                else {"enabled": False}
            ),
            "mysql_written": False,
            "v3_2_unavailable_logic": {
                "residual_model": "Mres is not trained; current output uses the Actual/Plan main path only.",
                "inventory_and_supply": "Formal warehouse, inbound ETA, protected inventory, lead schedule, MOQ and capacity require a constraints snapshot.",
                "quality": "CR_quality remains 1 unless the refund/sales semantic contract is verified.",
                "reliability": "CR_reliability is uncalibrated until sufficient future-window labels exist."
            },
        })
        logger.info("reorder_progress event=output_write_complete output_dir=%s", output_dir)
        logger.info("N1-N6 complete targets=%s", len(result.decisions))
    except Exception as exc:
        log_exception(logger, module="run_reorder", cutoff=args.decision_cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    status = (
        "historical_point_in_time_validation"
        if args.decision_age_days is not None
        else "partial_rule_scaffold_requires_training_and_business_inputs"
    )
    validation_line = (
        f"\nvalidation: {output_dir / 'reorder_historical_validation.csv'}"
        if args.decision_age_days is not None
        else ""
    )
    print(
        f"status: {status}\ndecisions: {output_dir / outputs[0]}"
        f"{validation_line}\nreport: {output_dir / outputs[4]}\nlog: {log_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
