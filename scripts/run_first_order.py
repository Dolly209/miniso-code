#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import (  # noqa: E402
    add_execution_arguments,
    parse_cutoff,
    print_dry_run,
)
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.decision_adapter import DecisionDataAdapter  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402
from first_order import calculate_first_order  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.2 M6/M7 first-order only; does not execute M1-M5.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, default=None,
                        help="Frozen Step5 trained model. Rebuild M3-M7 with its theta/tau/K and frozen CDF.")
    parser.add_argument("--recompute-input-dir", type=Path, default=None,
                        help="Canonical _recompute_inputs directory produced by placement/factors; no raw rescan.")
    parser.add_argument("--placement-dir", type=Path, default=Path("artifacts/round6_placement"))
    parser.add_argument("--factor-dir", type=Path, default=Path("artifacts/round7_factors"))
    parser.add_argument("--comparable-dir", type=Path, default=Path("artifacts/round5_comparable"))
    parser.add_argument("--constraints-file", type=Path, default=None)
    parser.add_argument("--prediction-interval-file", type=Path, default=None)
    parser.add_argument("--arrival-plan-file", type=Path, default=None,
                        help="Optional business-confirmed input, NOT actual arrivals: sku_code,store_code,arrival_week,available_time,confirmed. Without it normal launch starts all selected stores in week1.")
    parser.add_argument(
        "--product-snapshot-time",
        type=parse_cutoff,
        default=None,
        help=(
            "Optional fixed snapshot time for static product-master attributes. "
            "Dynamic inputs remain bounded by --decision-cutoff-time."
        ),
    )
    add_execution_arguments(parser, default_output="artifacts/round8_first_order")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("run_first_order", config.log_dir, config.timezone)
    outputs = ["first_order_decisions.csv", "first_order_store_targets.csv", "first_order_predictions.csv", "first_order_calculation_detail.csv", "first_order_report.json"]
    outputs.append("first_order_weekly_demand.csv")
    if args.model_file:
        outputs.extend(["trained_selected_comparables.csv", "trained_placement_plan.csv",
                        "trained_demand_curve.csv", "trained_demand_adjustments.csv"])
    if args.dry_run:
        print_dry_run("M6/M7 first order", args, outputs)
        print(
            "product_snapshot_time: "
            f"{(args.product_snapshot_time or args.decision_cutoff_time).isoformat()}"
        )
        return 0
    try:
        if args.model_file is not None:
            return _run_frozen_model(args, logger, log_path)
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        artifact = StageArtifactStore()
        logger.info("first_order_progress event=input_read_start source=adjustments")
        adjustments = artifact.read_frame(
            PROJECT_ROOT / args.factor_dir / "demand_adjustments.csv",
            sku_codes=args.sku_code, limit_skus=args.limit_skus,
        )
        target_skus = adjustments["sku_code"].astype(str).drop_duplicates().tolist()
        logger.info("first_order_progress event=input_read_complete source=adjustments rows=%s", len(adjustments))
        logger.info("first_order_progress event=input_read_start source=selected_placement targets=%s", len(target_skus))
        placement = artifact.read_frame(
            PROJECT_ROOT / args.placement_dir / "placement_plan.csv",
            columns=("sku_code", "store_code", "store_tier", "store_level", "selected", "v_anchor_week1"),
            optional_columns=("placement_origin", "productive_probability", "store_score", "tier_week1_anchor", "tier_anchor_status"),
            filters={"sku_code": target_skus, "selected": 1},
            chunksize=250_000,
        )
        logger.info("first_order_progress event=input_read_complete source=selected_placement rows=%s", len(placement))
        placement_summary = artifact.read_frame(PROJECT_ROOT / args.placement_dir / "placement_summary.csv", filters={"sku_code": target_skus})
        curve = artifact.read_frame(
            PROJECT_ROOT / args.placement_dir / "demand_curve.csv",
            columns=("sku_code", "store_tier", "lifecycle_week", "lifecycle_factor", "period_factor_h"),
            optional_columns=("curve_source", "curve_point_status", "curve_profile_validation", "lifecycle_component_observed", "lifecycle_component_interpolated", "lifecycle_component_edge_carried", "lifecycle_component_flat_default"),
            filters={"sku_code": target_skus},
            chunksize=250_000,
        )
        logger.info("first_order_progress event=input_read_complete source=demand_curve rows=%s", len(curve))
        comparable_summary = artifact.read_frame(
            PROJECT_ROOT / args.comparable_dir / "comparable_summary.csv",
            columns=("sku_code", "path_reliability", "selected_candidate_count"),
            filters={"sku_code": target_skus},
        )
        adapter = DecisionDataAdapter.from_project(project_root=PROJECT_ROOT, data_root=args.data_root, config=config)
        logger.info("first_order_progress event=input_read_start source=target_products")
        product_snapshot_time = (
            args.product_snapshot_time or args.decision_cutoff_time
        )
        targets = adapter.products(
            product_snapshot_time, sku_codes=target_skus
        ).products
        logger.info("first_order_progress event=input_read_complete source=target_products rows=%s", len(targets))
        constraints = None if args.constraints_file is None else artifact.read_frame(PROJECT_ROOT / args.constraints_file)
        intervals = None if args.prediction_interval_file is None else artifact.read_frame(PROJECT_ROOT / args.prediction_interval_file)
        arrival_plan = None
        if args.arrival_plan_file:
            arrival_plan = artifact.read_frame(PROJECT_ROOT / args.arrival_plan_file,
                required=("store_code", "arrival_week", "available_time", "confirmed"))
            available = pd.to_datetime(arrival_plan.available_time, errors="coerce", utc=True)
            timezone_declared = arrival_plan.available_time.astype(str).str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
            confirmed = arrival_plan.confirmed.astype(str).str.lower().isin(("1", "true", "yes"))
            if available.isna().any() or not timezone_declared.all() or available.gt(args.decision_cutoff_time).any() or not confirmed.all():
                raise ValueError("arrival plan must be confirmed and available by decision_cutoff_time; actual later arrivals cannot replace a prelaunch plan")
        from data.recompute import write_recompute_frames, seal_recompute_bundle
        bundle_root = PROJECT_ROOT / args.placement_dir / "_recompute_inputs"
        if (bundle_root / "manifest.json").is_file():
            write_recompute_frames(bundle_root, {"products": targets, "constraints": constraints, "arrival_plan": arrival_plan})
            seal_recompute_bundle(bundle_root, parameters, args.decision_cutoff_time.isoformat())
        result = calculate_first_order(
            targets=targets, placement_plan=placement, demand_curve=curve,
            adjustments=adjustments, comparable_summary=comparable_summary,
            config=parameters["first_order"], factor_config=parameters["factors"],
            constraints=constraints, prediction_intervals=intervals,
            arrival_plan=arrival_plan,
            placement_summary=placement_summary,
            progress=lambda event, details: logger.info(
                "first_order_progress event=%s details=%s", event, details
            ),
        )
        output_dir = PROJECT_ROOT / args.output_dir
        artifact.write_frame(output_dir / "first_order_weekly_demand.csv", result.weekly_demand)
        logger.info("first_order_progress event=output_write_start store_rows=%s", len(result.store_targets))
        first_order_frames = [
            frame.assign(feature_cutoff_time=args.decision_cutoff_time.isoformat())
            for frame in [
                result.decisions,
                result.store_targets,
                result.predictions,
                result.calculation_detail,
            ]
        ]
        for name, frame in zip(outputs[:4], first_order_frames):
            artifact.write_frame(output_dir / name, frame)
        logger.info("first_order_progress event=output_write_complete output_dir=%s", output_dir)
        confirmed_constraints = bool(args.constraints_file)
        blocked_count = int(result.decisions.decision_status.ne("calculated").sum())
        artifact.write_json(output_dir / outputs[4], {
            "spec_version": "3.2", "round": 8, "stage": "M6/M7 first order",
            "parameter_version": parameters["parameter_version"],
            "status": (
                "completed_with_blocked_targets"
                if blocked_count else
                "partial_v3_2_output_requires_real_training"
                if confirmed_constraints
                else "declared_reference_smoke_output"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "product_snapshot_time": product_snapshot_time.isoformat(),
            "product_snapshot_policy": (
                "fixed audited static product-master snapshot; dynamic facts remain "
                "bounded by decision_cutoff_time"
                if args.product_snapshot_time is not None
                else "latest product snapshot on or before decision_cutoff_time"
            ),
            "target_count": len(result.decisions), "warnings": list(result.warnings),
            "blocked_target_count": blocked_count,
            "default_parameter_basis": parameters["first_order"].get("default_parameter_basis", {}),
            "manual_parameter_status": parameters["manual_parameter_status"], "mysql_written": False,
        })
        logger.info("M6/M7 complete targets=%s", len(result.decisions))
    except Exception as exc:
        log_exception(logger, module="run_first_order", cutoff=args.decision_cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(f"status: {('completed_with_blocked_targets' if blocked_count else 'partial_v3_2_output_requires_real_training' if args.constraints_file else 'declared_reference_smoke_output')}\ndecisions: {output_dir / outputs[0]}\nreport: {output_dir / outputs[4]}\nlog: {log_path}")
    return 0


def _run_frozen_model(args, logger, log_path) -> int:
    from data.recompute import load_recompute_bundle
    from first_order.pipeline import full_first_order
    from training.comparable_tuning import ComparableParameterModel, select_comparable_parameters
    from training.section84 import DemandResidualModel
    if args.recompute_input_dir is None:
        raise ValueError("--model-file requires --recompute-input-dir; old scalar-only artifacts cannot reproduce full inference")
    if args.constraints_file or args.arrival_plan_file or args.prediction_interval_file:
        raise ValueError("trained inference uses the versioned business inputs embedded in its recompute bundle; supply these when generating that bundle instead of overriding them silently")
    store = StageArtifactStore()
    payload = store.read_json(PROJECT_ROOT / args.model_file)
    model = DemandResidualModel.from_dict(payload)
    if not model.frozen_transform or payload.get("pipeline_version") != "step5-full-chain-v1":
        raise ValueError("model has no frozen Step5 pipeline; train with updated canonical replay inputs")
    context = load_recompute_bundle(PROJECT_ROOT / args.recompute_input_dir)
    if pd.Timestamp(context["cutoff"]) != pd.Timestamp(args.decision_cutoff_time):
        raise ValueError("canonical inputs cutoff differs from requested decision cutoff")
    if context["parameters"] != payload["pipeline_parameters"]:
        raise ValueError("canonical rule/config version differs from frozen model; do not mix versions")
    comparable_model = ComparableParameterModel.from_dict(payload["comparable_model"])
    detail = context["frames"]["comparables"]
    codes = list(dict.fromkeys(detail.target_sku_code.astype(str)))
    if args.sku_code:
        missing = set(args.sku_code) - set(codes)
        if missing:
            raise ValueError(f"target SKUs absent in canonical inputs: {sorted(missing)}")
        codes = list(args.sku_code)
    if args.limit_skus is not None:
        codes = codes[:args.limit_skus]
    results = []
    selected_parts = []
    placement_parts, curve_parts, factor_parts = [], [], []
    for index, sku in enumerate(codes, 1):
        logger.info("trained_first_order_progress target=%s total=%s sku=%s", index, len(codes), sku)
        chosen = select_comparable_parameters(detail.loc[detail.target_sku_code.astype(str).eq(sku)], comparable_model)
        result, recomputed_placement, recomputed_factors = full_first_order({**context, "target_sku": sku}, chosen,
            frozen_transform=model.frozen_transform, learned_model=model)
        results.append(result)
        selected_parts.append(chosen)
        placement_parts.append(recomputed_placement.store_plan)
        curve_parts.append(recomputed_placement.demand_curve)
        factor_parts.append(recomputed_factors)
    output = PROJECT_ROOT / args.output_dir
    for attribute, filename in (("decisions", "first_order_decisions.csv"),
            ("store_targets", "first_order_store_targets.csv"), ("predictions", "first_order_predictions.csv"),
            ("calculation_detail", "first_order_calculation_detail.csv"), ("weekly_demand", "first_order_weekly_demand.csv")):
        store.write_frame(output / filename, pd.concat([getattr(r, attribute) for r in results], ignore_index=True).assign(model_id=payload["model_id"]))
    store.write_frame(output / "trained_selected_comparables.csv", pd.concat(selected_parts, ignore_index=True))
    for name, pieces in (("trained_placement_plan.csv", placement_parts),
                         ("trained_demand_curve.csv", curve_parts), ("trained_demand_adjustments.csv", factor_parts)):
        store.write_frame(output / name, pd.concat(pieces, ignore_index=True).assign(model_id=payload["model_id"]))
    store.write_json(output / "first_order_report.json", {
        "status": "frozen_trained_full_chain_inference", "model_id": payload["model_id"],
        "pipeline_version": payload["pipeline_version"], "cdf_transform_id": model.frozen_transform["transform_id"],
        "target_count": len(results), "model_refit": False,
        "blocked_target_count": sum(int(r.decisions.decision_status.ne("calculated").sum()) for r in results),
        "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
        "warnings": list(dict.fromkeys(w for r in results for w in r.warnings))})
    print(f"status: frozen_trained_full_chain_inference\nreport: {output / 'first_order_report.json'}\nlog: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
