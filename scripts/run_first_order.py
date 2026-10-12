#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import add_execution_arguments, print_dry_run  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.decision_adapter import DecisionDataAdapter  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402
from first_order import calculate_first_order  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.2 M6/M7 first-order only; does not execute M1-M5.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--placement-dir", type=Path, default=Path("artifacts/round6_placement"))
    parser.add_argument("--factor-dir", type=Path, default=Path("artifacts/round7_factors"))
    parser.add_argument("--comparable-dir", type=Path, default=Path("artifacts/round5_comparable"))
    parser.add_argument("--constraints-file", type=Path, default=None)
    parser.add_argument("--prediction-interval-file", type=Path, default=None)
    add_execution_arguments(parser, default_output="artifacts/round7_first_order")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("run_first_order", config.log_dir, config.timezone)
    outputs = ["first_order_decisions.csv", "first_order_store_targets.csv", "first_order_predictions.csv", "first_order_calculation_detail.csv", "first_order_report.json"]
    if args.dry_run:
        print_dry_run("M6/M7 first order", args, outputs)
        return 0
    try:
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        artifact = StageArtifactStore()
        placement = artifact.read_frame(PROJECT_ROOT / args.placement_dir / "placement_plan.csv")
        curve = artifact.read_frame(PROJECT_ROOT / args.placement_dir / "demand_curve.csv")
        adjustments = artifact.read_frame(
            PROJECT_ROOT / args.factor_dir / "demand_adjustments.csv",
            sku_codes=args.sku_code, limit_skus=args.limit_skus,
        )
        target_skus = adjustments["sku_code"].astype(str).drop_duplicates().tolist()
        placement = placement.loc[placement["sku_code"].astype(str).isin(target_skus)]
        curve = curve.loc[curve["sku_code"].astype(str).isin(target_skus)]
        comparable_summary = artifact.read_frame(PROJECT_ROOT / args.comparable_dir / "comparable_summary.csv")
        adapter = DecisionDataAdapter.from_project(project_root=PROJECT_ROOT, data_root=args.data_root, config=config)
        targets = adapter.products(args.decision_cutoff_time, sku_codes=target_skus).products
        constraints = None if args.constraints_file is None else artifact.read_frame(PROJECT_ROOT / args.constraints_file)
        intervals = None if args.prediction_interval_file is None else artifact.read_frame(PROJECT_ROOT / args.prediction_interval_file)
        result = calculate_first_order(
            targets=targets, placement_plan=placement, demand_curve=curve,
            adjustments=adjustments, comparable_summary=comparable_summary,
            config=parameters["first_order"], factor_config=parameters["factors"],
            constraints=constraints, prediction_intervals=intervals,
        )
        output_dir = PROJECT_ROOT / args.output_dir
        for name, frame in zip(outputs[:4], [result.decisions, result.store_targets, result.predictions, result.calculation_detail]):
            artifact.write_frame(output_dir / name, frame)
        confirmed_constraints = bool(args.constraints_file)
        artifact.write_json(output_dir / outputs[4], {
            "spec_version": "3.2", "round": 7, "stage": "M6/M7 first order",
            "parameter_version": parameters["parameter_version"],
            "status": (
                "partial_v3_2_output_requires_real_training"
                if confirmed_constraints
                else "declared_reference_smoke_output"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "target_count": len(result.decisions), "warnings": list(result.warnings),
            "default_parameter_basis": parameters["first_order"].get("default_parameter_basis", {}),
            "manual_parameter_status": parameters["manual_parameter_status"], "mysql_written": False,
        })
        logger.info("M6/M7 complete targets=%s", len(result.decisions))
    except Exception as exc:
        log_exception(logger, module="run_first_order", cutoff=args.decision_cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(f"status: {('partial_v3_2_output_requires_real_training' if args.constraints_file else 'declared_reference_smoke_output')}\ndecisions: {output_dir / outputs[0]}\nreport: {output_dir / outputs[4]}\nlog: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
