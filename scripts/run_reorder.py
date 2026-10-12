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
from data.decision_adapter import standardize_reorder_actuals  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402
from reorder import calculate_reorder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.2 N1-N6 reorder only from a frozen first-order snapshot.")
    parser.add_argument("--first-order-dir", type=Path, default=Path("artifacts/round7_first_order"))
    parser.add_argument("--placement-dir", type=Path, default=Path("artifacts/round6_placement"))
    parser.add_argument("--actual-file", type=Path, default=Path("artifacts/round4_restore/daily_exposure_audit.csv"))
    parser.add_argument("--constraints-file", type=Path, default=None)
    add_execution_arguments(parser, default_output="artifacts/round9_reorder")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("run_reorder", config.log_dir, config.timezone)
    outputs = ["reorder_decisions.csv", "reorder_attribution.csv", "reorder_demand_forecast.csv", "reorder_diagnostics.csv", "reorder_report.json"]
    if args.dry_run:
        print_dry_run("N1-N6 reorder", args, outputs)
        return 0
    try:
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        artifact = StageArtifactStore()
        decisions = artifact.read_frame(
            PROJECT_ROOT / args.first_order_dir / "first_order_decisions.csv",
            sku_codes=args.sku_code, limit_skus=args.limit_skus,
        )
        target_skus = decisions["sku_code"].astype(str).drop_duplicates().tolist()
        store_targets = artifact.read_frame(PROJECT_ROOT / args.first_order_dir / "first_order_store_targets.csv")
        store_targets = store_targets.loc[store_targets["sku_code"].astype(str).isin(target_skus)]
        curve = artifact.read_frame(PROJECT_ROOT / args.placement_dir / "demand_curve.csv")
        curve = curve.loc[curve["sku_code"].astype(str).isin(target_skus)]
        actual_raw = artifact.read_frame(PROJECT_ROOT / args.actual_file)
        actual = standardize_reorder_actuals(actual_raw)
        actual = actual.loc[actual["sku_code"].astype(str).isin(target_skus)]
        constraints = None if args.constraints_file is None else artifact.read_frame(PROJECT_ROOT / args.constraints_file)
        result = calculate_reorder(
            decision_cutoff_time=args.decision_cutoff_time,
            first_order_decisions=decisions,
            first_order_store_targets=store_targets,
            actual_daily=actual,
            demand_curve=curve,
            config=parameters["reorder"],
            constraints=constraints,
        )
        output_dir = PROJECT_ROOT / args.output_dir
        for name, frame in zip(outputs[:4], [result.decisions, result.attribution, result.demand_forecast, result.diagnostics]):
            artifact.write_frame(output_dir / name, frame)
        artifact.write_json(output_dir / outputs[4], {
            "spec_version": "3.2", "round": 9, "stage": "N1-N6 reorder", "status": "partial_rule_scaffold_requires_training_and_business_inputs",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "target_count": len(result.decisions), "warnings": list(result.warnings),
            "first_order_snapshot_frozen": True, "manual_parameter_status": parameters["manual_parameter_status"],
            "mysql_written": False,
            "v3_2_unavailable_logic": {
                "residual_model": "Mres is not trained; current output uses the Actual/Plan main path only.",
                "inventory_and_supply": "Formal warehouse, inbound ETA, protected inventory, lead schedule, MOQ and capacity require a constraints snapshot.",
                "quality": "CR_quality remains 1 unless the refund/sales semantic contract is verified.",
                "reliability": "CR_reliability is uncalibrated until sufficient future-window labels exist."
            },
        })
        logger.info("N1-N6 complete targets=%s", len(result.decisions))
    except Exception as exc:
        log_exception(logger, module="run_reorder", cutoff=args.decision_cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(f"status: partial_rule_scaffold_requires_training_and_business_inputs\ndecisions: {output_dir / outputs[0]}\nreport: {output_dir / outputs[4]}\nlog: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
