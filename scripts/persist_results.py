#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import add_execution_arguments, print_dry_run  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig, DatabaseConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402
from db.connection import Database  # noqa: E402
from db.result_writer import ResultWriter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist one completed stage to append-only MySQL tables.")
    parser.add_argument("--stage", choices=("comparable", "first_order", "reorder"), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    add_execution_arguments(parser, default_output="artifacts/persistence_receipts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("persist_results", config.log_dir, config.timezone)
    if args.dry_run:
        print_dry_run(f"persist {args.stage}", args, ["MySQL append-only rows"])
        return 0
    try:
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        artifact = StageArtifactStore()
        directory = PROJECT_ROOT / args.artifact_dir
        writer = ResultWriter(Database(DatabaseConfig.from_env()))
        metadata = {
            "parameter_version": parameters["parameter_version"],
            "rule_version": parameters["rule_version"],
            "trigger_source": "persist_results.py",
        }
        if args.stage == "comparable":
            frame = artifact.read_frame(directory / "candidate_detail.csv", sku_codes=args.sku_code, limit_skus=args.limit_skus, sku_column="target_sku_code")
            run_ids = writer.write_comparables(cutoff_time=args.decision_cutoff_time, candidates=frame, metadata=metadata)
        elif args.stage == "first_order":
            decisions = artifact.read_frame(directory / "first_order_decisions.csv", sku_codes=args.sku_code, limit_skus=args.limit_skus)
            details = artifact.read_frame(directory / "first_order_calculation_detail.csv")
            run_ids = writer.write_first_order(cutoff_time=args.decision_cutoff_time, decisions=decisions, calculations=details, metadata=metadata)
        else:
            decisions = artifact.read_frame(directory / "reorder_decisions.csv", sku_codes=args.sku_code, limit_skus=args.limit_skus)
            attribution = artifact.read_frame(directory / "reorder_attribution.csv")
            run_ids = writer.write_reorder(cutoff_time=args.decision_cutoff_time, decisions=decisions, attribution=attribution, metadata=metadata)
        logger.info("persistence complete stage=%s runs=%s", args.stage, len(run_ids))
    except Exception as exc:
        log_exception(logger, module="persist_results", cutoff=args.decision_cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(f"status: persisted\nrun_ids: {','.join(run_ids)}\nlog: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
