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
from data.domain import load_algorithm_parameters  # noqa: E402
from training import FittedModel, replay_test  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the locked V3.2 model on Test once.")
    parser.add_argument("--modeling-dataset", type=Path, default=Path("artifacts/training/modeling_dataset.csv"))
    parser.add_argument("--training-dir", type=Path, default=Path("artifacts/round8_training"))
    add_execution_arguments(parser, default_output="artifacts/round8_backtest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("backtest", config.log_dir, config.timezone)
    outputs = ["test_predictions.csv", "backtest_report.json"]
    if args.dry_run:
        print_dry_run("locked Test replay", args, outputs)
        return 0
    try:
        artifact = StageArtifactStore()
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        frame = artifact.read_frame(
            PROJECT_ROOT / args.modeling_dataset,
            sku_codes=args.sku_code, limit_skus=args.limit_skus,
        )
        model = FittedModel.from_dict(artifact.read_json(PROJECT_ROOT / args.training_dir / "model.json"))
        training_report = artifact.read_json(PROJECT_ROOT / args.training_dir / "training_report.json")
        predictions, report = replay_test(
            frame, model,
            interval_log_quantiles=training_report.get("oof_log_residual_quantiles"),
            grade_thresholds=parameters["first_order"]["grade_thresholds"],
        )
        report.update({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "model_selection_reopened": False,
        })
        output_dir = PROJECT_ROOT / args.output_dir
        artifact.write_frame(output_dir / outputs[0], predictions)
        artifact.write_json(output_dir / outputs[1], report)
        logger.info("backtest complete test_skus=%s", report["test_skus"])
    except Exception as exc:
        log_exception(logger, module="backtest", cutoff=args.decision_cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(f"status: test_replayed_once\npredictions: {output_dir / outputs[0]}\nreport: {output_dir / outputs[1]}\nlog: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
