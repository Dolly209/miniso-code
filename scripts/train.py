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

from _pipeline_common import add_execution_arguments, print_dry_run  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402
from training import rolling_oof_train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.2 Train-only rolling OOF and full-Train fit.")
    parser.add_argument("--modeling-dataset", type=Path, default=Path("artifacts/training/modeling_dataset.csv"))
    parser.add_argument("--feature-column", action="append", default=[])
    parser.add_argument("--allow-signed-coefficients", action="store_true")
    add_execution_arguments(parser, default_output="artifacts/round8_training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("train", config.log_dir, config.timezone)
    outputs = ["model.json", "oof_predictions.csv", "model_selection.csv", "training_report.json"]
    if args.dry_run:
        print_dry_run("rolling OOF training", args, outputs)
        return 0
    try:
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        artifact = StageArtifactStore()
        frame = artifact.read_frame(
            PROJECT_ROOT / args.modeling_dataset,
            sku_codes=args.sku_code, limit_skus=args.limit_skus,
        )
        features = args.feature_column or [
            name
            for name in frame.columns
            if name.startswith("feature__")
            and pd.to_numeric(frame[name], errors="coerce").notna().any()
        ]
        if not features:
            raise ValueError("no features selected; use --feature-column or feature__* columns")
        result = rolling_oof_train(
            frame, feature_columns=features, n_time_blocks=config.training.n_time_blocks,
            ridge_grid=parameters["training"]["ridge_grid"],
            l1_ratio_grid=parameters["training"]["l1_ratio_grid"],
            half_life_grid=parameters["training"]["half_life_weeks"],
            nonnegative=not args.allow_signed_coefficients,
        )
        output_dir = PROJECT_ROOT / args.output_dir
        artifact.write_json(output_dir / outputs[0], result.model.as_dict())
        artifact.write_frame(output_dir / outputs[1], result.oof_predictions)
        artifact.write_frame(output_dir / outputs[2], result.selection)
        report = dict(result.report)
        report.update({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "status": "partial_training_scaffold_not_full_v3_2",
            "real_training_required": True,
            "server_training_verified": False,
            "v3_2_missing_training_blocks": [
                "leave-one-SKU-out similarity weight/temperature/K tuning",
                "joint constrained demand-factor and price-elasticity fitting",
                "category lifecycle random-effects prior",
                "reorder residual model and reliability calibration"
            ],
        })
        artifact.write_json(output_dir / outputs[3], report)
        logger.info("training complete train_skus=%s oof_rows=%s", report["train_skus"], report["oof_rows"])
    except Exception as exc:
        log_exception(logger, module="train", cutoff=args.decision_cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(f"status: partial_training_scaffold_not_full_v3_2\nmodel: {output_dir / outputs[0]}\nreport: {output_dir / outputs[3]}\nlog: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
