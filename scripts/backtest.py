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
from training.pilot import evaluation_scope_metadata  # noqa: E402
from data.recompute import ReplayBundleStore  # noqa: E402
from first_order.pipeline import FullChainReplay  # noqa: E402
from training import (  # noqa: E402
    ComparableParameterModel,
    DemandResidualModel,
    FittedModel,
    replay_section84_locked_test,
    replay_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the locked V3.2 model on Test once.")
    parser.add_argument("--replay-dir", type=Path, default=None,
                        help="Final replay root with Step5 recompute bundles; required for frozen full-chain models.")
    parser.add_argument("--modeling-dataset", type=Path, default=Path("artifacts/training/modeling_dataset.csv"))
    parser.add_argument("--training-dir", type=Path, default=Path("artifacts/round8_training"))
    parser.add_argument(
        "--comparable-detail",
        type=Path,
        default=None,
        help=(
            "Final checkpoint-keyed comparable_candidate_detail.csv. Required "
            "for a Section 8.4 model so locked Test uses frozen theta/tau/K."
        ),
    )
    parser.add_argument(
        "--allow-unverified-point-in-time-smoke",
        action="store_true",
        help="Replay an explicitly non-formal smoke model/dataset.",
    )
    parser.add_argument(
        "--allow-partial-v3-2-scaffold-smoke",
        action="store_true",
        help=(
            "Replay the legacy generic training scaffold for code smoke only; "
            "never label the output as a formal V3.2 historical backtest."
        ),
    )
    parser.add_argument(
        "--allow-locked-test-rerun",
        action="store_true",
        help=(
            "Explicitly replace an existing locked-Test report. Omit for the "
            "V3.2 one-time final Test replay."
        ),
    )
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
        verified = (
            "point_in_time_verified" in frame
            and frame["point_in_time_verified"].astype(str).str.lower().isin({"true", "1"}).all()
        )
        if not verified and not args.allow_unverified_point_in_time_smoke:
            raise ValueError("modeling dataset is not point-in-time verified; formal backtest blocked")
        model_payload = artifact.read_json(
            PROJECT_ROOT / args.training_dir / "model.json"
        )
        training_report = artifact.read_json(PROJECT_ROOT / args.training_dir / "training_report.json")
        output_dir = PROJECT_ROOT / args.output_dir
        existing_report = output_dir / outputs[1]
        if existing_report.is_file() and not args.allow_locked_test_rerun:
            raise ValueError(
                "locked Test has already been replayed in this output directory; "
                "use a new --output-dir, or explicitly pass "
                "--allow-locked-test-rerun only when replacing an invalid run"
            )
        section84_model = (
            model_payload.get("artifact_type")
            == "v3_2_first_order_demand_residual_model"
        )
        if section84_model:
            if training_report.get("status") != (
                "section_8_4_first_order_train_complete_test_still_locked"
            ):
                raise ValueError(
                    "Section 8.4 model/report status mismatch; locked Test remains blocked"
                )
            model = DemandResidualModel.from_dict(model_payload)
            if args.comparable_detail is None:
                raise ValueError(
                    "--comparable-detail is required for the Section 8.4 "
                    "locked-Test replay"
                )
            comparable_payload = model_payload.get("comparable_model")
            if not isinstance(comparable_payload, dict):
                raise ValueError(
                    "Section 8.4 model artifact has no frozen comparable model"
                )
            comparable_model = ComparableParameterModel.from_dict(
                comparable_payload
            )
            comparable_detail = artifact.read_frame(
                PROJECT_ROOT / args.comparable_detail
            )
            grade_settings = parameters["first_order"]["grade_thresholds"]
            own_thresholds = [
                grade_settings["own_ip"][name]
                for name in ("C_max", "B_max", "A_max")
            ]
            licensed_thresholds = [
                grade_settings["licensed_ip"][name]
                for name in ("C_max", "B_max", "A_max")
            ]
            if own_thresholds != licensed_thresholds:
                raise ValueError(
                    "own/licensed grade thresholds differ; copyright-specific "
                    "locked-Test grading is required"
                )
            full_replay = None
            if model.frozen_transform:
                if args.replay_dir is None:
                    raise ValueError("--replay-dir is required for Step5 full-chain backtest")
                full_replay = FullChainReplay(ReplayBundleStore(PROJECT_ROOT / args.replay_dir, model_payload["pipeline_parameters"]),
                    key_column="prediction_checkpoint_key", base_demand_column=model.base_demand_column)
            predictions, report = replay_section84_locked_test(
                frame,
                model,
                candidate_detail=comparable_detail,
                comparable_model=comparable_model,
                interval_log_quantiles=model_payload[
                    "oof_log_residual_quantiles"
                ],
                grade_thresholds=own_thresholds,
                sales_band_edges=parameters["training"]["sales_band_edges"],
                full_replay=full_replay,
            )
            report.update(
                {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
                    "point_in_time_verified": verified,
                    "backtest_mode": "section_8_4_locked_test",
                    "training_status": training_report.get("status"),
                    "training_mode": training_report.get("training_mode", "legacy_unspecified"),
                    "fixed_hyperparameter_preset": model_payload.get("fixed_hyperparameter_preset"),
                    "model_id": model_payload.get("model_id"),
                    "dataset_version": model_payload.get("dataset_version"),
                    "dataset_hash": model_payload.get("dataset_hash"),
                }
            )
            report.update(evaluation_scope_metadata(frame))
            if report["pilot_versions"]:
                report["backtest_mode"] = "section_8_4_development_pilot"
            artifact.write_frame(output_dir / outputs[0], predictions)
            artifact.write_json(output_dir / outputs[1], report)
            logger.info("backtest complete test_skus=%s", report["test_skus"])
            print(
                "status: locked_test_replayed_once\n"
                f"predictions: {output_dir / outputs[0]}\n"
                f"report: {output_dir / outputs[1]}\n"
                f"log: {log_path}"
            )
            return 0

        model = FittedModel.from_dict(model_payload)
        full_v3_2_training = (
            training_report.get("status") == "full_v3_2_training_complete"
        )
        if (
            not full_v3_2_training
            and not args.allow_partial_v3_2_scaffold_smoke
        ):
            raise ValueError(
                "formal V3.2 backtest is blocked because the supplied model is not "
                "marked full_v3_2_training_complete; use "
                "--allow-partial-v3-2-scaffold-smoke only for code smoke tests"
            )
        if (
            not training_report.get("point_in_time_verified", False)
            and not args.allow_unverified_point_in_time_smoke
        ):
            raise ValueError("training model was not fitted from point-in-time verified features")
        predictions, report = replay_test(
            frame, model,
            interval_log_quantiles=training_report.get("oof_log_residual_quantiles"),
            grade_thresholds=parameters["first_order"]["grade_thresholds"],
        )
        report.update({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "model_selection_reopened": False,
            "point_in_time_verified": verified,
            "backtest_mode": (
                "formal_locked_test"
                if verified and full_v3_2_training
                else "partial_scaffold_smoke_only"
            ),
            "training_status": training_report.get("status"),
        })
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
