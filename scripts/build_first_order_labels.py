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
from data.prelaunch_artifacts import normalize_prelaunch_audit_metadata, validate_label_restoration_provenance, restoration_model_health  # noqa: E402
from training import (  # noqa: E402
    aggregate_weekly_label_inputs,
    build_first_order_wcover_labels,
    build_wcover_windows,
    combine_weekly_label_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build V3.2 section 8.1 D_first_ref labels from Round 4 restored "
            "natural sales inside each SKU's W_cover and re-split the feasible "
            "SKU set."
        )
    )
    parser.add_argument(
        "--schedule-file",
        type=Path,
        default=Path(
            "artifacts/prelaunch_row_coverage_release_dt_full/"
            "formal_eligible_prelaunch_schedule.csv"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Accepted for command compatibility but not used for formal labels. "
            "D_first_ref requires the Round 4 restored weekly artifact."
        ),
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=Path("artifacts/server_probe_round2"),
        help="Accepted for command compatibility; no raw source is read here.",
    )
    parser.add_argument(
        "--restore-weekly",
        type=Path,
        required=True,
        help=(
            "Round 4 restored_store_weekly_parts directory, CSV, or Parquet "
            "file. Formal labels cannot be built from the sales table alone."
        ),
    )
    parser.add_argument(
        "--restore-report", type=Path, default=None,
        help="Round 4 report; defaults to round4_report.json beside the weekly artifact. Required for verified launch-universe schedules.",
    )
    parser.add_argument(
        "--batch-skus",
        type=int,
        default=25,
        help="Accepted for command compatibility; Round 4 parts are streamed instead.",
    )
    parser.add_argument(
        "--constraints-file",
        type=Path,
        default=None,
        help=(
            "Optional one-row-per-SKU standardized artifact containing confirmed "
            "decision_weeks, buffer_weeks, lead_weeks, lifecycle_weeks, "
            "hard_life_weeks or hard-end dates. Versioned V3.2 defaults are used "
            "and disclosed when absent."
        ),
    )
    parser.add_argument(
        "--target-sku-file",
        type=Path,
        default=None,
        help=(
            "Optional standardized CSV/Parquet containing sku_code. Pass the "
            "same file to run_restore.py so both stages use exactly the same "
            "pilot SKU universe."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    add_execution_arguments(
        parser, default_output="artifacts/first_order_wcover_labels_v2"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger(
        "build_first_order_labels", config.log_dir, config.timezone
    )
    outputs = [
        "first_order_wcover_labels.csv",
        "formal_training_schedule.csv",
        "excluded_training_skus.csv",
        "first_order_label_report.json",
    ]
    if args.dry_run:
        print_dry_run("V3.2 W_cover label build", args, outputs)
        return 0
    try:
        if not 0 < args.train_ratio < 1:
            raise ValueError("--train-ratio must be between 0 and 1")
        artifact = StageArtifactStore()
        selected_skus = list(args.sku_code)
        if args.target_sku_file is not None:
            target_rows = artifact.read_frame(
                PROJECT_ROOT / args.target_sku_file,
                required=("sku_code",),
                columns=("sku_code",),
            )
            selected_skus.extend(target_rows["sku_code"].astype(str).tolist())
        selected_skus = list(dict.fromkeys(selected_skus))
        schedule = artifact.read_frame(
            PROJECT_ROOT / args.schedule_file,
            required=(
                "sku_code",
                "first_actual_sale_date",
                "sales_26w_qty",
                "prediction_checkpoint_key",
            ),
            sku_codes=selected_skus or None,
            limit_skus=None if selected_skus else args.limit_skus,
        )
        schedule = normalize_prelaunch_audit_metadata(schedule)
        sku_codes = schedule["sku_code"].astype(str).drop_duplicates().tolist()
        constraints = (
            None
            if args.constraints_file is None
            else artifact.read_frame(
                PROJECT_ROOT / args.constraints_file,
                required=("sku_code",),
                sku_codes=sku_codes,
            )
        )
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        windows = build_wcover_windows(schedule=schedule, first_order_config=parameters["first_order"], constraints=constraints)
        weekly_path = PROJECT_ROOT / args.restore_weekly
        report_path = PROJECT_ROOT / args.restore_report if args.restore_report else weekly_path.parent / "round4_report.json"
        restore_report = artifact.read_json(report_path) if report_path.is_file() else None
        provenance = validate_label_restoration_provenance(
            schedule=schedule, windows=windows, report=restore_report,
            available_through=args.decision_cutoff_time,
        )
        source_mode = "round4_standardized_weekly_reuse"
        weekly_parts = []
        per_sku_cover = windows.set_index("sku_code")["W_cover"]
        weekly_columns = (
            "sku_code",
            "lifecycle_week",
            "week_end",
            "clean_sales_qty",
            "restored_natural_sales",
            "restored_lost_sales",
            "stock_grain_unreliable",
            "unknown_exposure_days",
            "source_day_rows",
            "restored_exposure_days",
            "long_stockout_quality_weight",
            "complete_week",
        )
        for part_index, frame in enumerate(
            artifact.iter_frame_parts(
                weekly_path,
                required=(
                    "sku_code",
                    "lifecycle_week",
                    "week_end",
                    "clean_sales_qty",
                    "restored_natural_sales",
                ),
                columns=weekly_columns,
                sku_codes=sku_codes,
                require_all_skus=False,
            ),
            start=1,
        ):
            # Reduce before retaining any chunks; facts stay in the data layer.
            source_rows = len(frame)
            frame = frame.loc[
                frame["lifecycle_week"].between(1, frame["sku_code"].map(per_sku_cover))
            ].copy()
            if frame.empty:
                continue
            weekly_parts.append(aggregate_weekly_label_inputs(frame))
            logger.info(
                "wcover_label_progress event=weekly_part_reduced part=%s "
                "source_rows=%s aggregate_rows=%s",
                part_index,
                source_rows,
                len(weekly_parts[-1]),
            )
        if not weekly_parts:
            raise ValueError(
                "Round 4 weekly artifact contains no selected SKU rows"
            )
        weekly = combine_weekly_label_inputs(weekly_parts)
        restored_skus = set(weekly["sku_code"].astype(str))
        missing_restore_skus = sorted(set(sku_codes) - restored_skus)
        source_audit = {
            "weekly_parts": len(weekly_parts),
            "aggregated_sku_weeks": len(weekly),
            "formal_label_source": "restored_natural_sales",
            "observed_sales_audit_source": "clean_sales_qty",
            "selected_schedule_skus": len(sku_codes),
            "restored_weekly_skus": len(restored_skus),
            "missing_restore_sku_count": len(missing_restore_skus),
            "missing_restore_sku_examples": missing_restore_skus[:20],
        }
        result = build_first_order_wcover_labels(
            schedule=schedule,
            weekly=weekly,
            first_order_config=parameters["first_order"],
            available_through=args.decision_cutoff_time,
            constraints=constraints,
            train_ratio=args.train_ratio,
        )
        output_dir = PROJECT_ROOT / args.output_dir
        health = restoration_model_health(restore_report)
        # Do not silently promote fallback-derived labels into fully validated
        # ground truth. Keep the existing values/eligibility and expose lineage.
        for frame in (result.labels, result.formal_schedule, result.excluded_skus):
            frame["label_source_model_health"] = health["status"]
            frame["label_source_model_id"] = health["model_id"]
        if not health["all_scopes_converged"]:
            logger.warning("label source model health=%s; labels require disclosed development/audit use until restoration is validated", health["status"])
        artifact.write_frame(output_dir / outputs[0], result.labels)
        artifact.write_frame(output_dir / outputs[1], result.formal_schedule)
        artifact.write_frame(output_dir / outputs[2], result.excluded_skus)
        report = dict(result.report)
        report.update(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
                "schedule_file": str(args.schedule_file),
                "label_source_mode": source_mode,
                "data_root": None if args.data_root is None else str(args.data_root),
                "restore_weekly": (
                    None if args.restore_weekly is None else str(args.restore_weekly)
                ),
                "source_audit": source_audit,
                "restoration_provenance": provenance,
                "restoration_model_health": health,
                "label_model_validation_complete": health["all_scopes_converged"],
                "restore_report": str(report_path) if restore_report is not None else None,
                "constraints_file": (
                    None
                    if args.constraints_file is None
                    else str(args.constraints_file)
                ),
                "outputs": {
                    "labels": outputs[0],
                    "formal_training_schedule": outputs[1],
                    "excluded_skus": outputs[2],
                },
            }
        )
        artifact.write_json(output_dir / outputs[3], report)
        logger.info(
            "W_cover label build complete eligible_skus=%s train=%s test=%s rows=%s",
            report["eligible_skus"],
            report["train_skus"],
            report["test_skus"],
            report["checkpoint_rows"],
        )
    except Exception as exc:
        log_exception(
            logger,
            module="build_first_order_labels",
            cutoff=args.decision_cutoff_time,
            exc=exc,
        )
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(
        f"status: {report['status']}\n"
        f"labels: {output_dir / outputs[0]}\n"
        f"schedule: {output_dir / outputs[1]}\n"
        f"report: {output_dir / outputs[3]}\n"
        f"log: {log_path}"
    )
    return 0 if report["split_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
