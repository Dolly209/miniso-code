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
from comparable import select_comparables, summarize_restoration  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.decision_adapter import DecisionDataAdapter, select_target_skus  # noqa: E402
from data.domain import frame_hash, load_algorithm_parameters  # noqa: E402
from data.standard_ip import (  # noqa: E402
    load_standard_ip_mapping,
    path1_ip_membership,
    validate_resolved_standard_ip_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.2 M1 comparable selection only.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cohort-file", type=Path, default=Path("artifacts/round3_dataset/cohort.csv"))
    parser.add_argument("--restore-dir", type=Path, default=Path("artifacts/round4_restore"))
    parser.add_argument("--specified-comparables", type=Path, default=None)
    parser.add_argument(
        "--resolved-standard-ip-file",
        type=Path,
        default=Path("artifacts/standard_ip_mapping/cohort_sku_ip_mapping.csv"),
        help=(
            "Versioned cohort SKU-to-standard-IP snapshot produced by "
            "build_standard_ip_mapping.py; M1 performs lookup only."
        ),
    )
    parser.add_argument(
        "--resolved-standard-ip-report",
        type=Path,
        default=None,
        help=(
            "Companion snapshot report. Defaults to standard_ip_mapping_report.json "
            "beside --resolved-standard-ip-file."
        ),
    )
    add_execution_arguments(parser, default_output="artifacts/round5_comparable")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("run_comparable", config.log_dir, config.timezone)
    outputs = ["candidate_detail.csv", "comparable_summary.csv", "round5_report.json"]
    if args.dry_run:
        print_dry_run("M1 comparable", args, outputs)
        return 0
    try:
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        store = StageArtifactStore()
        cohort = store.read_frame(PROJECT_ROOT / args.cohort_file, required=("sku_code", "first_actual_sale_date", "split"))
        targets = select_target_skus(cohort, sku_codes=args.sku_code, limit_skus=args.limit_skus)
        restored = store.read_frame(PROJECT_ROOT / args.restore_dir / "restored_store_weekly.csv")
        restored_summary = summarize_restoration(restored)
        adapter = DecisionDataAdapter.from_project(project_root=PROJECT_ROOT, data_root=args.data_root, config=config)
        product_domain = adapter.products(args.decision_cutoff_time)
        resolved_mapping_path = PROJECT_ROOT / args.resolved_standard_ip_file
        resolved_report_path = (
            PROJECT_ROOT / args.resolved_standard_ip_report
            if args.resolved_standard_ip_report is not None
            else resolved_mapping_path.with_name("standard_ip_mapping_report.json")
        )
        standard_ip_mapping = load_standard_ip_mapping(resolved_mapping_path)
        standard_ip_report = store.read_json(resolved_report_path)
        standard_ip_dataset_version = validate_resolved_standard_ip_snapshot(
            standard_ip_mapping,
            cohort=cohort,
            report=standard_ip_report,
            decision_cutoff_time=args.decision_cutoff_time,
        )
        standard_ip_membership = path1_ip_membership(
            standard_ip_mapping,
            product_skus=product_domain.products["sku_code"],
        )
        mapped_skus = set(standard_ip_membership["sku_code"].astype(str))
        target_skus = targets["sku_code"].astype(str).drop_duplicates().tolist()
        unmapped_targets = [sku for sku in target_skus if sku not in mapped_skus]
        if unmapped_targets:
            logger.warning(
                "standard_ip_missing targets=%s; PATH1 disabled for these targets",
                ",".join(unmapped_targets),
            )
        specified = None if args.specified_comparables is None else store.read_frame(PROJECT_ROOT / args.specified_comparables)
        result = select_comparables(
            targets=targets,
            products=product_domain.products,
            ip_membership=standard_ip_membership,
            restored_summary=restored_summary,
            config=parameters["comparable"],
            specified_comparables=specified,
        )
        output_dir = PROJECT_ROOT / args.output_dir
        store.write_frame(output_dir / outputs[0], result.candidates)
        store.write_frame(output_dir / outputs[1], result.summary)
        report = {
            "spec_version": "3.2", "round": 5, "stage": "M1 comparable", "status": "partial_v3_2_output",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "source_product_partition": product_domain.source_partition,
            "resolved_standard_ip_file": str(args.resolved_standard_ip_file),
            "resolved_standard_ip_report": str(
                args.resolved_standard_ip_report
                if args.resolved_standard_ip_report is not None
                else args.resolved_standard_ip_file.with_name(
                    "standard_ip_mapping_report.json"
                )
            ),
            "resolved_standard_ip_dataset_version": standard_ip_dataset_version,
            "resolved_standard_ip_rows": len(standard_ip_mapping),
            "resolved_standard_ip_mapped_skus": len(mapped_skus),
            "resolved_standard_ip_status_counts": {
                str(key): int(value)
                for key, value in standard_ip_mapping.drop_duplicates("sku_code")[
                    "mapping_status"
                ]
                .value_counts()
                .items()
            },
            "standard_ip_rebuilt_in_stage": False,
            "unmapped_target_skus": unmapped_targets,
            "raw_proxy_ip_used_for_path1": False,
            "v3_2_unavailable_logic": {
                "emotion_48d_path2": "No delivered 48-dimensional emotion vector; similarity_emotion remains missing and Path2 cannot be formally activated.",
                "external_media": "No standardized external platform snapshot; Cext=1 is the V3.2 neutral missing-input treatment.",
                "trained_similarity_parameters": "Weights, temperature and K are reference config values until full Train OOF tuning is implemented and run."
            },
            "target_skus": target_skus,
            "candidate_rows": len(result.candidates), "warnings": list(result.warnings),
            "output_hash": frame_hash(result.candidates),
        }
        store.write_json(output_dir / outputs[2], report)
        logger.info("M1 complete targets=%s candidates=%s", len(result.summary), len(result.candidates))
    except Exception as exc:
        log_exception(logger, module="run_comparable", cutoff=args.decision_cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(f"status: partial_v3_2_output\nsummary: {output_dir / outputs[1]}\nreport: {output_dir / outputs[2]}\nlog: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
