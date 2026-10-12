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
from data.standard_ip import (  # noqa: E402
    load_standard_ip_mapping,
    path1_ip_membership,
    validate_resolved_standard_ip_snapshot,
)
from factors import calculate_demand_adjustments  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.2 M5 factor adjustment only.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cohort-file", type=Path, default=Path("artifacts/round3_dataset/cohort.csv"))
    parser.add_argument("--comparable-dir", type=Path, default=Path("artifacts/round5_comparable"))
    parser.add_argument("--restore-dir", type=Path, default=Path("artifacts/round4_restore"))
    parser.add_argument("--competition-rules", type=Path, default=None)
    parser.add_argument(
        "--resolved-standard-ip-file",
        type=Path,
        default=Path("artifacts/standard_ip_mapping/cohort_sku_ip_mapping.csv"),
        help="Frozen standard-IP CSV produced upstream; Round7 only looks it up.",
    )
    parser.add_argument(
        "--resolved-standard-ip-report",
        type=Path,
        default=None,
        help="Companion report; defaults to standard_ip_mapping_report.json beside the CSV.",
    )
    add_execution_arguments(parser, default_output="artifacts/round7_factors")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("run_factors", config.log_dir, config.timezone)
    outputs = ["factor_detail.csv", "demand_adjustments.csv", "transformed_features.csv", "factor_report.json"]
    if args.dry_run:
        print_dry_run("M5 factors", args, outputs)
        return 0
    try:
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        artifact = StageArtifactStore()
        cohort = artifact.read_frame(
            PROJECT_ROOT / args.cohort_file,
            required=("sku_code", "dataset_version", "split"),
        )
        resolved_mapping_path = PROJECT_ROOT / args.resolved_standard_ip_file
        resolved_report_path = (
            PROJECT_ROOT / args.resolved_standard_ip_report
            if args.resolved_standard_ip_report is not None
            else resolved_mapping_path.with_name("standard_ip_mapping_report.json")
        )
        standard_ip_mapping = load_standard_ip_mapping(resolved_mapping_path)
        standard_ip_report = artifact.read_json(resolved_report_path)
        standard_ip_dataset_version = validate_resolved_standard_ip_snapshot(
            standard_ip_mapping,
            cohort=cohort,
            report=standard_ip_report,
            decision_cutoff_time=args.decision_cutoff_time,
        )
        standard_ip_membership = path1_ip_membership(standard_ip_mapping)
        comparables = artifact.read_frame(PROJECT_ROOT / args.comparable_dir / "candidate_detail.csv")
        comparable_summary = artifact.read_frame(
            PROJECT_ROOT / args.comparable_dir / "comparable_summary.csv",
            sku_codes=args.sku_code, limit_skus=args.limit_skus,
        )
        targets = comparable_summary[["sku_code"]].drop_duplicates()
        selected = comparables.loc[comparables["selected"] == 1, "candidate_sku_code"].astype(str)
        restored_tier = artifact.read_frame(PROJECT_ROOT / args.restore_dir / "restored_tier_weekly.csv")
        all_skus = (
            targets["sku_code"].astype(str).tolist()
            + [value for value in selected if not value.startswith("__STATIC__")]
            + restored_tier["sku_code"].astype(str).drop_duplicates().tolist()
        )
        adapter = DecisionDataAdapter.from_project(project_root=PROJECT_ROOT, data_root=args.data_root, config=config)
        features, partitions = adapter.factor_features(
            args.decision_cutoff_time,
            sku_codes=list(dict.fromkeys(all_skus)),
            restored_weekly=restored_tier,
            standard_ip_membership=standard_ip_membership,
        )
        target_skus = set(targets["sku_code"].astype(str))
        mapped_skus = set(standard_ip_membership["sku_code"].astype(str))
        unmapped_targets = sorted(target_skus - mapped_skus)
        if unmapped_targets:
            logger.warning(
                "standard_ip_missing targets=%s; IP factor inputs remain missing",
                ",".join(unmapped_targets),
            )
        competition = None if args.competition_rules is None else artifact.read_frame(PROJECT_ROOT / args.competition_rules)
        result = calculate_demand_adjustments(
            targets=targets, features=features, comparables=comparables,
            comparable_summary=comparable_summary, config=parameters["factors"],
            competition_rules=competition,
        )
        output_dir = PROJECT_ROOT / args.output_dir
        for name, frame in zip(outputs[:3], [result.detail, result.adjustments, result.transformed_features]):
            artifact.write_frame(output_dir / name, frame)
        artifact.write_json(output_dir / outputs[3], {
            "spec_version": "3.2", "stage": "M5 factors", "status": "reference_parameter_smoke_output_requires_real_training",
            "parameter_version": parameters["parameter_version"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "source_partitions": partitions, "target_count": len(result.adjustments),
            "warnings": list(result.warnings),
            "resolved_standard_ip_file": str(args.resolved_standard_ip_file),
            "resolved_standard_ip_report": str(
                args.resolved_standard_ip_report
                if args.resolved_standard_ip_report is not None
                else args.resolved_standard_ip_file.with_name("standard_ip_mapping_report.json")
            ),
            "resolved_standard_ip_dataset_version": standard_ip_dataset_version,
            "resolved_standard_ip_rows": len(standard_ip_mapping),
            "resolved_standard_ip_mapped_skus": len(mapped_skus),
            "unmapped_target_skus": unmapped_targets,
            "raw_proxy_ip_used_for_factors": False,
            "missing_is_zero_policy": False,
            "source_lifecycle_status_used_as_numeric_factor": False,
            "lifecycle_persistence_definition": (
                "V3.2 PR-04/PF-02: exposure-weighted all-store neutral velocity "
                "sum for lifecycle weeks 5-10 divided by weeks 1-10; all ten "
                "weeks and a positive denominator are required."
            ),
            "lifecycle_persistence_status_counts": {
                str(key): int(value)
                for key, value in features["lifecycle_persistence_status"]
                .value_counts(dropna=False)
                .items()
            },
            "factor_input_missing_counts": {
                column: int(result.transformed_features[column].isna().sum())
                for column in parameters["factors"]["small_factor_weights"]["IP"]
                | parameters["factors"]["small_factor_weights"]["PRODUCT"]
                | parameters["factors"]["small_factor_weights"]["OPERATION"]
            },
            "v3_2_unavailable_logic": {
                "formal_small_factor_set": "Several V3.2 social, product and operational raw fields are not delivered; unavailable groups are neutralized rather than filled with zero.",
                "trained_weights_and_betas": "Current config values are references only; V3.2 requires Train-only OOF fitting.",
                "frozen_train_cdf": "Current smoke run scores the loaded target/comparable/restore pool; a frozen category-stratified Train CDF must be fitted and versioned by the training stage before production.",
                "competition": "Ccompetition=1 unless a versioned business competition-rule input is supplied."
            },
        })
        logger.info("M5 complete targets=%s", len(result.adjustments))
    except Exception as exc:
        log_exception(logger, module="run_factors", cutoff=args.decision_cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(f"status: reference_parameter_smoke_output_requires_real_training\nadjustments: {output_dir / outputs[1]}\nreport: {output_dir / outputs[3]}\nlog: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
