#!/usr/bin/env python3
"""Audit launch origins and prepare a new M3 universe without running models."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for item in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from _pipeline_common import add_execution_arguments, positive_int
from common.logging_utils import setup_run_logger, log_exception
from config.settings import AppConfig
from data.artifacts import StageArtifactStore
from data.decision_adapter import DecisionDataAdapter
from data.domain import load_algorithm_parameters
from data.launch_universe import read_verified_launch_universe, rebind_reviewed_ip_snapshot, validate_observed_cohort_report
from data.standard_ip import load_standard_ip_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cohort-file", type=Path, default=Path("artifacts/round3_dataset/cohort.csv"))
    parser.add_argument("--cohort-report", type=Path, help="Defaults to round3_report.json beside cohort.csv; verifies the original data cutoff.")
    parser.add_argument("--source-start-date", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--lead-months", type=positive_int, default=3)
    parser.add_argument("--label-quality-file", type=Path, help="Optional prior label-quality audit; only matching-origin quality gates are reused, never demand labels.")
    parser.add_argument("--resolved-standard-ip-file", type=Path, help="Optional reviewed mapping CSV to subset and rebind without rerunning text inference.")
    parser.add_argument("--resolved-standard-ip-report", type=Path)
    add_execution_arguments(parser, default_output="artifacts/verified_sku_universe_m3_v2")
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    outputs = {
        "launch_audit": "sku_launch_audit.csv",
        "candidate_cohort": "verified_candidate_cohort.csv",
        "target_schedule": "prelaunch_decision_schedule.csv",
        "source_windows": "source_window_audit.csv",
        "exclusion_summary": "exclusion_summary.csv",
        "report": "verified_sku_universe_report.json",
    }
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "source_start_date": args.source_start_date.isoformat(),
                          "lead_months": args.lead_months, "outputs": outputs,
                          "raw_fact_rows_scanned": 0, "training_started": False}, ensure_ascii=False, indent=2))
        return 0
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("build_verified_sku_universe", config.log_dir, config.timezone)
    try:
        store = StageArtifactStore()
        output = absolute(args.output_dir)
        if output.resolve() == args.data_root.resolve() or args.data_root.resolve() in output.resolve().parents:
            raise ValueError("output-dir must not be inside the raw data root")
        # A fresh directory keeps old audit versions and model inputs recoverable.
        if output.exists() and any(output.iterdir()):
            raise ValueError("output-dir is not empty; choose a fresh output directory")
        cohort = store.read_frame(
            absolute(args.cohort_file),
            required=("sku_code", "dataset_version", "first_actual_sale_date", "sales_26w_qty"),
            sku_codes=args.sku_code or None, limit_skus=args.limit_skus,
        )
        cohort_report_path = absolute(args.cohort_report) if args.cohort_report else absolute(args.cohort_file).with_name("round3_report.json")
        cohort_report = store.read_json(cohort_report_path)
        validate_observed_cohort_report(cohort, cohort_report, available_through=args.decision_cutoff_time)
        quality = store.read_frame(absolute(args.label_quality_file)) if args.label_quality_file else None
        parameters = load_algorithm_parameters(absolute(args.algorithm_config))
        adapter = DecisionDataAdapter.from_project(project_root=PROJECT_ROOT, data_root=args.data_root, config=config)
        result = read_verified_launch_universe(
            adapter=adapter, cohort=cohort, available_through=args.decision_cutoff_time,
            source_start_date=args.source_start_date, parameters=parameters,
            label_quality=quality, train_ratio=config.training.train_ratio,
            lead_months=args.lead_months,
            distribution_window_weeks=config.training.label_distribution_window_weeks,
            progress=lambda event, details: logger.info("sku_universe_progress event=%s details=%s", event, details),
        )
        result.report.update({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_cohort_file": str(absolute(args.cohort_file)),
            "input_cohort_report": str(cohort_report_path),
            "input_cohort_cutoff_verified": True,
            "label_quality_file": str(absolute(args.label_quality_file)) if args.label_quality_file else None,
            "scope": "selected_pilot" if args.sku_code or args.limit_skus else "full_input_cohort",
            "outputs": outputs,
        })
        rebound_mapping = None
        if args.resolved_standard_ip_file:
            mapping_path = absolute(args.resolved_standard_ip_file)
            report_path = absolute(args.resolved_standard_ip_report) if args.resolved_standard_ip_report else mapping_path.with_name("standard_ip_mapping_report.json")
            mapping, mapping_report = rebind_reviewed_ip_snapshot(
                mapping=load_standard_ip_mapping(mapping_path), mapping_report=store.read_json(report_path),
                source_cohort=cohort, verified_cohort=result.candidate_cohort,
                available_through=args.decision_cutoff_time,
            )
            mapping_report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
            mapping_report["cohort_file"] = str(output / outputs["candidate_cohort"])
            rebound_mapping = (mapping, mapping_report)
            result.report["reviewed_ip_mapping"] = "subset_and_rebound_without_ip_inference"
            outputs["ip_mapping"] = "standard_ip_mapping/cohort_sku_ip_mapping.csv"
            outputs["ip_mapping_report"] = "standard_ip_mapping/standard_ip_mapping_report.json"
        for key, frame in (("launch_audit", result.audit), ("candidate_cohort", result.candidate_cohort),
                           ("target_schedule", result.schedule), ("source_windows", result.source_window_detail),
                           ("exclusion_summary", result.exclusion_summary)):
            store.write_frame(output / outputs[key], frame)
        if rebound_mapping is not None:
            store.write_frame(output / outputs["ip_mapping"], rebound_mapping[0])
            store.write_json(output / outputs["ip_mapping_report"], rebound_mapping[1])
        store.write_json(output / outputs["report"], result.report)
        logger.info("verified universe complete candidates=%s targets=%s version=%s",
                    len(result.candidate_cohort), result.schedule["sku_code"].nunique(), result.report["dataset_version"])
        print(f"status: {result.report['status']}\ncandidates: {len(result.candidate_cohort)}\ntargets: {result.schedule['sku_code'].nunique()}\nreport: {output / outputs['report']}\nlog: {log_path}")
        return 0
    except Exception as exc:
        log_exception(logger, module="build_verified_sku_universe", cutoff=args.decision_cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
