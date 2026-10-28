#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import parse_cutoff, positive_int  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.decision_adapter import DecisionDataAdapter  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit physical first-store-arrival coverage and optionally compare "
            "the cohort lifecycle origin with the product launch date."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sku-code", action="append", required=True)
    parser.add_argument("--decision-cutoff-time", type=parse_cutoff, required=True)
    parser.add_argument("--product-snapshot-time", type=parse_cutoff, required=True)
    parser.add_argument(
        "--cohort-file", type=Path,
        help="Optional cohort CSV: compare first_actual_sale_date with the product launch origin; never overrides dates.",
    )
    parser.add_argument(
        "--launch-window-days",
        type=positive_int,
        help="Override the versioned V3.2 launch-window length.",
    )
    parser.add_argument(
        "--algorithm-config",
        type=Path,
        default=Path("config/algorithm_v3_2.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/placement_launch_source_audit"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    sku_codes = list(dict.fromkeys(str(value).strip() for value in args.sku_code))
    if any(not value for value in sku_codes):
        raise ValueError("--sku-code must be non-empty")
    parameters = load_algorithm_parameters(absolute(args.algorithm_config))
    launch_window_days = int(
        args.launch_window_days
        or parameters["placement"]["initial_defaults"]["launch_window_days"]
    )
    output_dir = absolute(args.output_dir)
    if args.dry_run:
        print("DRY RUN: no source table will be read.")
        print(f"sku_codes: {','.join(sku_codes)}")
        print(f"decision_cutoff_time: {args.decision_cutoff_time.isoformat()}")
        print(f"product_snapshot_time: {args.product_snapshot_time.isoformat()}")
        print(f"launch_window_days: {launch_window_days}")
        print(f"cohort_file: {absolute(args.cohort_file) if args.cohort_file else None}")
        print(f"output: {output_dir / 'placement_launch_source_audit.csv'}")
        return 0

    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger(
        "audit_placement_launch_source", config.log_dir, config.timezone
    )
    try:
        adapter = DecisionDataAdapter.from_project(
            project_root=PROJECT_ROOT,
            data_root=args.data_root,
            config=config,
        )
        product = adapter.products(
            args.product_snapshot_time,
            sku_codes=sku_codes,
            fallback_to_historical=False,
            progress=lambda event, details: logger.info(
                "product_snapshot_progress event=%s details=%s", event, details
            ),
        )
        cohort = (
            StageArtifactStore.read_frame(
                absolute(args.cohort_file),
                required=("sku_code", "first_actual_sale_date"),
                sku_codes=sku_codes,
            )
            if args.cohort_file is not None else None
        )
        domain = adapter.audit_placement_launch_source(
            args.decision_cutoff_time,
            candidate_products=product.products,
            launch_window_days=launch_window_days,
            candidate_cohort=cohort,
            progress=lambda event, details: logger.info(
                "placement_launch_audit_progress event=%s details=%s",
                event,
                details,
            ),
        )
        artifact = StageArtifactStore()
        summary_path = artifact.write_frame(
            output_dir / "placement_launch_source_audit.csv", domain.summary
        )
        product_path = artifact.write_frame(
            output_dir / "candidate_product_snapshot.csv",
            product.products.loc[
                :,
                [
                    column
                    for column in (
                        "sku_code",
                        "sku_name",
                        "release_dt",
                        "first_sale_date",
                    )
                    if column in product.products
                ],
            ],
        )
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "spec_version": "3.2",
            "stage": "placement physical launch-window source audit",
            "status": (
                "usable"
                if domain.summary["audit_status"]
                .eq("usable_first_arrival_evidence")
                .all()
                else "source_gap_detected"
            ),
            "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "product_snapshot_time": args.product_snapshot_time.isoformat(),
            "launch_window_days": launch_window_days,
            "sku_codes": sku_codes,
            "warnings": list(domain.warnings),
            "raw_source_access": "centralized_in_src/data/decision_adapter.py",
            "restored_exposure_proxy_applied": False,
            "cohort_file": str(absolute(args.cohort_file)) if args.cohort_file else None,
            "lifecycle_origin_audit": {
                "date_override_applied": False,
                "status_counts": (
                    domain.summary["lifecycle_origin_status"].value_counts().to_dict()
                    if "lifecycle_origin_status" in domain.summary else {}
                ),
            },
        }
        report_path = artifact.write_json(
            output_dir / "placement_launch_source_audit_report.json", report
        )
        logger.info(
            "placement launch source audit complete skus=%s status=%s",
            len(domain.summary),
            report["status"],
        )
        print(domain.summary.to_string(index=False))
        print(f"summary: {summary_path}")
        print(f"product: {product_path}")
        print(f"report: {report_path}")
        print(f"log: {log_path}")
        return 0
    except Exception as exc:
        log_exception(
            logger,
            exc,
            module="audit_placement_launch_source",
            cutoff=args.decision_cutoff_time,
        )
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"log: {log_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
