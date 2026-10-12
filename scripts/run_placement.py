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
from data.decision_adapter import DecisionDataAdapter  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402
from placement import plan_placement  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.2 M3/M4 placement only.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--comparable-dir", type=Path, default=Path("artifacts/round5_comparable"))
    parser.add_argument("--restore-dir", type=Path, default=Path("artifacts/round4_restore"))
    parser.add_argument(
        "--placement-policy-file",
        type=Path,
        help=(
            "Optional override CSV: store_tier,v_min,inv_max,tau_tier. "
            "When omitted, versioned defaults are used."
        ),
    )
    parser.add_argument(
        "--placement-evidence-file",
        type=Path,
        help=(
            "Optional standardized launch-window evidence CSV. When omitted, "
            "the data adapter builds it from first_store_arrival_time."
        ),
    )
    parser.add_argument(
        "--target-store-feature-file",
        type=Path,
        help=(
            "Optional standardized CSV containing target_sku_code,store_code," 
            "category_fit,execution and optional status columns."
        ),
    )
    add_execution_arguments(parser, default_output="artifacts/round6_placement")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("run_placement", config.log_dir, config.timezone)
    outputs = [
        "placement_plan.csv",
        "placement_summary.csv",
        "demand_curve.csv",
        "placement_diagnostics.csv",
        "v3_2_compliance.csv",
        "round6_report.json",
    ]
    if args.dry_run:
        print_dry_run("M3/M4 placement", args, outputs)
        return 0
    try:
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        artifact = StageArtifactStore()
        comparables = artifact.read_frame(PROJECT_ROOT / args.comparable_dir / "candidate_detail.csv")
        summary = artifact.read_frame(PROJECT_ROOT / args.comparable_dir / "comparable_summary.csv", sku_codes=args.sku_code, limit_skus=args.limit_skus)
        targets = summary[["sku_code"]].drop_duplicates()
        target_codes = set(targets["sku_code"].astype(str))
        comparables = comparables.loc[
            comparables["target_sku_code"].astype(str).isin(target_codes)
        ].copy()
        restored_store = artifact.read_frame(PROJECT_ROOT / args.restore_dir / "restored_store_weekly.csv")
        restored_tier = artifact.read_frame(PROJECT_ROOT / args.restore_dir / "restored_tier_weekly.csv")
        adapter = DecisionDataAdapter.from_project(project_root=PROJECT_ROOT, data_root=args.data_root, config=config)
        stores, store_partition = adapter.stores(args.decision_cutoff_time)
        selected_candidate_skus = sorted(
            {
                str(value)
                for value in comparables.loc[
                    pd.to_numeric(comparables["selected"], errors="coerce")
                    .fillna(0)
                    .eq(1),
                    "candidate_sku_code",
                ]
                if not str(value).startswith("__STATIC__")
            }
        )
        history_skus = sorted(restored_tier["sku_code"].astype(str).unique())
        all_product_skus = sorted(
            set(targets["sku_code"].astype(str))
            | set(selected_candidate_skus)
            | set(history_skus)
        )
        product = adapter.products(
            args.decision_cutoff_time,
            sku_codes=all_product_skus,
        )
        category_columns = [
            "sku_code",
            "cate_level1_code",
            "cate_level2_code",
            "cate_level3_code",
            "cate_level4_code",
        ]
        targets = targets.merge(
            product.products[category_columns],
            on="sku_code",
            how="left",
            validate="one_to_one",
        )
        restored_tier = restored_tier.drop(
            columns=[name for name in category_columns[1:] if name in restored_tier],
            errors="ignore",
        ).merge(
            product.products[category_columns],
            on="sku_code",
            how="left",
            validate="many_to_one",
        )
        placement_policy = (
            artifact.read_frame(
                PROJECT_ROOT / args.placement_policy_file,
                sku_column="store_tier",
            )
            if args.placement_policy_file
            else None
        )
        placement_evidence_source = "provided_standardized_snapshot"
        placement_evidence_partitions: list[str] = []
        adapter_warnings: list[str] = []
        if args.placement_evidence_file:
            placement_evidence = artifact.read_frame(
                PROJECT_ROOT / args.placement_evidence_file
            )
        elif selected_candidate_skus:
            try:
                evidence_domain = adapter.placement_launch_evidence(
                    args.decision_cutoff_time,
                    candidate_products=product.products.loc[
                        product.products["sku_code"].astype(str).isin(
                            selected_candidate_skus
                        )
                    ],
                    eligible_stores=stores.loc[
                        pd.to_numeric(stores["eligible"], errors="coerce")
                        .fillna(0)
                        .eq(1)
                    ],
                    restored_store_weekly=restored_store,
                    launch_window_days=int(
                        parameters["placement"]["initial_defaults"][
                            "launch_window_days"
                        ]
                    ),
                    progress=lambda event, details: logger.info(
                        "placement_evidence_progress event=%s details=%s",
                        event,
                        details,
                    ),
                )
                placement_evidence = (
                    evidence_domain.evidence
                    if not evidence_domain.evidence.empty
                    else None
                )
                placement_evidence_partitions = list(
                    evidence_domain.source_partitions
                )
                adapter_warnings.extend(evidence_domain.warnings)
                placement_evidence_source = (
                    "auto_first_store_arrival_time"
                    if placement_evidence is not None
                    else "restore_pool_empirical_proxy"
                )
            except Exception as evidence_exc:
                placement_evidence = None
                placement_evidence_source = "restore_pool_empirical_proxy"
                message = (
                    "automatic launch-window evidence unavailable; using the labeled "
                    f"restore-pool proxy: {type(evidence_exc).__name__}: {evidence_exc}"
                )
                adapter_warnings.append(message)
                logger.warning(message)
        else:
            placement_evidence = None
            placement_evidence_source = "not_applicable_path4"
        target_store_features = (
            artifact.read_frame(
                PROJECT_ROOT / args.target_store_feature_file,
                sku_column="target_sku_code",
            )
            if args.target_store_feature_file
            else None
        )
        result = plan_placement(
            targets=targets, comparables=comparables, stores=stores,
            restored_store_weekly=restored_store, restored_tier_weekly=restored_tier,
            config=parameters["placement"],
            placement_policy=placement_policy,
            placement_evidence=placement_evidence,
            target_store_features=target_store_features,
        )
        output_dir = PROJECT_ROOT / args.output_dir
        frames = [
            result.store_plan,
            result.summary,
            result.demand_curve,
            result.diagnostics,
            result.compliance,
        ]
        for name, frame in zip(outputs[:5], frames):
            artifact.write_frame(output_dir / name, frame)
        compliance_counts = {
            str(key): int(value)
            for key, value in result.compliance["status"].value_counts().items()
        }
        default_curve_rows = int(
            result.demand_curve["curve_status"]
            .eq("VERSIONED_INITIAL_PARAMETER")
            .sum()
        )
        formal_rate_count = int(result.summary["formal_v3_2_rate"].sum())
        default_statuses = {
            "VERSIONED_INITIAL_PARAMETER",
            "EMPIRICAL_INITIAL_MODEL",
            "MISSING_RENORMALIZED",
            "PARTIAL_SOURCE_FIELDS",
        }
        uses_defaults = result.compliance["status"].isin(default_statuses).any()
        run_status = (
            "implemented_with_versioned_defaults"
            if uses_defaults
            else "implemented_with_provided_calibration"
        )
        artifact.write_json(output_dir / outputs[5], {
            "spec_version": "3.2", "round": 6, "stage": "M3/M4 placement", "status": run_status,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "store_source_partition": store_partition,
            "product_source_partition": product.source_partition,
            "target_count": len(result.summary),
            "formal_v3_2_rate_target_count": formal_rate_count,
            "parameter_mode": "provided_overrides_plus_versioned_defaults",
            "default_curve_row_count": default_curve_rows,
            "compliance_status_counts": compliance_counts,
            "placement_evidence_source": placement_evidence_source,
            "placement_evidence_source_partitions": placement_evidence_partitions,
            "inputs": {
                "placement_policy_file": str(args.placement_policy_file) if args.placement_policy_file else None,
                "placement_evidence_file": str(args.placement_evidence_file) if args.placement_evidence_file else None,
                "target_store_feature_file": str(args.target_store_feature_file) if args.target_store_feature_file else None,
            },
            "warnings": adapter_warnings + list(result.warnings),
            "shelf_capacity_status": "missing_not_hard_filtered",
        })
        logger.info("M3/M4 complete targets=%s selected_stores=%s", len(result.summary), int(result.store_plan["selected"].sum()))
    except Exception as exc:
        log_exception(logger, module="run_placement", cutoff=args.decision_cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(f"status: {run_status}\nsummary: {output_dir / outputs[1]}\ncompliance: {output_dir / outputs[4]}\nreport: {output_dir / outputs[5]}\nlog: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
