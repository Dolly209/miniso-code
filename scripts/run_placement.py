#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import add_execution_arguments, parse_cutoff, print_dry_run  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.decision_adapter import DecisionDataAdapter  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402
from data.launch_contract import validate_launch_origin_contract  # noqa: E402
from data.placement_checkpoint import PlacementCheckpointStore, placement_fingerprint  # noqa: E402
from placement import build_target_store_features, plan_placement  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.2 M3/M4 placement only.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--comparable-dir", type=Path, default=Path("artifacts/round5_comparable"))
    parser.add_argument("--restore-dir", type=Path, default=Path("artifacts/round4_restore"))
    parser.add_argument(
        "--launch-cohort-file", type=Path,
        help="Versioned canonical candidate origins; use the same cohort as M2, without replacing product master attributes.",
    )
    parser.add_argument(
        "--product-snapshot-time",
        type=parse_cutoff,
        default=None,
        help=(
            "Optional fixed product-master snapshot for static SKU attributes. "
            "Store and launch-window facts remain bounded by the decision cutoff."
        ),
    )
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
    parser.add_argument("--no-resume", action="store_true", help="Recompute target checkpoints instead of reusing completed targets.")
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
        "target_store_features.csv",
        "placement_evidence.csv",
    ]
    if args.dry_run:
        print_dry_run("M3/M4 placement", args, outputs)
        return 0
    try:
        logger.info("placement_progress event=input_read_start")
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
        launch_origins = None
        if args.launch_cohort_file:
            launch_origins = artifact.read_frame(
                PROJECT_ROOT / args.launch_cohort_file,
                required=("launch_universe_version", "first_actual_sale_date"),
            )
            validate_launch_origin_contract(launch_origins, name="placement launch cohort")
            launch_origins = launch_origins.loc[launch_origins["sku_code"].isin(selected_candidate_skus)].copy()
        history_skus = sorted(restored_tier["sku_code"].astype(str).unique())
        all_product_skus = sorted(
            set(targets["sku_code"].astype(str))
            | set(selected_candidate_skus)
            | set(history_skus)
        )
        product_snapshot_time = (
            args.product_snapshot_time or args.decision_cutoff_time
        )
        product = adapter.products(
            product_snapshot_time,
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
        restored_store = restored_store.drop(
            columns=[name for name in category_columns[1:] if name in restored_store],
            errors="ignore",
        ).merge(
            product.products[category_columns],
            on="sku_code",
            how="left",
            validate="many_to_one",
        )
        restored_store = adapter.restoration_clock(restored_store, product.products, verified_origins=launch_origins)
        restored_tier = adapter.restoration_clock(restored_tier, product.products, verified_origins=launch_origins)
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
                    # The adapter reconstructs each comparable's launch-time
                    # denominator from open/close dates. Passing only today's
                    # eligible stores would silently use the wrong universe.
                    eligible_stores=stores,
                    restored_store_weekly=restored_store,
                    candidate_launch_origins=launch_origins,
                    launch_window_days=int(
                        parameters["placement"]["initial_defaults"][
                            "launch_window_days"
                        ]
                    ),
                    stock_pressure_recent_weeks=int(
                        parameters["placement"]["initial_defaults"][
                            "stock_pressure_recent_weeks"
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
                    else "missing_launch_window_evidence"
                )
                if placement_evidence is not None:
                    missing_evidence = set(selected_candidate_skus) - set(placement_evidence["sku_code"].astype(str))
                    if missing_evidence:
                        placement_evidence_source = (
                            "partial_launch_window_evidence_missing_candidates"
                        )
                        logger.warning(
                            "true launch-window evidence unavailable for SKUs=%s; "
                            "later Round-4 exposure will not be used as a launch proxy",
                            sorted(missing_evidence),
                        )
            except Exception:
                # Unexpected code/schema failures must not silently downgrade all candidates.
                logger.exception("launch-window evidence failed; completed target checkpoints remain reusable")
                raise
        else:
            placement_evidence = None
            placement_evidence_source = "not_applicable_path4"
        if args.target_store_feature_file:
            target_store_features = artifact.read_frame(
                PROJECT_ROOT / args.target_store_feature_file,
                sku_column="target_sku_code",
            )
            target_store_feature_source = "provided_standardized_snapshot"
        else:
            logger.info("placement_progress event=target_store_features_start")
            target_store_features = build_target_store_features(
                targets=targets,
                stores=stores.loc[
                    pd.to_numeric(stores["eligible"], errors="coerce").fillna(0).eq(1)
                ],
                restored_store_weekly=restored_store,
                cutoff_date=args.decision_cutoff_time,
                lookback_days=90,
            )
            target_store_feature_source = "auto_m2_previous_90d"
            logger.info(
                "placement_progress event=target_store_features_complete rows=%s",
                len(target_store_features),
            )
        logger.info("placement_progress event=checkpoint_validation_start")
        fingerprint = placement_fingerprint(
            [targets, comparables, stores, restored_store, restored_tier,
             placement_policy, placement_evidence, target_store_features, launch_origins],
            {"cutoff": args.decision_cutoff_time.isoformat(), "config": parameters["placement"],
             "code": {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [Path(__file__), PROJECT_ROOT / "src/placement/engine.py",
                                   PROJECT_ROOT / "src/data/decision_adapter.py", PROJECT_ROOT / "src/data/artifacts.py",
                                   PROJECT_ROOT / "src/data/placement_checkpoint.py", PROJECT_ROOT / "src/data/launch_contract.py",
                                   PROJECT_ROOT / "src/data/restoration_audit.py", PROJECT_ROOT / "src/placement/lifecycle.py"]}},
        )
        checkpoints = PlacementCheckpointStore(PROJECT_ROOT / args.output_dir / "_checkpoints", fingerprint, resume=not args.no_resume)
        logger.info("placement_progress event=model_start targets=%s checkpoint=%s", len(targets), checkpoints.root)
        result = plan_placement(
            targets=targets, comparables=comparables, stores=stores,
            restored_store_weekly=restored_store, restored_tier_weekly=restored_tier,
            config=parameters["placement"],
            placement_policy=placement_policy,
            placement_evidence=placement_evidence,
            target_store_features=target_store_features,
            progress=lambda event, details: logger.info("placement_progress event=%s details=%s", event, details),
            load_target=checkpoints.load,
            save_target=checkpoints.save,
        )
        output_dir = PROJECT_ROOT / args.output_dir
        # Sufficient statistics for Step5 parameter replay. Store history enters
        # M3 only through tier policy medians and week-one store speeds; retain
        # the complete tier curve, not the large daily/store-week audit.
        from data.recompute import write_recompute_frames
        from placement.engine import _normalize_policy
        policy_tiers = set(stores["store_tier"].astype(str))
        if placement_evidence is not None and not placement_evidence.empty:
            policy_tiers.update(placement_evidence.store_tier.astype(str))
        policy_history = restored_store.copy()
        if "usable_for_launch_curve" in policy_history:
            policy_history.loc[~policy_history.usable_for_launch_curve.fillna(False), "v_neutral"] = float("nan")
        frozen_policy, frozen_policy_source = _normalize_policy(
            policy=placement_policy, store_tiers=sorted(policy_tiers),
            history=policy_history, config=parameters["placement"])
        sufficient_columns = [c for c in (
            "sku_code", "store_code", "store_tier", "lifecycle_week", "v_neutral",
            "stockout_days", "source_day_rows", "restored_exposure_days", "usable_for_launch_curve")
            if c in restored_store]
        write_recompute_frames(output_dir / "_recompute_inputs", {
            "stores": stores, "comparables": comparables,
            "restored_store_weekly": restored_store.loc[
                pd.to_numeric(restored_store.lifecycle_week, errors="coerce").eq(1), sufficient_columns],
            "restored_tier_weekly": restored_tier,
            "placement_policy": frozen_policy.assign(step5_policy_origin=frozen_policy_source),
            "placement_evidence": placement_evidence,
            "target_store_features": target_store_features,
        })
        clock_columns = [c for c in ("sku_code", "lifecycle_origin_date", "product_first_sale_date", "lifecycle_clock_status", "usable_for_launch_curve") if c in restored_store]
        artifact.write_frame(output_dir / "restoration_clock_audit.csv", restored_store[clock_columns].drop_duplicates())
        logger.info("placement_progress event=output_write_start")
        frames = [
            result.store_plan,
            result.summary.assign(
                feature_cutoff_time=args.decision_cutoff_time.isoformat()
            ),
            result.demand_curve,
            result.diagnostics,
            result.compliance,
        ]
        for name, frame in zip(outputs[:5], frames):
            artifact.write_frame(output_dir / name, frame)
        artifact.write_frame(output_dir / outputs[6], target_store_features)
        # Preserve the standardized, point-in-time comparable-store evidence
        # used by M3.  This makes p_prior, p_perf and StockPressure traceable
        # without re-reading the raw server partitions during an audit.
        artifact.write_frame(
            output_dir / outputs[7],
            placement_evidence if placement_evidence is not None else pd.DataFrame(),
        )
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
        blocked_count = int(result.summary["planned_store_count"].eq(0).sum())
        default_statuses = {
            "VERSIONED_INITIAL_PARAMETER",
            "EMPIRICAL_INITIAL_MODEL",
            "MISSING_RENORMALIZED",
            "PARTIAL_SOURCE_FIELDS",
        }
        uses_defaults = result.compliance["status"].isin(default_statuses).any()
        run_status = (
            "completed_with_blocked_targets"
            if blocked_count else
            "implemented_with_versioned_defaults"
            if uses_defaults
            else "implemented_with_provided_calibration"
        )
        artifact.write_json(output_dir / outputs[5], {
            "spec_version": "3.2", "round": 6, "stage": "M3/M4 placement", "status": run_status,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "product_snapshot_time": product_snapshot_time.isoformat(),
            "store_source_partition": store_partition,
            "product_source_partition": product.source_partition,
            "target_count": len(result.summary),
            "blocked_target_count": blocked_count,
            "formal_v3_2_rate_target_count": formal_rate_count,
            "parameter_mode": "provided_overrides_plus_versioned_defaults",
            "default_curve_row_count": default_curve_rows,
            "compliance_status_counts": compliance_counts,
            "placement_evidence_source": placement_evidence_source,
            "launch_origin_policy": "canonical_candidate_sales_observed_origin" if launch_origins is not None else "legacy_product_first_sale_or_release",
            "placement_evidence_source_partitions": placement_evidence_partitions,
            "inputs": {
                "launch_cohort_file": str(args.launch_cohort_file) if args.launch_cohort_file else None,
                "placement_policy_file": str(args.placement_policy_file) if args.placement_policy_file else None,
                "placement_evidence_file": str(args.placement_evidence_file) if args.placement_evidence_file else None,
                "target_store_feature_file": str(args.target_store_feature_file) if args.target_store_feature_file else None,
            },
            "target_store_feature_source": target_store_feature_source,
            "target_store_feature_artifact": outputs[6],
            "placement_evidence_artifact": outputs[7],
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
