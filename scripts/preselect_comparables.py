#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import add_execution_arguments, parse_cutoff, print_dry_run  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from comparable import preselect_comparables  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.decision_adapter import DecisionDataAdapter, select_target_skus  # noqa: E402
from data.domain import frame_hash, load_algorithm_parameters  # noqa: E402
from data.domestic import DataContractViolation  # noqa: E402
from data.launch_contract import validate_launch_universe_pair  # noqa: E402
from data.standard_ip import (  # noqa: E402
    load_standard_ip_mapping,
    path1_ip_membership,
    validate_resolved_standard_ip_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run V3.2 M1 Path/hard-filter Top-K retrieval before M2. This "
            "stage selects candidate identities only; restored quality and "
            "alpha weights are calculated after M2."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-file", type=Path, required=True)
    parser.add_argument("--candidate-cohort-file", type=Path, required=True)
    parser.add_argument("--specified-comparables", type=Path, default=None)
    parser.add_argument(
        "--product-snapshot-time",
        type=parse_cutoff,
        default=None,
        help="Fixed audited product-master snapshot; defaults to decision cutoff.",
    )
    parser.add_argument(
        "--resolved-standard-ip-file",
        type=Path,
        default=Path("artifacts/standard_ip_mapping_v2/cohort_sku_ip_mapping.csv"),
    )
    parser.add_argument("--resolved-standard-ip-report", type=Path, default=None)
    add_execution_arguments(
        parser,
        default_output="artifacts/preselected_comparables",
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger(
        "preselect_comparables", config.log_dir, config.timezone
    )
    output_names = [
        "candidate_detail.csv",
        "selected_comparables.csv",
        "selected_candidate_cohort.csv",
        "product_snapshot_audit.csv",
        "preselection_report.json",
    ]
    if args.dry_run:
        print_dry_run("M1 pre-restoration Top-K retrieval", args, output_names)
        return 0
    try:
        store = StageArtifactStore()
        parameters = load_algorithm_parameters(_absolute(args.algorithm_config))
        targets_all = store.read_frame(
            _absolute(args.target_file),
            required=("sku_code", "first_actual_sale_date", "split", "dataset_version"),
        )
        targets = select_target_skus(
            targets_all,
            sku_codes=args.sku_code,
            limit_skus=args.limit_skus,
        )
        candidates = store.read_frame(
            _absolute(args.candidate_cohort_file),
            required=("sku_code", "first_actual_sale_date", "split", "dataset_version"),
        ).copy()
        candidates["sku_code"] = candidates["sku_code"].astype(str)
        validate_launch_universe_pair(candidates, targets)
        candidates = candidates.drop_duplicates("sku_code", keep="first")
        target_skus = targets["sku_code"].astype(str).drop_duplicates().tolist()
        candidate_skus = candidates["sku_code"].tolist()
        requested_skus = list(dict.fromkeys(target_skus + candidate_skus))
        snapshot_time = args.product_snapshot_time or args.decision_cutoff_time
        adapter = DecisionDataAdapter.from_project(
            project_root=PROJECT_ROOT,
            data_root=args.data_root,
            config=config,
        )
        product_domain = adapter.products(
            snapshot_time,
            sku_codes=requested_skus,
            allow_missing=True,
            progress=lambda event, details: logger.info(
                "preselection_product_progress event=%s %s",
                event,
                " ".join(f"{key}={value}" for key, value in details.items()),
            ),
        )
        products = product_domain.products
        available = set(products["sku_code"].astype(str))
        missing_targets = sorted(set(target_skus) - available)
        if missing_targets:
            raise DataContractViolation(
                "pre-restoration M1 targets have no product snapshot: "
                f"{missing_targets}"
            )
        missing_candidates = sorted(set(candidate_skus) - available)
        if missing_candidates:
            logger.warning(
                "preselection excludes candidates without product snapshot count=%s examples=%s",
                len(missing_candidates),
                missing_candidates[:20],
            )
            candidates = candidates.loc[
                ~candidates["sku_code"].isin(missing_candidates)
            ].reset_index(drop=True)

        mapping_path = _absolute(args.resolved_standard_ip_file)
        report_path = (
            _absolute(args.resolved_standard_ip_report)
            if args.resolved_standard_ip_report is not None
            else mapping_path.with_name("standard_ip_mapping_report.json")
        )
        mapping = load_standard_ip_mapping(mapping_path)
        mapping_report = store.read_json(report_path)
        mapping_cohort = pd.concat(
            [
                targets[["sku_code", "dataset_version"]],
                candidates[["sku_code", "dataset_version"]],
            ],
            ignore_index=True,
        ).drop_duplicates("sku_code", keep="first")
        dataset_version = validate_resolved_standard_ip_snapshot(
            mapping,
            cohort=mapping_cohort,
            report=mapping_report,
            decision_cutoff_time=args.decision_cutoff_time,
            reference_snapshot_time=snapshot_time,
            allow_cohort_subset=True,
        )
        membership = path1_ip_membership(
            mapping, product_skus=products["sku_code"]
        )
        specified = (
            None
            if args.specified_comparables is None
            else store.read_frame(_absolute(args.specified_comparables))
        )
        result = preselect_comparables(
            targets=targets,
            candidates=candidates,
            products=products,
            ip_membership=membership,
            config=parameters["comparable"],
            specified_comparables=specified,
        )
        selected_pairs = result.candidates.loc[
            result.candidates["selected"].eq(1)
            & ~result.candidates["candidate_sku_code"].astype(str).str.startswith("__STATIC__"),
            [
                "target_sku_code",
                "candidate_sku_code",
                "path_code",
                "path_priority",
                "similarity_score",
                "similarity_completeness",
            ],
        ].copy()
        selected_codes = selected_pairs["candidate_sku_code"].astype(str).drop_duplicates()
        selected_cohort = candidates.loc[
            candidates["sku_code"].isin(set(selected_codes))
        ].copy()
        output_dir = _absolute(args.output_dir)
        product_audit = product_domain.snapshot_audit.copy()
        product_audit["is_target"] = product_audit["sku_code"].isin(target_skus)
        product_audit["is_candidate_universe"] = product_audit["sku_code"].isin(
            candidate_skus
        )
        store.write_frame(output_dir / output_names[0], result.candidates)
        store.write_frame(output_dir / output_names[1], selected_pairs)
        store.write_frame(output_dir / output_names[2], selected_cohort)
        store.write_frame(output_dir / output_names[3], product_audit)
        selected_counts = (
            selected_pairs.groupby("target_sku_code").size().to_dict()
            if not selected_pairs.empty
            else {}
        )
        report = {
            "spec_version": "3.2",
            "stage": "M1 pre-restoration Top-K retrieval",
            "status": "implemented_output",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "product_snapshot_time": snapshot_time.isoformat(),
            "dataset_version": dataset_version,
            "ordering_policy": "M1 Path/hard-filter Top-K before M2 restoration",
            "top_k": int(parameters["comparable"]["top_k_internal"]),
            "pre_restore_ranking": (
                "hard path priority then documented static similarity; restored "
                "quality Q_i and alpha_i are deliberately deferred until M2"
            ),
            "target_sku_count": len(target_skus),
            "candidate_universe_sku_count": len(candidate_skus),
            "candidate_with_product_sku_count": len(candidates),
            "selected_pair_count": len(selected_pairs),
            "selected_unique_candidate_sku_count": len(selected_cohort),
            "selected_count_by_target": {
                str(key): int(value) for key, value in selected_counts.items()
            },
            "missing_candidate_product_count": len(missing_candidates),
            "missing_candidate_product_examples": missing_candidates[:20],
            "warnings": list(result.warnings),
            "candidate_detail_hash": frame_hash(result.candidates),
            "outputs": output_names[:-1],
        }
        store.write_json(output_dir / output_names[4], report)
        logger.info(
            "preselection complete targets=%s universe=%s selected_pairs=%s selected_unique=%s",
            len(target_skus),
            len(candidate_skus),
            len(selected_pairs),
            len(selected_cohort),
        )
    except Exception as exc:
        log_exception(
            logger,
            module="preselect_comparables",
            cutoff=args.decision_cutoff_time.isoformat(),
            exc=exc,
        )
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print(f"log: {log_path}")
        return 1

    print("status: implemented_output")
    print(f"selected_pairs: {len(selected_pairs)}")
    print(f"selected_unique_candidates: {len(selected_cohort)}")
    print(f"selected_candidate_cohort: {output_dir / output_names[2]}")
    print(f"report: {output_dir / output_names[4]}")
    print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
