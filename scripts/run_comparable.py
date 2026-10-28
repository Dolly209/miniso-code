#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import add_execution_arguments, parse_cutoff, print_dry_run  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from comparable import (  # noqa: E402
    apply_external_media_correction,
    select_comparables,
    summarize_restoration,
)
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.decision_adapter import (  # noqa: E402
    DataLayerError,
    DecisionDataAdapter,
    SOURCES,
    select_target_skus,
)
from data.domain import frame_hash, load_algorithm_parameters  # noqa: E402
from data.domestic import DataContractViolation  # noqa: E402
from data.standard_ip import (  # noqa: E402
    load_standard_ip_mapping,
    path1_ip_membership,
    validate_resolved_standard_ip_snapshot,
)
from data.social_ip_aliases import load_social_ip_alias_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.2 M1 comparable selection only.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cohort-file", type=Path, default=Path("artifacts/round3_dataset/cohort.csv"))
    parser.add_argument("--restore-dir", type=Path, default=Path("artifacts/round4_restore"))
    parser.add_argument(
        "--product-snapshot-time",
        type=parse_cutoff,
        default=None,
        help=(
            "Optional fixed product-master/standard-IP reference snapshot. "
            "Dynamic sales, social and restoration evidence remain bounded by "
            "--decision-cutoff-time."
        ),
    )
    parser.add_argument("--specified-comparables", type=Path, default=None)
    parser.add_argument(
        "--preselected-comparables",
        type=Path,
        default=None,
        help=(
            "Optional M1 pre-restoration selected_comparables.csv. When set, "
            "post-M2 quality and alpha weights are calculated only for those "
            "preselected candidate identities; no new candidate is introduced."
        ),
    )
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
        "--social-media-mode",
        choices=("required", "optional", "disabled"),
        default="required",
        help=(
            "required fails when the standardized social source cannot be read; "
            "optional records the failure and uses neutral C_ext=1; disabled skips it."
        ),
    )
    parser.add_argument(
        "--social-ip-alias-file",
        type=Path,
        default=Path("config/social_ip_aliases.csv"),
        help=(
            "Versioned standard-IP-to-social-ip_name crosswalk. Round 5 only "
            "performs exact lookups from this reviewed CSV; it does not infer "
            "new social aliases."
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
    outputs = [
        "candidate_detail.csv",
        "comparable_summary.csv",
        "round5_report.json",
        "product_snapshot_audit.csv",
        "external_media_diagnostics.csv",
    ]
    if args.dry_run:
        print_dry_run("M1 comparable", args, outputs)
        return 0
    try:
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        store = StageArtifactStore()
        cohort = store.read_frame(PROJECT_ROOT / args.cohort_file, required=("sku_code", "first_actual_sale_date", "split"))
        targets = select_target_skus(cohort, sku_codes=args.sku_code, limit_skus=args.limit_skus)
        target_skus = targets["sku_code"].astype(str).drop_duplicates().tolist()
        restored = store.read_frame(PROJECT_ROOT / args.restore_dir / "restored_store_weekly.csv")
        restored_summary = summarize_restoration(
            restored,
            minimum_mature_weeks=int(
                parameters["comparable"].get("minimum_mature_weeks", 4)
            ),
        )
        candidate_skus = restored_summary["sku_code"].astype(str).drop_duplicates().tolist()
        adapter = DecisionDataAdapter.from_project(project_root=PROJECT_ROOT, data_root=args.data_root, config=config)
        product_snapshot_time = (
            args.product_snapshot_time or args.decision_cutoff_time
        )
        product_domain = adapter.products(
            product_snapshot_time,
            sku_codes=list(dict.fromkeys(target_skus + candidate_skus)),
            allow_missing=True,
            progress=lambda event, details: logger.info(
                "product_progress event=%s %s", event,
                " ".join(f"{key}={value}" for key, value in details.items()),
            ),
        )
        output_dir = PROJECT_ROOT / args.output_dir
        product_audit = product_domain.snapshot_audit.copy()
        product_audit["is_target"] = product_audit["sku_code"].isin(target_skus)
        product_audit["is_restored_candidate"] = product_audit["sku_code"].isin(candidate_skus)
        store.write_frame(output_dir / outputs[3], product_audit)
        available_products = set(product_domain.products["sku_code"].astype(str))
        missing_targets = sorted(set(target_skus) - available_products)
        missing_candidates = sorted(set(candidate_skus) - available_products)
        fallback = product_audit.loc[
            product_audit["product_snapshot_status"].eq("historical_fallback")
        ]
        product_report = {
            "source_product_partition": product_domain.source_partition,
            "product_snapshot_time": product_snapshot_time.isoformat(),
            "product_snapshot_policy": (
                "fixed_reference_snapshot_with_per_sku_historical_fallback"
                if args.product_snapshot_time is not None
                else "latest_per_requested_sku_on_or_before_cutoff"
            ),
            "product_snapshot_audit_file": outputs[3],
            "product_snapshot_status_counts": {
                str(key): int(value) for key, value in
                product_audit["product_snapshot_status"].value_counts().items()
            },
            "product_fallback_partitions": dict(zip(
                fallback["sku_code"], fallback["source_product_partition"]
            )),
            "missing_product_target_skus": missing_targets,
            "missing_product_candidate_skus": missing_candidates,
        }
        if missing_targets:
            # Missing product attributes cannot be replaced by invented names,
            # categories or prices. Report every unresolved target up front.
            store.write_json(output_dir / outputs[2], {
                "spec_version": "3.2", "round": 5, "stage": "M1 comparable",
                "status": "blocked_missing_product_snapshot",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
                "product_snapshot_time": product_snapshot_time.isoformat(),
                "target_skus": target_skus,
                "comparable_outputs_written": False,
                **product_report,
            })
            raise DataContractViolation(
                f"no product snapshot on/before cutoff for target SKUs: {missing_targets}; "
                f"see {output_dir / outputs[3]} and {output_dir / outputs[2]}; "
                "no comparable outputs were written in this run"
            )
        adapter_warnings = []
        if missing_candidates:
            message = (
                "historical candidates excluded because no product snapshot exists "
                f"on/before cutoff: {missing_candidates}"
            )
            adapter_warnings.append(message)
            logger.warning(message)
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
            reference_snapshot_time=product_snapshot_time,
            allow_cohort_subset=args.product_snapshot_time is not None,
        )
        standard_ip_membership = path1_ip_membership(
            standard_ip_mapping,
            product_skus=product_domain.products["sku_code"],
        )
        mapped_skus = set(standard_ip_membership["sku_code"].astype(str))
        unmapped_targets = [sku for sku in target_skus if sku not in mapped_skus]
        if unmapped_targets:
            logger.warning(
                "standard_ip_missing targets=%s; PATH1 disabled for these targets",
                ",".join(unmapped_targets),
            )
        specified = None if args.specified_comparables is None else store.read_frame(PROJECT_ROOT / args.specified_comparables)
        preselected = (
            None
            if args.preselected_comparables is None
            else store.read_frame(
                args.preselected_comparables
                if args.preselected_comparables.is_absolute()
                else PROJECT_ROOT / args.preselected_comparables,
                required=("target_sku_code", "candidate_sku_code"),
            )
        )
        result = select_comparables(
            targets=targets,
            products=product_domain.products,
            ip_membership=standard_ip_membership,
            restored_summary=restored_summary,
            config=parameters["comparable"],
            specified_comparables=specified,
            candidate_allowlist=preselected,
        )
        social_partitions: tuple[str, ...] = ()
        social_warnings: list[str] = []
        external_summary = result.summary
        external_diagnostics = pd.DataFrame()
        social_start = None
        social_end = None
        social_alias_mapping = pd.DataFrame()
        social_alias_path = PROJECT_ROOT / args.social_ip_alias_file
        social_alias_sha256 = None
        if args.social_media_mode != "disabled":
            social_alias_mapping = load_social_ip_alias_mapping(social_alias_path)
            digest = hashlib.sha256()
            with social_alias_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            social_alias_sha256 = digest.hexdigest()
            launch_dates = pd.to_datetime(
                targets["first_actual_sale_date"], errors="coerce"
            ).dt.date.dropna()
            if launch_dates.empty:
                raise DataContractViolation(
                    "target cohort has no valid first_actual_sale_date for external-media windows"
                )
            prelaunch_days = int(
                parameters["comparable"]["external_media"]["prelaunch_days"]
            )
            social_ends = [
                min(value - timedelta(days=1), args.decision_cutoff_time.date())
                for value in launch_dates
            ]
            social_start = min(social_ends) - timedelta(
                days=max(prelaunch_days - 1, 0)
            )
            social_end = max(social_ends)
            try:
                social = adapter.external_media_history(
                    args.decision_cutoff_time,
                    start_date=social_start,
                    end_date=social_end,
                    progress=lambda event, details: logger.info(
                        "social_media_progress event=%s %s",
                        event,
                        " ".join(f"{key}={value}" for key, value in details.items()),
                    ),
                )
                social_partitions = social.source_partitions
                social_warnings.extend(social.warnings)
                external = apply_external_media_correction(
                    comparable_summary=result.summary,
                    candidates=result.candidates,
                    targets=targets,
                    standard_ip_mapping=standard_ip_mapping,
                    social_ip_alias_mapping=social_alias_mapping,
                    media_history=social.history,
                    config=parameters["comparable"],
                    decision_cutoff_date=args.decision_cutoff_time.date(),
                )
                external_summary = external.summary
                external_diagnostics = external.diagnostics
                social_warnings.extend(external.warnings)
            except DataLayerError as exc:
                if args.social_media_mode == "required":
                    raise
                social_warnings.append(
                    f"optional social source unavailable: {type(exc).__name__}: {exc}"
                )
                logger.warning(social_warnings[-1])
        else:
            # Keep a schema-stable diagnostic artifact even in deliberately
            # disabled smoke runs. This is not the default production path.
            external_diagnostics = pd.DataFrame(
                {
                    "sku_code": external_summary["sku_code"].astype(str),
                    "external_direction_coefficient": 1.0,
                    "external_status": "neutral_social_media_disabled",
                }
            )
            external_summary = external_summary.copy()
            external_summary["external_direction_coefficient"] = 1.0
            external_summary["external_status"] = "neutral_social_media_disabled"
        external_summary = external_summary.copy()
        external_summary["feature_cutoff_time"] = args.decision_cutoff_time.isoformat()
        candidate_output = result.candidates.assign(
            feature_cutoff_time=args.decision_cutoff_time.isoformat()
        )
        store.write_frame(output_dir / outputs[0], candidate_output)
        store.write_frame(output_dir / outputs[1], external_summary)
        store.write_frame(output_dir / outputs[4], external_diagnostics)
        report = {
            "spec_version": "3.2", "round": 5, "stage": "M1 comparable", "status": "partial_v3_2_output",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "product_snapshot_time": product_snapshot_time.isoformat(),
            **product_report,
            "comparable_outputs_written": True,
            "preselected_comparables_file": (
                None
                if args.preselected_comparables is None
                else str(args.preselected_comparables)
            ),
            "candidate_identity_policy": (
                "M1 pre-restoration Top-K fixed; post-M2 recomputes Q_i and alpha_i only"
                if args.preselected_comparables is not None
                else "single-stage selection from restored candidate universe"
            ),
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
                "trained_similarity_parameters": "Weights, temperature and K are reference config values until full Train OOF tuning is implemented and run."
            },
            "external_media": {
                "mode": args.social_media_mode,
                "source_table": SOURCES["social"],
                "alias_file": str(args.social_ip_alias_file),
                "alias_file_sha256": social_alias_sha256,
                "alias_mapping_version": (
                    None
                    if social_alias_mapping.empty
                    else str(social_alias_mapping.loc[
                        social_alias_mapping["enabled"].eq(1), "mapping_version"
                    ].iloc[0])
                ),
                "alias_enabled_rows": int(
                    social_alias_mapping["enabled"].eq(1).sum()
                ) if not social_alias_mapping.empty else 0,
                "alias_match_method_counts": (
                    {
                        str(key): int(value)
                        for key, value in social_alias_mapping.loc[
                            social_alias_mapping["enabled"].eq(1), "match_method"
                        ].value_counts().items()
                    }
                    if not social_alias_mapping.empty
                    else {}
                ),
                "alias_policy": (
                    "reviewed exact CSV lookup only; no fuzzy matching or "
                    "algorithm-code name list"
                ),
                "stat_date_source": "business date column, contract-checked equal to ds partition",
                "window_aggregation": "daily counts summed; rates weighted; trend and stability use daily series",
                "excluded_pending_data_confirmation": ["interaction_rate"],
                "window_start": None if social_start is None else social_start.isoformat(),
                "window_end": None if social_end is None else social_end.isoformat(),
                "source_partitions": list(social_partitions),
                "diagnostics_file": outputs[4],
                "status_counts": {
                    str(key): int(value)
                    for key, value in external_summary["external_status"]
                    .value_counts(dropna=False)
                    .items()
                },
                "coefficient_bounds": parameters["comparable"]["external_coefficient_bounds"],
                "parameter_status": parameters["comparable"]["external_media"]["parameter_status"],
                "never_creates_sales_placement_or_inventory_proxy": True,
            },
            "target_skus": target_skus,
            "candidate_rows": len(result.candidates),
            "warnings": adapter_warnings + list(result.warnings) + social_warnings,
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
