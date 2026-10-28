#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.decision_adapter import DataLayerError, DecisionDataAdapter  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402
from data.standard_ip import (  # noqa: E402
    load_standard_ip_mapping,
    path1_ip_membership,
    validate_resolved_standard_ip_snapshot,
)
from data.social_ip_aliases import (  # noqa: E402
    build_sku_social_aliases,
    load_social_ip_alias_mapping,
    normalize_social_ip_name,
)
from factors import (  # noqa: E402
    build_internal_small_factors,
    build_social_small_factors,
    calculate_demand_adjustments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.2 M5 factor adjustment only.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cohort-file", type=Path, default=Path("artifacts/round3_dataset/cohort.csv"))
    parser.add_argument("--comparable-dir", type=Path, default=Path("artifacts/round5_comparable"))
    parser.add_argument("--restore-dir", type=Path, default=Path("artifacts/round4_restore"))
    parser.add_argument(
        "--product-snapshot-time",
        type=parse_cutoff,
        default=None,
        help=(
            "Optional fixed product-master/standard-IP reference snapshot. "
            "Supplier, member and social facts remain bounded by the decision cutoff."
        ),
    )
    parser.add_argument("--competition-rules", type=Path, default=None)
    parser.add_argument(
        "--factor-inputs-file",
        type=Path,
        default=None,
        help=(
            "Optional reviewed SKU inputs for form_play_fit, design_innovation "
            "or marketing_resource. Objective values computed from delivered "
            "data take precedence unless the objective field is unavailable."
        ),
    )
    parser.add_argument(
        "--social-media-mode",
        choices=("required", "optional", "disabled"),
        default="required",
    )
    parser.add_argument(
        "--social-ip-alias-file",
        type=Path,
        default=Path("config/social_ip_aliases.csv"),
    )
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
    parser.add_argument("--placement-dir", type=Path, default=None,
                        help="Canonical placement output to seal Step5 recompute inputs; optional for isolated factor smoke.")
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
        product_snapshot_time = (
            args.product_snapshot_time or args.decision_cutoff_time
        )
        standard_ip_dataset_version = validate_resolved_standard_ip_snapshot(
            standard_ip_mapping,
            cohort=cohort,
            report=standard_ip_report,
            decision_cutoff_time=args.decision_cutoff_time,
            reference_snapshot_time=product_snapshot_time,
            allow_cohort_subset=args.product_snapshot_time is not None,
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
            product_snapshot_time=product_snapshot_time,
        )
        factor_settings = parameters["factors"]
        features = build_internal_small_factors(
            features=features,
            restored_weekly=restored_tier,
            standard_ip_membership=standard_ip_membership,
            cutoff_date=args.decision_cutoff_time,
            competition_window_days=int(
                factor_settings["competition_initial_rules"]["same_ip_launch_window_days"]
            ),
        )
        social_partitions: tuple[str, ...] = ()
        social_warnings: list[str] = []
        media = None
        aliases = {}
        if args.social_media_mode != "disabled":
            alias_mapping = load_social_ip_alias_mapping(
                PROJECT_ROOT / args.social_ip_alias_file
            )
            aliases, _ = build_sku_social_aliases(
                standard_ip_mapping, alias_mapping
            )
            releases = pd.to_datetime(features["release_dt"], errors="coerce").dt.date.dropna()
            if not releases.empty:
                prelaunch_days = int(factor_settings["social_prelaunch_days"])
                social_ends = [
                    min(
                        value - timedelta(days=1),
                        args.decision_cutoff_time.date(),
                    )
                    for value in releases
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
                            "factor_social_progress event=%s details=%s", event, details
                        ),
                    )
                    media = social.history.copy()
                    media["external_ip_key"] = media["ip_name"].map(
                        normalize_social_ip_name
                    )
                    features = build_social_small_factors(
                        features=features,
                        media_history=media,
                        sku_social_aliases=aliases,
                        prelaunch_days=int(factor_settings["social_prelaunch_days"]),
                        momentum_days=int(factor_settings["social_momentum_days"]),
                        minimum_days=int(factor_settings["minimum_social_days"]),
                        decision_cutoff_date=args.decision_cutoff_time.date(),
                    )
                    social_partitions = social.source_partitions
                    social_warnings.extend(social.warnings)
                except DataLayerError as exc:
                    if args.social_media_mode == "required":
                        raise
                    social_warnings.append(
                        f"optional social source unavailable: {type(exc).__name__}: {exc}"
                    )
            elif args.social_media_mode == "required":
                raise ValueError("formal IP factors require at least one valid release_dt")

        if args.factor_inputs_file is not None:
            reviewed = artifact.read_frame(PROJECT_ROOT / args.factor_inputs_file)
            required_reviewed = {"sku_code"}
            if not required_reviewed.issubset(reviewed.columns):
                raise ValueError("factor inputs file requires sku_code")
            allowed = [
                column
                for column in ("form_play_fit", "design_innovation", "marketing_resource")
                if column in reviewed
            ]
            reviewed = reviewed[["sku_code", *allowed]].copy()
            reviewed["sku_code"] = reviewed["sku_code"].astype(str)
            features = features.merge(
                reviewed.rename(columns={column: f"manual__{column}" for column in allowed}),
                on="sku_code",
                how="left",
                validate="one_to_one",
            )
            for column in allowed:
                manual = pd.to_numeric(features[f"manual__{column}"], errors="coerce")
                objective_missing = features[column].isna()
                features.loc[objective_missing, column] = manual.loc[objective_missing]
                features.loc[objective_missing & manual.notna(), f"{column}_status"] = (
                    "reviewed_manual_0_10_fallback"
                )
        target_skus = set(targets["sku_code"].astype(str))
        mapped_skus = set(standard_ip_membership["sku_code"].astype(str))
        unmapped_targets = sorted(target_skus - mapped_skus)
        if unmapped_targets:
            logger.warning(
                "standard_ip_missing targets=%s; IP factor inputs remain missing",
                ",".join(unmapped_targets),
            )
        if args.competition_rules is None:
            competition_settings = factor_settings["competition_initial_rules"]
            threshold = int(
                competition_settings["same_ip_effective_new_sku_threshold"]
            )
            competition = features.loc[
                features["sku_code"].isin(target_skus),
                ["sku_code", "same_ip_launch_count"],
            ].copy()
            competition["rule_code"] = "CO01_SAME_IP_MULTI_LAUNCH"
            competition["triggered"] = pd.to_numeric(
                competition["same_ip_launch_count"], errors="coerce"
            ).ge(threshold).astype(int)
            competition["coefficient"] = float(
                competition_settings["same_ip_coefficient"]
            )
        else:
            competition = artifact.read_frame(PROJECT_ROOT / args.competition_rules)
        result = calculate_demand_adjustments(
            targets=targets, features=features, comparables=comparables,
            comparable_summary=comparable_summary, config=parameters["factors"],
            competition_rules=competition,
        )
        output_dir = PROJECT_ROOT / args.output_dir
        from data.recompute import write_recompute_frames, seal_recompute_bundle
        bundle_root = PROJECT_ROOT / args.placement_dir / "_recompute_inputs" if args.placement_dir else None
        # run_factors is also usable independently. The replay driver supplies
        # the canonical placement sibling without any raw-table duplication.
        if bundle_root is None:
            bundle_root = output_dir.parent / "round6" / "_recompute_inputs"
        artifact.write_frame(output_dir / "raw_factor_features.parquet", features)
        if bundle_root.is_dir():
            feature_codes = set(features.sku_code.astype(str))
            social_columns = ["stat_date", "external_ip_key", "platform_code", "post_count", "unique_author_count",
                              "ordinary_author_count", "koc_author_count", "kol_author_count",
                              "avg_author_follower_count", "max_author_follower_count"]
            write_recompute_frames(bundle_root, {
                "features": features, "comparable_summary": comparable_summary,
                "competition_rules": competition,
                "social_history": None if media is None else media[social_columns],
                "social_aliases": (pd.DataFrame([
                    {"sku_code": sku, "external_ip_key": alias}
                    for sku, values in aliases.items() if sku in feature_codes
                    for alias in sorted(values)]) if aliases else None),
            })
            seal_recompute_bundle(bundle_root, parameters, args.decision_cutoff_time.isoformat())
        factor_frames = [
            frame.assign(feature_cutoff_time=args.decision_cutoff_time.isoformat())
            for frame in [result.detail, result.adjustments, result.transformed_features]
        ]
        for name, frame in zip(outputs[:3], factor_frames):
            artifact.write_frame(output_dir / name, frame)
        artifact.write_json(output_dir / outputs[3], {
            "spec_version": "3.2", "stage": "M5 factors", "status": "reference_parameter_smoke_output_requires_real_training",
            "parameter_version": parameters["parameter_version"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "product_snapshot_time": product_snapshot_time.isoformat(),
            "source_partitions": partitions, "target_count": len(result.adjustments),
            "warnings": list(result.warnings) + social_warnings,
            "social_source_partitions": list(social_partitions),
            "social_media_mode": args.social_media_mode,
            "social_ip_alias_file": str(args.social_ip_alias_file),
            "social_small_factors": ["IP-01", "IP-02", "IP-03"],
            "category_prosperity_factor": "IP-04 from PR-01/PR-02",
            "same_social_fact_deduplication": (
                "C_ext is forced to 1 whenever the same social source is available in C_IP"
            ),
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
                "formal_small_factor_set": "PF-01/02/03 are enabled only where objective or reviewed manual inputs exist; OF-01 remains neutral without an internal marketing plan/resource input.",
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
