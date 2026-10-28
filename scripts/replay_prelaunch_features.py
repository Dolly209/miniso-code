#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic
from typing import Any, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import (  # noqa: E402
    add_execution_arguments,
    positive_int,
)
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402
from data.domestic import DataContractViolation  # noqa: E402
from data.launch_contract import require_verified_training_schedule  # noqa: E402
from training.checkpoint_replay import (  # noqa: E402
    align_candidate_cohort_to_schedule,
    attach_checkpoint_identity,
    candidate_pool_at_cutoff,
    checkpoint_group_token,
    normalize_replay_schedule,
    replay_fingerprint,
)


COMBINED_OUTPUTS = {
    "comparable_summary.csv": ("round5", "comparable_summary.csv"),
    "placement_summary.csv": ("round6", "placement_summary.csv"),
    "demand_adjustments.csv": ("round7", "demand_adjustments.csv"),
    "transformed_features.csv": ("round7", "transformed_features.csv"),
    "first_order_decisions.csv": ("round8", "first_order_decisions.csv"),
}

COMBINED_MANY_OUTPUTS = {
    "comparable_candidate_detail.csv": (
        "round5",
        "candidate_detail.csv",
        "target_sku_code",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay V3.2 M1-M7 features at a pre-launch checkpoint without "
            "future fact leakage. M1 selects Top-K before M2 restores only "
            "those selected internal comparables."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--probe-dir", type=Path, default=Path("artifacts/server_probe_round2")
    )
    parser.add_argument(
        "--schedule-file",
        type=Path,
        default=Path(
            "artifacts/first_order_wcover_labels_v2/formal_training_schedule.csv"
        ),
    )
    parser.add_argument(
        "--candidate-cohort-file",
        type=Path,
        default=Path("artifacts/round3_dataset/cohort.csv"),
        help=(
            "Historical SKU universe for comparable restoration. Only SKUs with "
            "the minimum observed lifecycle weeks by each checkpoint are used."
        ),
    )
    parser.add_argument(
        "--prediction-checkpoint",
        choices=("all", "M5", "M4", "M3"),
        default="M3",
        help=(
            "Replay one selected node; defaults to M3 (T-3M). Use 'all' only "
            "for an explicitly requested three-node study. "
            "The complete M5/M4/M3 schedule is still validated before the "
            "requested node is selected."
        ),
    )
    parser.add_argument(
        "--limit-train-skus",
        type=positive_int,
        default=None,
        help=(
            "Smoke-only number of existing Train SKUs to sample evenly along "
            "the first-sale timeline. Must be used with --limit-test-skus."
        ),
    )
    parser.add_argument(
        "--limit-test-skus",
        type=positive_int,
        default=None,
        help=(
            "Smoke-only number of locked Test SKUs to sample evenly along the "
            "first-sale timeline. Must be used with --limit-train-skus."
        ),
    )
    parser.add_argument(
        "--resolved-standard-ip-file",
        type=Path,
        default=Path("artifacts/standard_ip_mapping_v2/cohort_sku_ip_mapping.csv"),
    )
    parser.add_argument("--resolved-standard-ip-report", type=Path, default=None)
    parser.add_argument("--specified-comparables", type=Path, default=None)
    parser.add_argument(
        "--social-ip-alias-file",
        type=Path,
        default=Path("config/social_ip_aliases.csv"),
    )
    parser.add_argument(
        "--social-media-mode",
        choices=("required", "optional", "disabled"),
        default="required",
    )
    parser.add_argument("--factor-inputs-file", type=Path, default=None)
    parser.add_argument(
        "--constraints-file",
        type=Path,
        default=None,
        help=(
            "Optional point-in-time first-order constraints used by M6/M7. "
            "When omitted, the centralized V3.2 reference values are retained "
            "and disclosed in the Round8 report."
        ),
    )
    parser.add_argument("--event-calendar", type=Path, default=None)
    parser.add_argument("--placement-policy-file", type=Path, default=None)
    parser.add_argument("--placement-evidence-file", type=Path, default=None)
    parser.add_argument("--target-store-feature-file", type=Path, default=None)
    parser.add_argument(
        "--restoration-model-registry",
        type=Path,
        default=None,
        help=(
            "CSV written by train_restoration_oof.py. Every selected target "
            "SKU must map to one frozen v_model; replay never fits v_model."
        ),
    )
    parser.add_argument(
        "--registry-selects-skus",
        action="store_true",
        help=(
            "Restrict the already selected schedule to SKU keys present in the "
            "registry. Use it with the Train feature registry (B1 seed plus "
            "B2-B5 OOF), which intentionally omits locked Test. Never drops "
            "rows silently without this flag."
        ),
    )
    parser.add_argument(
        "--batch-skus",
        type=positive_int,
        default=25,
        help="Sequential M2 source-read batch size inside each cutoff group.",
    )
    parser.add_argument(
        "--shared-source-cache-dir",
        type=Path,
        default=None,
        help=(
            "Cross-cutoff raw source cache. Defaults to _shared_source_cache "
            "inside the replay output directory."
        ),
    )
    parser.add_argument(
        "--keep-group-intermediates",
        action="store_true",
        help=(
            "Keep large Round4 group artifacts after Round5-7 consume them. "
            "The default compact mode preserves the Round4 report and all "
            "Round5-8 downstream artifacts while deleting consumed M2 rows."
        ),
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=10.0,
        help=(
            "Stop safely between completed checkpoint groups when filesystem "
            "free space falls below this threshold. Defaults to 10 GiB."
        ),
    )
    parser.add_argument(
        "--candidate-limit",
        type=positive_int,
        default=None,
        help=(
            "Smoke-only Train-prioritized cap on historical candidates per cutoff. "
            "Omit for formal replay because a cap changes the candidate universe."
        ),
    )
    parser.add_argument(
        "--limit-checkpoint-groups",
        type=positive_int,
        default=None,
        help="Smoke-only cap on unique cutoff groups after SKU selection.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed checkpoint-group manifests and recompute them.",
    )
    parser.add_argument(
        "--keep-daily-exposure-audit",
        action="store_true",
        help=(
            "Persist each checkpoint group's daily_exposure_audit.csv for a "
            "targeted diagnostic run. It is omitted by default because Round5-7 "
            "consume only the restored weekly artifacts."
        ),
    )
    add_execution_arguments(
        parser,
        default_output="artifacts/prelaunch_feature_replay_v1",
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _time_spread_split_skus(
    schedule: pd.DataFrame, *, split: str, limit: int
) -> list[str]:
    identities = schedule[
        ["sku_code", "split", "first_actual_sale_date"]
    ].drop_duplicates("sku_code")
    pool = identities.loc[
        identities["split"].astype(str).str.lower().eq(split.lower())
    ].copy()
    pool["first_actual_sale_date"] = pd.to_datetime(
        pool["first_actual_sale_date"], errors="raise"
    )
    pool = pool.sort_values(
        ["first_actual_sale_date", "sku_code"], kind="stable"
    ).reset_index(drop=True)
    if limit > len(pool):
        raise DataContractViolation(
            f"requested {limit} {split} SKUs but schedule contains only {len(pool)}"
        )
    if limit == 1:
        positions = [len(pool) // 2]
    else:
        positions = [
            index * (len(pool) - 1) // (limit - 1)
            for index in range(limit)
        ]
    return pool.iloc[positions]["sku_code"].astype(str).tolist()


def _attach_restoration_models(
    schedule: pd.DataFrame,
    registry_path: Path,
    *,
    registry_selects_skus: bool = False,
) -> pd.DataFrame:
    registry = StageArtifactStore.read_frame(
        registry_path,
        required=(
            "sku_code",
            "model_id",
            "model_file",
            "model_train_end_time",
            "usage_scope",
        ),
    )
    registry = registry.copy()
    if "launch_universe_version" in schedule:
        if "launch_universe_version" not in registry or set(
            registry["launch_universe_version"].astype(str)
        ) != set(schedule["launch_universe_version"].astype(str)):
            raise DataContractViolation(
                "frozen restoration-model registry does not match the verified launch universe; old model registries cannot be reused"
            )
    registry["sku_code"] = registry["sku_code"].astype(str)
    if registry["sku_code"].duplicated().any():
        raise DataContractViolation(
            "restoration-model registry is not unique by sku_code"
        )
    if registry_selects_skus:
        schedule = schedule.loc[
            schedule["sku_code"].astype(str).isin(set(registry["sku_code"]))
        ].copy()
        if schedule.empty:
            raise DataContractViolation(
                "model registry and selected schedule have no common SKUs"
            )
    registry["restoration_model_file"] = registry["model_file"].map(
        lambda value: str(
            (registry_path.parent / str(value)).resolve()
            if not Path(str(value)).is_absolute()
            else Path(str(value)).resolve()
        )
    )
    result = schedule.merge(
        registry[
            [
                "sku_code",
                "model_id",
                "model_train_end_time",
                "usage_scope",
                "restoration_model_file",
            ]
        ],
        on="sku_code",
        how="left",
        validate="many_to_one",
        indicator="__model_merge",
    )
    missing = result["__model_merge"].ne("both")
    if missing.any():
        raise DataContractViolation(
            "selected replay SKUs have no frozen v_model mapping: "
            f"{result.loc[missing, 'sku_code'].drop_duplicates().head(20).tolist()}"
        )
    result = result.drop(columns="__model_merge")
    cutoffs = pd.to_datetime(
        result["decision_cutoff_time"], errors="coerce", utc=True
    )
    train_end = pd.to_datetime(
        result["model_train_end_time"], errors="coerce", utc=True
    )
    if cutoffs.isna().any() or train_end.isna().any():
        raise DataContractViolation(
            "restoration-model registry contains invalid cutoff timestamps"
        )
    # Full-Train refitting is allowed to transform its own Train rows with the
    # final upstream model. OOF validation and locked Test must remain strictly
    # point-in-time safe.
    future = train_end.gt(cutoffs) & ~result["usage_scope"].astype(str).eq(
        "final_fit"
    )
    if future.any():
        examples = result.loc[
            future,
            ["sku_code", "decision_cutoff_time", "model_train_end_time"],
        ].head(5)
        raise DataContractViolation(
            "frozen v_model uses data after its target decision cutoff: "
            f"{examples.to_dict(orient='records')}"
        )
    missing_files = sorted(
        {
            value
            for value in result["restoration_model_file"].astype(str)
            if not Path(value).is_file()
        }
    )
    if missing_files:
        raise DataContractViolation(
            "restoration-model files are missing: " + ", ".join(missing_files[:10])
        )
    result["__cutoff_utc"] = cutoffs
    return result


def _model_aware_groups(
    schedule: pd.DataFrame,
) -> list[tuple[datetime, pd.DataFrame, Path, str, str, str]]:
    groups: list[tuple[datetime, pd.DataFrame, Path, str, str, str]] = []
    for (cutoff_utc, model_id, usage_scope), group in schedule.groupby(
        ["__cutoff_utc", "model_id", "usage_scope"],
        sort=True,
        dropna=False,
    ):
        first = group.iloc[0]
        cutoff = datetime.fromisoformat(
            str(first["decision_cutoff_time"]).replace("Z", "+00:00")
        )
        model_path = Path(str(first["restoration_model_file"]))
        safe_model = "".join(
            character if character.isalnum() else "_"
            for character in str(model_id)
        )[-16:]
        safe_scope = "".join(
            character if character.isalnum() else "_"
            for character in str(usage_scope)
        )
        token = f"{checkpoint_group_token(cutoff)}_{safe_model}_{safe_scope}"
        groups.append(
            (
                cutoff,
                group.drop(columns="__cutoff_utc").reset_index(drop=True),
                model_path,
                str(model_id),
                str(usage_scope),
                token,
            )
        )
    return groups


def _run_stage(
    command: Sequence[str],
    *,
    logger: Any,
    group_index: int,
    group_count: int,
    stage: str,
) -> None:
    logger.info(
        "checkpoint_replay_progress event=stage_start group=%s groups=%s stage=%s command=%s",
        group_index,
        group_count,
        stage,
        " ".join(command),
    )
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"checkpoint group {group_index}/{group_count} stage {stage} "
            f"failed with exit code {completed.returncode}"
        )
    logger.info(
        "checkpoint_replay_progress event=stage_complete group=%s groups=%s stage=%s",
        group_index,
        group_count,
        stage,
    )


def _append_optional(command: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        command.extend([flag, str(_absolute(value))])


def _write_static_restore(
    root: Path,
    *,
    cutoff_time: datetime,
    product_snapshot_time: datetime,
    reason: str,
) -> None:
    """Create a schema-valid empty M2 artifact for documented Path4 fallback."""

    store = StageArtifactStore()
    store_columns = [
        "sku_code",
        "store_code",
        "store_tier",
        "lifecycle_week",
        "week_end",
        "v_neutral",
        "n_eff",
        "stockout_days",
        "nondisplay_days",
        "source_day_rows",
        "restored_exposure_days",
        "unknown_exposure_days",
        "price_observation_days",
        "restored_lost_sales",
        "restored_natural_sales",
        "stock_grain_unreliable",
    ]
    tier_columns = [
        "sku_code",
        "store_tier",
        "lifecycle_week",
        "week_end",
        "v_neutral",
        "n_eff",
        "stockout_days",
        "source_day_rows",
        "restored_exposure_days",
    ]
    store.write_frame(root / "restored_store_weekly.csv", pd.DataFrame(columns=store_columns))
    store.write_frame(root / "restored_tier_weekly.csv", pd.DataFrame(columns=tier_columns))
    store.write_json(
        root / "round4_report.json",
        {
            "spec_version": "3.2",
            "round": 4,
            "module": "M2 historical operating-condition restoration",
            "status": "not_applicable_static_path4",
            "decision_cutoff_time": cutoff_time.isoformat(),
            "product_snapshot_time": product_snapshot_time.isoformat(),
            "selected_sku_count": 0,
            "fit_train_sku_count": 0,
            "fallback": "V3.2 Path4 static/category prior",
            "reason": reason,
            "future_fact_rows_used": 0,
        },
    )


def _required_group_outputs(group_root: Path) -> list[Path]:
    one_row = [
        group_root / stage / name
        for _, (stage, name) in COMBINED_OUTPUTS.items()
    ]
    many_rows = [
        group_root / stage / name
        for _, (stage, name, _) in COMBINED_MANY_OUTPUTS.items()
    ]
    return [*one_row, *many_rows]


def _group_complete(
    group_root: Path,
    *,
    expected_fingerprint: str,
) -> bool:
    manifest_path = group_root / "replay_manifest.json"
    try:
        payload = StageArtifactStore.read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "complete"
        and payload.get("fingerprint") == expected_fingerprint
        and all(path.is_file() for path in _required_group_outputs(group_root))
    )


def _run_group(
    *,
    args: argparse.Namespace,
    cutoff_time: datetime,
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    group_root: Path,
    fingerprint: str,
    logger: Any,
    group_index: int,
    group_count: int,
    shared_source_cache_dir: Path,
    restoration_model_path: Path,
    restoration_model_id: str,
    restoration_model_usage_scope: str,
) -> dict[str, Any]:
    store = StageArtifactStore()
    group_root.mkdir(parents=True, exist_ok=True)
    targets_path = group_root / "target_schedule.csv"
    candidates_path = group_root / "m1_candidate_universe.csv"
    store.write_frame(targets_path, targets)
    store.write_frame(candidates_path, candidates)
    product_snapshot_time = args.decision_cutoff_time
    standard_report = (
        args.resolved_standard_ip_report
        if args.resolved_standard_ip_report is not None
        else args.resolved_standard_ip_file.with_name("standard_ip_mapping_report.json")
    )
    preselection_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/preselect_comparables.py"),
        "--data-root",
        str(args.data_root),
        "--target-file",
        str(targets_path),
        "--candidate-cohort-file",
        str(candidates_path),
        "--resolved-standard-ip-file",
        str(_absolute(args.resolved_standard_ip_file)),
        "--resolved-standard-ip-report",
        str(_absolute(standard_report)),
        "--output-dir",
        str(group_root / "round5_preselection"),
        "--decision-cutoff-time",
        cutoff_time.isoformat(),
        "--product-snapshot-time",
        product_snapshot_time.isoformat(),
        "--algorithm-config",
        str(_absolute(args.algorithm_config)),
    ]
    _append_optional(
        preselection_command,
        "--specified-comparables",
        args.specified_comparables,
    )
    _run_stage(
        preselection_command,
        logger=logger,
        group_index=group_index,
        group_count=group_count,
        stage="round5_preselect_top3",
    )
    selected_candidates_path = (
        group_root / "round5_preselection" / "selected_candidate_cohort.csv"
    )
    selected_pairs_path = (
        group_root / "round5_preselection" / "selected_comparables.csv"
    )
    selected_candidates = store.read_frame(selected_candidates_path)
    history_mode = "historical_m2_frozen_model"
    if selected_candidates.empty:
        history_mode = "static_path4"
        reason = "M1 preselection found no internal comparable"
        _write_static_restore(
            group_root / "round4",
            cutoff_time=cutoff_time,
            product_snapshot_time=product_snapshot_time,
            reason=reason,
        )
        logger.info(
            "checkpoint_replay_progress event=static_path4 group=%s groups=%s reason=%s",
            group_index,
            group_count,
            reason,
        )
    else:
        restore_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_restore.py"),
            "--data-root",
            str(args.data_root),
            "--probe-dir",
            str(_absolute(args.probe_dir)),
            "--cohort-file",
            str(selected_candidates_path),
            "--target-sku-file",
            str(selected_candidates_path),
            "--output-dir",
            str(group_root / "round4"),
            "--decision-cutoff-time",
            cutoff_time.isoformat(),
            "--product-snapshot-time",
            product_snapshot_time.isoformat(),
            "--batch-skus",
            str(args.batch_skus),
            "--shared-source-cache-dir",
            str(shared_source_cache_dir),
            "--skip-partition-parts",
            "--restoration-model-file",
            str(restoration_model_path),
            "--model-application-scope",
            restoration_model_usage_scope,
        ]
        _append_optional(restore_command, "--event-calendar", args.event_calendar)
        if args.no_resume:
            restore_command.append("--no-resume")
        if not args.keep_daily_exposure_audit:
            restore_command.append("--skip-daily-exposure-audit")
        _run_stage(
            restore_command,
            logger=logger,
            group_index=group_index,
            group_count=group_count,
            stage="round4_restore",
        )

    comparable_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_comparable.py"),
        "--data-root",
        str(args.data_root),
        "--cohort-file",
        str(targets_path),
        "--restore-dir",
        str(group_root / "round4"),
        "--preselected-comparables",
        str(selected_pairs_path),
        "--resolved-standard-ip-file",
        str(_absolute(args.resolved_standard_ip_file)),
        "--resolved-standard-ip-report",
        str(_absolute(standard_report)),
        "--social-ip-alias-file",
        str(_absolute(args.social_ip_alias_file)),
        "--social-media-mode",
        args.social_media_mode,
        "--output-dir",
        str(group_root / "round5"),
        "--decision-cutoff-time",
        cutoff_time.isoformat(),
        "--product-snapshot-time",
        product_snapshot_time.isoformat(),
        "--algorithm-config",
        str(_absolute(args.algorithm_config)),
    ]
    _run_stage(
        comparable_command,
        logger=logger,
        group_index=group_index,
        group_count=group_count,
        stage="round5_comparable",
    )

    placement_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_placement.py"),
        "--data-root",
        str(args.data_root),
        "--comparable-dir",
        str(group_root / "round5"),
        "--restore-dir",
        str(group_root / "round4"),
        "--output-dir",
        str(group_root / "round6"),
        "--decision-cutoff-time",
        cutoff_time.isoformat(),
        "--product-snapshot-time",
        product_snapshot_time.isoformat(),
        "--algorithm-config",
        str(_absolute(args.algorithm_config)),
    ]
    if "launch_origin_policy" in selected_candidates:
        _append_optional(placement_command, "--launch-cohort-file", selected_candidates_path)
    _append_optional(
        placement_command, "--placement-policy-file", args.placement_policy_file
    )
    _append_optional(
        placement_command,
        "--placement-evidence-file",
        args.placement_evidence_file,
    )
    _append_optional(
        placement_command,
        "--target-store-feature-file",
        args.target_store_feature_file,
    )
    if args.no_resume:
        placement_command.append("--no-resume")
    _run_stage(
        placement_command,
        logger=logger,
        group_index=group_index,
        group_count=group_count,
        stage="round6_placement",
    )

    factor_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_factors.py"),
        "--data-root",
        str(args.data_root),
        "--cohort-file",
        str(targets_path),
        "--comparable-dir",
        str(group_root / "round5"),
        "--restore-dir",
        str(group_root / "round4"),
        "--resolved-standard-ip-file",
        str(_absolute(args.resolved_standard_ip_file)),
        "--resolved-standard-ip-report",
        str(_absolute(standard_report)),
        "--social-ip-alias-file",
        str(_absolute(args.social_ip_alias_file)),
        "--social-media-mode",
        args.social_media_mode,
        "--output-dir",
        str(group_root / "round7"),
        "--decision-cutoff-time",
        cutoff_time.isoformat(),
        "--product-snapshot-time",
        product_snapshot_time.isoformat(),
        "--algorithm-config",
        str(_absolute(args.algorithm_config)),
    ]
    _append_optional(factor_command, "--factor-inputs-file", args.factor_inputs_file)
    factor_command.extend(["--placement-dir", str(group_root / "round6")])
    _run_stage(
        factor_command,
        logger=logger,
        group_index=group_index,
        group_count=group_count,
        stage="round7_factors",
    )

    first_order_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_first_order.py"),
        "--data-root",
        str(args.data_root),
        "--placement-dir",
        str(group_root / "round6"),
        "--factor-dir",
        str(group_root / "round7"),
        "--comparable-dir",
        str(group_root / "round5"),
        "--output-dir",
        str(group_root / "round8"),
        "--decision-cutoff-time",
        cutoff_time.isoformat(),
        "--product-snapshot-time",
        product_snapshot_time.isoformat(),
        "--algorithm-config",
        str(_absolute(args.algorithm_config)),
    ]
    _append_optional(
        first_order_command, "--constraints-file", args.constraints_file
    )
    _run_stage(
        first_order_command,
        logger=logger,
        group_index=group_index,
        group_count=group_count,
        stage="round8_first_order",
    )
    manifest = {
        "spec_version": "3.2",
        "status": "complete",
        "fingerprint": fingerprint,
        "decision_cutoff_time": cutoff_time.isoformat(),
        "product_snapshot_time": product_snapshot_time.isoformat(),
        "target_skus": targets["sku_code"].astype(str).tolist(),
        "prediction_checkpoint_keys": targets[
            "prediction_checkpoint_key"
        ].astype(str).tolist(),
        "candidate_skus": int(len(selected_candidates)),
        "candidate_universe_skus": int(len(candidates)),
        "preselected_candidate_skus": int(len(selected_candidates)),
        "preselected_candidate_pairs": int(
            len(store.read_frame(selected_pairs_path))
        ),
        "fit_train_candidate_skus": 0,
        "restoration_model_id": restoration_model_id,
        "restoration_model_file": str(restoration_model_path),
        "restoration_model_usage_scope": restoration_model_usage_scope,
        "stage_order": [
            "M1_preselect_top_k",
            "M2_apply_frozen_fold_model_to_selected_only",
            "M1_post_restore_quality_and_weights",
            "M3_placement",
            "M5_factors",
            "M6_M7_first_order",
        ],
        "history_mode": history_mode,
        "round4_quality": {},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    round4_report_path = group_root / "round4" / "round4_report.json"
    if round4_report_path.is_file():
        round4_report = store.read_json(round4_report_path)
        quality = round4_report.get("data_quality", {})
        manifest["round4_quality"] = {
            "status": round4_report.get("status"),
            "dataset_version": round4_report.get("dataset_version"),
            "warnings": quality.get("warnings", []),
            "model_gate": quality.get("model_gate", {}),
            "exposure": quality.get("exposure", {}),
        }
    store.write_json(group_root / "replay_manifest.json", manifest)
    return manifest


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _compact_completed_group(group_root: Path) -> tuple[int, int]:
    """Remove only intermediates already consumed by completed Round5-7.

    The canonical Round4 report is copied to the group root before M2 rows and
    checkpoints are removed. Comparable detail, placement outputs, factor
    outputs, and the replay manifest remain available for training/audit.
    """

    before = _directory_bytes(group_root)
    round4 = group_root / "round4"
    report = round4 / "round4_report.json"
    if report.is_file():
        destination = group_root / "round4_report.json"
        temporary = destination.with_suffix(".json.tmp")
        shutil.copy2(report, temporary)
        temporary.replace(destination)
    if round4.exists():
        shutil.rmtree(round4)
    placement_checkpoints = group_root / "round6" / "_checkpoints"
    if placement_checkpoints.exists():
        shutil.rmtree(placement_checkpoints)
    after = _directory_bytes(group_root)
    return before, after


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger(
        "replay_prelaunch_features", config.log_dir, config.timezone
    )
    try:
        if args.decision_cutoff_time.tzinfo is None:
            raise ValueError("--decision-cutoff-time must include a UTC offset")
        if args.min_free_gb < 0:
            raise ValueError("--min-free-gb must be non-negative")
        balanced_limits = (
            args.limit_train_skus is not None
            or args.limit_test_skus is not None
        )
        if balanced_limits and (
            args.limit_train_skus is None or args.limit_test_skus is None
        ):
            raise ValueError(
                "--limit-train-skus and --limit-test-skus must be used together"
            )
        if balanced_limits and (args.limit_skus is not None or args.sku_code):
            raise ValueError(
                "balanced Train/Test limits cannot be combined with --limit-skus "
                "or --sku-code"
            )
        store = StageArtifactStore()
        schedule = store.read_frame(_absolute(args.schedule_file))
        selected = normalize_replay_schedule(
            schedule,
            available_through=args.decision_cutoff_time,
            sku_codes=args.sku_code,
            limit_skus=args.limit_skus,
        )
        require_verified_training_schedule(selected)
        if balanced_limits:
            selected_skus = {
                *_time_spread_split_skus(
                    selected, split="Train", limit=args.limit_train_skus
                ),
                *_time_spread_split_skus(
                    selected, split="Test", limit=args.limit_test_skus
                ),
            }
            selected = selected.loc[
                selected["sku_code"].astype(str).isin(selected_skus)
            ].reset_index(drop=True)
        if args.prediction_checkpoint != "all":
            selected = selected.loc[
                selected["prediction_checkpoint"].astype(str).eq(
                    args.prediction_checkpoint
                )
            ].reset_index(drop=True)
            if selected.empty:
                raise DataContractViolation(
                    "no replay rows remain for prediction checkpoint "
                    f"{args.prediction_checkpoint}"
                )
        if args.restoration_model_registry is None:
            raise ValueError(
                "--restoration-model-registry is required: formal replay must "
                "load fold/final v_model artifacts and may not fit on Top-K"
            )
        selected = _attach_restoration_models(
            selected,
            _absolute(args.restoration_model_registry),
            registry_selects_skus=args.registry_selects_skus,
        )
        candidate_universe, split_boundary = align_candidate_cohort_to_schedule(
            store.read_frame(_absolute(args.candidate_cohort_file)),
            selected,
        )
        parameters = load_algorithm_parameters(_absolute(args.algorithm_config))
        minimum_mature_weeks = int(
            parameters["comparable"].get("minimum_mature_weeks", 4)
        )
        groups = _model_aware_groups(selected)
        if args.limit_checkpoint_groups is not None:
            groups = groups[: args.limit_checkpoint_groups]
        if not groups:
            raise ValueError("no checkpoint groups selected")
        dry_plan = {
            "status": "dry_run" if args.dry_run else "planned",
            "stage": "historical prelaunch feature replay",
            "selected_skus": int(selected["sku_code"].nunique()),
            "selected_checkpoint_rows": int(
                sum(len(group) for _, group, _, _, _, _ in groups)
            ),
            "unique_cutoff_groups": len(groups),
            "minimum_mature_weeks": minimum_mature_weeks,
            "temporal_split_boundary_date": split_boundary.isoformat(),
            "candidate_split_policy": "refit from final W_cover-feasible schedule boundary",
            "candidate_weight_policy": "refit V3.2 26-week sales-band weights on final Train candidates only",
            "candidate_limit": args.candidate_limit,
            "prediction_checkpoint_scope": args.prediction_checkpoint,
            "stage_order": (
                "M1 Path/hard-filter Top-3 -> M2 selected-only restoration -> "
                "M1 restored Q_i/alpha_i -> downstream features"
            ),
            "restoration_scope": "at most Top-K internal comparables per target",
            "restoration_model_registry": str(args.restoration_model_registry),
            "registry_selects_skus": args.registry_selects_skus,
            "restoration_model_policy": (
                "frozen OOF/final model only; Top-K candidates never fit v_model"
            ),
            "sample_scope": {
                "limit_train_skus": args.limit_train_skus,
                "limit_test_skus": args.limit_test_skus,
                "policy": (
                    "time_spread_within_locked_split"
                    if balanced_limits
                    else "full_or_existing_selection"
                ),
            },
            "product_snapshot_time": args.decision_cutoff_time.isoformat(),
            "dynamic_fact_policy": "every source row must be visible at the scheduled checkpoint",
            "static_dimension_policy": "product master and reviewed standard-IP mapping fixed at the global 2026-09-30-style snapshot supplied by --decision-cutoff-time",
            "resume": not args.no_resume,
            "daily_exposure_audit_persisted": args.keep_daily_exposure_audit,
            "m2_checkpoint_grain": (
                "daily only when --keep-daily-exposure-audit; weekly otherwise"
            ),
            "group_intermediate_policy": (
                "retained"
                if args.keep_group_intermediates
                else "round4_consumed_rows_pruned_after_round7"
            ),
            "shared_source_cache_dir": (
                str(args.shared_source_cache_dir)
                if args.shared_source_cache_dir is not None
                else "<output-dir>/_shared_source_cache"
            ),
            "min_free_gb": args.min_free_gb,
            "output_dir": str(args.output_dir),
        }
        if args.dry_run:
            print(json.dumps(dry_plan, ensure_ascii=False, indent=2))
            print(f"log: {log_path}")
            return 0

        output_dir = _absolute(args.output_dir)
        store.write_frame(output_dir / "selected_replay_schedule.csv", selected)
        group_root_base = output_dir / "checkpoint_groups"
        shared_source_cache_dir = (
            _absolute(args.shared_source_cache_dir)
            if args.shared_source_cache_dir is not None
            else output_dir / "_shared_source_cache"
        )
        shared_source_cache_dir.mkdir(parents=True, exist_ok=True)
        candidate_counts = [
            len(
                candidate_pool_at_cutoff(
                    candidate_universe,
                    cutoff_time=cutoff_time,
                    minimum_mature_weeks=minimum_mature_weeks,
                    candidate_limit=args.candidate_limit,
                )
            )
            for cutoff_time, _, _, _, _, _ in groups
        ]
        top_k = int(parameters["comparable"]["top_k_internal"])
        restore_work_by_group = [
            min(candidate_count, top_k * len(group_targets))
            for candidate_count, (_, group_targets, _, _, _, _) in zip(
                candidate_counts, groups
            )
        ]
        # Every group also has fixed Round5-7 work even when Path4 has no
        # historical candidates. A small baseline makes the ETA less erratic
        # for the earliest empty candidate pools.
        work_by_group = [count + 30 for count in restore_work_by_group]
        total_work = sum(work_by_group)
        manifests: list[dict[str, Any]] = []
        group_records: list[dict[str, Any]] = []
        computed_seconds = 0.0
        computed_work = 0
        completed_work = 0
        replay_started = monotonic()
        for group_index, (
            cutoff_time,
            targets,
            restoration_model_path,
            restoration_model_id,
            restoration_model_usage_scope,
            token,
        ) in enumerate(groups, start=1):
            free_before_gb = shutil.disk_usage(output_dir).free / 1024**3
            if free_before_gb < args.min_free_gb:
                raise RuntimeError(
                    "checkpoint replay stopped before the next group because "
                    f"free disk space is {free_before_gb:.2f} GiB, below "
                    f"--min-free-gb={args.min_free_gb:.2f}; rerun the same "
                    "command after freeing space to resume completed groups"
                )
            group_started = monotonic()
            candidates = candidate_pool_at_cutoff(
                candidate_universe,
                cutoff_time=cutoff_time,
                minimum_mature_weeks=minimum_mature_weeks,
                candidate_limit=args.candidate_limit,
            )
            group_root = group_root_base / token
            options = {
                "decision_cutoff_time": cutoff_time.isoformat(),
                "product_snapshot_time": args.decision_cutoff_time.isoformat(),
                "batch_skus": args.batch_skus,
                "candidate_limit": args.candidate_limit,
                "pre_restore_top_k": top_k,
                "stage_order_version": "m1-topk-before-m2-through-m7-step5-recompute-v3",
                "minimum_mature_weeks": minimum_mature_weeks,
                "temporal_split_boundary_date": split_boundary.isoformat(),
                "social_media_mode": args.social_media_mode,
                "algorithm_config": str(args.algorithm_config),
                "resolved_standard_ip_file": str(args.resolved_standard_ip_file),
                "social_ip_alias_file": str(args.social_ip_alias_file),
                "constraints_file": (
                    None
                    if args.constraints_file is None
                    else str(args.constraints_file)
                ),
                "daily_exposure_audit_persisted": args.keep_daily_exposure_audit,
                "restoration_model_id": restoration_model_id,
                "restoration_model_file": str(restoration_model_path),
                "restoration_model_usage_scope": restoration_model_usage_scope,
            }
            fingerprint = replay_fingerprint(
                targets=targets,
                candidates=candidates,
                options=options,
            )
            logger.info(
                "checkpoint_replay_progress event=group_start group=%s groups=%s cutoff=%s targets=%s candidates=%s",
                group_index,
                len(groups),
                cutoff_time.isoformat(),
                len(targets),
                len(candidates),
            )
            reused = (
                not args.no_resume
                and _group_complete(
                    group_root, expected_fingerprint=fingerprint
                )
            )
            if reused:
                manifest = store.read_json(group_root / "replay_manifest.json")
                logger.info(
                    "checkpoint_replay_progress event=group_checkpoint_reused group=%s groups=%s cutoff=%s",
                    group_index,
                    len(groups),
                    cutoff_time.isoformat(),
                )
            else:
                manifest = _run_group(
                    args=args,
                    cutoff_time=cutoff_time,
                    targets=targets,
                    candidates=candidates,
                    group_root=group_root,
                    fingerprint=fingerprint,
                    logger=logger,
                    group_index=group_index,
                    group_count=len(groups),
                    shared_source_cache_dir=shared_source_cache_dir,
                    restoration_model_path=restoration_model_path,
                    restoration_model_id=restoration_model_id,
                    restoration_model_usage_scope=restoration_model_usage_scope,
                )
            if not args.keep_group_intermediates:
                bytes_before, bytes_after = _compact_completed_group(group_root)
                logger.info(
                    "checkpoint_replay_progress event=group_compacted group=%s groups=%s before_mb=%.3f after_mb=%.3f reclaimed_mb=%.3f",
                    group_index,
                    len(groups),
                    bytes_before / 1024**2,
                    bytes_after / 1024**2,
                    max(bytes_before - bytes_after, 0) / 1024**2,
                )
            manifests.append(manifest)
            group_records.append(
                {
                    "group_index": group_index,
                    "group_token": token,
                    "decision_cutoff_time": cutoff_time.isoformat(),
                    "target_rows": len(targets),
                    "candidate_skus": manifest["preselected_candidate_skus"],
                    "candidate_universe_skus": len(candidates),
                    "preselected_candidate_skus": manifest[
                        "preselected_candidate_skus"
                    ],
                    "fit_train_candidate_skus": manifest[
                        "fit_train_candidate_skus"
                    ],
                    "history_mode": manifest["history_mode"],
                    "restoration_model_id": restoration_model_id,
                    "restoration_model_file": str(restoration_model_path),
                    "restoration_model_usage_scope": restoration_model_usage_scope,
                    "reused": reused,
                    "status": "complete",
                }
            )
            group_elapsed = monotonic() - group_started
            group_work = work_by_group[group_index - 1]
            completed_work += group_work
            if not reused:
                computed_seconds += group_elapsed
                computed_work += group_work
            seconds_per_work = (
                computed_seconds / computed_work if computed_work else 0.0
            )
            eta_seconds = max(total_work - completed_work, 0) * seconds_per_work
            free_gb = shutil.disk_usage(output_dir).free / 1024**3
            logger.info(
                "checkpoint_replay_progress event=group_complete group=%s groups=%s cutoff=%s reused=%s candidates=%s candidate_work=%s total_candidate_work=%s group_elapsed_seconds=%.3f elapsed_seconds=%.3f eta_seconds=%.3f free_gb=%.3f",
                group_index,
                len(groups),
                cutoff_time.isoformat(),
                reused,
                manifest["preselected_candidate_skus"],
                completed_work,
                total_work,
                group_elapsed,
                monotonic() - replay_started,
                eta_seconds,
                free_gb,
            )
            if free_gb < args.min_free_gb and group_index < len(groups):
                raise RuntimeError(
                    "checkpoint replay stopped after a completed resumable "
                    f"group because free disk space is {free_gb:.2f} GiB, "
                    f"below --min-free-gb={args.min_free_gb:.2f}"
                )

        combined: dict[str, list[pd.DataFrame]] = {
            name: [] for name in COMBINED_OUTPUTS
        }
        for cutoff_time, targets, _, _, _, token in groups:
            group_root = group_root_base / token
            for output_name, (stage, source_name) in COMBINED_OUTPUTS.items():
                source = store.read_frame(group_root / stage / source_name)
                combined[output_name].append(
                    attach_checkpoint_identity(source, targets)
                )
        for output_name, parts in combined.items():
            frame = pd.concat(parts, ignore_index=True)
            if frame["prediction_checkpoint_key"].duplicated().any():
                raise DataContractViolation(
                    f"combined {output_name} is not unique by prediction_checkpoint_key"
                )
            store.write_frame(output_dir / output_name, frame)
        for output_name, (stage, source_name, target_column) in (
            COMBINED_MANY_OUTPUTS.items()
        ):
            parts: list[pd.DataFrame] = []
            for _, targets, _, _, _, token in groups:
                source = store.read_frame(
                    group_root_base / token / stage / source_name
                ).copy()
                identity_columns = [
                    column
                    for column in (
                        "prediction_checkpoint_key",
                        "prediction_checkpoint",
                        "lead_months",
                        "decision_cutoff_time",
                        "decision_date",
                        "release_dt",
                        "first_actual_sale_date",
                        "split",
                    )
                    if column in targets
                ]
                identity = targets[["sku_code", *identity_columns]].rename(
                    columns={"sku_code": target_column}
                )
                source[target_column] = source[target_column].astype(str)
                identity[target_column] = identity[target_column].astype(str)
                attached = identity.merge(
                    source,
                    on=target_column,
                    how="left",
                    validate="one_to_many",
                    indicator="__candidate_merge",
                )
                if attached["__candidate_merge"].ne("both").any():
                    raise DataContractViolation(
                        f"{output_name} is missing a target checkpoint"
                    )
                attached = attached.drop(columns="__candidate_merge")
                scheduled = pd.to_datetime(
                    attached["decision_cutoff_time"], errors="coerce", utc=True
                )
                actual = pd.to_datetime(
                    attached["feature_cutoff_time"], errors="coerce", utc=True
                )
                if scheduled.isna().any() or actual.isna().any() or actual.ne(
                    scheduled
                ).any():
                    raise DataContractViolation(
                        f"{output_name} cutoff does not equal scheduled checkpoint"
                    )
                parts.append(attached)
            many = pd.concat(parts, ignore_index=True)
            duplicate_key = [
                "prediction_checkpoint_key",
                "target_sku_code",
                "candidate_sku_code",
            ]
            if many.duplicated(duplicate_key).any():
                raise DataContractViolation(
                    f"combined {output_name} has duplicate target-candidate rows"
                )
            store.write_frame(output_dir / output_name, many)
        manifest_frame = pd.DataFrame(group_records)
        store.write_frame(output_dir / "replay_manifest.csv", manifest_frame)
        formal_scope = (
            args.limit_skus is None
            and not args.sku_code
            and not balanced_limits
            and args.limit_checkpoint_groups is None
            and args.candidate_limit is None
        )
        first_order_rows = pd.concat(combined["first_order_decisions.csv"], ignore_index=True)
        if "decision_status" in first_order_rows:
            blocked_rows = first_order_rows.loc[first_order_rows.decision_status.eq("blocked_no_supported_stores")].copy()
        else:
            blocked_rows = first_order_rows.iloc[:0].copy()
        blocked_columns = [c for c in ("prediction_checkpoint_key", "sku_code", "decision_cutoff_time",
            "decision_status", "decision_reason", "training_feature_eligible", "planned_store_count") if c in blocked_rows]
        store.write_frame(output_dir / "blocked_forecasts.csv", blocked_rows[blocked_columns])
        report = {
            "spec_version": "3.2",
            "status": (
                "checkpoint_features_replayed"
                if formal_scope
                else "partial_checkpoint_replay_smoke"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_skus": int(selected["sku_code"].nunique()),
            "checkpoint_rows": int(
                sum(len(group) for _, group, _, _, _, _ in groups)
            ),
            "unique_cutoff_groups": len(groups),
            "completed_groups": len(manifests),
            "blocked_forecast_count": len(blocked_rows),
            "blocked_forecast_skus": blocked_rows.sku_code.astype(str).tolist(),
            "training_features_complete": blocked_rows.empty,
            "reused_groups": int(manifest_frame["reused"].sum()),
            "static_path4_groups": int(
                manifest_frame["history_mode"].eq("static_path4").sum()
            ),
            "minimum_mature_weeks": minimum_mature_weeks,
            "temporal_split_boundary_date": split_boundary.isoformat(),
            "candidate_split_policy": "final schedule boundary; Test candidates never fit M2",
            "candidate_weight_policy": "V3.2 26-week sales-band weights refit on final Train candidates",
            "candidate_limit": args.candidate_limit,
            "pre_restore_top_k": top_k,
            "stage_order": (
                "M1 Path/hard-filter Top-K before M2; M2 reads only selected "
                "internal comparables; restored Q_i and alpha_i are then "
                "recomputed without changing candidate identities"
            ),
            "prediction_checkpoint_scope": args.prediction_checkpoint,
            "sample_scope": {
                "limit_train_skus": args.limit_train_skus,
                "limit_test_skus": args.limit_test_skus,
                "policy": (
                    "time_spread_within_locked_split"
                    if balanced_limits
                    else "full_or_existing_selection"
                ),
            },
            "point_in_time_verified": True,
            "dynamic_fact_policy": "all fact reads are bounded by each scheduled checkpoint",
            "product_snapshot_time": args.decision_cutoff_time.isoformat(),
            "product_dimension_policy": "fixed audited static snapshot, per confirmed business rule",
            "standard_ip_policy": "fixed reviewed mapping snapshot; target cohort subset only",
            "future_fact_rows_allowed": 0,
            "daily_exposure_audit_policy": (
                "persisted_for_diagnostics"
                if args.keep_daily_exposure_audit
                else "not_persisted; compact daily quality summaries only, restored outputs are weekly"
            ),
            "shared_source_cache_dir": str(shared_source_cache_dir),
            "group_intermediate_policy": (
                "retained"
                if args.keep_group_intermediates
                else "round4_consumed_rows_pruned_after_round7"
            ),
            "candidate_universe_scan_total": int(sum(candidate_counts)),
            "planned_restoration_sku_work_upper_bound": int(
                sum(restore_work_by_group)
            ),
            "min_free_gb": args.min_free_gb,
            "formal_scope": formal_scope,
            "outputs": [*COMBINED_OUTPUTS, *COMBINED_MANY_OUTPUTS],
            "selected_schedule": "selected_replay_schedule.csv",
        }
        store.write_json(output_dir / "prelaunch_feature_replay_report.json", report)
        logger.info(
            "checkpoint replay complete skus=%s rows=%s groups=%s reused=%s",
            report["selected_skus"],
            report["checkpoint_rows"],
            report["unique_cutoff_groups"],
            report["reused_groups"],
        )
    except KeyboardInterrupt:
        logger.warning(
            "checkpoint replay interrupted; completed checkpoint groups remain resumable"
        )
        print("INTERRUPTED: rerun the same command to resume completed groups.")
        print(f"log: {log_path}")
        return 130
    except Exception as exc:
        log_exception(
            logger,
            module="replay_prelaunch_features",
            cutoff=getattr(args, "decision_cutoff_time", None),
            exc=exc,
        )
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(f"status: {report['status']}")
    print(f"counts: skus={report['selected_skus']} checkpoints={report['checkpoint_rows']} groups={report['unique_cutoff_groups']}")
    print(f"report: {output_dir / 'prelaunch_feature_replay_report.json'}")
    print(f"manifest: {output_dir / 'replay_manifest.csv'}")
    print(f"selected_schedule: {output_dir / 'selected_replay_schedule.csv'}")
    print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
