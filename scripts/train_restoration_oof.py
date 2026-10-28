#!/usr/bin/env python3
"""Train V3.2 M2 with four expanding Train folds plus one final model.

This entry point owns *model fitting*.  M1/M2 replay owns only model
application: it selects Top-K comparables and loads one frozen artifact from
the registries written here.  Keeping those responsibilities separate prevents
Top-K candidates from silently becoming the v_model fitting population.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts._pipeline_common import parse_cutoff, positive_int  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402
from data.domestic import DataContractViolation  # noqa: E402
from data.launch_contract import require_verified_training_schedule  # noqa: E402
from training.checkpoint_replay import (  # noqa: E402
    align_candidate_cohort_to_schedule,
    candidate_pool_at_cutoff,
    collapse_candidate_cohort,
    normalize_replay_schedule,
)
from training.split import (  # noqa: E402
    SkuSample,
    build_time_blocks,
    expanding_window_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a point-in-time B1 feature seed, V3.2 v_model in B1->B2 "
            "... B1..B4->B5 rolling folds, then refit one final Train-only "
            "model for final replay and locked Test."
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
            "artifacts/first_order_wcover_labels_prelaunch300_v1/"
            "formal_training_schedule.csv"
        ),
    )
    parser.add_argument(
        "--candidate-cohort-file",
        type=Path,
        default=Path("artifacts/round3_dataset/cohort.csv"),
    )
    parser.add_argument(
        "--prediction-checkpoint",
        choices=("M3", "M4", "M5"),
        default="M3",
    )
    parser.add_argument("--limit-train-skus", type=positive_int, default=None)
    parser.add_argument("--limit-test-skus", type=positive_int, default=None)
    parser.add_argument(
        "--v-model-fit-sku-limit",
        type=positive_int,
        default=None,
        help=(
            "Smoke-only cap on the eligible historical Train pool for each "
            "v_model. Omit for formal V3.2 training."
        ),
    )
    parser.add_argument("--batch-skus", type=positive_int, default=25)
    parser.add_argument(
        "--huber-max-iter", type=positive_int, default=None,
        help="Numerical solve budget forwarded to every seed/fold/final fit; does not change delta, penalties or convergence tolerance.",
    )
    parser.add_argument("--event-calendar", type=Path, default=None)
    parser.add_argument("--shared-source-cache-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--decision-cutoff-time", type=parse_cutoff, required=True
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/restoration_oof_models_v1"),
    )
    parser.add_argument(
        "--algorithm-config",
        type=Path,
        default=Path("config/algorithm_v3_2.json"),
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _time_spread_skus(
    schedule: pd.DataFrame, *, split: str, limit: int | None
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
    if limit is None:
        return pool["sku_code"].astype(str).tolist()
    if limit > len(pool):
        raise DataContractViolation(
            f"requested {limit} {split} SKUs but only {len(pool)} are available"
        )
    positions = (
        [len(pool) // 2]
        if limit == 1
        else [index * (len(pool) - 1) // (limit - 1) for index in range(limit)]
    )
    return pool.iloc[positions]["sku_code"].astype(str).tolist()


def _run(command: Sequence[str], *, logger: Any, stage: str) -> None:
    logger.info("restoration_oof_progress event=stage_start stage=%s", stage)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"restoration model stage {stage} failed with exit code "
            f"{completed.returncode}"
        )
    logger.info("restoration_oof_progress event=stage_complete stage=%s", stage)


def _model_cutoff(rows: pd.DataFrame) -> datetime:
    parsed = pd.to_datetime(
        rows["decision_cutoff_time"], errors="raise", utc=True
    )
    earliest = parsed.min()
    if pd.isna(earliest):
        raise DataContractViolation("model validation/Test rows have no cutoff")
    # Preserve the business timezone representation used by the schedule.
    original = datetime.fromisoformat(
        str(rows.loc[parsed.idxmin(), "decision_cutoff_time"]).replace(
            "Z", "+00:00"
        )
    )
    if original.tzinfo is None or original.utcoffset() is None:
        raise DataContractViolation("decision cutoff must include a UTC offset")
    return original


def _fit_population(
    candidates: pd.DataFrame,
    *,
    cutoff_time: datetime,
    minimum_mature_weeks: int,
    fit_sku_limit: int | None,
) -> pd.DataFrame:
    pool = candidate_pool_at_cutoff(
        candidates,
        cutoff_time=cutoff_time,
        minimum_mature_weeks=minimum_mature_weeks,
    )
    pool = pool.loc[
        pool["split"].astype(str).str.lower().eq("train")
    ].copy()
    if fit_sku_limit is not None:
        pool = pool.sort_values(
            ["first_actual_sale_date", "sku_code"],
            ascending=[False, True],
            kind="stable",
        ).head(fit_sku_limit)
    if pool.empty:
        raise DataContractViolation(
            "no mature historical Train SKU is available before "
            f"v_model cutoff {cutoff_time.isoformat()}"
        )
    return pool.reset_index(drop=True)


def _fit_one(
    *,
    args: argparse.Namespace,
    candidates: pd.DataFrame,
    cutoff_time: datetime,
    minimum_mature_weeks: int,
    root: Path,
    shared_cache: Path,
    logger: Any,
    stage: str,
) -> dict[str, Any]:
    store = StageArtifactStore()
    fit_population = _fit_population(
        candidates,
        cutoff_time=cutoff_time,
        minimum_mature_weeks=minimum_mature_weeks,
        fit_sku_limit=args.v_model_fit_sku_limit,
    )
    root.mkdir(parents=True, exist_ok=True)
    cohort_path = root / "fit_population.csv"
    store.write_frame(cohort_path, fit_population)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_restore.py"),
        "--data-root",
        str(_absolute(args.data_root)),
        "--probe-dir",
        str(_absolute(args.probe_dir)),
        "--cohort-file",
        str(cohort_path),
        "--target-sku-file",
        str(cohort_path),
        "--output-dir",
        str(root),
        "--decision-cutoff-time",
        cutoff_time.isoformat(),
        "--product-snapshot-time",
        args.decision_cutoff_time.isoformat(),
        "--batch-skus",
        str(args.batch_skus),
        "--shared-source-cache-dir",
        str(shared_cache),
        "--skip-daily-exposure-audit",
        "--skip-partition-parts",
        "--fit-model-only",
    ]
    if args.event_calendar is not None:
        command.extend(
            ["--event-calendar", str(_absolute(args.event_calendar))]
        )
    if args.no_resume:
        command.append("--no-resume")
    if args.huber_max_iter is not None:
        command.extend(["--huber-max-iter", str(args.huber_max_iter)])
    _run(command, logger=logger, stage=stage)
    model = store.read_json(root / "restoration_model.json")
    nonconverged = model["report"].get("nonconverged_scopes", [])
    global_report = model["report"].get("category_models", {}).get("__GLOBAL__", {})
    if nonconverged:
        logger.warning("restoration_oof_progress event=model_fit_warning stage=%s nonconverged_scopes=%s", stage, nonconverged)
    return {
        "model_id": model["report"]["model_id"],
        "model_file": str((root / "restoration_model.json").relative_to(_absolute(args.output_dir))),
        "model_train_end_time": model["report"]["train_end_time"],
        "fit_candidate_skus": int(fit_population["sku_code"].nunique()),
        "global_model_converged": bool(global_report.get("converged", False)),
        "nonconverged_scope_count": len(nonconverged),
        "nonconverged_scopes": "|".join(str(value) for value in nonconverged),
        "huber_max_iter": model["report"].get("solver_controls", {}).get("huber_max_iter"),
    }


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger(
        "train_restoration_oof", config.log_dir, config.timezone
    )
    try:
        if (args.limit_train_skus is None) != (args.limit_test_skus is None):
            raise ValueError(
                "--limit-train-skus and --limit-test-skus must be supplied together"
            )
        store = StageArtifactStore()
        schedule_all = normalize_replay_schedule(
            store.read_frame(_absolute(args.schedule_file)),
            available_through=args.decision_cutoff_time,
        )
        require_verified_training_schedule(schedule_all)
        selected_skus = set(
            _time_spread_skus(
                schedule_all, split="Train", limit=args.limit_train_skus
            )
            + _time_spread_skus(
                schedule_all, split="Test", limit=args.limit_test_skus
            )
        )
        schedule = schedule_all.loc[
            schedule_all["sku_code"].astype(str).isin(selected_skus)
            & schedule_all["prediction_checkpoint"].astype(str).eq(
                args.prediction_checkpoint
            )
        ].copy()
        train_schedule = schedule.loc[
            schedule["split"].astype(str).str.lower().eq("train")
        ].copy()
        test_schedule = schedule.loc[
            schedule["split"].astype(str).str.lower().eq("test")
        ].copy()
        if train_schedule.empty or test_schedule.empty:
            raise DataContractViolation(
                "restoration OOF training requires non-empty fixed Train and Test"
            )
        train_identities = train_schedule[
            ["sku_code", "first_actual_sale_date"]
        ].drop_duplicates("sku_code")
        samples = [
            SkuSample(
                pd.Timestamp(row.first_actual_sale_date).date(),
                str(row.sku_code),
            )
            for row in train_identities.itertuples(index=False)
        ]
        blocks = build_time_blocks(
            samples, n_time_blocks=config.training.n_time_blocks
        )
        folds = expanding_window_folds(blocks)
        parameters = load_algorithm_parameters(
            PROJECT_ROOT / args.algorithm_config
        )
        minimum_mature_weeks = int(
            parameters["comparable"].get("minimum_mature_weeks", 4)
        )
        candidates_raw = collapse_candidate_cohort(
            store.read_frame(_absolute(args.candidate_cohort_file))
        )
        candidates, split_boundary = align_candidate_cohort_to_schedule(
            candidates_raw, schedule_all
        )
        output_dir = _absolute(args.output_dir)
        shared_cache = (
            _absolute(args.shared_source_cache_dir)
            if args.shared_source_cache_dir is not None
            else output_dir / "_shared_source_cache"
        )
        plan = {
            "spec_version": "3.2",
            "launch_universe_version": (
                str(schedule["launch_universe_version"].iloc[0])
                if "launch_universe_version" in schedule else None
            ),
            "time_blocks": config.training.n_time_blocks,
            "rolling_folds": len(folds),
            "train_skus": int(train_schedule["sku_code"].nunique()),
            "test_skus": int(test_schedule["sku_code"].nunique()),
            "prediction_checkpoint": args.prediction_checkpoint,
            "final_model_cutoff": _model_cutoff(test_schedule).isoformat(),
            "fit_sku_limit": args.v_model_fit_sku_limit,
            "huber_max_iter": args.huber_max_iter or config.restoration.huber_max_iter,
            "pilot_versions": sorted(schedule["pilot_version"].astype(str).unique()) if "pilot_version" in schedule else [],
        }
        if args.dry_run:
            print(json.dumps({"status": "dry_run", **plan}, ensure_ascii=False, indent=2))
            print(f"log: {log_path}")
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        shared_cache.mkdir(parents=True, exist_ok=True)
        oof_rows: list[dict[str, Any]] = []
        train_feature_rows: list[dict[str, Any]] = []
        block_rows: list[dict[str, Any]] = []
        for block_index, block in enumerate(blocks, start=1):
            for sample in block:
                block_rows.append(
                    {
                        "sku_code": sample.sku_code,
                        "first_actual_sale_date": sample.first_actual_sale_date,
                        "time_block": block_index,
                    }
                )
        # B1 is the warm-up training block and therefore has no outer-fold OOF
        # score.  It still needs a point-in-time-safe upstream feature snapshot
        # so that the B1->B2 first-order fit can be executed.  Fit one seed
        # v_model no later than the earliest B1 decision cutoff; this row is
        # never counted as an OOF validation prediction.
        b1_skus = {sample.sku_code for sample in blocks[0]}
        b1_rows = train_schedule.loc[
            train_schedule["sku_code"].astype(str).isin(b1_skus)
        ]
        b1_cutoff = _model_cutoff(b1_rows)
        b1_metadata = _fit_one(
            args=args,
            candidates=candidates,
            cutoff_time=b1_cutoff,
            minimum_mature_weeks=minimum_mature_weeks,
            root=output_dir / "b1_seed",
            shared_cache=shared_cache,
            logger=logger,
            stage="b1_point_in_time_seed",
        )
        for sku in sorted(b1_skus):
            train_feature_rows.append(
                {
                    "sku_code": sku,
                    "split": "Train",
                    "usage_scope": "oof_seed",
                    "fold": 0,
                    **b1_metadata,
                }
            )
        for fold_index, (_, validation_samples) in enumerate(folds, start=1):
            validation_skus = {sample.sku_code for sample in validation_samples}
            validation_rows = train_schedule.loc[
                train_schedule["sku_code"].astype(str).isin(validation_skus)
            ]
            cutoff = _model_cutoff(validation_rows)
            metadata = _fit_one(
                args=args,
                candidates=candidates,
                cutoff_time=cutoff,
                minimum_mature_weeks=minimum_mature_weeks,
                root=output_dir / f"fold_{fold_index}",
                shared_cache=shared_cache,
                logger=logger,
                stage=f"fold_{fold_index}",
            )
            for sku in sorted(validation_skus):
                row = {
                    "sku_code": sku,
                    "split": "Train",
                    "usage_scope": "oof",
                    "fold": fold_index,
                    **metadata,
                }
                oof_rows.append(row)
                train_feature_rows.append(row)

        final_cutoff = _model_cutoff(test_schedule)
        final_metadata = _fit_one(
            args=args,
            candidates=candidates,
            cutoff_time=final_cutoff,
            minimum_mature_weeks=minimum_mature_weeks,
            root=output_dir / "final",
            shared_cache=shared_cache,
            logger=logger,
            stage="final_full_train",
        )
        final_rows = [
            {
                "sku_code": str(row.sku_code),
                "split": str(row.split),
                "usage_scope": (
                    "final_fit"
                    if str(row.split).lower() == "train"
                    else "locked_test"
                ),
                "fold": None,
                **final_metadata,
            }
            for row in schedule[["sku_code", "split"]]
            .drop_duplicates("sku_code")
            .itertuples(index=False)
        ]
        registry_version = (
            str(schedule["launch_universe_version"].iloc[0])
            if "launch_universe_version" in schedule else None
        )
        if registry_version is not None:
            for rows in (oof_rows, train_feature_rows, final_rows):
                for row in rows:
                    row["launch_universe_version"] = registry_version
        store.write_frame(output_dir / "time_blocks.csv", pd.DataFrame(block_rows))
        store.write_frame(
            output_dir / "oof_model_registry.csv", pd.DataFrame(oof_rows)
        )
        store.write_frame(
            output_dir / "train_feature_model_registry.csv",
            pd.DataFrame(train_feature_rows),
        )
        store.write_frame(
            output_dir / "final_model_registry.csv", pd.DataFrame(final_rows)
        )
        report = {
            **plan,
            "status": (
                "smoke_limited_fit_population"
                if args.v_model_fit_sku_limit is not None
                else "formal_v3_2_restoration_oof_models"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "temporal_split_boundary_date": split_boundary.isoformat(),
            "fold_models": len(folds),
            "final_model_id": final_metadata["model_id"],
            "b1_policy": (
                "warm-up block has no OOF score; a model ending at the earliest "
                "B1 checkpoint is used only to create point-in-time B1 fit features"
            ),
            "b1_seed_model_id": b1_metadata["model_id"],
            "leakage_policy": (
                "each fold model ends no later than the earliest validation "
                "decision cutoff; final model ends no later than the earliest "
                "locked-Test decision cutoff"
            ),
            "stage_order": (
                "offline fit frozen v_model; replay then executes M1 Top-K and "
                "applies, never refits, M2"
            ),
            "fit_health": {
                "b1_seed": b1_metadata,
                "folds": [dict(row) for row in {int(item["fold"]): item for item in oof_rows}.values()],
                "final": final_metadata,
                "all_global_models_converged": all(bool(row["global_model_converged"]) for row in [b1_metadata, final_metadata, *oof_rows]),
                "all_scopes_converged": all(int(row["nonconverged_scope_count"]) == 0 for row in [b1_metadata, final_metadata, *oof_rows]),
            },
        }
        if not report["fit_health"]["all_scopes_converged"] or not report["fit_health"]["all_global_models_converged"]:
            report["status"] = "restoration_oof_models_with_fit_warnings"
        report["validation_scope"] = "development_pilot_not_final_acceptance" if "pilot_version" in schedule else "full_fixed_schedule"
        store.write_json(output_dir / "restoration_oof_report.json", report)
    except Exception as exc:
        log_exception(logger, module="train_restoration_oof", exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1

    print("status: " + report["status"])
    print(f"oof_registry: {output_dir / 'oof_model_registry.csv'}")
    print(
        "train_feature_registry: "
        f"{output_dir / 'train_feature_model_registry.csv'}"
    )
    print(f"final_registry: {output_dir / 'final_model_registry.csv'}")
    print(f"report: {output_dir / 'restoration_oof_report.json'}")
    print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
