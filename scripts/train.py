#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import add_execution_arguments, print_dry_run  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.domain import load_algorithm_parameters  # noqa: E402
from training.pilot import evaluation_scope_metadata  # noqa: E402
from training import fix_section84_dataset, train_section84_first_order  # noqa: E402
from training.fit_population import prepare_fit_population  # noqa: E402


DEFAULT_FEATURES = (
    "feature__adjustments__delta_IP",
    "feature__adjustments__delta_PRODUCT",
    "feature__adjustments__delta_OPERATION",
    "feature__adjustments__delta_log_price",
    "feature__adjustments__competition_trigger_count",
)


def _feature_sign(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "feature sign must be COLUMN=nonnegative|nonpositive|free"
        )
    column, sign = value.split("=", 1)
    if not column or sign not in {"nonnegative", "nonpositive", "free"}:
        raise argparse.ArgumentTypeError(
            "feature sign must be COLUMN=nonnegative|nonpositive|free"
        )
    return column, sign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the V3.2 section 8.4 first-order training sequence: freeze "
            "data, construct B1-B5, produce four expanding-window OOF folds, "
            "select configuration on OOF only, refit on all Train, and leave "
            "Test locked for backtest.py."
        )
    )
    parser.add_argument("--oof-replay-dir", type=Path, required=True,
                        help="Train replay root containing Step5 canonical recompute bundles.")
    parser.add_argument("--final-replay-dir", type=Path, required=True,
                        help="Final Train/Test replay root; only Train is used for fitting.")
    parser.add_argument("--fixed-hyperparameters", action="store_true",
                        help="Fix one outer hyperparameter preset and K, while still learning M1 theta/tau and first-order coefficients in each OOF fold and final Train.")
    parser.add_argument("--fixed-reference-parameters", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ablation-baseline-dataset", type=Path, default=None,
                        help="Optional paired frozen corrected-quantity baseline, same checkpoint keys and labels.")
    parser.add_argument("--ablation-placement-dataset", type=Path, default=None,
                        help="Optional paired placement-only correction version; do not substitute unrelated SKUs.")
    parser.add_argument(
        "--oof-modeling-dataset",
        type=Path,
        required=True,
        help=(
            "Train-only point-in-time features generated with the B1 seed and "
            "B2-B5 frozen upstream fold models."
        ),
    )
    parser.add_argument(
        "--final-modeling-dataset",
        type=Path,
        required=True,
        help=(
            "Separately replayed all-Train plus locked-Test features generated "
            "with the frozen final upstream model. Test labels are not read here."
        ),
    )
    parser.add_argument(
        "--oof-comparable-detail",
        type=Path,
        required=True,
        help=(
            "Train-only checkpoint-keyed comparable_candidate_detail.csv from "
            "the rolling replay. Each target contains only its own PIT M1 Top-K."
        ),
    )
    parser.add_argument(
        "--final-comparable-detail",
        type=Path,
        required=True,
        help=(
            "All-Train plus locked-Test comparable detail replayed with the "
            "final upstream model. Test rows are not used during training."
        ),
    )
    parser.add_argument(
        "--feature-column",
        action="append",
        default=[],
        help="Relative-factor input column. Repeat for more columns.",
    )
    parser.add_argument(
        "--feature-sign",
        action="append",
        type=_feature_sign,
        default=[],
        help="Per-feature sign constraint COLUMN=nonnegative|nonpositive|free.",
    )
    parser.add_argument(
        "--base-demand-column",
        default="feature__first_order__D_demand_base",
        help=(
            "Pure M6 sales-demand baseline before demand multipliers. The V3.2 "
            "training ratio is D_first_ref / D_demand_base; Q_base contains "
            "safety/display ordering policy and is intentionally excluded."
        ),
    )
    parser.add_argument(
        "--upstream-restoration-registry",
        type=Path,
        default=None,
        help=(
            "Optional train_feature_model_registry.csv used to audit that every "
            "fixed Train SKU used an oof_seed/oof M2 model."
        ),
    )
    add_execution_arguments(parser, default_output="artifacts/section8_4_training")
    args = parser.parse_args()
    if args.fixed_reference_parameters:
        parser.error("obsolete --fixed-reference-parameters freezes trainable theta/tau; use the corrected --fixed-hyperparameters runner")
    return args


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _fixed_hyperparameter_settings(parameters: dict[str, Any]) -> dict[str, Any]:
    """Use existing configured values, never select a preset from Test/OOF."""
    comparable = parameters["comparable"]
    training = parameters["training"]
    return {
        "similarity_weights_initialization_only": dict(comparable["similarity_weights"]),
        "temperature_training_grid": list(training["comparable_temperature_grid"]),
        "top_k": int(comparable["top_k_internal"]),
        "similarity_regularization": float(training["similarity_regularization_grid"][0]),
        "ridge": float(training["ridge_grid"][0]),
        "l1_ratio": float(training["l1_ratio_grid"][0]),
        "half_life_weeks": training["half_life_weeks"][0],
        "selection_policy": "existing outer hyperparameter values fixed before evaluation; Section8.3B theta and tau still trained",
    }


def _training_code_hash(paths: list[Path]) -> str:
    digest = sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _default_sign(column: str) -> str:
    if column.endswith("delta_log_price") or column.endswith(
        "competition_trigger_count"
    ):
        return "nonpositive"
    return "nonnegative"


def _validate_point_in_time(frame: pd.DataFrame, *, name: str) -> None:
    required = {
        "prediction_checkpoint_key",
        "decision_cutoff_time",
        "first_actual_sale_date",
        "split",
        "dataset_version",
        "label",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame["prediction_checkpoint_key"].astype(str).duplicated().any():
        raise ValueError(f"{name} is not unique by prediction_checkpoint_key")
    if "point_in_time_verified" in frame:
        verified = (
            frame["point_in_time_verified"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "t", "yes", "y"})
        )
        if not verified.all():
            raise ValueError(f"{name} contains non-verified point-in-time rows")
    cutoff = pd.to_datetime(
        frame["decision_cutoff_time"], errors="coerce", utc=True
    )
    first_sale = pd.to_datetime(
        frame["first_actual_sale_date"], errors="coerce", utc=True
    )
    if cutoff.isna().any() or first_sale.isna().any():
        raise ValueError(f"{name} contains invalid cutoff/first-sale timestamps")
    if cutoff.dt.date.ge(first_sale.dt.date).any():
        raise ValueError(f"{name} contains a cutoff on/after first sale")


def _audit_restoration_registry(
    registry: pd.DataFrame,
    fixed_rows: pd.DataFrame,
) -> dict[str, Any]:
    required = {
        "sku_code",
        "usage_scope",
        "fold",
        "model_id",
        "model_train_end_time",
    }
    missing = sorted(required - set(registry.columns))
    if missing:
        raise ValueError(
            "upstream restoration registry is missing columns: " f"{missing}"
        )
    source = registry.copy()
    source["sku_code"] = source["sku_code"].astype(str)
    if source["sku_code"].duplicated().any():
        raise ValueError("upstream restoration registry is not unique by SKU")
    fixed_train = fixed_rows.loc[
        fixed_rows["split"].astype(str).str.lower().eq("train")
    ].copy()
    expected = set(fixed_train["sku_code"].astype(str))
    actual = set(source["sku_code"])
    if actual != expected:
        raise ValueError(
            "upstream restoration registry does not exactly cover fixed Train: "
            f"missing={sorted(expected-actual)[:10]} extra={sorted(actual-expected)[:10]}"
        )
    block = fixed_train[["sku_code", "time_block"]].drop_duplicates("sku_code")
    checked = block.merge(source, on="sku_code", how="left", validate="one_to_one")
    b1_bad = checked["time_block"].eq("B1") & ~checked["usage_scope"].astype(
        str
    ).eq("oof_seed")
    later_bad = checked["time_block"].ne("B1") & ~checked[
        "usage_scope"
    ].astype(str).eq("oof")
    if b1_bad.any() or later_bad.any():
        raise ValueError(
            "restoration registry must use oof_seed for B1 and oof for B2-B5"
        )
    expected_fold = checked["time_block"].str[1:].astype(int) - 1
    later = checked["time_block"].ne("B1")
    parsed_fold = pd.to_numeric(checked["fold"], errors="coerce")
    if parsed_fold.loc[later].ne(expected_fold.loc[later]).any():
        raise ValueError("restoration registry fold does not match B2-B5")
    return {
        "status": "verified",
        "rows": int(len(source)),
        "models": int(source["model_id"].nunique()),
        "usage_scope_counts": {
            str(key): int(value)
            for key, value in source["usage_scope"].value_counts().items()
        },
        "model_fit_health_available": "nonconverged_scope_count" in source and "global_model_converged" in source,
        "has_nonconverged_scopes": (
            bool(pd.to_numeric(source["nonconverged_scope_count"], errors="coerce").fillna(1).gt(0).any())
            or not source["global_model_converged"].astype(str).str.lower().isin(("true", "1")).all()
        ) if {"nonconverged_scope_count", "global_model_converged"}.issubset(source) else True,
    }


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("train", config.log_dir, config.timezone)
    output_names = [
        "fixed_dataset.csv",
        "excluded_rows.csv",
        "time_blocks.csv",
        "fold_plan.csv",
        "model.json",
        "oof_predictions.csv",
        "model_selection.csv",
        "fold_models.csv",
        "locked_test_keys.csv",
        "training_report.json",
        "comparable_model.json",
        "train_ablation.csv",
        "fit_exclusions.csv",
    ]
    if args.dry_run:
        print_dry_run("V3.2 section 8.4 training", args, output_names)
        return 0
    try:
        store = StageArtifactStore()
        parameters = load_algorithm_parameters(_absolute(args.algorithm_config))
        training_grid = parameters["training"]
        fixed_reference = _fixed_hyperparameter_settings(parameters) if args.fixed_hyperparameters else None
        if fixed_reference is not None:
            logger.info("fixed_hyperparameters preset=%s configurations=1 learned_parameters=theta,tau,first_order_coefficients", fixed_reference)
        oof_frame = store.read_frame(_absolute(args.oof_modeling_dataset))
        final_frame = store.read_frame(_absolute(args.final_modeling_dataset))
        oof_candidate_detail = store.read_frame(
            _absolute(args.oof_comparable_detail)
        )
        final_candidate_detail = store.read_frame(
            _absolute(args.final_comparable_detail)
        )
        _validate_point_in_time(oof_frame, name="OOF modeling dataset")
        _validate_point_in_time(final_frame, name="final modeling dataset")
        fixed = fix_section84_dataset(
            final_frame, n_time_blocks=config.training.n_time_blocks
        )
        # Freeze the original 20/5 identities and B1-B5 BEFORE omitting an
        # explicitly unavailable Train forecast. Never rebuild these blocks
        # from a shortened cohort or silently substitute another target.
        fit_fixed, fit_oof, fit_final, fit_exclusions = prepare_fit_population(fixed, oof_frame, final_frame)
        logger.info("parameter_fit_population fixed_train=%s usable_train=%s test_locked=%s excluded=%s original_blocks_retained=True",
                    fixed.manifest["train_skus"], fit_fixed.manifest["train_skus"],
                    fixed.manifest["test_skus"], fit_exclusions.to_dict(orient="records"))
        from data.recompute import ReplayBundleStore
        from first_order.pipeline import FullChainReplay
        replay_options = {"key_column": fixed.manifest["primary_key"],
                          "base_demand_column": args.base_demand_column,
                          "progress": lambda d: logger.info("step5_recompute_progress details=%s", d)}
        oof_replay = FullChainReplay(ReplayBundleStore(_absolute(args.oof_replay_dir), parameters), **replay_options)
        final_replay = FullChainReplay(ReplayBundleStore(_absolute(args.final_replay_dir), parameters), **replay_options)
        learned_small_factors = sorted(
            column
            for column in final_frame
            if column.startswith("feature__adjustments__delta_factor__")
        )
        feature_columns = args.feature_column or [
            *learned_small_factors,
            *[
                column
                for column in DEFAULT_FEATURES
                if column in final_frame
                and not (
                    learned_small_factors
                    and column.endswith(("delta_IP", "delta_PRODUCT", "delta_OPERATION"))
                )
            ],
        ]
        if not feature_columns:
            raise ValueError(
                "no Section 8.3 C relative-factor columns are available; "
                "rerun Round7 with delta_* outputs or pass --feature-column"
            )
        explicit_signs = dict(args.feature_sign)
        unknown_signs = sorted(set(explicit_signs) - set(feature_columns))
        if unknown_signs:
            raise ValueError(
                "--feature-sign references a column not selected for training: "
                f"{unknown_signs}"
            )
        feature_signs = {
            column: explicit_signs.get(column, _default_sign(column))
            for column in feature_columns
        }
        factor_to_group = {
            factor: group
            for group, group_weights in parameters["factors"][
                "small_factor_weights"
            ].items()
            for factor in group_weights
        }
        feature_groups: dict[str, str] = {}
        for column in feature_columns:
            if "delta_factor__" in column:
                factor_name = column.split("delta_factor__", 1)[1]
                if factor_name not in factor_to_group:
                    raise ValueError(
                        "small-factor delta has no V3.2 group mapping: "
                        f"{factor_name}"
                    )
                feature_groups[column] = factor_to_group[factor_name]
            elif column.endswith("delta_IP"):
                feature_groups[column] = "IP"
            elif column.endswith("delta_PRODUCT"):
                feature_groups[column] = "PRODUCT"
            elif column.endswith("delta_OPERATION"):
                feature_groups[column] = "OPERATION"
            elif column.endswith("delta_log_price"):
                feature_groups[column] = "PRICE"
            elif column.endswith("competition_trigger_count"):
                feature_groups[column] = "COMPETITION"
            else:
                raise ValueError(
                    "feature has no production-formula group; pass only V3.2 "
                    f"relative-factor inputs: {column}"
                )
        grade_settings = parameters["first_order"]["grade_thresholds"]
        own_thresholds = [
            grade_settings["own_ip"][name]
            for name in ("C_max", "B_max", "A_max")
        ]
        licensed_thresholds = [
            grade_settings["licensed_ip"][name]
            for name in ("C_max", "B_max", "A_max")
        ]
        if own_thresholds != licensed_thresholds:
            raise ValueError(
                "own/licensed grade thresholds differ; add copyright-specific "
                "BandMAE evaluation before training this parameter version"
            )
        result = train_section84_first_order(
            fixed_hyperparameters=bool(fixed_reference),
            oof_replay=oof_replay, final_replay=final_replay,
            progress=lambda d: logger.info("step5_training_progress details=%s", d),
            fixed=fit_fixed,
            oof_feature_rows=fit_oof,
            final_feature_rows=fit_final,
            oof_candidate_detail=oof_candidate_detail,
            final_candidate_detail=final_candidate_detail,
            initial_similarity_weights=parameters["comparable"][
                "similarity_weights"
            ],
            comparable_temperature_grid=(fixed_reference["temperature_training_grid"] if fixed_reference else training_grid["comparable_temperature_grid"]),
            comparable_top_k_grid=([fixed_reference["top_k"]] if fixed_reference else training_grid["comparable_top_k_grid"]),
            similarity_regularization_grid=([fixed_reference["similarity_regularization"]] if fixed_reference else training_grid["similarity_regularization_grid"]),
            feature_columns=feature_columns,
            feature_signs=feature_signs,
            feature_groups=feature_groups,
            base_demand_column=args.base_demand_column,
            ridge_grid=([fixed_reference["ridge"]] if fixed_reference else training_grid["ridge_grid"]),
            l1_ratio_grid=([fixed_reference["l1_ratio"]] if fixed_reference else training_grid["l1_ratio_grid"]),
            half_life_grid=([fixed_reference["half_life_weeks"]] if fixed_reference else training_grid["half_life_weeks"]),
            demand_bounds=tuple(parameters["factors"]["demand_bounds"]),
            coefficient_bounds=parameters["factors"]["coefficient_bounds"],
            price_bounds=tuple(parameters["factors"]["price_bounds"]),
            grade_thresholds=own_thresholds,
            sales_band_edges=parameters["training"]["sales_band_edges"],
        )
        restoration_audit: dict[str, Any]
        if args.upstream_restoration_registry is None:
            restoration_audit = {
                "status": "not_supplied",
                "formal_full_chain_claim_allowed": False,
            }
        else:
            restoration_audit = _audit_restoration_registry(
                store.read_frame(_absolute(args.upstream_restoration_registry)),
                fixed.rows,
            )
            restoration_audit["formal_full_chain_claim_allowed"] = (
                restoration_audit["model_fit_health_available"] and not restoration_audit["has_nonconverged_scopes"]
            )

        evaluation_metadata = evaluation_scope_metadata(fixed.rows)
        if not evaluation_metadata["final_acceptance_claim_allowed"]:
            restoration_audit["formal_full_chain_claim_allowed"] = False
        if fixed_reference is not None:
            restoration_audit["formal_full_chain_claim_allowed"] = False

        output_dir = _absolute(args.output_dir)
        code_hash = _training_code_hash(
            [
                PROJECT_ROOT / "scripts/train.py",
                PROJECT_ROOT / "src/training/section84.py",
                PROJECT_ROOT / "src/training/comparable_tuning.py",
                PROJECT_ROOT / "src/first_order/pipeline.py",
                PROJECT_ROOT / "src/factors/frozen.py",
                PROJECT_ROOT / "src/factors/engine.py",
                PROJECT_ROOT / "src/placement/lifecycle.py",
                PROJECT_ROOT / "src/data/recompute.py",
                PROJECT_ROOT / "src/placement/engine.py",
                PROJECT_ROOT / "src/first_order/engine.py",
                PROJECT_ROOT / "src/training/pilot.py",
                PROJECT_ROOT / "src/data/modeling_dataset.py",
                PROJECT_ROOT / "src/data/forecast_eligibility.py",
                PROJECT_ROOT / "src/training/fit_population.py",
                _absolute(args.algorithm_config),
            ]
        )
        store.write_frame(output_dir / output_names[0], fixed.rows)
        store.write_frame(output_dir / output_names[1], fixed.exclusions)
        store.write_frame(output_dir / output_names[2], fixed.time_blocks)
        store.write_frame(output_dir / output_names[3], fit_fixed.folds)
        store.write_frame(output_dir / "fit_exclusions.csv", fit_exclusions)
        model_payload = result.model.as_dict()
        model_payload.update(evaluation_metadata)
        model_payload.update(
            {
                "model_id": result.report["final_model_id"],
                "dataset_version": fixed.manifest["dataset_version"],
                "dataset_hash": fixed.manifest["dataset_hash"],
                "oof_log_residual_quantiles": result.report[
                    "oof_log_residual_quantiles"
                ],
                "parameter_version": parameters["parameter_version"],
                "rule_version": parameters["rule_version"],
                "comparable_model": result.comparable_model.as_dict(),
                "training_code_hash": code_hash,
                "randomness_policy": "deterministic_no_random_sampling",
                "pipeline_version": "step5-full-chain-v1",
                "pipeline_parameters": parameters,
                "fit_population": fit_fixed.manifest,
                "fit_exclusions": fit_exclusions.to_dict(orient="records"),
                "training_mode": result.report["training_mode"],
                "fixed_hyperparameter_preset": fixed_reference,
            }
        )
        store.write_json(output_dir / output_names[4], model_payload)
        store.write_frame(output_dir / output_names[5], result.oof_predictions)
        store.write_frame(output_dir / output_names[6], result.model_selection)
        store.write_frame(output_dir / output_names[7], result.fold_models)
        from training.ablation import train_ablation
        ablation = train_ablation(oof=result.oof_predictions,
            key=fixed.manifest["primary_key"], base_column=args.base_demand_column,
            grade_thresholds=own_thresholds, sales_band_edges=parameters["training"]["sales_band_edges"],
            baseline=None if args.ablation_baseline_dataset is None else store.read_frame(_absolute(args.ablation_baseline_dataset)),
            placement=None if args.ablation_placement_dataset is None else store.read_frame(_absolute(args.ablation_placement_dataset)))
        store.write_frame(output_dir / "train_ablation.csv", ablation)
        locked_test = fixed.rows.loc[
            fixed.rows["split"].astype(str).str.lower().eq("test"),
            [
                "prediction_checkpoint_key",
                "sku_code",
                "first_actual_sale_date",
                "dataset_version",
            ],
        ].copy()
        store.write_frame(output_dir / output_names[8], locked_test)
        store.write_json(
            output_dir / output_names[10], result.comparable_model.as_dict()
        )
        report = dict(result.report)
        report["train_ablation"] = ablation.fillna("").to_dict(orient="records")
        report["historical_stage_attribution_complete"] = bool(ablation.status.eq("evaluated").all())
        report.update(evaluation_metadata)
        report.update(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
                "fixed_dataset": fixed.manifest,
                "fit_population": fit_fixed.manifest,
                "fit_exclusions": fit_exclusions.to_dict(orient="records"),
                "fixed_hyperparameter_preset": fixed_reference,
                "feature_columns": feature_columns,
                "feature_signs": feature_signs,
                "feature_groups": feature_groups,
                "base_demand_column": args.base_demand_column,
                "parameter_version": parameters["parameter_version"],
                "rule_version": parameters["rule_version"],
                "training_code_hash": code_hash,
                "randomness_policy": "deterministic_no_random_sampling",
                "comparable_candidate_scope": (
                    "per-target PIT M1 Top-3; theta/tau refitted in each historical Train fold; K fixed at configured value"
                    if fixed_reference else "per-target point-in-time M1 Top-3 identities; fold-local "
                    "theta/tau/K refit uses only the historical fit blocks"
                ),
                "search_grid_status": (
                    "outer_grid_not_run; theta/tau_and_demand_parameters_still_learned"
                    if fixed_reference else "V3.2 specifies Train-only grid selection but not exact "
                    "grid values; the versioned grids are disclosed in the "
                    "algorithm config and selected only from OOF results"
                ),
                "upstream_restoration_registry_audit": restoration_audit,
                "section_8_4_steps": {
                    "1_fixed_training_data": "complete",
                    "2_B1_B5_rolling_plan": "complete",
                    "3_fold_refit_first_order_parameters": "complete",
                    "3_fold_refit_historical_restore": restoration_audit[
                        "status"
                    ],
                    "3_fold_refit_comparable_parameters": "complete",
                    "3_fold_refit_reorder_modules": (
                        "not_part_of_first_order_training_artifact"
                    ),
                    "4_oof_selection_and_interval_calibration": "OOF_metrics_and_interval_calibration_only_no_configuration_selection" if fixed_reference else "complete",
                    "5_full_train_refit": "complete",
                    "6_version_locked": "complete",
                    "7_test_final_replay": "pending_backtest_py_once",
                },
                "formal_scope": (
                    "diagnostic_single_outer_hyperparameter_preset; Section8.3B theta/tau and Section8.3C demand coefficients fitted"
                    if fixed_reference else "first_order_section_8_4_training including Section 8.3 B/C; "
                    "reorder D remains a separate mature decision-node model"
                ),
                "server_training_verified": False,
            }
        )
        store.write_json(output_dir / output_names[9], report)
        logger.info(
            "section 8.4 training complete train_skus=%s oof_rows=%s test_locked=%s",
            report["train_skus"],
            report["oof_rows"],
            report["test_skus_locked"],
        )
    except Exception as exc:
        log_exception(
            logger, module="train", cutoff=args.decision_cutoff_time, exc=exc
        )
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1

    print("status: section_8_4_first_order_train_complete_test_still_locked")
    print(f"model: {output_dir / output_names[4]}")
    print(f"oof: {output_dir / output_names[5]}")
    print(f"report: {output_dir / output_names[9]}")
    print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
