#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig, TableSpec, load_table_specs  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.domestic import DataContractViolation  # noqa: E402
from data.normalization import normalize_identifier  # noqa: E402
from data.launch_contract import validate_launch_origin_contract  # noqa: E402
from data.restoration_adapter import (  # noqa: E402
    RESTORATION_SOURCE_NAMES,
    RestorationDataAdapter,
)
from data.validation import validate_probe_evidence  # noqa: E402
from restore.batched import run_batched_restoration  # noqa: E402
from restore.historical import FittedRestorationModel  # noqa: E402


def _parse_cutoff(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cutoff must be ISO-8601, for example 2026-09-30T23:59:59+08:00"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("cutoff must include a UTC offset")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _restoration_probe_specs(table_specs: list[TableSpec]) -> list[TableSpec]:
    """Select only the physical sources that ROUND 4 actually reads."""

    by_name = {spec.table_name: spec for spec in table_specs}
    missing = sorted(set(RESTORATION_SOURCE_NAMES) - set(by_name))
    if missing:
        raise ValueError(
            "ROUND 4 source configuration is incomplete: " + ", ".join(missing)
        )
    return [by_name[name] for name in RESTORATION_SOURCE_NAMES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a bounded V3.2 ROUND 4 M2 historical-restoration smoke sample."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--probe-dir", type=Path, default=Path("artifacts/server_probe_round2")
    )
    parser.add_argument(
        "--cohort-file", type=Path, default=Path("artifacts/round3_dataset/cohort.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/round4_restore")
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Optional existing source/pre-model checkpoint root. Allows new outputs while preserving baseline CSV/report files.",
    )
    parser.add_argument(
        "--cutoff-time", "--decision-cutoff-time", dest="cutoff_time",
        type=_parse_cutoff, required=True,
    )
    parser.add_argument(
        "--product-snapshot-time",
        type=_parse_cutoff,
        default=None,
        help=(
            "Optional fixed product-master snapshot time. Fact sources remain "
            "strictly bounded by --decision-cutoff-time; only static product "
            "attributes are read from this separately audited snapshot."
        ),
    )
    parser.add_argument(
        "--sku-limit", "--limit-skus", dest="sku_limit",
        type=_positive_int,
        default=5,
        help="Latest mature Train SKUs to read when --sku-code is not supplied.",
    )
    parser.add_argument(
        "--batch-skus",
        type=_positive_int,
        default=5,
        help=(
            "Sequential source-read batch size. Defaults to 5; lower it when memory "
            "is constrained. This does not change the joint M2 model fit."
        ),
    )
    parser.add_argument(
        "--window-weeks", type=_positive_int, default=None,
        help="Optional lifecycle read horizon for offline W_cover labels; default is the existing 26-week distribution horizon. This never changes the stored sales_26w_qty audit.",
    )
    parser.add_argument(
        "--huber-max-iter", type=_positive_int, default=None,
        help="Numerical fitting budget only. Does not change Huber delta, penalties, convergence tolerance, exposure or weekly cache contents. Omit for the configured budget.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore completed source checkpoints and reread every selected SKU. "
            "By default the same command resumes completed batches."
        ),
    )
    parser.add_argument(
        "--skip-daily-exposure-audit",
        action="store_true",
        help=(
            "Do not persist daily_exposure_audit.csv or its final Parquet parts. "
            "The daily panel is still computed for M2 quality checks, and resumable "
            "batch checkpoints are retained. Use this for feature replay stages "
            "whose downstream modules read only restored weekly artifacts."
        ),
    )
    parser.add_argument(
        "--shared-source-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional cross-cutoff SKU source cache used by historical replay. "
            "Cached fact rows are always sliced back to the current cutoff and "
            "SKU lifecycle window before model fitting."
        ),
    )
    parser.add_argument(
        "--skip-partition-parts",
        action="store_true",
        help=(
            "Write only the canonical restored CSV artifacts, not duplicate "
            "Parquet part directories. Intended for a checkpoint group that is "
            "consumed immediately and compacted by the replay orchestrator."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sku-code",
        action="append",
        default=[],
        help=(
            "Explicit ROUND 3 cohort SKU; repeat for multiple SKUs. Explicit selection "
            "overrides --sku-limit and must contain at least one Train SKU unless "
            "--fit-sku-limit supplies Train fit anchors."
        ),
    )
    parser.add_argument(
        "--target-sku-file",
        type=Path,
        default=None,
        help=(
            "Optional standardized CSV/Parquet containing sku_code, for example "
            "a ROUND 8 first_order_decisions.csv. Its SKUs are combined with any "
            "repeated --sku-code values and override --sku-limit."
        ),
    )
    parser.add_argument(
        "--fit-sku-limit",
        type=_positive_int,
        default=None,
        help=(
            "When explicit targets do not contain enough Train SKUs, prepend the "
            "latest mature Train SKUs until this fit count is reached. Use the same "
            "count and --batch-skus as an earlier ROUND 4 run to reuse its checkpoints."
        ),
    )
    parser.add_argument(
        "--event-calendar",
        type=Path,
        default=None,
        help=(
            "Optional business-owned CSV: event_code,start_date,end_date and optional "
            "cate_level4_code. No event dates are hard-coded."
        ),
    )
    model_mode = parser.add_mutually_exclusive_group()
    model_mode.add_argument(
        "--restoration-model-file",
        type=Path,
        default=None,
        help=(
            "Frozen Train/OOF M2 model artifact to apply to the selected SKUs. "
            "When supplied, this run never fits v_model on those SKUs."
        ),
    )
    model_mode.add_argument(
        "--fit-model-only",
        action="store_true",
        help=(
            "Fit and persist restoration_model.json from the selected Train "
            "normal-exposure rows; do not materialize restored output rows."
        ),
    )
    parser.add_argument(
        "--model-application-scope",
        choices=("online", "oof_seed", "oof", "final_fit", "locked_test"),
        default="online",
        help=(
            "Audit scope for a frozen model. oof_seed is the point-in-time-safe "
            "B1 warm-up model; oof is used for B2-B5 validation rows. Only "
            "final_fit may transform its own historical Train rows with a model "
            "refit on all Train; OOF, locked Test and online application remain "
            "cutoff-safe."
        ),
    )
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _read_cohort(
    path: Path,
    explicit_skus: list[str],
    limit: int,
    *,
    fit_sku_limit: int | None = None,
    require_train: bool = True,
) -> pd.DataFrame:
    cohort = pd.read_csv(path, dtype={"sku_code": str})
    validate_launch_origin_contract(cohort, name="ROUND 4 cohort")
    required = {
        "dataset_version",
        "sku_code",
        "first_actual_sale_date",
        "split",
        "long_tail_weight",
    }
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise DataContractViolation(
            f"ROUND 3 cohort is missing columns: {', '.join(missing)}"
        )
    try:
        cohort["sku_code"] = normalize_identifier(
            cohort["sku_code"], field_name="cohort.sku_code"
        )
    except ValueError as exc:
        raise DataContractViolation(str(exc)) from exc
    cohort["first_actual_sale_date"] = pd.to_datetime(
        cohort["first_actual_sale_date"], errors="coerce"
    ).dt.date
    if cohort["first_actual_sale_date"].isna().any():
        raise DataContractViolation("ROUND 3 cohort has invalid first_actual_sale_date")
    if cohort["sku_code"].duplicated().any():
        # A formal prelaunch schedule contains one row per SKU x M5/M4/M3
        # checkpoint. M2 restoration is SKU-level, so collapse those rows only
        # when every restoration-relevant SKU attribute is identical.
        identity_columns = (
            "dataset_version",
            "first_actual_sale_date",
            "split",
            "long_tail_weight",
        )
        conflicts = cohort.groupby("sku_code", dropna=False)[
            list(identity_columns)
        ].nunique(dropna=False)
        bad = conflicts.gt(1).any(axis=1)
        if bad.any():
            examples = bad.loc[bad].head(5).index.astype(str).tolist()
            raise DataContractViolation(
                "ROUND 3/checkpoint cohort has conflicting duplicate SKU "
                f"attributes; examples={examples}"
            )
        cohort = cohort.drop_duplicates("sku_code", keep="first").reset_index(
            drop=True
        )
    if cohort["dataset_version"].nunique(dropna=False) != 1:
        raise DataContractViolation(
            "ROUND 3 cohort contains more than one dataset_version"
        )
    if explicit_skus:
        requested_series = pd.Series(explicit_skus, dtype="object")
        try:
            requested = list(
                dict.fromkeys(
                    normalize_identifier(
                        requested_series, field_name="--sku-code"
                    ).tolist()
                )
            )
        except ValueError as exc:
            raise DataContractViolation(str(exc)) from exc
        selected = cohort.loc[cohort["sku_code"].isin(requested)].copy()
        missing_skus = sorted(set(requested) - set(selected["sku_code"]))
        if missing_skus:
            raise DataContractViolation(
                f"requested SKUs are absent from ROUND 3 cohort: {missing_skus}"
            )
        selected["__request_order"] = selected["sku_code"].map(
            {sku: index for index, sku in enumerate(requested)}
        )
        selected = selected.sort_values("__request_order").drop(
            columns="__request_order"
        )
        if fit_sku_limit is not None:
            selected_train_count = int((selected["split"] == "Train").sum())
            additional_fit_count = max(fit_sku_limit - selected_train_count, 0)
            if additional_fit_count:
                selected_keys = set(selected["sku_code"])
                fit_anchors = (
                    cohort.loc[
                        (cohort["split"] == "Train")
                        & ~cohort["sku_code"].isin(selected_keys)
                    ]
                    .sort_values(
                        ["first_actual_sale_date", "sku_code"],
                        ascending=[False, True],
                    )
                    .head(additional_fit_count)
                    .copy()
                )
                if len(fit_anchors) != additional_fit_count:
                    raise DataContractViolation(
                        "ROUND 3 cohort has insufficient Train SKUs for "
                        f"--fit-sku-limit={fit_sku_limit}; available total after "
                        f"selection={selected_train_count + len(fit_anchors)}"
                    )
                selected = pd.concat(
                    [fit_anchors, selected], ignore_index=True
                )
    else:
        selected = (
            cohort.loc[cohort["split"] == "Train"]
            .sort_values(["first_actual_sale_date", "sku_code"], ascending=[False, True])
            .head(limit)
            .copy()
        )
    if selected.empty:
        raise DataContractViolation("no cohort SKUs were selected for ROUND 4")
    if require_train and not selected["split"].astype(str).str.lower().eq("train").any():
        split_summary = selected[["sku_code", "split"]].to_dict(orient="records")
        raise DataContractViolation(
            "ROUND 4 explicit selection must contain at least one Train SKU to fit "
            f"v_model; selected={split_summary}"
        )
    selected["long_tail_weight"] = pd.to_numeric(
        selected["long_tail_weight"], errors="coerce"
    ).fillna(1.0)
    return selected.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    checkpoint_root = args.checkpoint_dir or args.output_dir / "_checkpoints"
    config = AppConfig.load(PROJECT_ROOT, args.config)
    logger, log_path = setup_run_logger(
        "build_historical_restore", config.log_dir, config.timezone
    )
    logger.info(
        "starting ROUND 4 M2 data_root=%s cutoff=%s product_snapshot=%s cohort=%s sku_limit=%s "
        "explicit_skus=%s target_sku_file=%s fit_sku_limit=%s "
        "batch_skus=%s resume=%s",
        args.data_root,
        args.cutoff_time.isoformat(),
        (args.product_snapshot_time or args.cutoff_time).isoformat(),
        args.cohort_file,
        args.sku_limit,
        args.sku_code,
        args.target_sku_file,
        args.fit_sku_limit,
        args.batch_skus,
        not args.no_resume,
    )
    logger.info("checkpoint_root=%s (source/pre-model reuse only; model will refit unless explicitly frozen)", checkpoint_root)
    if args.dry_run:
        print("status: dry_run")
        print("stage: M2 historical restoration only")
        print(f"cutoff: {args.cutoff_time.isoformat()}")
        print(
            "product_snapshot_time: "
            f"{(args.product_snapshot_time or args.cutoff_time).isoformat()}"
        )
        print(f"limit_skus: {args.sku_limit}")
        print(f"sku_codes: {args.sku_code}")
        print(f"target_sku_file: {args.target_sku_file}")
        print(f"fit_sku_limit: {args.fit_sku_limit}")
        print(f"batch_skus: {args.batch_skus}")
        print(f"window_weeks: {args.window_weeks or config.training.label_distribution_window_weeks}")
        print(f"resume: {not args.no_resume}")
        print(f"checkpoint_dir: {checkpoint_root}")
        print(f"restoration_model_file: {args.restoration_model_file}")
        print(f"huber_max_iter: {args.huber_max_iter or config.restoration.huber_max_iter}")
        print(f"fit_model_only: {args.fit_model_only}")
        print(f"shared_source_cache_dir: {args.shared_source_cache_dir}")
        planned_outputs = [
            "round4_report.json",
            "restored_store_weekly.csv",
            "restored_tier_weekly.csv",
        ]
        if not args.skip_daily_exposure_audit:
            planned_outputs.insert(1, "daily_exposure_audit.csv")
        print("planned_outputs: " + ",".join(planned_outputs))
        print(
            "daily_exposure_audit: "
            + ("not_persisted" if args.skip_daily_exposure_audit else "persisted")
        )
        print(f"log: {log_path}")
        return 0
    try:
        for path, label in (
            (args.output_dir, "output-dir"),
            (checkpoint_root, "checkpoint-dir"),
            (args.probe_dir, "probe-dir"),
            (args.cohort_file, "cohort-file"),
        ):
            if not _inside(path, PROJECT_ROOT):
                raise ValueError(f"{label} must be inside project root {PROJECT_ROOT}")
        if args.target_sku_file is not None and not _inside(
            args.target_sku_file, PROJECT_ROOT
        ):
            raise ValueError(
                f"target-sku-file must be inside project root {PROJECT_ROOT}"
            )
        if args.shared_source_cache_dir is not None and not _inside(
            args.shared_source_cache_dir, PROJECT_ROOT
        ):
            raise ValueError(
                f"shared-source-cache-dir must be inside project root {PROJECT_ROOT}"
            )
        if _inside(args.output_dir, args.data_root):
            raise ValueError("output-dir must not be inside the read-only data root")
        if _inside(checkpoint_root, args.data_root):
            raise ValueError("checkpoint-dir must not be inside the read-only data root")
        if args.event_calendar is not None and not _inside(
            args.event_calendar, PROJECT_ROOT
        ):
            raise ValueError(
                f"event-calendar must be inside project root {PROJECT_ROOT}"
            )
        if args.restoration_model_file is not None and not _inside(
            args.restoration_model_file, PROJECT_ROOT
        ):
            raise ValueError(
                f"restoration-model-file must be inside project root {PROJECT_ROOT}"
            )
        specs = _restoration_probe_specs(
            load_table_specs(config.probe.table_specs_path)
        )
        evidence = validate_probe_evidence(
            probe_dir=args.probe_dir, table_specs=specs
        )
        if evidence["status"] != "ok":
            raise ValueError(
                "ROUND 2 probe evidence failed: " + "; ".join(evidence["errors"])
            )
        for warning in evidence["warnings"]:
            logger.warning("probe_evidence=%s", warning)

        requested_skus = list(args.sku_code)
        if args.target_sku_file is not None:
            target_frame = StageArtifactStore.read_frame(
                args.target_sku_file,
                required=("sku_code",),
                columns=("sku_code",),
            )
            requested_skus.extend(
                target_frame["sku_code"].astype(str).drop_duplicates().tolist()
            )
        selected = _read_cohort(
            args.cohort_file,
            requested_skus,
            args.sku_limit,
            fit_sku_limit=args.fit_sku_limit,
            require_train=args.restoration_model_file is None,
        )
        frozen_model = None
        if args.restoration_model_file is not None:
            frozen_model = FittedRestorationModel.from_dict(
                StageArtifactStore.read_json(args.restoration_model_file)
            )
            raw_train_end = frozen_model.report.get("train_end_time")
            if not raw_train_end:
                raise ValueError(
                    "frozen M2 model does not record train_end_time"
                )
            model_train_end = datetime.fromisoformat(
                str(raw_train_end).replace("Z", "+00:00")
            )
            if model_train_end.tzinfo is None or model_train_end.utcoffset() is None:
                raise ValueError("frozen M2 model train_end_time is not timezone-aware")
            if (
                model_train_end > args.cutoff_time
                and args.model_application_scope != "final_fit"
            ):
                raise ValueError(
                    "frozen M2 model uses future data: "
                    f"train_end_time={model_train_end.isoformat()} > "
                    f"decision_cutoff_time={args.cutoff_time.isoformat()}"
                )
        window_days = (args.window_weeks or config.training.label_distribution_window_weeks) * 7

        def report_progress(event: str, details: Mapping[str, Any]) -> None:
            rendered = " ".join(
                f"{key}={value}" for key, value in details.items()
            )
            logger.info("m2_progress event=%s %s", event, rendered)

        events = (
            None
            if args.event_calendar is None
            else pd.read_csv(args.event_calendar, dtype={"event_code": str})
        )
        logger.info(
            "m2_progress event=optimized_pipeline_start skus=%s batch_skus=%s "
            "raw_sources_materialized_together=false",
            len(selected),
            args.batch_skus,
        )
        model_started = monotonic()
        optimized, source_bundle = run_batched_restoration(
            adapter=RestorationDataAdapter.from_project(
                project_root=PROJECT_ROOT, data_root=args.data_root, config=config
            ),
            selected=selected,
            cutoff_time=args.cutoff_time,
            product_snapshot_time=args.product_snapshot_time,
            window_days=window_days,
            batch_size=args.batch_skus,
            huber_max_iter=args.huber_max_iter,
            checkpoint_dir=checkpoint_root,
            output_dir=args.output_dir,
            config=config.restoration,
            events=events,
            resume=not args.no_resume,
            write_daily_audit=not args.skip_daily_exposure_audit,
            shared_source_cache_dir=args.shared_source_cache_dir,
            write_partition_parts=not args.skip_partition_parts,
            fitted_model=frozen_model,
            fit_model_only=args.fit_model_only,
            progress=report_progress,
        )
        if args.fit_model_only:
            StageArtifactStore.write_json(
                optimized.paths["model"], optimized.fitted_model.as_dict()
            )
        logger.info(
            "m2_progress event=model_complete elapsed_seconds=%.3f daily_rows=%s "
            "weekly_rows=%s tier_weekly_rows=%s",
            monotonic() - model_started,
            optimized.stats.daily_panel_rows,
            optimized.stats.weekly_rows,
            optimized.stats.tier_weekly_rows,
        )
        warnings = list(evidence["warnings"])
        warnings.append(
            "The delivered source has no formal promotion flag; price_ratio_proxy is "
            "audit evidence only and is not represented as a confirmed campaign."
        )
        if events is None:
            warnings.append(
                "No business-owned event calendar was supplied; H(c,e)=1 and period "
                "neutralization remains explicitly neutral."
            )
        warnings.extend(optimized.quality_report["warnings"])
        for warning in optimized.quality_report["warnings"]:
            logger.warning("round4_data_quality=%s", warning)
        report_status = "warning" if warnings else "ok"
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "spec_version": "3.2",
            "round": 4,
            "module": "M2 historical operating-condition restoration",
            "status": report_status,
            "decision_cutoff_time": args.cutoff_time.isoformat(),
            "product_snapshot_time": (
                args.product_snapshot_time or args.cutoff_time
            ).isoformat(),
            "round3_dataset_version": str(selected.iloc[0]["dataset_version"]),
            "selected_sku_count": len(selected),
            "fit_train_sku_count": int(
                optimized.fitted_model.report.get("fit_sku_count", 0)
            ),
            "model_execution_mode": (
                "fit_train_only"
                if args.fit_model_only
                else "apply_frozen_model"
                if frozen_model is not None
                else "legacy_fit_and_apply"
            ),
            "restoration_model_id": optimized.fitted_model.report.get("model_id"),
            "restoration_model_train_end_time": optimized.fitted_model.report.get(
                "train_end_time"
            ),
            "model_application_scope": args.model_application_scope,
            "source_batching": {
                "batch_skus": args.batch_skus,
                "resume_enabled": not args.no_resume,
                "reused_batch_count": source_bundle.reused_batch_count,
                "computed_batch_count": source_bundle.computed_batch_count,
                "reused_pre_model_batch_count": optimized.reused_pre_model_batches,
                "computed_pre_model_batch_count": optimized.computed_pre_model_batches,
                "checkpoint_run_dir": str(source_bundle.checkpoint_run_dir),
                "pipeline_mode": "batched_two_pass_vectorized",
                "raw_sources_materialized_together": False,
                "daily_batches_materialized_together": False,
                "fit_weight_mean": optimized.fit_weight_mean,
            },
            "artifact_policy": {
                "daily_exposure_audit": (
                    "persisted"
                    if optimized.daily_audit_persisted
                    else (
                        "not_persisted; daily exposure is computed transiently and "
                        "reduced to compact per-SKU quality counters"
                    )
                ),
                "restored_store_weekly": (
                    "not_materialized_model_fit_only"
                    if args.fit_model_only
                    else "persisted_required_downstream"
                ),
                "restored_tier_weekly": (
                    "not_materialized_model_fit_only"
                    if args.fit_model_only
                    else "persisted_required_downstream"
                ),
                "resumable_batch_checkpoints": (
                    "weekly_pre_model plus compact daily quality summary; full "
                    "SKU-store-day checkpoint only with daily audit enabled"
                ),
                "shared_source_cache": (
                    str(args.shared_source_cache_dir)
                    if args.shared_source_cache_dir is not None
                    else "disabled"
                ),
                "partition_parts": (
                    "not_persisted"
                    if args.skip_partition_parts
                    else "persisted"
                ),
            },
            "selection_mode": (
                "target_sku_file"
                if args.target_sku_file is not None
                else "explicit"
                if args.sku_code
                else "latest_train_limit"
            ),
            "requested_target_sku_count": len(dict.fromkeys(requested_skus)),
            "selected_skus": source_bundle.source_windows,
            "window_days": window_days,
            "window_horizon_source": "explicit_window_weeks" if args.window_weeks else "distribution_window_default",
            "point_in_time_policy": {
                "partition_rule": "ds <= per-SKU window end <= decision cutoff date",
                "sales_event_rule": "latest order-line version is then filtered by order_time <= cutoff",
                "test_fit_rule": "Test SKUs never enter v_model fitting",
                "product_dimension_rule": (
                    "fixed audited product snapshot; static attributes only"
                    if args.product_snapshot_time is not None
                    else "latest product snapshot on or before decision cutoff"
                ),
            },
            "restoration_policy": {
                "normal_exposure": "Etotal = Eopen * Estock * Edisplay with nullable missing state",
                "source_value_normalization": "known source aliases become 1/0; unmapped non-null values remain NA and are audited",
                "stock_conflict": "contradictory inventory/placement evidence remains NA; no source silently wins",
                "stockout": "weekly R1 days and R2 share are audited; known censored days receive lost-sales restoration",
                "unreliable_stock_grain": "legacy heuristic risk flag, not proof of inventory-grain error; opening stock zero can coexist with intraday replenishment and sales. Keep the existing lower-bound/no-uplift hold pending review; do not automatically allocate parent inventory or admit labels to training",
                "unknown_exposure_sales": "observed sales are retained in business totals but unknown-exposure days do not enter n_eff or model fitting",
                "nondisplay": "in_stock_not_displayed_flag=1 is censored rather than zero demand",
                "abnormal_orders": "Train-sample category/global P99.5 marks large positive orders for anomaly-confidence/Huber downweighting; signed sales quantities are retained, not deleted or capped",
                "price": "paid/retail is modeled separately; no formal promotion flag is fabricated",
                "shrinkage": "v*=r*vobs+(1-r)*vmodel, r=neff/(neff+k)",
                "period": "H(c,e) is learned only from a supplied business event calendar; otherwise 1",
                "store_tier": "g=(store_business_state,store_level), unknown values remain explicit",
            },
            "warnings": warnings,
        }
        report["dataset_version"] = optimized.dataset_version
        report["stats"] = asdict(optimized.stats)
        report["model"] = optimized.model_report
        report["data_quality"] = optimized.quality_report
        paths = optimized.paths
        report_tmp = paths["report"].with_suffix(".json.tmp")
        report_tmp.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report_tmp.replace(paths["report"])
        logger.info(
            "ROUND 4 complete status=%s dataset_version=%s skus=%s daily=%s weekly=%s tier_weekly=%s",
            report_status,
            optimized.dataset_version,
            len(selected),
            optimized.stats.daily_panel_rows,
            optimized.stats.weekly_rows,
            optimized.stats.tier_weekly_rows,
        )
    except KeyboardInterrupt:
        logger.warning(
            "ROUND 4 interrupted; completed source batches remain resumable under %s",
            checkpoint_root,
        )
        print("INTERRUPTED: completed source batches were kept; rerun the same command to resume.")
        print(f"checkpoint_dir: {checkpoint_root}")
        print(f"log: {log_path}")
        return 130
    except Exception as exc:
        log_exception(
            logger,
            module="build_historical_restore",
            cutoff=args.cutoff_time.isoformat(),
            exc=exc,
        )
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print(f"log: {log_path}")
        return 1

    print(f"status: {report_status}")
    print(f"dataset_version: {optimized.dataset_version}")
    if args.fit_model_only:
        print(f"model: {paths['model']}")
        print(f"report: {paths['report']}")
        print(f"checkpoint_dir: {source_bundle.checkpoint_run_dir}")
        print(f"log: {log_path}")
        return 0
    print(
        "counts: "
        f"skus={len(selected)} daily={optimized.stats.daily_panel_rows} "
        f"weekly={optimized.stats.weekly_rows} tier_weekly={optimized.stats.tier_weekly_rows}"
    )
    print(
        "diagnostics: "
        f"stockout_days={optimized.stats.stockout_days} "
        f"nondisplay_days={optimized.stats.nondisplay_days} "
        f"unknown_exposure_days={optimized.stats.unknown_exposure_days} "
        f"abnormal_order_lines={optimized.stats.abnormal_order_lines}"
    )
    print(
        "quality: "
        f"status={optimized.quality_report['status']} "
        f"warnings={len(optimized.quality_report['warnings'])} "
        f"stock_conflicts={optimized.quality_report['exposure']['stock_state_conflict_days']} "
        f"fitted_train_skus={optimized.quality_report['model_gate']['fitted_train_sku_count']} "
        f"stock_grain_unreliable_skus={optimized.quality_report['exposure']['stock_grain_unreliable_sku_count']} "
        f"nonconverged_scopes={len(optimized.quality_report['model_gate']['nonconverged_scopes'])}"
    )
    print(f"report: {paths['report']}")
    print(f"weekly: {paths['weekly']}")
    print(f"tier_weekly: {paths['tier_weekly']}")
    print(
        "daily_audit: "
        + (
            str(paths["daily"])
            if optimized.daily_audit_persisted
            else "not_persisted (--skip-daily-exposure-audit)"
        )
    )
    print(f"checkpoint_dir: {source_bundle.checkpoint_run_dir}")
    print(
        "checkpoint_batches: "
        f"reused={source_bundle.reused_batch_count} "
        f"computed={source_bundle.computed_batch_count} "
        f"pre_model_reused={optimized.reused_pre_model_batches} "
        f"pre_model_computed={optimized.computed_pre_model_batches}"
    )
    print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
