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
from config.settings import AppConfig, load_table_specs  # noqa: E402
from data.domestic import DataContractViolation  # noqa: E402
from data.normalization import normalize_identifier  # noqa: E402
from data.restoration_adapter import RestorationDataAdapter  # noqa: E402
from data.validation import validate_probe_evidence  # noqa: E402
from restore.batched import run_batched_restoration  # noqa: E402


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
        "--cutoff-time", "--decision-cutoff-time", dest="cutoff_time",
        type=_parse_cutoff, required=True,
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
        "--no-resume",
        action="store_true",
        help=(
            "Ignore completed source checkpoints and reread every selected SKU. "
            "By default the same command resumes completed batches."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sku-code",
        action="append",
        default=[],
        help=(
            "Explicit ROUND 3 cohort SKU; repeat for multiple SKUs. Explicit selection "
            "overrides --sku-limit and must contain at least one Train SKU."
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
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _read_cohort(path: Path, explicit_skus: list[str], limit: int) -> pd.DataFrame:
    cohort = pd.read_csv(path, dtype={"sku_code": str})
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
        raise DataContractViolation("ROUND 3 cohort has duplicate sku_code")
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
    else:
        selected = (
            cohort.loc[cohort["split"] == "Train"]
            .sort_values(["first_actual_sale_date", "sku_code"], ascending=[False, True])
            .head(limit)
            .copy()
        )
    if selected.empty:
        raise DataContractViolation("no cohort SKUs were selected for ROUND 4")
    if not (selected["split"] == "Train").any():
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
    config = AppConfig.load(PROJECT_ROOT, args.config)
    logger, log_path = setup_run_logger(
        "build_historical_restore", config.log_dir, config.timezone
    )
    logger.info(
        "starting ROUND 4 M2 data_root=%s cutoff=%s cohort=%s sku_limit=%s "
        "explicit_skus=%s batch_skus=%s resume=%s",
        args.data_root,
        args.cutoff_time.isoformat(),
        args.cohort_file,
        args.sku_limit,
        args.sku_code,
        args.batch_skus,
        not args.no_resume,
    )
    if args.dry_run:
        print("status: dry_run")
        print("stage: M2 historical restoration only")
        print(f"cutoff: {args.cutoff_time.isoformat()}")
        print(f"limit_skus: {args.sku_limit}")
        print(f"sku_codes: {args.sku_code}")
        print(f"batch_skus: {args.batch_skus}")
        print(f"resume: {not args.no_resume}")
        print(f"checkpoint_dir: {args.output_dir / '_checkpoints'}")
        print("planned_outputs: round4_report.json,daily_exposure_audit.csv,restored_store_weekly.csv,restored_tier_weekly.csv")
        print(f"log: {log_path}")
        return 0
    try:
        for path, label in (
            (args.output_dir, "output-dir"),
            (args.probe_dir, "probe-dir"),
            (args.cohort_file, "cohort-file"),
        ):
            if not _inside(path, PROJECT_ROOT):
                raise ValueError(f"{label} must be inside project root {PROJECT_ROOT}")
        if _inside(args.output_dir, args.data_root):
            raise ValueError("output-dir must not be inside the read-only data root")
        if args.event_calendar is not None and not _inside(
            args.event_calendar, PROJECT_ROOT
        ):
            raise ValueError(
                f"event-calendar must be inside project root {PROJECT_ROOT}"
            )
        specs = load_table_specs(config.probe.table_specs_path)
        evidence = validate_probe_evidence(
            probe_dir=args.probe_dir, table_specs=specs
        )
        if evidence["status"] != "ok":
            raise ValueError(
                "ROUND 2 probe evidence failed: " + "; ".join(evidence["errors"])
            )
        for warning in evidence["warnings"]:
            logger.warning("probe_evidence=%s", warning)

        selected = _read_cohort(args.cohort_file, args.sku_code, args.sku_limit)
        window_days = config.training.label_distribution_window_weeks * 7

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
            window_days=window_days,
            batch_size=args.batch_skus,
            checkpoint_dir=args.output_dir / "_checkpoints",
            output_dir=args.output_dir,
            config=config.restoration,
            events=events,
            resume=not args.no_resume,
            progress=report_progress,
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
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "spec_version": "3.2",
            "round": 4,
            "module": "M2 historical operating-condition restoration",
            "status": "ok",
            "decision_cutoff_time": args.cutoff_time.isoformat(),
            "round3_dataset_version": str(selected.iloc[0]["dataset_version"]),
            "selected_sku_count": len(selected),
            "fit_train_sku_count": len(source_bundle.fit_skus),
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
            "selection_mode": "explicit" if args.sku_code else "latest_train_limit",
            "selected_skus": source_bundle.source_windows,
            "window_days": window_days,
            "point_in_time_policy": {
                "partition_rule": "ds <= per-SKU window end <= decision cutoff date",
                "sales_event_rule": "latest order-line version is then filtered by order_time <= cutoff",
                "test_fit_rule": "Test SKUs never enter v_model fitting",
            },
            "restoration_policy": {
                "normal_exposure": "Etotal = Eopen * Estock * Edisplay with nullable missing state",
                "source_value_normalization": "known source aliases become 1/0; unmapped non-null values remain NA and are audited",
                "stock_conflict": "contradictory inventory/placement evidence remains NA; no source silently wins",
                "stockout": "weekly R1 days and R2 share are audited; known censored days receive lost-sales restoration",
                "nondisplay": "in_stock_not_displayed_flag=1 is censored rather than zero demand",
                "abnormal_orders": "positive order lines strictly above Train-sample category/global P99.5 are excluded and retained in audit columns",
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
            "ROUND 4 complete status=ok dataset_version=%s skus=%s daily=%s weekly=%s tier_weekly=%s",
            optimized.dataset_version,
            len(selected),
            optimized.stats.daily_panel_rows,
            optimized.stats.weekly_rows,
            optimized.stats.tier_weekly_rows,
        )
    except KeyboardInterrupt:
        logger.warning(
            "ROUND 4 interrupted; completed source batches remain resumable under %s",
            args.output_dir / "_checkpoints",
        )
        print("INTERRUPTED: completed source batches were kept; rerun the same command to resume.")
        print(f"checkpoint_dir: {args.output_dir / '_checkpoints'}")
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

    print("status: ok")
    print(f"dataset_version: {optimized.dataset_version}")
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
        f"fitted_train_skus={optimized.quality_report['model_gate']['fitted_train_sku_count']}"
    )
    print(f"report: {paths['report']}")
    print(f"weekly: {paths['weekly']}")
    print(f"tier_weekly: {paths['tier_weekly']}")
    print(f"daily_audit: {paths['daily']}")
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
