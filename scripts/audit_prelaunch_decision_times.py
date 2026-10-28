#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import (  # noqa: E402
    add_execution_arguments,
    dry_run_payload,
    positive_int,
)
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.decision_adapter import DecisionDataAdapter  # noqa: E402
from data.prelaunch_training_audit import (  # noqa: E402
    DEFAULT_AUDIT_SOURCE_KEYS,
    PrelaunchTrainingAuditAdapter,
)
from training.decision_schedule import (  # noqa: E402
    DEFAULT_PRELAUNCH_LEAD_MONTHS,
    build_prelaunch_decision_schedule,
    normalize_lead_months,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V3.2 historical feature cutoffs at configurable calendar-month "
            "prelaunch forecast checkpoints without starting training."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--cohort-file",
        type=Path,
        default=Path("artifacts/round3_dataset/cohort.csv"),
    )
    parser.add_argument(
        "--lead-months",
        action="append",
        type=positive_int,
        default=[],
        help=(
            "Calendar months before release_dt. Repeat the option; "
            "defaults to the confirmed M3-only scope (3)."
        ),
    )
    parser.add_argument(
        "--source-key",
        action="append",
        choices=DEFAULT_AUDIT_SOURCE_KEYS,
        default=[],
        help="Source to audit. Repeat the option; defaults to all model sources.",
    )
    add_execution_arguments(
        parser,
        default_output="artifacts/prelaunch_decision_time_audit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    leads = normalize_lead_months(
        args.lead_months or DEFAULT_PRELAUNCH_LEAD_MONTHS
    )
    source_keys = tuple(args.source_key or DEFAULT_AUDIT_SOURCE_KEYS)
    outputs = [
        "prelaunch_decision_schedule.csv",
        "product_release_date_audit.csv",
        "source_partition_detail.csv",
        "source_partition_summary.csv",
        "prelaunch_decision_audit_report.json",
    ]
    if args.dry_run:
        payload = dry_run_payload("prelaunch decision-time audit", args, outputs)
        payload.update(
            {
                "lead_months": list(leads),
                "source_keys": list(source_keys),
                "training_started": False,
                "test_consumed": False,
            }
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger(
        "audit_prelaunch_decision_times", config.log_dir, config.timezone
    )
    try:
        artifact = StageArtifactStore()
        cohort = artifact.read_frame(
            PROJECT_ROOT / args.cohort_file,
            required=("sku_code", "first_actual_sale_date", "split"),
            sku_codes=args.sku_code,
            limit_skus=args.limit_skus,
        )
        target_skus = cohort["sku_code"].astype(str).drop_duplicates().tolist()
        product_adapter = DecisionDataAdapter.from_project(
            project_root=PROJECT_ROOT,
            data_root=args.data_root,
            config=config,
        )
        logger.info(
            "prelaunch_decision_audit_progress event=product_snapshot_read_start "
            "cutoff=%s sku_count=%s",
            args.decision_cutoff_time.isoformat(),
            len(target_skus),
        )
        product_domain = product_adapter.products(
            args.decision_cutoff_time,
            sku_codes=target_skus,
            allow_missing=True,
            fallback_to_historical=False,
            progress=lambda event, details: logger.info(
                "prelaunch_decision_audit_progress event=%s %s",
                event,
                " ".join(f"{key}={value}" for key, value in details.items()),
            ),
        )
        expected_product_partition = args.decision_cutoff_time.astimezone(
            ZoneInfo(config.timezone)
        ).date().isoformat()
        if product_domain.source_partition != expected_product_partition:
            raise ValueError(
                "product snapshot is not the exact requested cutoff partition: "
                f"expected={expected_product_partition} "
                f"actual={product_domain.source_partition}"
            )

        release_rows = product_domain.products[
            ["sku_code", "release_dt"]
        ].copy()
        release_audit = cohort[
            ["sku_code", "first_actual_sale_date", "split"]
        ].merge(
            product_domain.snapshot_audit,
            on="sku_code",
            how="left",
            validate="one_to_one",
        ).merge(
            release_rows,
            on="sku_code",
            how="left",
            validate="one_to_one",
        )
        release_audit["product_snapshot_partition"] = (
            product_domain.source_partition
        )
        parsed_release = pd.to_datetime(
            release_audit["release_dt"], errors="coerce", format="mixed"
        )
        parsed_first_sale = pd.to_datetime(
            release_audit["first_actual_sale_date"],
            errors="coerce",
            format="mixed",
        )
        snapshot_available = release_audit["product_snapshot_status"].eq(
            "latest_partition"
        )
        release_audit["release_dt_audit_status"] = "valid"
        release_audit.loc[
            ~snapshot_available, "release_dt_audit_status"
        ] = "sku_missing_in_fixed_product_snapshot"
        release_audit.loc[
            snapshot_available & parsed_release.isna(),
            "release_dt_audit_status",
        ] = "release_dt_missing_or_invalid"
        release_audit["first_sale_minus_release_days"] = (
            parsed_first_sale - parsed_release
        ).dt.days
        release_audit["release_dt"] = parsed_release.dt.date.astype("string")
        release_audit.loc[parsed_release.isna(), "release_dt"] = pd.NA

        eligible_release = release_audit.loc[
            release_audit["release_dt_audit_status"].eq("valid"),
            [
                "sku_code",
                "release_dt",
                "product_snapshot_partition",
                "product_snapshot_status",
            ],
        ]
        cohort_with_release = cohort.merge(
            eligible_release,
            on="sku_code",
            how="inner",
            validate="one_to_one",
        )
        if cohort_with_release.empty:
            raise ValueError(
                "no cohort SKU has a valid release_dt in the exact "
                f"{expected_product_partition} product snapshot"
            )
        logger.info(
            "prelaunch_decision_audit_progress event=product_snapshot_read_complete "
            "partition=%s cohort_skus=%s eligible_release_skus=%s",
            product_domain.source_partition,
            len(target_skus),
            cohort_with_release["sku_code"].nunique(),
        )
        logger.info(
            "prelaunch_decision_audit_progress event=schedule_build_start "
            "cohort_skus=%s lead_months=%s",
            cohort_with_release["sku_code"].astype(str).nunique(),
            ",".join(str(value) for value in leads),
        )
        schedule = build_prelaunch_decision_schedule(
            cohort_with_release,
            lead_months=leads,
            timezone=config.timezone,
            available_through=args.decision_cutoff_time,
        )
        logger.info(
            "prelaunch_decision_audit_progress event=schedule_build_complete nodes=%s",
            len(schedule),
        )
        adapter = PrelaunchTrainingAuditAdapter.from_project(
            project_root=PROJECT_ROOT,
            data_root=args.data_root,
            config=config,
        )
        logger.info(
            "prelaunch_decision_audit_progress event=partition_audit_start "
            "sources=%s",
            ",".join(source_keys),
        )
        audit = adapter.audit(schedule, source_keys=source_keys)
        logger.info(
            "prelaunch_decision_audit_progress event=partition_audit_complete "
            "detail_rows=%s",
            len(audit.source_detail),
        )

        output_dir = PROJECT_ROOT / args.output_dir
        artifact.write_frame(output_dir / outputs[0], audit.schedule)
        artifact.write_frame(output_dir / outputs[1], release_audit)
        artifact.write_frame(output_dir / outputs[2], audit.source_detail)
        artifact.write_frame(output_dir / outputs[3], audit.source_summary)

        checkpoint_ready = audit.schedule[
            "all_audited_sources_available"
        ].fillna(False)
        lead_summary = (
            audit.schedule.groupby("lead_months", as_index=False)
            .agg(
                prediction_checkpoints=("prediction_checkpoint_key", "size"),
                skus=("sku_code", "nunique"),
                checkpoints_with_all_audited_sources=(
                    "all_audited_sources_available",
                    "sum",
                ),
                earliest_decision_date=("decision_date", "min"),
                latest_decision_date=("decision_date", "max"),
            )
        )
        lead_summary["all_source_partition_coverage_rate"] = (
            lead_summary["checkpoints_with_all_audited_sources"]
            / lead_summary["prediction_checkpoints"]
        )
        report = {
            "spec_version": "3.2",
            "stage": "prelaunch multi-checkpoint first-order forecast audit",
            "status": "partition_audit_complete_requires_row_level_replay",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "available_through": args.decision_cutoff_time.isoformat(),
            "product_snapshot_partition": product_domain.source_partition,
            "product_snapshot_policy": (
                f"exact {product_domain.source_partition} partition only; no "
                "later partition and no per-SKU historical fallback"
            ),
            "decision_rule": (
                "end of the calendar date exactly N calendar months before "
                "release_dt from the fixed product snapshot"
            ),
            "lead_months": list(leads),
            "input_cohort_skus": int(cohort["sku_code"].nunique()),
            "cohort_skus": int(audit.schedule["sku_code"].nunique()),
            "skus_missing_from_fixed_product_snapshot": int(
                release_audit["release_dt_audit_status"].eq(
                    "sku_missing_in_fixed_product_snapshot"
                ).sum()
            ),
            "skus_missing_or_invalid_release_dt": int(
                release_audit["release_dt_audit_status"].eq(
                    "release_dt_missing_or_invalid"
                ).sum()
            ),
            "prediction_checkpoints": len(audit.schedule),
            "prediction_checkpoints_per_sku": len(leads),
            "forecast_scope_per_checkpoint": "complete_first_order_demand",
            "checkpoint_weight_policy": (
                "do not apply the V3.2 reorder-node w_i/m_i rule; report and "
                "evaluate M5, M4 and M3 first-order forecasts separately"
            ),
            "split_sku_counts": {
                str(key): int(value)
                for key, value in cohort_with_release.groupby("split")[
                    "sku_code"
                ].nunique().items()
            },
            "checkpoints_with_all_audited_sources": int(checkpoint_ready.sum()),
            "checkpoints_missing_one_or_more_sources": int(
                (~checkpoint_ready).sum()
            ),
            "lead_month_summary": lead_summary.to_dict(orient="records"),
            "audited_source_keys": list(source_keys),
            "audit_scope": (
                "partition metadata only; this does not prove row-level SKU/IP "
                "coverage or build model features"
            ),
            "training_started": False,
            "test_consumed": False,
            "outputs": {
                "decision_schedule": outputs[0],
                "product_release_date_audit": outputs[1],
                "source_partition_detail": outputs[2],
                "source_partition_summary": outputs[3],
            },
        }
        artifact.write_json(output_dir / outputs[4], report)
        logger.info(
            "prelaunch decision-time audit complete cohort_skus=%s "
            "checkpoints=%s checkpoints_with_all_sources=%s",
            report["cohort_skus"],
            report["prediction_checkpoints"],
            report["checkpoints_with_all_audited_sources"],
        )
    except Exception as exc:
        log_exception(
            logger,
            module="audit_prelaunch_decision_times",
            cutoff=args.decision_cutoff_time.isoformat(),
            exc=exc,
        )
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1

    print(
        "status: partition_audit_complete_requires_row_level_replay\n"
        f"schedule: {output_dir / outputs[0]}\n"
        f"product release audit: {output_dir / outputs[1]}\n"
        f"source summary: {output_dir / outputs[3]}\n"
        f"report: {output_dir / outputs[4]}\n"
        f"log: {log_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
