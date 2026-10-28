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
from data.domain import filter_skus, load_algorithm_parameters  # noqa: E402
from data.prelaunch_artifacts import merge_row_audit_schedule  # noqa: E402
from data.prelaunch_row_audit import (  # noqa: E402
    PrelaunchRowAuditAdapter,
    select_partition_replay_candidates,
)
from data.social_ip_aliases import load_social_ip_alias_mapping  # noqa: E402
from data.standard_ip import load_standard_ip_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit real row-grain source availability at independent M5/M4/M3 "
            "first-order forecast checkpoints without training a model."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--schedule-file",
        type=Path,
        default=Path(
            "artifacts/prelaunch_decision_time_audit/"
            "prelaunch_decision_schedule.csv"
        ),
    )
    parser.add_argument(
        "--resolved-standard-ip-file",
        type=Path,
        default=Path(
            "artifacts/standard_ip_mapping_v2/cohort_sku_ip_mapping.csv"
        ),
    )
    parser.add_argument(
        "--social-ip-alias-file",
        type=Path,
        default=Path("config/social_ip_aliases.csv"),
    )
    parser.add_argument(
        "--social-window-days",
        type=positive_int,
        default=None,
        help=(
            "Recent social history ending at each prediction checkpoint. "
            "Defaults to V3.2 comparable.external_media.prelaunch_days."
        ),
    )
    parser.add_argument("--batch-size", type=positive_int, default=50_000)
    add_execution_arguments(
        parser,
        default_output="artifacts/prelaunch_row_coverage_audit",
    )
    return parser.parse_args()


def _lead_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
    summary = (
        frame.groupby("lead_months", as_index=False)
        .agg(
            prediction_checkpoints=("prediction_checkpoint_key", "size"),
            skus=("sku_code", "nunique"),
            checkpoints_core_row_ready=(
                "core_row_ready_for_feature_replay",
                "sum",
            ),
            checkpoints_with_predecision_target_history=(
                "predecision_target_history_rows",
                lambda value: int((value > 0).sum()),
            ),
        )
        .sort_values("lead_months")
    )
    summary["core_row_ready_rate"] = (
        summary["checkpoints_core_row_ready"]
        / summary["prediction_checkpoints"]
    )
    return summary.to_dict(orient="records")


def main() -> int:
    args = parse_args()
    outputs = {
        "partition_candidate_skus": "partition_replay_candidate_skus.csv",
        "partition_candidate_schedule": "partition_replay_candidate_schedule.csv",
        "checkpoint_coverage": "checkpoint_row_coverage.csv",
        "source_detail": "source_row_detail.csv",
        "source_summary": "source_row_summary.csv",
        "common_core_skus": "common_core_row_skus.csv",
        "formal_eligible_skus": "formal_eligible_prelaunch_skus.csv",
        "formal_eligible_schedule": "formal_eligible_prelaunch_schedule.csv",
        "report": "prelaunch_row_audit_report.json",
    }
    if args.dry_run:
        payload = dry_run_payload(
            "prelaunch row-level coverage audit", args, list(outputs.values())
        )
        payload.update(
            {
                "schedule_file": str(args.schedule_file),
                "resolved_standard_ip_file": str(args.resolved_standard_ip_file),
                "social_ip_alias_file": str(args.social_ip_alias_file),
                "training_started": False,
                "test_consumed": False,
            }
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger(
        "audit_prelaunch_row_coverage", config.log_dir, config.timezone
    )
    try:
        parameters = load_algorithm_parameters(PROJECT_ROOT / args.algorithm_config)
        social_window_days = int(
            args.social_window_days
            or parameters["comparable"]["external_media"]["prelaunch_days"]
        )
        artifact = StageArtifactStore()
        complete_schedule = artifact.read_frame(
            PROJECT_ROOT / args.schedule_file,
            required=(
                "prediction_checkpoint_key",
                "sku_code",
                "release_dt",
                "first_actual_sale_date",
                "lead_months",
                "decision_date",
                "decision_cutoff_time",
                "product_snapshot_partition",
                "product_snapshot_status",
                "all_audited_sources_available",
                "within_available_through",
                "sales_26w_qty",
            ),
        )
        expected_leads = tuple(
            sorted(
                pd.to_numeric(
                    complete_schedule["lead_months"], errors="raise"
                ).astype(int).unique()
            )
        )
        partition_candidates = select_partition_replay_candidates(
            complete_schedule,
            expected_lead_months=expected_leads,
            label_as_of_date=args.decision_cutoff_time.astimezone(
                ZoneInfo(config.timezone)
            ).date(),
        )
        candidate_schedule = partition_candidates.schedule
        if candidate_schedule.empty:
            raise ValueError(
                "no SKU passes the partition-level M5/M4/M3, temporal-order, "
                "and mature-label eligibility gates"
            )
        schedule = candidate_schedule.copy()
        if args.sku_code or args.limit_skus is not None:
            schedule = filter_skus(
                candidate_schedule,
                sku_codes=args.sku_code,
                limit_skus=args.limit_skus,
            )
        if schedule.empty:
            raise ValueError("selected row-audit schedule is empty")
        schedule_cutoffs = pd.to_datetime(
            schedule["decision_cutoff_time"], errors="raise", utc=True
        )
        global_ceiling = pd.Timestamp(args.decision_cutoff_time).tz_convert("UTC")
        if schedule_cutoffs.gt(global_ceiling).any():
            raise ValueError(
                "schedule contains a prediction checkpoint after "
                "--decision-cutoff-time"
            )

        selected_skus = int(schedule["sku_code"].astype(str).nunique())
        complete_skus = int(complete_schedule["sku_code"].astype(str).nunique())
        candidate_skus = int(
            candidate_schedule["sku_code"].astype(str).nunique()
        )
        full_candidate_universe_audited = selected_skus == candidate_skus
        logger.info(
            "prelaunch_row_audit_progress event=audit_start selected_skus=%s "
            "complete_skus=%s partition_candidate_skus=%s checkpoints=%s "
            "full_candidate_universe=%s",
            selected_skus,
            complete_skus,
            candidate_skus,
            len(schedule),
            full_candidate_universe_audited,
        )
        standard_ip_mapping = load_standard_ip_mapping(
            PROJECT_ROOT / args.resolved_standard_ip_file
        )
        social_alias_mapping = load_social_ip_alias_mapping(
            PROJECT_ROOT / args.social_ip_alias_file
        )
        adapter = PrelaunchRowAuditAdapter.from_project(
            project_root=PROJECT_ROOT,
            data_root=args.data_root,
            config=config,
            batch_size=args.batch_size,
        )
        audit = adapter.audit(
            schedule,
            standard_ip_mapping=standard_ip_mapping,
            social_ip_alias_mapping=social_alias_mapping,
            social_window_days=social_window_days,
            progress=lambda event, details: logger.info(
                "prelaunch_row_audit_progress event=%s %s",
                event,
                " ".join(f"{key}={value}" for key, value in details.items()),
            ),
            emit_formal_eligibility=full_candidate_universe_audited,
        )

        output_dir = PROJECT_ROOT / args.output_dir
        artifact.write_frame(
            output_dir / outputs["partition_candidate_skus"],
            partition_candidates.sku_audit,
        )
        artifact.write_frame(
            output_dir / outputs["partition_candidate_schedule"],
            candidate_schedule,
        )
        artifact.write_frame(
            output_dir / outputs["checkpoint_coverage"],
            audit.checkpoint_summary,
        )
        artifact.write_frame(
            output_dir / outputs["source_detail"], audit.source_detail
        )
        artifact.write_frame(
            output_dir / outputs["source_summary"], audit.source_summary
        )
        artifact.write_frame(
            output_dir / outputs["common_core_skus"], audit.common_core_skus
        )

        formal_schedule = pd.DataFrame()
        if full_candidate_universe_audited:
            artifact.write_frame(
                output_dir / outputs["formal_eligible_skus"],
                audit.formal_eligible_skus,
            )
            if not audit.formal_eligible_skus.empty:
                split_mapping = audit.formal_eligible_skus[
                    [
                        "sku_code",
                        "split",
                        "temporal_split_boundary_date",
                        "split_rule",
                        "eligibility_scope",
                        "training_ready",
                        "requires_full_feature_replay_before_training",
                        "requires_train_only_weight_refit",
                    ]
                ]
                formal_schedule = merge_row_audit_schedule(candidate_schedule, split_mapping)
            artifact.write_frame(
                output_dir / outputs["formal_eligible_schedule"],
                formal_schedule,
            )

        common_ready = int(
            audit.common_core_skus["common_core_row_ready"].fillna(False).sum()
        )
        source_errors = int(
            audit.source_detail["source_error"].fillna("").ne("").sum()
        )
        formal_split_counts = (
            {
                str(key): int(value)
                for key, value in audit.formal_eligible_skus.groupby(
                    "split"
                )["sku_code"].nunique().items()
            }
            if full_candidate_universe_audited
            and not audit.formal_eligible_skus.empty
            else {}
        )
        reason_series = partition_candidates.sku_audit[
            "partition_ineligibility_reasons"
        ].fillna("").str.split("|")
        exclusion_reasons = {
            reason: int(reason_series.map(lambda values: reason in values).sum())
            for reason in (
                "incomplete_M5_M4_M3_checkpoint_set",
                "one_or_more_source_partitions_unavailable",
                "decision_not_strictly_before_first_sale",
                "checkpoint_after_global_cutoff",
                "sales_26w_label_missing",
                "sales_26w_label_not_mature",
            )
        }
        report = {
            "spec_version": "3.2",
            "stage": "prelaunch multi-checkpoint row-level coverage audit",
            "status": (
                "core_row_audit_complete_requires_historical_feature_replay"
                if source_errors == 0
                else "row_audit_completed_with_source_errors"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "social_window_days": social_window_days,
            "selected_skus": selected_skus,
            "complete_schedule_skus": complete_skus,
            "partition_replay_candidate_skus": candidate_skus,
            "partition_ineligible_skus": complete_skus - candidate_skus,
            "partition_ineligibility_reason_counts": exclusion_reasons,
            "prediction_checkpoints": len(audit.checkpoint_summary),
            "full_candidate_universe_audited": full_candidate_universe_audited,
            "common_core_row_ready_skus": common_ready,
            "formal_replay_eligible_skus": int(
                len(audit.formal_eligible_skus)
                if full_candidate_universe_audited
                else 0
            ),
            "source_error_rows": source_errors,
            "lead_month_summary": _lead_summary(audit.checkpoint_summary),
            "row_grain_policy": {
                "product": (
                    "required target-SKU attributes and release_dt from the exact "
                    "fixed product snapshot recorded in the schedule"
                ),
                "store": "required nonempty global store snapshot at or before checkpoint",
                "packaging": "goods_unit/pkg_unit/box_unit are read from the fixed product-master snapshot; the duplicate pack table is not a replay dependency",
                "supplier_member": "optional target-SKU diagnostic values; missing remains missing/neutral, never zero-filled",
                "sales_inventory_placement": "diagnostic target-SKU history before checkpoint; zero rows are expected for a genuine prelaunch target and do not cause rejection",
                "social": "optional reviewed standard-IP/alias rows in the recent point-in-time window",
            },
            "formal_replay_split_emitted": full_candidate_universe_audited,
            "formal_replay_split_counts": formal_split_counts,
            "formal_replay_scope": (
                "Formal eligibility and temporal split for historical feature "
                "replay. Training remains blocked until Path 0-4/comparable, "
                "restore, placement and factor features are replayed at every "
                "checkpoint and Train-only weights are refitted."
            ),
            "training_started": False,
            "test_consumed": False,
            "outputs": {
                "partition_candidate_skus": outputs[
                    "partition_candidate_skus"
                ],
                "partition_candidate_schedule": outputs[
                    "partition_candidate_schedule"
                ],
                "checkpoint_coverage": outputs["checkpoint_coverage"],
                "source_detail": outputs["source_detail"],
                "source_summary": outputs["source_summary"],
                "common_core_skus": outputs["common_core_skus"],
                "formal_eligible_skus": (
                    outputs["formal_eligible_skus"]
                    if full_candidate_universe_audited
                    else None
                ),
                "formal_eligible_schedule": (
                    outputs["formal_eligible_schedule"]
                    if full_candidate_universe_audited
                    else None
                ),
            },
        }
        artifact.write_json(output_dir / outputs["report"], report)
        logger.info(
            "prelaunch row-level audit complete selected_skus=%s checkpoints=%s "
            "common_core_ready=%s source_error_rows=%s",
            selected_skus,
            len(audit.checkpoint_summary),
            common_ready,
            source_errors,
        )
    except Exception as exc:
        log_exception(
            logger,
            module="audit_prelaunch_row_coverage",
            cutoff=args.decision_cutoff_time.isoformat(),
            exc=exc,
        )
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1

    print(
        f"status: {report['status']}\n"
        f"partition candidates: "
        f"{output_dir / outputs['partition_candidate_skus']}\n"
        f"checkpoint coverage: {output_dir / outputs['checkpoint_coverage']}\n"
        f"source summary: {output_dir / outputs['source_summary']}\n"
        f"common core SKUs: {output_dir / outputs['common_core_skus']}\n"
        f"formal eligible SKUs: "
        f"{output_dir / outputs['formal_eligible_skus'] if full_candidate_universe_audited else 'not emitted for partial audit'}\n"
        f"formal eligible schedule: "
        f"{output_dir / outputs['formal_eligible_schedule'] if full_candidate_universe_audited else 'not emitted for partial audit'}\n"
        f"report: {output_dir / outputs['report']}\n"
        f"log: {log_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
