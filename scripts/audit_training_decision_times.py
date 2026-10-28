#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import add_execution_arguments, positive_int, print_dry_run  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.decision_adapter import (  # noqa: E402
    DecisionDataAdapter,
    SOURCES,
    select_target_skus,
)


DATE_COLUMNS = ("changed_date", "approval_date", "confirm_date", "arrival_date")
TYPE_COLUMNS = (
    "demand_type",
    "demand_type_desc",
    "order_type",
    "order_type_desc",
    "document_status",
    "approval_status_desc",
    "deletion_flag",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Table 18 fields needed to identify each historical SKU's "
            "real first-order decision time before V3.2 training replay."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--cohort-file",
        type=Path,
        default=Path("artifacts/round3_dataset/cohort.csv"),
    )
    parser.add_argument("--batch-size", type=positive_int, default=50_000)
    add_execution_arguments(
        parser,
        default_output="artifacts/training_decision_time_audit",
    )
    return parser.parse_args()


def _joined_tokens(values: pd.Series) -> str:
    tokens = sorted(
        {
            str(value).strip()
            for value in values
            if pd.notna(value) and str(value).strip()
        }
    )
    return "|".join(tokens)


def _iso_or_blank(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()


def _timeline(cohort: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    base = cohort[
        [
            column
            for column in (
                "sku_code",
                "first_actual_sale_date",
                "split",
                "time_block",
                "dataset_version",
            )
            if column in cohort
        ]
    ].copy()
    base["sku_code"] = base["sku_code"].astype(str)
    base["first_actual_sale_date"] = pd.to_datetime(
        base["first_actual_sale_date"], errors="raise"
    )
    if history.empty:
        for column in (
            "purchase_line_count",
            "purchase_order_count",
            *[f"earliest_{value}" for value in DATE_COLUMNS],
            *TYPE_COLUMNS,
        ):
            base[column] = 0 if column.endswith("_count") else ""
        base["purchase_history_status"] = "no_purchase_rows"
        return base

    grouped = history.groupby("sku_code", sort=False)
    counts = grouped.agg(
        purchase_line_count=("purchase_order_line_no", "size"),
        purchase_order_count=("purchase_order_no", "nunique"),
    )
    for column in DATE_COLUMNS:
        counts[f"earliest_{column}"] = grouped[column].min()
    for column in TYPE_COLUMNS:
        counts[column] = grouped[column].agg(_joined_tokens)
    result = base.merge(counts.reset_index(), on="sku_code", how="left")
    result["purchase_line_count"] = result["purchase_line_count"].fillna(0).astype(int)
    result["purchase_order_count"] = result["purchase_order_count"].fillna(0).astype(int)
    for column in TYPE_COLUMNS:
        result[column] = result[column].fillna("")
    for column in DATE_COLUMNS:
        date_column = f"earliest_{column}"
        result[f"{column}_days_before_first_sale"] = (
            result["first_actual_sale_date"] - pd.to_datetime(
                result[date_column], errors="coerce"
            )
        ).dt.total_seconds() / 86_400.0
        result[date_column] = result[date_column].map(_iso_or_blank)
    result["purchase_history_status"] = np.where(
        result["purchase_line_count"].gt(0),
        "purchase_rows_found_needs_first_order_rule",
        "no_purchase_rows",
    )
    result["first_actual_sale_date"] = result["first_actual_sale_date"].dt.date.astype(str)
    return result


def _type_summary(cohort: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *TYPE_COLUMNS,
        "purchase_line_count",
        "purchase_order_count",
        "sku_count",
        *[f"{value}_nonnull_rows" for value in DATE_COLUMNS],
        *[f"{value}_before_or_on_first_sale_rows" for value in DATE_COLUMNS],
        "target_qty_sum",
        "received_qty_sum",
        "first_arrival_partition",
        "last_arrival_partition",
    ]
    if history.empty:
        return pd.DataFrame(columns=columns)
    first_sale = cohort[["sku_code", "first_actual_sale_date"]].copy()
    first_sale["sku_code"] = first_sale["sku_code"].astype(str)
    first_sale["first_actual_sale_date"] = pd.to_datetime(
        first_sale["first_actual_sale_date"], errors="raise"
    )
    frame = history.merge(first_sale, on="sku_code", how="left", validate="many_to_one")
    for column in TYPE_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    for column in DATE_COLUMNS:
        frame[f"__{column}_nonnull"] = frame[column].notna().astype(int)
        frame[f"__{column}_before"] = (
            frame[column].notna()
            & frame["first_actual_sale_date"].notna()
            & frame[column].le(frame["first_actual_sale_date"])
        ).astype(int)
    summary = (
        frame.groupby(list(TYPE_COLUMNS), dropna=False, as_index=False)
        .agg(
            purchase_line_count=("purchase_order_line_no", "size"),
            purchase_order_count=("purchase_order_no", "nunique"),
            sku_count=("sku_code", "nunique"),
            changed_date_nonnull_rows=("__changed_date_nonnull", "sum"),
            approval_date_nonnull_rows=("__approval_date_nonnull", "sum"),
            confirm_date_nonnull_rows=("__confirm_date_nonnull", "sum"),
            arrival_date_nonnull_rows=("__arrival_date_nonnull", "sum"),
            changed_date_before_or_on_first_sale_rows=("__changed_date_before", "sum"),
            approval_date_before_or_on_first_sale_rows=("__approval_date_before", "sum"),
            confirm_date_before_or_on_first_sale_rows=("__confirm_date_before", "sum"),
            arrival_date_before_or_on_first_sale_rows=("__arrival_date_before", "sum"),
            target_qty_sum=("target_qty", "sum"),
            received_qty_sum=("received_qty", "sum"),
            first_arrival_partition=("ds", "min"),
            last_arrival_partition=("ds", "max"),
        )
        .sort_values(
            ["sku_count", "purchase_line_count", "demand_type_desc"],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    for column in ("first_arrival_partition", "last_arrival_partition"):
        summary[column] = summary[column].map(
            lambda value: "" if pd.isna(value) else value.isoformat()
        )
    return summary[columns]


def main() -> int:
    args = parse_args()
    outputs = [
        "purchase_demand_type_summary.csv",
        "sku_purchase_timeline.csv",
        "training_decision_time_audit_report.json",
    ]
    if args.dry_run:
        print_dry_run("training decision-time audit", args, outputs)
        return 0

    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger(
        "audit_training_decision_times", config.log_dir, config.timezone
    )
    try:
        artifact = StageArtifactStore()
        cohort = artifact.read_frame(
            PROJECT_ROOT / args.cohort_file,
            required=("sku_code", "first_actual_sale_date", "split"),
        )
        cohort = select_target_skus(
            cohort,
            sku_codes=args.sku_code,
            limit_skus=args.limit_skus,
        )
        target_skus = cohort["sku_code"].astype(str).drop_duplicates().tolist()
        adapter = DecisionDataAdapter.from_project(
            project_root=PROJECT_ROOT,
            data_root=args.data_root,
            config=config,
        )
        purchase = adapter.purchase_order_history(
            args.decision_cutoff_time,
            sku_codes=target_skus,
            batch_size=args.batch_size,
            progress=lambda event, details: logger.info(
                "training_decision_time_progress event=%s %s",
                event,
                " ".join(f"{key}={value}" for key, value in details.items()),
            ),
        )
        timeline = _timeline(cohort, purchase.history)
        type_summary = _type_summary(cohort, purchase.history)
        output_dir = PROJECT_ROOT / args.output_dir
        artifact.write_frame(output_dir / outputs[0], type_summary)
        artifact.write_frame(output_dir / outputs[1], timeline)

        sku_with_rows = int(timeline["purchase_line_count"].gt(0).sum())
        date_coverage: dict[str, object] = {}
        for column in DATE_COLUMNS:
            earliest = pd.to_datetime(
                timeline[f"earliest_{column}"], errors="coerce"
            )
            delta = pd.to_numeric(
                timeline[f"{column}_days_before_first_sale"], errors="coerce"
            )
            date_coverage[column] = {
                "sku_count": int(earliest.notna().sum()),
                "coverage_rate": float(earliest.notna().mean()) if len(timeline) else 0.0,
                "on_or_before_first_sale_skus": int(delta.ge(0).sum()),
                "after_first_sale_skus": int(delta.lt(0).sum()),
                "median_days_before_first_sale": (
                    None if delta.dropna().empty else float(delta.median())
                ),
            }
        report = {
            "spec_version": "3.2",
            "stage": "historical first-order decision-time audit",
            "status": "requires_business_confirmation_before_training_replay",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "source_table": SOURCES["purchase"],
            "source_partition_semantics": "arrival_date_partition_not_order_date",
            "source_partitions": list(purchase.source_partitions),
            "cohort_skus": len(timeline),
            "split_sku_counts": {
                str(key): int(value)
                for key, value in timeline.groupby("split")["sku_code"].nunique().items()
            },
            "skus_with_purchase_rows": sku_with_rows,
            "skus_without_purchase_rows": int(len(timeline) - sku_with_rows),
            "purchase_line_rows": len(purchase.history),
            "demand_type_combinations": len(type_summary),
            "date_field_coverage": date_coverage,
            "required_confirmations": [
                "exact demand_type/demand_type_desc values that mean an initial order rather than a reorder",
                "whether approval_date or confirm_date is the business decision timestamp t_j",
                "which deletion_flag, document_status and approval_status_desc values are valid executable orders",
            ],
            "training_started": False,
            "test_consumed": False,
            "outputs": {
                "type_summary": outputs[0],
                "sku_timeline": outputs[1],
            },
        }
        artifact.write_json(output_dir / outputs[2], report)
        logger.info(
            "training decision-time audit complete cohort_skus=%s purchase_skus=%s rows=%s",
            len(timeline),
            sku_with_rows,
            len(purchase.history),
        )
    except Exception as exc:
        log_exception(
            logger,
            module="audit_training_decision_times",
            cutoff=args.decision_cutoff_time,
            exc=exc,
        )
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1

    print(
        "status: requires_business_confirmation_before_training_replay\n"
        f"type summary: {output_dir / outputs[0]}\n"
        f"sku timeline: {output_dir / outputs[1]}\n"
        f"report: {output_dir / outputs[2]}\n"
        f"log: {log_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
