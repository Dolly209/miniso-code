#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import add_execution_arguments, positive_int, print_dry_run  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.blindbox_audit import (  # noqa: E402
    BlindBoxAuditDataAdapter,
    audit_windows,
    classify_blindbox_variants,
)
from data.decision_adapter import DecisionDataAdapter, SOURCES, select_target_skus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit blind-box child/open/hidden/end-box SKUs with positive signed sales "
            "but missing or nonpositive store inventory evidence."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--cohort-file", type=Path, default=Path("artifacts/round3_dataset/cohort.csv")
    )
    parser.add_argument(
        "--window-days",
        type=positive_int,
        default=182,
        help="Observation window from each SKU's first actual sale; 182 days = 26 weeks.",
    )
    parser.add_argument(
        "--batch-skus",
        type=positive_int,
        default=5,
        help="Serial SKU batch size; lower this when a batch exceeds the configured row limit.",
    )
    parser.add_argument(
        "--classification-only",
        action="store_true",
        help=(
            "Only classify the potentially affected SKU scope from the product "
            "snapshot. Do not read historical sales or inventory."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    add_execution_arguments(parser, default_output="artifacts/blindbox_inventory_grain_audit")
    return parser.parse_args()


def _fingerprint(args: argparse.Namespace, entity_windows: pd.DataFrame) -> str:
    payload = {
        "version": "blindbox-inventory-grain-audit-v2",
        "cutoff": args.decision_cutoff_time.isoformat(),
        "window_days": args.window_days,
        "batch_skus": args.batch_skus,
        "batch_strategy": "window_start_window_end_sku",
        "entities": entity_windows.astype(str).to_dict(orient="records"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_matches_batch(metrics: pd.DataFrame, batch: pd.DataFrame) -> bool:
    if "sku_code" not in metrics.columns:
        return False
    expected = set(batch["sku_code"].astype(str))
    actual = set(metrics["sku_code"].astype(str))
    return len(metrics) == len(expected) and actual == expected


def _classification_scope_family(
    family_summary: pd.DataFrame, variants: pd.DataFrame
) -> pd.DataFrame:
    if variants.empty:
        return family_summary.iloc[0:0].copy()
    scoped = variants.assign(
        family_identifier=variants["parent_sku_code"].where(
            variants["parent_sku_code"].notna(), variants["family_key"]
        )
    )
    counts = (
        scoped.groupby("family_identifier", as_index=False, dropna=False)
        .agg(
            scoped_variant_sku_count=("sku_code", "nunique"),
            scoped_variant_types=(
                "variant_type", lambda values: "|".join(sorted(set(values)))
            ),
        )
    )
    return family_summary.merge(
        counts, on="family_identifier", how="inner", validate="one_to_one"
    )


def _classification_report(
    *,
    args: argparse.Namespace,
    scope_skus: set[str],
    resolved_scope_skus: set[str],
    variants: pd.DataFrame,
) -> dict[str, object]:
    denominator = len(scope_skus)
    variant_count = int(variants["sku_code"].nunique()) if not variants.empty else 0
    variant_counts = (
        {
            str(key): int(value)
            for key, value in variants["variant_type"].value_counts().items()
        }
        if not variants.empty
        else {}
    )
    method_counts = (
        {
            str(key): int(value)
            for key, value in variants["classification_method"]
            .fillna("unresolved")
            .value_counts()
            .items()
        }
        if not variants.empty
        else {}
    )
    parent_match_counts = (
        {
            str(key): int(value)
            for key, value in variants["parent_match_method"]
            .fillna("unresolved")
            .value_counts()
            .items()
        }
        if not variants.empty
        else {}
    )
    return {
        "spec_version": "3.2",
        "stage": "blind-box potentially affected SKU scope",
        "status": "classification_complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
        "cohort_file": str(args.cohort_file),
        "cohort_scope_skus": denominator,
        "product_snapshot_resolved_scope_skus": len(resolved_scope_skus),
        "product_snapshot_missing_scope_skus": denominator - len(resolved_scope_skus),
        "product_snapshot_resolution_rate": (
            len(resolved_scope_skus) / denominator if denominator else None
        ),
        "potentially_affected_variant_skus": variant_count,
        "potentially_affected_share_of_cohort": (
            variant_count / denominator if denominator else None
        ),
        "variant_type_counts": variant_counts,
        "classification_method_counts": method_counts,
        "parent_match_method_counts": parent_match_counts,
        "unresolved_parent_variant_skus": int(
            variants.get("parent_sku_code", pd.Series(dtype=object)).isna().sum()
        ),
        "interpretation": (
            "Potentially affected scope only. Sales and inventory were not read, "
            "so these counts are not proven inventory anomalies."
        ),
        "classification_contract": {
            "confirmed": "端盒/整盒、隐藏、明盒 direct product-name tokens",
            "inferred": (
                "child variant requires a blind-box cue and one unique product-master "
                "SPU or normalized-series parent"
            ),
            "missing_spu_values_are_not_relationships": True,
            "ambiguous_parent_candidates_are_not_forced": True,
        },
        "outputs": {
            "detail": "blindbox_variant_scope.csv",
            "family": "blindbox_family_scope.csv",
        },
    }


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger(
        "audit_blindbox_inventory_grain", config.log_dir, config.timezone
    )
    outputs = (
        [
            "blindbox_variant_scope.csv",
            "blindbox_family_scope.csv",
            "blindbox_scope_report.json",
            "product_snapshot_audit.csv",
        ]
        if args.classification_only
        else [
            "blindbox_variant_inventory_audit.csv",
            "blindbox_family_summary.csv",
            "blindbox_inventory_audit_report.json",
            "product_snapshot_audit.csv",
            "spu_relationship_audit.csv",
        ]
    )
    if args.dry_run:
        print_dry_run("blind-box inventory-grain audit", args, outputs)
        return 0
    try:
        artifact = StageArtifactStore()
        cohort = artifact.read_frame(
            PROJECT_ROOT / args.cohort_file,
            required=("sku_code", "first_actual_sale_date", "split"),
        )
        scope = select_target_skus(
            cohort, sku_codes=args.sku_code, limit_skus=args.limit_skus
        )
        scope_skus = set(scope["sku_code"].astype(str))
        all_cohort_skus = cohort["sku_code"].astype(str).drop_duplicates().tolist()
        decision = DecisionDataAdapter.from_project(
            project_root=PROJECT_ROOT, data_root=args.data_root, config=config
        )
        products = decision.products(
            args.decision_cutoff_time,
            sku_codes=all_cohort_skus,
            allow_missing=True,
            progress=lambda event, details: logger.info(
                "blindbox_audit_progress event=%s %s",
                event,
                " ".join(f"{key}={value}" for key, value in details.items()),
            ),
        )
        output_dir = PROJECT_ROOT / args.output_dir
        artifact.write_frame(output_dir / outputs[3], products.snapshot_audit)
        if args.classification_only:
            classification = classify_blindbox_variants(products.products)
            variants = classification.variants.loc[
                classification.variants["sku_code"].astype(str).isin(scope_skus)
            ].copy()
            variants["potentially_affected_inventory_grain"] = 1
            family = _classification_scope_family(
                classification.family_summary, variants
            )
            resolved_scope_skus = set(
                products.products["sku_code"].astype(str)
            ) & scope_skus
            report = _classification_report(
                args=args,
                scope_skus=scope_skus,
                resolved_scope_skus=resolved_scope_skus,
                variants=variants,
            )
            artifact.write_frame(output_dir / outputs[0], variants)
            artifact.write_frame(output_dir / outputs[1], family)
            artifact.write_json(output_dir / outputs[2], report)
            logger.info(
                "blind-box classification complete scope=%s potentially_affected=%s share=%.6f",
                len(scope_skus),
                len(variants),
                report["potentially_affected_share_of_cohort"] or 0.0,
            )
            print(
                "status: classification_complete\n"
                f"scope_skus: {len(scope_skus)}\n"
                f"potentially_affected_variant_skus: {len(variants)}\n"
                "potentially_affected_share_of_cohort: "
                f"{report['potentially_affected_share_of_cohort'] or 0.0:.6%}\n"
                f"detail: {output_dir / outputs[0]}\n"
                f"family: {output_dir / outputs[1]}\n"
                f"report: {output_dir / outputs[2]}\n"
                f"log: {log_path}"
            )
            return 0
        source_adapter = BlindBoxAuditDataAdapter(decision.repository)
        direct_classification = classify_blindbox_variants(
            products.products,
            allow_name_parent_match=False,
        )
        unresolved_direct = direct_classification.variants.loc[
            direct_classification.variants["sku_code"].astype(str).isin(scope_skus)
            & direct_classification.variants["parent_sku_code"].isna()
            & direct_classification.variants["classification_method"].eq(
                "direct_name_token"
            )
        ].copy()
        unresolved_windows = audit_windows(
            cohort,
            sku_codes=unresolved_direct["sku_code"].astype(str).tolist(),
            cutoff_time=args.decision_cutoff_time,
            window_days=args.window_days,
        )
        spu_relationship_audit = source_adapter.resolve_spu_ids(
            entities=unresolved_windows,
            cutoff_time=args.decision_cutoff_time,
            progress=lambda event, details: logger.info(
                "blindbox_audit_progress event=%s %s",
                event,
                " ".join(f"{key}={value}" for key, value in details.items()),
            ),
        )
        artifact.write_frame(output_dir / outputs[4], spu_relationship_audit)
        spu_overrides = {
            str(row.sku_code): str(row.resolved_observed_spu_id)
            for row in spu_relationship_audit.itertuples(index=False)
            if str(row.resolved_observed_spu_id).strip()
        }
        classification = classify_blindbox_variants(
            products.products,
            sku_spu_overrides=spu_overrides,
            allow_name_parent_match=True,
        )
        variants = classification.variants.loc[
            classification.variants["sku_code"].astype(str).isin(scope_skus)
        ].copy()
        parent_skus = set(
            variants["parent_sku_code"].dropna().astype(str)
        ) & set(all_cohort_skus)
        entity_skus = sorted(set(variants["sku_code"].astype(str)) | parent_skus)
        entity_windows = audit_windows(
            cohort,
            sku_codes=entity_skus,
            cutoff_time=args.decision_cutoff_time,
            window_days=args.window_days,
        ).sort_values(
            ["window_start", "window_end", "sku_code"], kind="stable"
        ).reset_index(drop=True)
        fingerprint = _fingerprint(args, entity_windows)
        checkpoint_dir = output_dir / ".checkpoints" / fingerprint[:16]
        metric_frames: list[pd.DataFrame] = []
        batch_count = (len(entity_windows) + args.batch_skus - 1) // args.batch_skus
        for offset in range(0, len(entity_windows), args.batch_skus):
            index = offset // args.batch_skus + 1
            batch = entity_windows.iloc[offset : offset + args.batch_skus].copy()
            checkpoint = checkpoint_dir / f"batch_{index:05d}.csv"
            reuse_checkpoint = False
            if not args.no_resume and checkpoint.is_file():
                try:
                    checkpoint_metrics = artifact.read_frame(checkpoint)
                    reuse_checkpoint = _checkpoint_matches_batch(
                        checkpoint_metrics, batch
                    )
                except Exception as exc:
                    logger.warning(
                        "blindbox_audit_progress event=batch_checkpoint_unreadable "
                        "batch=%s batches=%s exception=%s",
                        index,
                        batch_count,
                        f"{type(exc).__name__}: {exc}",
                    )
                if reuse_checkpoint:
                    metrics = checkpoint_metrics
                else:
                    logger.warning(
                        "blindbox_audit_progress event=batch_checkpoint_mismatch "
                        "batch=%s batches=%s expected_skus=%s",
                        index,
                        batch_count,
                        ",".join(batch["sku_code"].astype(str)),
                    )
            if reuse_checkpoint:
                logger.info(
                    "blindbox_audit_progress event=batch_checkpoint_reused batch=%s batches=%s skus=%s",
                    index,
                    batch_count,
                    ",".join(batch["sku_code"].astype(str)),
                )
            else:
                logger.info(
                    "blindbox_audit_progress event=batch_start batch=%s batches=%s skus=%s",
                    index,
                    batch_count,
                    ",".join(batch["sku_code"].astype(str)),
                )
                metrics = source_adapter.entity_metrics(
                    entities=batch,
                    cutoff_time=args.decision_cutoff_time,
                    progress=lambda event, details, batch_index=index: logger.info(
                        "blindbox_audit_progress event=%s batch=%s batches=%s %s",
                        event,
                        batch_index,
                        batch_count,
                        " ".join(f"{key}={value}" for key, value in details.items()),
                    ),
                )
                artifact.write_frame(checkpoint, metrics)
                logger.info(
                    "blindbox_audit_progress event=batch_checkpoint_written batch=%s batches=%s rows=%s",
                    index,
                    batch_count,
                    len(metrics),
                )
            metric_frames.append(metrics)
        metrics = (
            pd.concat(metric_frames, ignore_index=True).drop_duplicates("sku_code")
            if metric_frames
            else pd.DataFrame(columns=[
                "sku_code", "window_start", "window_end", "net_sale_qty",
                "positive_sale_qty", "negative_sale_qty", "latest_order_line_count",
                "selling_store_count", "inventory_row_count",
                "positive_available_row_count", "zero_available_row_count",
                "null_available_row_count", "available_qty_store_day_sum",
                "max_available_qty", "positive_inventory_row_rate",
                "inventory_evidence_status", "sales_inventory_issue",
            ])
        )
        result = variants.merge(metrics, on="sku_code", how="left", validate="one_to_one")
        if not spu_relationship_audit.empty:
            result = result.merge(
                spu_relationship_audit,
                on="sku_code",
                how="left",
                validate="one_to_one",
            )
        if parent_skus and not result.empty:
            parent_columns = [
                "sku_code", "net_sale_qty", "inventory_row_count",
                "positive_available_row_count", "positive_inventory_row_rate",
                "inventory_evidence_status", "sales_inventory_issue",
            ]
            parents = metrics.loc[metrics["sku_code"].isin(parent_skus), parent_columns].rename(
                columns={
                    column: f"parent_{column}"
                    for column in parent_columns
                    if column != "sku_code"
                }
            ).rename(columns={"sku_code": "parent_sku_code"})
            result = result.merge(
                parents, on="parent_sku_code", how="left", validate="many_to_one"
            )
        result["affects_inventory_based_algorithm"] = result["sales_inventory_issue"].isin(
            {
                "sales_positive_inventory_missing",
                "sales_positive_inventory_all_nonpositive",
            }
        ).astype(int)
        family = classification.family_summary.copy()
        if not result.empty:
            scoped_family_ids = set(
                result["parent_sku_code"].where(
                    result["parent_sku_code"].notna(), result["family_key"]
                )
            )
            family = family.loc[
                family["family_identifier"].isin(scoped_family_ids)
            ].copy()
            issue_family = (
                result.assign(
                    family_identifier=result["parent_sku_code"].where(
                        result["parent_sku_code"].notna(), result["family_key"]
                    )
                )
                .groupby("family_identifier", as_index=False, dropna=False)
                .agg(
                    scoped_variant_sku_count=("sku_code", "nunique"),
                    positive_sales_variant_count=("net_sale_qty", lambda value: int((value > 0).sum())),
                    affected_variant_sku_count=("affects_inventory_based_algorithm", "sum"),
                    missing_inventory_variant_count=(
                        "sales_inventory_issue",
                        lambda value: int((value == "sales_positive_inventory_missing").sum()),
                    ),
                    all_nonpositive_inventory_variant_count=(
                        "sales_inventory_issue",
                        lambda value: int((value == "sales_positive_inventory_all_nonpositive").sum()),
                    ),
                )
            )
            family = family.merge(
                issue_family, on="family_identifier", how="left", validate="one_to_one"
            )
        else:
            family = family.iloc[0:0].copy()
        artifact.write_frame(output_dir / outputs[0], result)
        artifact.write_frame(output_dir / outputs[1], family)
        issue_counts = (
            {str(key): int(value) for key, value in result["sales_inventory_issue"].value_counts().items()}
            if not result.empty
            else {}
        )
        variant_counts = (
            {str(key): int(value) for key, value in result["variant_type"].value_counts().items()}
            if not result.empty
            else {}
        )
        parent_match_counts = (
            {
                str(key): int(value)
                for key, value in result["parent_match_method"]
                .fillna("unresolved")
                .value_counts()
                .items()
            }
            if not result.empty
            else {}
        )
        spu_resolution_counts = (
            {
                str(key): int(value)
                for key, value in spu_relationship_audit[
                    "observed_spu_resolution_status"
                ].value_counts().items()
            }
            if not spu_relationship_audit.empty
            else {}
        )
        report = {
            "spec_version": "3.2",
            "stage": "blind-box inventory-grain audit",
            "status": "audit_complete",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "cohort_file": str(args.cohort_file),
            "cohort_scope_skus": len(scope_skus),
            "product_snapshot_resolved_skus": int(products.products["sku_code"].nunique()),
            "classified_variant_skus": len(result),
            "variant_type_counts": variant_counts,
            "parent_match_method_counts": parent_match_counts,
            "unresolved_parent_variant_skus": int(
                result.get("parent_sku_code", pd.Series(dtype=object)).isna().sum()
            ),
            "observed_spu_resolution_counts": spu_resolution_counts,
            "sales_inventory_issue_counts": issue_counts,
            "affected_inventory_based_algorithm_skus": int(
                result.get("affects_inventory_based_algorithm", pd.Series(dtype=int)).sum()
            ),
            "observation_window_days": args.window_days,
            "batch_skus": args.batch_skus,
            "batch_strategy": "window_start_window_end_sku",
            "sales_source": {
                "table": SOURCES["sales"],
                "field": "sale_qty",
                "contract": "same-ds signed events netted, then latest order-line version retained",
            },
            "inventory_source": {
                "table": SOURCES["inventory"],
                "field": "available_stock_qty",
                "contract": "row absence and rows with no positive available quantity are distinct",
            },
            "classification_contract": {
                "confirmed": "端盒/整盒、隐藏、明盒 direct product-name tokens",
                "inferred": "child variant requires blind-box cue plus unique structured-SPU or unique normalized-series parent",
                "parent_resolution_order": [
                    "valid product-master spu_code",
                    "one consistent sales/inventory spu_id",
                    "one unambiguous normalized series-name parent",
                ],
                "inferred_rows_are_not_claimed_as_confirmed_master_data": True,
                "missing_spu_values_are_not_relationships": True,
                "ambiguous_parent_candidates_are_not_forced": True,
            },
            "checkpoint_fingerprint": fingerprint,
            "checkpoint_directory": str(checkpoint_dir.relative_to(PROJECT_ROOT)),
            "outputs": {
                "detail": outputs[0],
                "family": outputs[1],
                "spu_relationship_audit": outputs[4],
            },
        }
        artifact.write_json(output_dir / outputs[2], report)
        logger.info(
            "blind-box audit complete scope=%s variants=%s affected=%s",
            len(scope_skus),
            len(result),
            report["affected_inventory_based_algorithm_skus"],
        )
    except Exception as exc:
        log_exception(
            logger,
            module="audit_blindbox_inventory_grain",
            cutoff=args.decision_cutoff_time,
            exc=exc,
        )
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(
        "status: audit_complete\n"
        f"detail: {output_dir / outputs[0]}\n"
        f"family: {output_dir / outputs[1]}\n"
        f"report: {output_dir / outputs[2]}\n"
        f"log: {log_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
