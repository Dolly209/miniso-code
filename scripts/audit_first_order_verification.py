#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from data.artifacts import StageArtifactStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a V3.2 four-SKU trace from Top-K comparables through placement, "
            "weekly demand, ordering policy, model prediction and W_cover label."
        )
    )
    parser.add_argument("--final-replay-dir", type=Path, required=True)
    parser.add_argument(
        "--modeling-dataset",
        type=Path,
        help="Optional Section 8.4 modeling snapshot; omit for upstream-only audit.",
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        help="Optional locked-Test predictions; omit before corrected retraining.",
    )
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument(
        "--label-restore-weekly",
        type=Path,
        help=(
            "Round-4 restored_store_weekly_parts directory/CSV used by the label "
            "builder. Supply this for week-level label-source reconciliation."
        ),
    )
    parser.add_argument("--sku-code", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/first_order_verification_4sku_v1"),
    )
    parser.add_argument("--large-deviation-threshold", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_filtered(
    path: Path,
    skus: list[str],
    *,
    column: str = "sku_code",
    require_all: bool = True,
) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    if not skus:
        return pd.DataFrame()
    try:
        if not require_all:
            return StageArtifactStore.read_frame(
                path,
                sku_column=column,
                filters={column: [str(value) for value in skus]},
                chunksize=100_000 if path.suffix.lower() == ".csv" else None,
            )
        return StageArtifactStore.read_frame(
            path,
            sku_codes=skus,
            sku_column=column,
            chunksize=100_000 if path.suffix.lower() == ".csv" else None,
        )
    except pd.errors.EmptyDataError:
        # Empty evidence is a valid outcome when the selected comparable has no
        # raw launch-window rows and Round 6 uses its disclosed versioned prior.
        return pd.DataFrame()


def checkpoint_groups(replay_dir: Path, skus: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for schedule_path in sorted((replay_dir / "checkpoint_groups").glob("*/target_schedule.csv")):
        schedule = read_filtered(schedule_path, skus, require_all=False)
        for sku in schedule.get("sku_code", pd.Series(dtype=str)).astype(str).unique():
            if sku in result:
                raise ValueError(f"more than one final replay group contains SKU {sku}")
            result[sku] = schedule_path.parent
    missing = sorted(set(skus) - set(result))
    if missing:
        raise ValueError(f"verification SKUs missing from final replay groups: {missing}")
    return result


def weekly_label_trace(
    path: Path | None,
    labels: pd.DataFrame,
    skus: list[str],
) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    columns = (
        "sku_code",
        "lifecycle_week",
        "week_end",
        "clean_sales_qty",
        "restored_natural_sales",
        "restored_lost_sales",
        "restored_exposure_days",
        "unknown_exposure_days",
        "source_day_rows",
    )
    pieces: list[pd.DataFrame] = []
    for part in StageArtifactStore.iter_frame_parts(
        absolute(path),
        required=("sku_code", "lifecycle_week", "restored_natural_sales"),
        columns=columns,
        sku_codes=skus,
        require_all_skus=False,
    ):
        pieces.append(
            part.groupby(["sku_code", "lifecycle_week"], as_index=False).agg(
                week_end=("week_end", "max"),
                observed_clean_sales=("clean_sales_qty", "sum"),
                restored_natural_sales=("restored_natural_sales", "sum"),
                restored_lost_sales=("restored_lost_sales", "sum"),
                restored_exposure_days=("restored_exposure_days", "sum"),
                unknown_exposure_days=("unknown_exposure_days", "sum"),
                source_day_rows=("source_day_rows", "sum"),
            )
        )
    if not pieces:
        return pd.DataFrame()
    combined = pd.concat(pieces, ignore_index=True).groupby(
        ["sku_code", "lifecycle_week"], as_index=False
    ).agg(
        week_end=("week_end", "max"),
        observed_clean_sales=("observed_clean_sales", "sum"),
        restored_natural_sales=("restored_natural_sales", "sum"),
        restored_lost_sales=("restored_lost_sales", "sum"),
        restored_exposure_days=("restored_exposure_days", "sum"),
        unknown_exposure_days=("unknown_exposure_days", "sum"),
        source_day_rows=("source_day_rows", "sum"),
    )
    cover = labels[["sku_code", "W_cover"]].drop_duplicates("sku_code")
    combined = combined.merge(cover, on="sku_code", how="left", validate="many_to_one")
    return combined.loc[combined["lifecycle_week"].le(combined["W_cover"])].copy()


def main() -> int:
    args = parse_args()
    skus = args.sku_code
    if not skus:
        raise ValueError("supply one or more --sku-code values")
    if not 0 <= args.large_deviation_threshold:
        raise ValueError("large-deviation-threshold must be non-negative")
    output = absolute(args.output_dir)
    if args.dry_run:
        print(f"DRY RUN: audit SKUs={','.join(skus)} output={output}")
        return 0

    store = StageArtifactStore()
    replay_dir = absolute(args.final_replay_dir)
    modeling = (
        read_filtered(absolute(args.modeling_dataset), skus)
        if args.modeling_dataset is not None else pd.DataFrame()
    )
    predictions = (
        read_filtered(absolute(args.prediction_file), skus)
        if args.prediction_file is not None else pd.DataFrame()
    )
    labels = read_filtered(absolute(args.label_file), skus)
    groups = checkpoint_groups(replay_dir, skus)

    candidate_parts: list[pd.DataFrame] = []
    placement_parts: list[pd.DataFrame] = []
    placement_summary_parts: list[pd.DataFrame] = []
    placement_evidence_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    calculation_parts: list[pd.DataFrame] = []
    store_target_parts: list[pd.DataFrame] = []
    for sku, group in groups.items():
        candidate = read_filtered(
            group / "round5/candidate_detail.csv", [sku], column="target_sku_code"
        )
        if not candidate.empty:
            candidate_parts.append(candidate)
            evidence = read_filtered(
                group / "round6/placement_evidence.csv",
                candidate.loc[
                    pd.to_numeric(candidate.get("selected"), errors="coerce")
                    .fillna(0).eq(1),
                    "candidate_sku_code",
                ].astype(str).unique().tolist(),
                require_all=False,
            )
            if not evidence.empty:
                selected_pairs = candidate.loc[
                    pd.to_numeric(candidate.get("selected"), errors="coerce")
                    .fillna(0).eq(1),
                    ["target_sku_code", "candidate_sku_code"],
                ].drop_duplicates()
                placement_evidence_parts.append(
                    selected_pairs.merge(
                        evidence,
                        left_on="candidate_sku_code",
                        right_on="sku_code",
                        how="inner",
                        validate="many_to_many",
                    )
                )
        for relative, destination in (
            ("round6/placement_plan.csv", placement_parts),
            ("round6/placement_summary.csv", placement_summary_parts),
            ("round6/demand_curve.csv", curve_parts),
            ("round8/first_order_decisions.csv", decision_parts),
            ("round8/first_order_calculation_detail.csv", calculation_parts),
            ("round8/first_order_store_targets.csv", store_target_parts),
        ):
            frame = read_filtered(group / relative, [sku])
            if not frame.empty:
                destination.append(frame)

    candidates = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    placement = pd.concat(placement_parts, ignore_index=True) if placement_parts else pd.DataFrame()
    placement_summary = (
        pd.concat(placement_summary_parts, ignore_index=True)
        if placement_summary_parts else pd.DataFrame()
    )
    placement_evidence = (
        pd.concat(placement_evidence_parts, ignore_index=True)
        if placement_evidence_parts else pd.DataFrame()
    )
    curves = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()
    decisions = pd.concat(decision_parts, ignore_index=True) if decision_parts else pd.DataFrame()
    calculations = pd.concat(calculation_parts, ignore_index=True) if calculation_parts else pd.DataFrame()
    store_targets = pd.concat(store_target_parts, ignore_index=True) if store_target_parts else pd.DataFrame()

    if not candidates.empty:
        candidates = candidates.loc[
            pd.to_numeric(candidates.get("selected"), errors="coerce").fillna(0).eq(1)
        ].copy()
    if not placement.empty:
        placement["selected"] = pd.to_numeric(
            placement["selected"], errors="coerce"
        ).fillna(0).astype(int)
    selected_placement = placement.loc[placement.get("selected", 0).eq(1)].copy()

    tier_velocity = pd.DataFrame()
    if not selected_placement.empty:
        tier_velocity = selected_placement.groupby(
            ["sku_code", "store_tier"], as_index=False
        ).agg(
            selected_store_count=("store_code", "nunique"),
            mean_week1_velocity=("v_anchor_week1", "mean"),
            median_week1_velocity=("v_anchor_week1", "median"),
            historical_coverage_stores=(
                "placement_origin",
                lambda values: int(
                    pd.Series(values).eq("historical_launch_coverage").sum()
                ),
            ) if "placement_origin" in selected_placement else ("store_code", "size"),
        )

    weekly_demand_rows: list[dict[str, Any]] = []
    if not tier_velocity.empty and not curves.empty:
        for curve in curves.itertuples(index=False):
            tier = tier_velocity.loc[
                tier_velocity["sku_code"].astype(str).eq(str(curve.sku_code))
                & tier_velocity["store_tier"].astype(str).eq(str(curve.store_tier))
            ]
            if tier.empty:
                continue
            row = tier.iloc[0]
            neutral = (
                float(row["selected_store_count"])
                * float(row["mean_week1_velocity"])
                * float(curve.lifecycle_factor)
            )
            sellable = neutral * float(curve.period_factor_h)
            weekly_demand_rows.append(
                {
                    "sku_code": str(curve.sku_code),
                    "store_tier": str(curve.store_tier),
                    "lifecycle_week": int(curve.lifecycle_week),
                    "selected_store_count": int(row["selected_store_count"]),
                    "mean_week1_velocity": float(row["mean_week1_velocity"]),
                    "lifecycle_factor": float(curve.lifecycle_factor),
                    "period_factor_h": float(curve.period_factor_h),
                    "neutral_demand": neutral,
                    "sellable_demand": sellable,
                }
            )
    weekly_demand = pd.DataFrame(weekly_demand_rows)

    label_weekly = weekly_label_trace(args.label_restore_weekly, labels, skus)
    if not label_weekly.empty:
        label_reconcile = label_weekly.groupby("sku_code", as_index=False).agg(
            recomputed_D_first_ref=("restored_natural_sales", "sum"),
            observed_sales_wcover=("observed_clean_sales", "sum"),
            restored_lost_sales_wcover=("restored_lost_sales", "sum"),
            label_week_count=("lifecycle_week", "nunique"),
        )
        labels = labels.merge(label_reconcile, on="sku_code", how="left")
        labels["label_reconciles_to_weekly_source"] = np.isclose(
            pd.to_numeric(labels["D_first_ref"], errors="coerce"),
            pd.to_numeric(labels["recomputed_D_first_ref"], errors="coerce"),
            equal_nan=False,
        )
    else:
        labels["label_reconciles_to_weekly_source"] = False

    prediction_column = "prediction" if "prediction" in predictions else None
    summary = decisions.copy()
    if summary.empty:
        raise ValueError("no Round-8 decisions found for verification SKUs")
    label_keep = [
        column
        for column in (
            "sku_code", "D_first_ref", "W_cover", "label_start_date",
            "label_end_date", "label_source_column", "label_definition",
            "observed_clean_sales_wcover", "restored_demand_wcover",
            "restoration_uplift_wcover", "label_quality_status",
            "label_reconciles_to_weekly_source",
        )
        if column in labels
    ]
    summary = summary.merge(
        labels[label_keep].drop_duplicates("sku_code"),
        on="sku_code", how="left", validate="one_to_one",
    )
    if not placement_summary.empty:
        placement_keep = [
            column for column in (
                "sku_code", "path", "selected_store_count",
                "historical_coverage_store_count",
                "counterfactual_expansion_store_count",
                "evidence_supported_store_count", "effective_store_count",
                "placement_rate", "placement_selection_pool_store_count",
                "placement_evidence_source", "missing_launch_evidence_candidate_count",
                "formal_v3_2_rate", "category_fit_coverage",
                "execution_coverage",
            ) if column in placement_summary
        ]
        summary = summary.merge(
            placement_summary[placement_keep].drop_duplicates("sku_code"),
            on="sku_code", how="left", validate="one_to_one",
            suffixes=("", "__placement"),
        )
    if prediction_column:
        summary = summary.merge(
            predictions[["sku_code", prediction_column]].drop_duplicates("sku_code"),
            on="sku_code", how="left", validate="one_to_one",
        ).rename(columns={prediction_column: "final_model_prediction"})
    else:
        summary["final_model_prediction"] = np.nan
    if not modeling.empty:
        modeling_keep = [
            column
            for column in (
                "sku_code",
                "prediction_checkpoint_key",
                "decision_cutoff_time",
                "feature__first_order__D_demand_base",
                "feature__first_order__Q_base",
                "label",
            )
            if column in modeling
        ]
        summary = summary.merge(
            modeling[modeling_keep]
            .drop_duplicates("sku_code")
            .rename(
                columns={
                    column: f"modeling_snapshot__{column}"
                    for column in modeling_keep
                    if column != "sku_code"
                }
            ),
            on="sku_code",
            how="left",
            validate="one_to_one",
        )

    # Pre-fix artifacts used D_sell/D_first for the same two demand concepts.
    # Preserve compatibility without inferring any policy quantity: the source
    # column is disclosed so the frozen baseline remains auditable.
    if "D_demand_base" not in summary and "D_sell" in summary:
        summary["D_demand_base"] = summary["D_sell"]
        summary["D_demand_base_trace_source"] = "legacy_D_sell"
    elif "D_demand_base" in summary:
        summary["D_demand_base_trace_source"] = "D_demand_base"
    if "D_demand_adjusted" not in summary and "D_first" in summary:
        summary["D_demand_adjusted"] = summary["D_first"]
        summary["D_demand_adjusted_trace_source"] = "legacy_D_first"
    elif "D_demand_adjusted" in summary:
        summary["D_demand_adjusted_trace_source"] = "D_demand_adjusted"

    chain_fields = (
        "D_demand_base", "D_demand_adjusted", "SS_first",
        "minimum_floor_uplift_base", "Display_first", "Q_base", "Q_nat",
        "Q_plan", "Q_feas", "final_model_prediction", "D_first_ref",
    )
    chain_rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        label = float(getattr(row, "D_first_ref"))
        first_large = "none"
        for sequence, field in enumerate(chain_fields, start=1):
            value = pd.to_numeric(pd.Series([getattr(row, field, np.nan)]), errors="coerce").iloc[0]
            relative_error = (
                float((value - label) / max(abs(label), 1.0))
                if pd.notna(value) else np.nan
            )
            is_large = bool(
                pd.notna(relative_error)
                and abs(relative_error) >= args.large_deviation_threshold
            )
            if first_large == "none" and is_large:
                first_large = field
            chain_rows.append(
                {
                    "sku_code": str(row.sku_code),
                    "calc_sequence": sequence,
                    "calc_item_code": field,
                    "calc_value": value,
                    "label_D_first_ref": label,
                    "difference_vs_label": value - label if pd.notna(value) else np.nan,
                    "relative_error_vs_label": relative_error,
                    "large_deviation": is_large,
                }
            )
        summary.loc[summary["sku_code"].astype(str).eq(str(row.sku_code)), "first_large_deviation_step"] = first_large
    chain = pd.DataFrame(chain_rows)

    if not store_targets.empty:
        demand_column = (
            "pure_demand_base"
            if "pure_demand_base" in store_targets
            else "sellable_demand"
        )
        store_diag = store_targets.groupby("sku_code", as_index=False).agg(
            potential_demand_recomputed=(demand_column, "sum"),
            safety_stock_recomputed=("safety_stock", "sum"),
            display_recomputed=("display_units", "sum"),
            q_base_recomputed=("base_target", "sum"),
        )
        if "placement_origin" in store_targets:
            historical = (
                store_targets.loc[
                    store_targets["placement_origin"].eq(
                        "historical_launch_coverage"
                    )
                ]
                .groupby("sku_code")[demand_column]
                .sum()
                .rename("comparable_historical_supported_potential_demand")
            )
            expansion = (
                store_targets.loc[
                    store_targets["placement_origin"].eq(
                        "modeled_counterfactual_trial"
                    )
                ]
                .groupby("sku_code")[demand_column]
                .sum()
                .rename("counterfactual_expansion_demand")
            )
            store_diag = store_diag.merge(
                historical, on="sku_code", how="left"
            ).merge(expansion, on="sku_code", how="left")
        summary = summary.merge(store_diag, on="sku_code", how="left")

    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sku_trace_summary.csv": summary,
        "calculation_chain.csv": chain,
        "candidate_trace.csv": candidates,
        "placement_store_trace.csv": placement,
        "placement_summary_trace.csv": placement_summary,
        "candidate_store_evidence_trace.csv": placement_evidence,
        "tier_velocity_trace.csv": tier_velocity,
        "weekly_potential_demand_trace.csv": weekly_demand,
        "weekly_label_source_trace.csv": label_weekly,
        "round8_store_target_trace.csv": store_targets,
        "round8_calculation_detail.csv": calculations,
        "label_trace.csv": labels,
    }
    for name, frame in outputs.items():
        store.write_frame(output / name, frame)

    label_reconciled = bool(
        not label_weekly.empty
        and labels["label_reconciles_to_weekly_source"].fillna(False).all()
    )
    report = {
        "spec_version": "3.2",
        "status": "verification_trace_built",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_skus": skus,
        "trace_summary_rows": len(summary),
        "candidate_rows": len(candidates),
        "placement_store_rows": len(placement),
        "candidate_store_evidence_rows": len(placement_evidence),
        "final_model_prediction_status": (
            "included" if prediction_column else "not_supplied_upstream_only_audit"
        ),
        "weekly_potential_rows": len(weekly_demand),
        "weekly_label_source_rows": len(label_weekly),
        "label_source_reconciled": label_reconciled,
        "label_source_reconciliation_status": (
            "complete" if label_reconciled else
            "missing_or_mismatched_restore_weekly_source"
        ),
        "target_actual_coverage_policy": (
            "target post-launch actual coverage is label/offline-diagnostic only; "
            "it is never merged into prelaunch model inputs. Comparable historical "
            "launch evidence remains a valid M3 point-in-time prior."
        ),
        "quantity_scope": (
            "D_demand_* and D_first_ref are pure sales-demand piece quantities; "
            "SS/display/Q_* are separate order-policy quantities"
        ),
        "large_deviation_threshold": args.large_deviation_threshold,
    }
    store.write_json(output / "verification_report.json", report)
    print(f"status: verification_trace_built\nsummary: {output / 'sku_trace_summary.csv'}")
    print(f"report: {output / 'verification_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
