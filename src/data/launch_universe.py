"""Lightweight launch-origin and source-window audit; no raw fact scans or fit."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from data.domain import frame_hash, json_hash, normalize_sku_column, require_columns
from data.domestic import DataContractViolation
from data.launch_contract import SALES_OBSERVED_ORIGIN_POLICY, validate_launch_origin_contract
from training.decision_schedule import build_prelaunch_decision_schedule
from training.split import SkuSample, chronological_ratio_split
from training.weights import long_tail_weights


CORE_HISTORY = ("sales", "inventory", "placement")


@dataclass(frozen=True)
class VerifiedLaunchUniverse:
    audit: pd.DataFrame
    candidate_cohort: pd.DataFrame
    schedule: pd.DataFrame
    source_window_detail: pd.DataFrame
    exclusion_summary: pd.DataFrame
    report: dict[str, Any]


def validate_observed_cohort_report(
    cohort: pd.DataFrame, report: Mapping[str, Any], *, available_through: datetime,
) -> None:
    """Do not label a later-cohort extract as the September-30 baseline."""
    versions = set(cohort["dataset_version"].dropna().astype(str))
    if versions != {str(report.get("dataset_version", ""))}:
        raise DataContractViolation("source cohort report dataset_version does not match cohort CSV")
    try:
        cutoff = datetime.fromisoformat(str(report["decision_cutoff_time"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise DataContractViolation("source cohort report lacks a valid decision_cutoff_time") from exc
    if cutoff.tzinfo is None or cutoff.utcoffset() is None or cutoff != available_through:
        raise DataContractViolation(
            "source cohort cutoff does not match requested available-through time; use the matching Round-3 CSV/report pair"
        )


def read_verified_launch_universe(
    *, adapter: Any, cohort: pd.DataFrame, available_through: datetime,
    source_start_date: date, parameters: Mapping[str, Any],
    label_quality: pd.DataFrame | None = None, train_ratio: float = 0.8,
    lead_months: int = 3, distribution_window_weeks: int = 26,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> VerifiedLaunchUniverse:
    from data.decision_adapter import SOURCES
    from data.parquet_repository import DataLayerError

    product = adapter.products(
        available_through, sku_codes=cohort["sku_code"].astype(str).tolist(),
        allow_missing=True, fallback_to_historical=False, progress=progress,
    )
    if product.source_partition != available_through.date().isoformat():
        raise DataContractViolation("verified launch universe requires the exact fixed product partition")
    partitions = {}
    errors = {}
    for key in (*CORE_HISTORY, "store", "social"):
        try:
            partitions[key] = tuple(day for day in adapter.repository.available_partitions(SOURCES[key]) if day <= available_through.date())
        except DataLayerError as exc:
            partitions[key] = ()
            errors[key] = f"{type(exc).__name__}: {exc}"
        if progress:
            progress("source_partition_index_complete", {"source": key, "partitions": len(partitions[key])})
    result = build_verified_launch_universe(
        cohort=cohort, products=product.products, source_partitions=partitions,
        available_through=available_through, source_start_date=source_start_date,
        parameters=parameters, label_quality=label_quality, train_ratio=train_ratio,
        lead_months=lead_months, distribution_window_weeks=distribution_window_weeks,
    )
    result.report["source_partition_errors"] = errors
    result.report["source_partition_ranges"] = {
        key: {"first": min(days).isoformat() if days else None,
              "last": max(days).isoformat() if days else None, "count": len(days)}
        for key, days in partitions.items()
    }
    return result


def rebind_reviewed_ip_snapshot(
    *, mapping: pd.DataFrame, mapping_report: Mapping[str, Any],
    source_cohort: pd.DataFrame, verified_cohort: pd.DataFrame,
    available_through: datetime,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebind reviewed SKU identities; never infer or edit an IP name/code."""
    from data.standard_ip import validate_resolved_standard_ip_snapshot

    parent_version = validate_resolved_standard_ip_snapshot(
        mapping, cohort=source_cohort[["sku_code", "dataset_version"]],
        report=mapping_report, decision_cutoff_time=available_through,
        allow_cohort_subset=True,
    )
    context_keys = ("dataset_version", "first_actual_sale_date", "split", "long_tail_weight", "time_block")
    selected = mapping.loc[mapping["sku_code"].astype(str).isin(verified_cohort["sku_code"])].copy()
    selected = selected.drop(columns=[key for key in context_keys if key in selected])
    selected = selected.merge(
        verified_cohort[["sku_code", *[key for key in context_keys if key in verified_cohort]]],
        on="sku_code", how="left", validate="many_to_one",
    ).sort_values(["sku_code", "standard_ip_code"], kind="stable").reset_index(drop=True)
    selected["parent_mapping_dataset_version"] = parent_version
    mapped = selected.loc[selected["path1_eligible"].eq(1) & selected["standard_ip_code"].ne(""), "sku_code"].nunique()
    report = dict(mapping_report)
    report.update({
        "cohort_skus": int(verified_cohort["sku_code"].nunique()),
        "mapped_cohort_skus": int(mapped),
        "cohort_skus_found_in_product_snapshot": len(verified_cohort),
        "missing_product_snapshot_skus": 0,
        "mapping_coverage": float(mapped / len(verified_cohort)) if len(verified_cohort) else 0.0,
        "mapping_status_counts": selected.drop_duplicates("sku_code")["mapping_status"].value_counts().to_dict(),
        "mapping_output_hash": frame_hash(selected),
        "audit_output_hash": None,
        "parent_mapping_dataset_version": parent_version,
        "parent_mapping_output_hash": mapping_report.get("mapping_output_hash"),
        "rebinding_policy": "reviewed rows subsetted by SKU; only dataset/split context rebound; IP identities and inference provenance unchanged",
        "ip_inference_rerun": False,
    })
    if not verified_cohort.empty:
        validate_resolved_standard_ip_snapshot(
            selected, cohort=verified_cohort, report=report,
            decision_cutoff_time=available_through,
        )
    return selected, report


def _dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed").dt.normalize()


def _window_counts(
    partitions: Sequence[date], starts: pd.Series, ends: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Two binary searches per window, not one directory scan per SKU/day."""
    ordinals = np.asarray(sorted({item.toordinal() for item in partitions}), dtype=np.int64)
    left = starts.map(lambda value: value.date().toordinal() if pd.notna(value) else 0).to_numpy()
    right = ends.map(lambda value: value.date().toordinal() if pd.notna(value) else -1).to_numpy()
    observed = np.searchsorted(ordinals, right, side="right") - np.searchsorted(ordinals, left, side="left")
    expected = (ends - starts).dt.days.add(1).fillna(0).clip(lower=0).to_numpy(dtype=int)
    return np.maximum(observed, 0), expected


def _reason_text(index: pd.Index, checks: Mapping[str, pd.Series]) -> pd.Series:
    text = pd.Series("", index=index, dtype=object)
    for reason, mask in checks.items():
        text.loc[mask.fillna(False)] += reason + "|"
    return text.str.rstrip("|")


def build_verified_launch_universe(
    *, cohort: pd.DataFrame, products: pd.DataFrame,
    source_partitions: Mapping[str, Sequence[date]],
    available_through: datetime, source_start_date: date,
    parameters: Mapping[str, Any], label_quality: pd.DataFrame | None = None,
    train_ratio: float = 0.8, lead_months: int = 3,
    distribution_window_weeks: int = 26,
) -> VerifiedLaunchUniverse:
    if available_through.tzinfo is None or available_through.utcoffset() is None:
        raise ValueError("available_through must include a UTC offset")
    if not 0 < train_ratio < 1 or lead_months <= 0 or distribution_window_weeks <= 0:
        raise ValueError("invalid train ratio, lead months or distribution window")
    require_columns(cohort, ("sku_code", "dataset_version", "first_actual_sale_date", "sales_26w_qty"), "source cohort")
    require_columns(products, ("sku_code", "first_sale_date", "release_dt"), "fixed product snapshot")
    original = normalize_sku_column(cohort)
    if original["sku_code"].duplicated().any():
        raise DataContractViolation("source cohort must contain one row per SKU")
    if original["dataset_version"].nunique(dropna=False) != 1:
        raise DataContractViolation("source cohort must contain one dataset_version")
    product = normalize_sku_column(products)
    product_columns = [c for c in ("sku_code", "sku_name", "first_sale_date", "release_dt") if c in product]
    product = product[product_columns].drop_duplicates()
    if product["sku_code"].duplicated().any():
        raise DataContractViolation("product launch attributes conflict across rows for one SKU")
    product = product.rename(columns={"first_sale_date": "product_first_sale_date"})
    audit = original[["sku_code", "dataset_version", "first_actual_sale_date", "sales_26w_qty"]].rename(
        columns={"dataset_version": "source_dataset_version", "first_actual_sale_date": "observed_first_sale_date"}
    ).merge(product, on="sku_code", how="left", validate="one_to_one")
    cutoff_day = pd.Timestamp(available_through.date())
    audit["product_first_sale_date"] = _dates(audit["product_first_sale_date"])
    audit["observed_first_sale_date"] = _dates(audit["observed_first_sale_date"])
    audit["release_dt"] = _dates(audit["release_dt"])
    audit["source_start_date"] = pd.Timestamp(source_start_date)
    audit["product_snapshot_partition"] = available_through.date().isoformat()
    audit["product_snapshot_status"] = np.where(audit["sku_code"].isin(product["sku_code"]), "latest_partition", "missing_in_latest_partition")
    audit["origin_offset_days"] = (audit["observed_first_sale_date"] - audit["product_first_sale_date"]).dt.days.astype("Int64")
    audit["product_lifecycle_week_at_observed_start"] = (
        audit["origin_offset_days"].floordiv(7).add(1).where(audit["origin_offset_days"].ge(0)).astype("Int64")
    )
    audit["product_observed_dates_agree"] = (
        audit["product_first_sale_date"].notna()
        & audit["product_first_sale_date"].eq(audit["observed_first_sale_date"])
    )
    sales_days = [day for day in source_partitions.get("sales", ()) if day <= available_through.date()]
    left_boundary = pd.Timestamp(max(source_start_date, min(sales_days))) if sales_days else pd.Timestamp(source_start_date)
    audit["source_left_boundary_date"] = left_boundary
    audit["known_pre_source_sale_evidence"] = audit["product_first_sale_date"].lt(left_boundary)
    audit["observed_on_source_left_boundary"] = audit["observed_first_sale_date"].eq(left_boundary)
    audit["unconfirmed_source_boundary_origin"] = audit["observed_on_source_left_boundary"] & ~audit["product_observed_dates_agree"]
    audit["launch_origin_eligible"] = (
        audit["observed_first_sale_date"].notna()
        & audit["observed_first_sale_date"].ge(left_boundary)
        & audit["observed_first_sale_date"].le(cutoff_day)
        & ~audit["known_pre_source_sale_evidence"]
        & ~audit["unconfirmed_source_boundary_origin"]
    )
    audit["launch_origin_verified"] = audit["product_observed_dates_agree"] & audit["launch_origin_eligible"]
    audit["launch_origin_policy"] = SALES_OBSERVED_ORIGIN_POLICY
    audit["first_sale_origin_source"] = "sales_order_first_positive_from_cohort"
    audit["launch_origin_status"] = np.select(
        [audit["observed_first_sale_date"].isna(), audit["known_pre_source_sale_evidence"],
         audit["unconfirmed_source_boundary_origin"], audit["launch_origin_verified"],
         audit["product_first_sale_date"].isna()],
        ["missing_observed_origin", "known_pre_source_sale_evidence", "source_boundary_origin_unconfirmed",
         "sales_origin_corroborated_by_product", "sales_observed_product_first_sale_missing"],
        default="sales_observed_product_date_disagrees",
    )
    audit["origin_audit_notes"] = _reason_text(audit.index, {
        "product_first_sale_missing_not_a_hard_filter": audit["product_first_sale_date"].isna(),
        "product_date_disagrees_not_a_hard_filter": audit["product_first_sale_date"].notna() & audit["observed_first_sale_date"].notna() & ~audit["product_observed_dates_agree"],
        "product_first_sale_after_available_through": audit["product_first_sale_date"].gt(cutoff_day),
        "release_before_source_start": audit["release_dt"].lt(pd.Timestamp(source_start_date)),
    })
    # All windows, sales_26w_qty and old quality evidence now share the original
    # sales-observation origin. A product-date mismatch does not move that clock.
    audit["first_actual_sale_date"] = audit["observed_first_sale_date"]
    numeric_sales = pd.to_numeric(audit["sales_26w_qty"], errors="coerce")
    audit["sales_26w_qty"] = numeric_sales
    audit["distribution_window_end"] = audit["first_actual_sale_date"] + pd.Timedelta(days=distribution_window_weeks * 7 - 1)
    audit["distribution_window_mature"] = audit["distribution_window_end"].le(cutoff_day)
    launch_days = int(parameters["placement"]["initial_defaults"]["launch_window_days"])
    if launch_days <= 0:
        raise ValueError("launch_window_days must be positive")
    audit["launch_window_end"] = audit["first_actual_sale_date"] + pd.Timedelta(days=launch_days - 1)
    details: list[pd.DataFrame] = []
    core_complete = pd.Series(True, index=audit.index)
    for key in CORE_HISTORY:
        partitions = tuple(day for day in source_partitions.get(key, ()) if day <= available_through.date())
        count, expected = _window_counts(partitions, audit["first_actual_sale_date"], audit["launch_window_end"])
        complete = pd.Series((expected == launch_days) & (count == expected), index=audit.index)
        audit[f"{key}_launch_partition_days"] = count
        core_complete &= complete
        detail = audit[["sku_code"]].copy()
        detail["window_role"] = "candidate_launch"
        detail["source_key"] = key
        detail["window_start"] = audit["first_actual_sale_date"]
        detail["window_end"] = audit["launch_window_end"]
        detail["expected_partition_days"] = expected
        detail["observed_partition_days"] = count
        detail["window_complete"] = complete
        detail["required_for_this_gate"] = True
        details.append(detail)
    audit["core_launch_partition_windows_complete"] = core_complete

    audit["prior_label_quality_status"] = "not_audited_requires_fresh_label_build"
    audit["prior_label_origin_matches"] = False
    audit["known_unrestorable_stock_grain"] = False
    audit["known_target_label_quality_rejected"] = False
    if label_quality is not None:
        require_columns(label_quality, ("sku_code", "label_start_date", "label_eligible", "stock_grain_unreliable_rows"), "prior label-quality audit")
        quality = normalize_sku_column(label_quality)
        if quality["sku_code"].duplicated().any():
            raise DataContractViolation("prior label-quality audit must contain one row per SKU")
        quality = quality.set_index("sku_code")
        prior_origin = _dates(audit["sku_code"].map(quality["label_start_date"]))
        matched = prior_origin.eq(audit["first_actual_sale_date"]) & prior_origin.notna()
        bad_grain = pd.to_numeric(audit["sku_code"].map(quality["stock_grain_unreliable_rows"]), errors="coerce").gt(0)
        eligible = audit["sku_code"].map(quality["label_eligible"]).astype(str).str.lower().isin(("true", "1"))
        audit["prior_label_origin_matches"] = matched
        audit["known_unrestorable_stock_grain"] = matched & bad_grain
        audit["known_target_label_quality_rejected"] = matched & ~eligible
        audit["prior_label_quality_status"] = np.select(
            [matched & ~eligible, matched & eligible, audit["sku_code"].isin(quality.index)],
            ["previously_rejected_same_origin", "previously_eligible_fresh_label_validation_required", "prior_origin_differs_not_reused"],
            default="not_audited_requires_fresh_label_build",
        )
    candidate_checks = {
        "missing_product_snapshot": audit["product_snapshot_status"].ne("latest_partition"),
        "missing_observed_first_sale_date": audit["observed_first_sale_date"].isna(),
        "observed_first_sale_before_source_start": audit["observed_first_sale_date"].lt(left_boundary),
        "observed_first_sale_after_available_through": audit["observed_first_sale_date"].gt(cutoff_day),
        "known_pre_source_sale_evidence": audit["known_pre_source_sale_evidence"],
        "source_boundary_origin_unconfirmed": audit["unconfirmed_source_boundary_origin"],
        "incomplete_core_launch_partition_window": ~core_complete,
        "incomplete_26w_distribution_window": ~audit["distribution_window_mature"],
        "invalid_26w_audit_quantity": ~np.isfinite(numeric_sales) | numeric_sales.lt(0),
        "known_unrestorable_inventory_grain": audit["known_unrestorable_stock_grain"],
    }
    audit["candidate_exclusion_reasons"] = _reason_text(audit.index, candidate_checks)
    audit["candidate_launch_eligible"] = audit["candidate_exclusion_reasons"].eq("")

    audit["decision_date"] = audit["release_dt"] - pd.DateOffset(months=lead_months)
    minimum_weeks = int(parameters["comparable"]["minimum_mature_weeks"])
    window_specs = [(key, minimum_weeks * 7, True) for key in CORE_HISTORY]
    window_specs += [("social", int(parameters["factors"]["social_prelaunch_days"]), False)]
    window_specs += [(key, 90, False) for key in CORE_HISTORY]
    decision_core_ready = pd.Series(True, index=audit.index)
    for key, days, required in window_specs:
        partitions = tuple(day for day in source_partitions.get(key, ()) if day <= available_through.date())
        start = audit["decision_date"] - pd.Timedelta(days=days - 1)
        count, expected = _window_counts(partitions, start, audit["decision_date"])
        complete = pd.Series((expected == days) & (count == expected), index=audit.index)
        role = "decision_core_history" if required else ("decision_social_history" if key == "social" else "decision_category_history")
        audit[f"{role}_{key}_complete"] = complete
        if required:
            decision_core_ready &= complete
        detail = audit[["sku_code"]].copy()
        detail["window_role"] = role
        detail["source_key"] = key
        detail["window_start"] = start
        detail["window_end"] = audit["decision_date"]
        detail["expected_partition_days"] = expected
        detail["observed_partition_days"] = count
        detail["window_complete"] = complete
        detail["required_for_this_gate"] = required
        details.append(detail)
    store_partitions = np.asarray(sorted({d.toordinal() for d in source_partitions.get("store", ()) if d <= available_through.date()}), dtype=np.int64)
    decision_ordinals = audit["decision_date"].map(lambda d: d.date().toordinal() if pd.notna(d) else -1).to_numpy()
    audit["store_snapshot_visible_at_decision"] = np.searchsorted(store_partitions, decision_ordinals, side="right") > 0
    verified_dates = audit.loc[audit["candidate_launch_eligible"], "first_actual_sale_date"].sort_values()
    eligible_ordinals = verified_dates.map(lambda d: d.date().toordinal()).to_numpy(dtype=np.int64)
    audit["mature_internal_candidate_count_at_decision"] = np.searchsorted(
        eligible_ordinals, decision_ordinals - (minimum_weeks * 7 - 1), side="right"
    )
    target_checks = {
        "candidate_launch_quality_gate_failed": ~audit["candidate_launch_eligible"],
        "missing_release_dt": audit["release_dt"].isna(),
        "decision_before_source_start": audit["decision_date"].lt(pd.Timestamp(source_start_date)),
        "decision_not_before_first_sale": audit["decision_date"].ge(audit["first_actual_sale_date"]),
        "decision_after_available_through": audit["decision_date"].gt(cutoff_day),
        "incomplete_core_predecision_history": ~decision_core_ready,
        "missing_predecision_store_snapshot": ~audit["store_snapshot_visible_at_decision"],
        "known_target_label_quality_rejected": audit["known_target_label_quality_rejected"],
    }
    audit["target_exclusion_reasons"] = _reason_text(audit.index, target_checks)
    audit["prelaunch_target_preflight_eligible"] = audit["target_exclusion_reasons"].eq("")
    audit["training_ready"] = False

    metadata = {
        "policy": "sales-observed-launch-universe-v2", "source_start_date": source_start_date.isoformat(),
        "available_through": available_through.isoformat(), "lead_months": lead_months,
        "train_ratio": train_ratio, "distribution_window_weeks": distribution_window_weeks,
        "launch_window_days": launch_days, "minimum_mature_weeks": minimum_weeks,
        "audit_sha256": frame_hash(audit.sort_values("sku_code").reset_index(drop=True)),
        "source_partition_hash": json_hash({key: sorted(set(days)) for key, days in source_partitions.items()}),
    }
    version = f"v3.2-launch-{available_through.date().isoformat()}-{json_hash(metadata)[:16]}"
    audit["dataset_version"] = version
    audit["launch_universe_version"] = version
    candidates = audit.loc[audit["candidate_launch_eligible"]].copy()
    targets = audit.loc[audit["prelaunch_target_preflight_eligible"]].copy()
    split_boundary: date | None = None
    if targets["first_actual_sale_date"].nunique() >= 2:
        _, _, split_boundary = chronological_ratio_split(
            [SkuSample(row.first_actual_sale_date.date(), str(row.sku_code)) for row in targets.itertuples(index=False)],
            train_ratio=train_ratio,
        )
    for frame in (audit, candidates, targets):
        frame["split"] = np.where(
            frame["first_actual_sale_date"].le(pd.Timestamp(split_boundary)) if split_boundary else pd.Series(True, index=frame.index),
            "Train", "Test",
        )
        frame["split_status"] = "provisional_until_row_and_restored_label_validation"
        if split_boundary:
            frame["temporal_split_boundary_date"] = split_boundary.isoformat()
    training_sales = {str(row.sku_code): float(row.sales_26w_qty) for row in candidates.loc[candidates["split"].eq("Train")].itertuples(index=False)}
    weights = {row.sku_code: row.weight for row in long_tail_weights(training_sales)} if training_sales else {}
    for frame in (audit, candidates, targets):
        frame["long_tail_weight"] = frame["sku_code"].map(weights)
        frame["weight_source"] = "provisional_candidate_Train_only; refit_after_final_eligibility"
    for frame in (audit, candidates, targets):
        for key in ("first_actual_sale_date", "product_first_sale_date", "observed_first_sale_date", "release_dt", "source_start_date", "source_left_boundary_date", "distribution_window_end", "launch_window_end", "decision_date"):
            frame[key] = frame[key].dt.strftime("%Y-%m-%d")
    validate_launch_origin_contract(candidates, name="verified candidate cohort")
    # Empty schedules still have headers so exclusion reports survive a zero-SKU result.
    schedule = build_prelaunch_decision_schedule(
        targets, lead_months=(lead_months,), available_through=available_through,
    )
    schedule["all_audited_sources_available"] = True
    schedule["audited_source_count"] = len(CORE_HISTORY) + 1
    schedule["available_source_count"] = len(CORE_HISTORY) + 1
    schedule["eligibility_scope"] = "launch_and_partition_preflight_only; requires_row_and_restored_label_validation"
    window_detail = pd.concat(details, ignore_index=True)
    for key in ("window_start", "window_end"):
        window_detail[key] = window_detail[key].dt.strftime("%Y-%m-%d")
    reasons = []
    for scope, column in (("candidate", "candidate_exclusion_reasons"), ("target", "target_exclusion_reasons")):
        counts = audit[column].str.split("|").explode()
        reasons.extend({"scope": scope, "reason": reason, "sku_count": int(count)} for reason, count in counts.loc[counts.ne("")].value_counts().items())
    report = {
        "spec_version": "3.2", "stage": "verified launch universe preflight",
        "status": "preflight_complete_requires_row_and_restored_label_validation" if not targets.empty else "no_eligible_targets_review_exclusions",
        "dataset_version": version, "policy": metadata,
        "input_skus": len(audit), "verified_candidate_skus": len(candidates),
        "target_preflight_skus": len(targets), "checkpoint_rows": len(schedule),
        "target_split_counts_provisional": targets["split"].value_counts().to_dict(),
        "temporal_split_boundary_date_provisional": split_boundary.isoformat() if split_boundary else None,
        "split_status": "provisional_requires_final_quality_resplit" if split_boundary else "insufficient_distinct_target_dates_for_train_test_split",
        "train_ratio_requested": train_ratio,
        "train_ratio_provisional": float(targets["split"].eq("Train").mean()) if len(targets) else None,
        "same_first_sale_day_split": False,
        "target_distinct_first_sale_dates": int(targets["first_actual_sale_date"].nunique()),
        "provisional_train_distinct_first_sale_dates": int(targets.loc[targets["split"].eq("Train"), "first_actual_sale_date"].nunique()),
        "target_candidate_counts_at_decision": {
            "minimum": int(targets["mature_internal_candidate_count_at_decision"].min()) if len(targets) else None,
            "median": float(targets["mature_internal_candidate_count_at_decision"].median()) if len(targets) else None,
            "zero_candidate_targets": int(targets["mature_internal_candidate_count_at_decision"].eq(0).sum()),
            "fewer_than_top_k_targets": int(targets["mature_internal_candidate_count_at_decision"].lt(int(parameters["comparable"]["top_k_internal"])).sum()),
            "count_role": "mature verified universe size only; not same-IP/Path matching or confirmed usable Top-K count",
        },
        "origin_policy": "sales-observed first positive sale defines lifecycle and label windows; product first-sale missingness/date differences are audit-only; independently evidenced pre-source sales and unconfirmed source-boundary origins remain excluded",
        "origin_evidence_counts_candidates": candidates["launch_origin_status"].value_counts().to_dict(),
        "origin_evidence_counts_targets": targets["launch_origin_status"].value_counts().to_dict(),
        "product_first_sale_required": False,
        "product_observed_date_equality_required": False,
        "independently_verified_origin_claim_for_all_rows": False,
        "candidate_policy": "same origin/source-start/launch-window gates; at each checkpoint filter again by mature weeks before cutoff",
        "required_window_policy": "core source launch window and predecision minimum mature-history window must have all daily partitions; row coverage remains unverified",
        "optional_window_policy": "social 30d and category 90d are diagnostics; missing features follow V3.2 missing-value rules, not automatic SKU exclusion",
        "prior_label_policy": "only same-origin negative quality evidence is used; no old D_first_ref is installed as a new training label",
        "formal_label": "requires fresh validated restored_natural_sales inside SKU-specific W_cover; 26w quantity remains distribution/weight audit only",
        "raw_fact_rows_scanned": 0, "training_started": False, "training_ready": False,
        "test_consumed": False, "dates_overridden": False,
    }
    return VerifiedLaunchUniverse(audit, candidates, schedule, window_detail,
                                  pd.DataFrame(reasons, columns=["scope", "reason", "sku_count"]), report)
