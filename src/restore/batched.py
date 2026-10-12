from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from time import monotonic
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from config.settings import RestorationConfig
from data.domestic import DataContractViolation
from data.restoration_adapter import RestorationDataAdapter, RestorationSourceBatch
from restore.historical import (
    RestorationStats,
    _aggregate_weekly,
    aggregate_store_tiers,
    apply_period_factors,
    apply_restoration_model,
    assign_event_codes,
    build_daily_panel,
    build_restoration_quality_report,
    fit_restoration_model,
    neutralize_period_effects,
    normalize_order_lines,
    period_effect_evidence,
    prepare_sales_daily,
    prepare_weekly_for_model,
)


PREPROCESS_VERSION = "m2-pre-model-v1"
ProgressCallback = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class BatchRecord:
    index: int
    count: int
    selected: pd.DataFrame
    source_dir: Path
    processed_dir: Path
    order_stats: dict[str, int]


@dataclass(frozen=True)
class BatchedRestorationOutput:
    paths: dict[str, Path]
    stats: RestorationStats
    model_report: dict[str, Any]
    quality_report: dict[str, Any]
    dataset_version: str
    fit_weight_mean: float
    reused_pre_model_batches: int
    computed_pre_model_batches: int


def _emit(callback: ProgressCallback | None, event: str, **details: Any) -> None:
    if callback is not None:
        callback(event, details)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _preprocess_fingerprint(
    *,
    thresholds: Mapping[str, float],
    config: RestorationConfig,
    cutoff_time: datetime,
) -> str:
    payload = {
        "version": PREPROCESS_VERSION,
        "cutoff_time": cutoff_time.isoformat(),
        "config": asdict(config),
        "thresholds": {
            str(key): None if not np.isfinite(value) else float(value)
            for key, value in sorted(thresholds.items())
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _normalize_batch_sales(
    batch: RestorationSourceBatch,
    *,
    cutoff_time: datetime,
    resume: bool,
    progress: ProgressCallback | None,
) -> BatchRecord:
    if batch.checkpoint_batch_dir is None:
        raise ValueError("batched restoration requires checkpoint_dir")
    processed = batch.checkpoint_batch_dir / "processed"
    manifest_path = processed / "normalized_sales.json"
    sales_path = processed / "latest_sales.parquet"
    stats: dict[str, int] | None = None
    if resume and manifest_path.is_file() and sales_path.is_file():
        try:
            manifest = _read_json(manifest_path)
            if (
                manifest.get("version") == PREPROCESS_VERSION
                and int(manifest.get("rows", -1))
                == len(pd.read_parquet(sales_path, columns=["sku_code"]))
            ):
                stats = {str(k): int(v) for k, v in manifest["order_stats"].items()}
                _emit(
                    progress,
                    "normalized_sales_checkpoint_reused",
                    batch=batch.batch_index,
                    batches=batch.batch_count,
                    rows=manifest["rows"],
                )
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            stats = None
    if stats is None:
        started = monotonic()
        latest, stats = normalize_order_lines(
            batch.frames["sales"], cutoff_time=cutoff_time
        )
        _write_parquet_atomic(sales_path, latest)
        _write_json_atomic(
            manifest_path,
            {
                "version": PREPROCESS_VERSION,
                "rows": len(latest),
                "order_stats": stats,
            },
        )
        _emit(
            progress,
            "normalized_sales_checkpoint_written",
            batch=batch.batch_index,
            batches=batch.batch_count,
            rows=len(latest),
            elapsed_seconds=round(monotonic() - started, 3),
        )
    return BatchRecord(
        index=batch.batch_index,
        count=batch.batch_count,
        selected=batch.selected.copy(),
        source_dir=batch.checkpoint_batch_dir,
        processed_dir=processed,
        order_stats=stats,
    )


def _abnormal_thresholds(
    records: list[BatchRecord],
    *,
    products: pd.DataFrame,
    fit_skus: set[str],
    quantile: float,
) -> dict[str, float]:
    attributes = products[["sku_code", "cate_level4_code"]].copy()
    attributes["sku_code"] = attributes["sku_code"].astype(str)
    attributes["cate_level4_code"] = (
        attributes["cate_level4_code"].fillna("__UNKNOWN__").astype(str)
    )
    category_by_sku = attributes.set_index("sku_code")["cate_level4_code"]
    values: dict[str, list[np.ndarray]] = {}
    for record in records:
        frame = pd.read_parquet(
            record.processed_dir / "latest_sales.parquet",
            columns=["sku_code", "sale_qty"],
        )
        if frame.empty:
            continue
        frame["sku_code"] = frame["sku_code"].astype(str)
        selected = frame["sale_qty"].gt(0) & frame["sku_code"].isin(fit_skus)
        frame = frame.loc[selected]
        if frame.empty:
            continue
        categories = frame["sku_code"].map(category_by_sku)
        if categories.isna().any():
            missing = frame.loc[categories.isna(), "sku_code"].head(5).tolist()
            raise DataContractViolation(
                f"normalized sales SKUs have no category attributes: {missing}"
            )
        for category, group in frame.groupby(categories, sort=False):
            values.setdefault(str(category), []).append(
                group["sale_qty"].to_numpy(float, copy=True)
            )
    category_arrays = {
        category: np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        for category, chunks in values.items()
    }
    all_values = (
        np.concatenate(list(category_arrays.values()))
        if category_arrays
        else np.array([], dtype=float)
    )
    global_threshold = (
        float(pd.Series(all_values).quantile(quantile))
        if len(all_values) >= 20
        else float("inf")
    )
    thresholds: dict[str, float] = {}
    for category in attributes["cate_level4_code"].unique():
        category_values = category_arrays.get(str(category), np.array([], dtype=float))
        thresholds[str(category)] = (
            float(pd.Series(category_values).quantile(quantile))
            if len(category_values) >= 20
            else global_threshold
        )
    return thresholds


def _prepare_batch(
    record: BatchRecord,
    *,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    fit_skus: set[str],
    training_weights: Mapping[str, float],
    cutoff_time: datetime,
    window_days: int,
    config: RestorationConfig,
    thresholds: Mapping[str, float],
    fingerprint: str,
    resume: bool,
    progress: ProgressCallback | None,
) -> tuple[bool, dict[str, Any]]:
    manifest_path = record.processed_dir / "pre_model.json"
    daily_path = record.processed_dir / "daily_panel.parquet"
    weekly_path = record.processed_dir / "weekly_pre_model.parquet"
    if (
        resume
        and manifest_path.is_file()
        and daily_path.is_file()
        and weekly_path.is_file()
    ):
        try:
            manifest = _read_json(manifest_path)
            if manifest.get("fingerprint") == fingerprint:
                _emit(
                    progress,
                    "pre_model_checkpoint_reused",
                    batch=record.index,
                    batches=record.count,
                    daily_rows=manifest["daily_rows"],
                    weekly_rows=manifest["weekly_rows"],
                )
                return True, manifest
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            pass
    started = monotonic()
    selected_skus = set(record.selected["sku_code"].astype(str))
    latest = pd.read_parquet(record.processed_dir / "latest_sales.parquet")
    inventory = pd.read_parquet(record.source_dir / "inventory.parquet")
    placement = pd.read_parquet(record.source_dir / "placement.parquet")
    product = products.loc[
        products["sku_code"].astype(str).isin(selected_skus)
    ].copy()
    sales_daily, anomaly = prepare_sales_daily(
        latest,
        product,
        abnormal_order_quantile=config.abnormal_order_quantile,
        threshold_fit_skus=fit_skus,
        abnormal_thresholds=thresholds,
    )
    first_sale_dates = dict(
        zip(record.selected["sku_code"].astype(str), record.selected["first_actual_sale_date"])
    )
    daily = build_daily_panel(
        sales_daily=sales_daily,
        inventory=inventory,
        placement=placement,
        stores=stores,
        first_sale_dates=first_sale_dates,
        cutoff_date=cutoff_time.date(),
        window_days=window_days,
        promotion_price_ratio_threshold=config.promotion_price_ratio_threshold,
    )
    weekly = _aggregate_weekly(daily, config)
    weekly, _ = prepare_weekly_for_model(
        weekly,
        fit_skus=fit_skus,
        training_weights=training_weights,
        fit_weight_mean=1.0,
    )
    _write_parquet_atomic(daily_path, daily)
    _write_parquet_atomic(weekly_path, weekly)
    manifest = {
        "version": PREPROCESS_VERSION,
        "fingerprint": fingerprint,
        "daily_rows": len(daily),
        "weekly_rows": len(weekly),
        "abnormal_order_lines": int(anomaly["abnormal_order_lines"]),
        "skus": sorted(selected_skus),
    }
    _write_json_atomic(manifest_path, manifest)
    _emit(
        progress,
        "pre_model_checkpoint_written",
        batch=record.index,
        batches=record.count,
        daily_rows=len(daily),
        weekly_rows=len(weekly),
        elapsed_seconds=round(monotonic() - started, 3),
    )
    return False, manifest


def _combine_flag_audits(reports: list[dict[str, Any]]) -> dict[str, Any]:
    labels = reports[0]["flag_normalization"] if reports else {}
    combined: dict[str, Any] = {}
    for label in labels:
        numeric = {
            key: sum(int(report["flag_normalization"][label][key]) for report in reports)
            for key in (
                "rows", "true_rows", "false_rows", "missing_rows", "unrecognized_rows"
            )
        }
        raw_counts: dict[str, int] = {}
        bad_counts: dict[str, int] = {}
        for report in reports:
            audit = report["flag_normalization"][label]
            for key, value in audit["raw_value_counts"].items():
                raw_counts[str(key)] = raw_counts.get(str(key), 0) + int(value)
            for key, value in audit["unrecognized_value_counts"].items():
                bad_counts[str(key)] = bad_counts.get(str(key), 0) + int(value)
        numeric["raw_value_counts"] = dict(
            sorted(raw_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
        )
        numeric["unrecognized_value_counts"] = dict(
            sorted(bad_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
        )
        combined[label] = numeric
    return combined


def _combine_quality_reports(
    reports: list[dict[str, Any]],
    *,
    requested_fit_skus: set[str],
    model_report: Mapping[str, Any],
) -> dict[str, Any]:
    flag_audits = _combine_flag_audits(reports)
    exposure_keys = (
        "component_unknown_days",
        "open_days",
        "unknown_exposure_days",
        "stockout_days",
        "positive_sales_on_stockout_days",
        "positive_sales_qty_on_stockout_days",
        "stock_state_conflict_days",
    )
    exposure = {
        key: sum(report["exposure"][key] for report in reports)
        for key in exposure_keys
    }
    daily_rows = sum(int(report["flag_normalization"]["inventory_out_of_stock"]["rows"]) for report in reports)
    exposure["unknown_exposure_rate"] = (
        exposure["unknown_exposure_days"] / exposure["open_days"]
        if exposure["open_days"]
        else 0.0
    )
    exposure["stockout_rate"] = (
        exposure["stockout_days"] / daily_rows if daily_rows else 0.0
    )
    per_sku = [row for report in reports for row in report["per_sku"]]
    all_stockout = sorted(
        str(row["sku_code"])
        for row in per_sku
        if bool(row["all_open_days_stockout"])
    )
    exposure["all_stockout_skus"] = all_stockout
    fitted_count = int(model_report.get("fit_sku_count", 0))
    weekly_rows = sum(int(report["model_gate"]["weekly_rows"]) for report in reports)
    near_zero = sum(int(report["model_gate"]["near_zero_model_rows"]) for report in reports)
    restored_total = sum(float(report["model_gate"]["restored_lost_sales_total"]) for report in reports)
    warnings: list[str] = []
    for label, audit in flag_audits.items():
        if int(audit["unrecognized_rows"]) > 0:
            warnings.append(
                f"{label} has {audit['unrecognized_rows']} non-null unmapped values; "
                "they remain unknown and are listed in data_quality.flag_normalization"
            )
    if exposure["stock_state_conflict_days"]:
        warnings.append(
            f"{exposure['stock_state_conflict_days']} rows contain contradictory stock evidence; "
            "Estock remains unknown for those rows"
        )
    if exposure["unknown_exposure_rate"] > 0.05:
        warnings.append(
            f"unknown exposure rate is {exposure['unknown_exposure_rate']:.2%}, above the 5% review threshold"
        )
    if all_stockout:
        warnings.append(
            "all open days are stockout for selected SKUs: " + ", ".join(all_stockout)
        )
    if fitted_count < len(requested_fit_skus):
        warnings.append(
            f"only {fitted_count} of {len(requested_fit_skus)} requested Train SKUs "
            "have eligible exposure rows for model fitting"
        )
    if fitted_count < 2:
        warnings.append(
            "fewer than two Train SKUs entered model fitting; treat this as a bounded "
            "pipeline check rather than production model validation"
        )
    if weekly_rows and near_zero == weekly_rows:
        warnings.append(
            "all v_model predictions are effectively zero; this run validates plumbing "
            "but not lost-sales restoration"
        )
    return {
        "status": "warning" if warnings else "ok",
        "policy": reports[0]["policy"] if reports else {},
        "warnings": warnings,
        "flag_normalization": flag_audits,
        "exposure": exposure,
        "model_gate": {
            "requested_train_sku_count": len(requested_fit_skus),
            "fitted_train_sku_count": fitted_count,
            "minimum_fit_sku_review_threshold": 2,
            "near_zero_model_rows": near_zero,
            "weekly_rows": weekly_rows,
            "restored_lost_sales_total": restored_total,
        },
        "per_sku": per_sku,
    }


def _append_csv(path: Path, frame: pd.DataFrame, *, first: bool) -> None:
    frame.to_csv(path, mode="w" if first else "a", header=first, index=False)


def run_batched_restoration(
    *,
    adapter: RestorationDataAdapter,
    selected: pd.DataFrame,
    cutoff_time: datetime,
    window_days: int,
    batch_size: int,
    checkpoint_dir: Path,
    output_dir: Path,
    config: RestorationConfig,
    events: pd.DataFrame | None,
    resume: bool,
    progress: ProgressCallback | None = None,
) -> tuple[BatchedRestorationOutput, Any]:
    """Run M2 without ever concatenating all raw or daily source frames."""

    records: list[BatchRecord] = []

    def normalize_callback(batch: RestorationSourceBatch) -> None:
        records.append(
            _normalize_batch_sales(
                batch,
                cutoff_time=cutoff_time,
                resume=resume,
                progress=progress,
            )
        )

    source_bundle = adapter.load_batched(
        selected=selected,
        cutoff_time=cutoff_time,
        window_days=window_days,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        progress=progress,
        batch_callback=normalize_callback,
        materialize_sources=False,
    )
    records.sort(key=lambda record: record.index)
    fit_skus = set(selected.loc[selected["split"] == "Train", "sku_code"].astype(str))
    training_weights = dict(
        zip(selected["sku_code"].astype(str), selected["long_tail_weight"].astype(float))
    )
    thresholds = _abnormal_thresholds(
        records,
        products=source_bundle.product_attributes,
        fit_skus=fit_skus,
        quantile=config.abnormal_order_quantile,
    )
    pre_fingerprint = _preprocess_fingerprint(
        thresholds=thresholds, config=config, cutoff_time=cutoff_time
    )
    reused_pre = 0
    computed_pre = 0
    manifests: list[dict[str, Any]] = []
    for record in records:
        reused, manifest = _prepare_batch(
            record,
            products=source_bundle.product_attributes,
            stores=source_bundle.stores,
            fit_skus=fit_skus,
            training_weights=training_weights,
            cutoff_time=cutoff_time,
            window_days=window_days,
            config=config,
            thresholds=thresholds,
            fingerprint=pre_fingerprint,
            resume=resume,
            progress=progress,
        )
        reused_pre += int(reused)
        computed_pre += int(not reused)
        manifests.append(manifest)

    fit_parts: list[pd.DataFrame] = []
    fit_columns = [
        "sku_code",
        "store_code",
        "cate_level4_code",
        "v_observed_raw",
        "mean_price_log_ratio",
        "fit_weight",
        "model_fit_eligible",
    ]
    for record in records:
        part = pd.read_parquet(
            record.processed_dir / "weekly_pre_model.parquet",
            columns=fit_columns,
        )
        part = part.loc[part["model_fit_eligible"].fillna(False)].copy()
        if not part.empty:
            fit_parts.append(part)
    if not fit_parts:
        raise DataContractViolation(
            "M2 has no clean Train exposure rows for the Huber fallback model"
        )
    fit_rows = pd.concat(fit_parts, ignore_index=True)
    fit_weight_mean = float(fit_rows["fit_weight"].mean())
    if not np.isfinite(fit_weight_mean) or fit_weight_mean <= 0:
        raise DataContractViolation("M2 fit weights have no positive finite mean")
    fit_rows["fit_weight"] /= fit_weight_mean
    _emit(
        progress,
        "model_fit_start",
        fit_rows=len(fit_rows),
        fit_skus=fit_rows["sku_code"].nunique(),
    )
    started = monotonic()
    fitted = fit_restoration_model(fit_rows, config=config)
    _emit(
        progress,
        "model_fit_complete",
        elapsed_seconds=round(monotonic() - started, 3),
    )
    del fit_parts, fit_rows

    modeled_paths: list[Path] = []
    evidence_parts: list[pd.DataFrame] = []
    for record in records:
        weekly = pd.read_parquet(record.processed_dir / "weekly_pre_model.parquet")
        eligible = weekly["model_fit_eligible"].fillna(False)
        weekly.loc[eligible, "fit_weight"] /= fit_weight_mean
        weekly = apply_restoration_model(weekly, fitted=fitted, config=config)
        weekly = assign_event_codes(weekly, events)
        modeled_path = record.processed_dir / "weekly_modeled.parquet"
        _write_parquet_atomic(modeled_path, weekly)
        modeled_paths.append(modeled_path)
        if events is not None and not events.empty:
            evidence = period_effect_evidence(weekly)
            if not evidence.empty:
                evidence_parts.append(evidence)
    if evidence_parts:
        period_factors = (
            pd.concat(evidence_parts, ignore_index=True)
            .groupby(["cate_level4_code", "event_code"])["event_ratio"]
            .median()
        )
    else:
        period_factors = pd.Series(dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_dir / "round4_report.json",
        "daily": output_dir / "daily_exposure_audit.csv",
        "weekly": output_dir / "restored_store_weekly.csv",
        "tier_weekly": output_dir / "restored_tier_weekly.csv",
    }
    temporary_paths = {
        key: path.with_suffix(path.suffix + ".tmp")
        for key, path in paths.items()
        if key != "report"
    }
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)
    part_roots = {
        "daily": output_dir / "daily_exposure_audit_parts.tmp",
        "weekly": output_dir / "restored_store_weekly_parts.tmp",
        "tier_weekly": output_dir / "restored_tier_weekly_parts.tmp",
    }
    for root in part_roots.values():
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

    quality_parts: list[dict[str, Any]] = []
    weekly_total = 0.0
    tier_rows = 0
    unknown_days = 0
    stockout_days = 0
    nondisplay_days = 0
    event_rows = 0
    first_output = True
    for record, modeled_path in zip(records, modeled_paths):
        started = monotonic()
        daily = pd.read_parquet(record.processed_dir / "daily_panel.parquet")
        weekly = pd.read_parquet(modeled_path)
        weekly = (
            apply_period_factors(weekly, period_factors)
            if len(period_factors)
            else neutralize_period_effects(weekly)
        )
        tier = aggregate_store_tiers(weekly)
        local_fit_count = int(
            weekly.loc[weekly["model_fit_eligible"], "sku_code"].nunique()
        )
        quality_parts.append(
            build_restoration_quality_report(
                daily,
                weekly,
                requested_fit_skus=set(record.selected["sku_code"].astype(str)),
                model_report={"fit_sku_count": local_fit_count},
            )
        )
        _append_csv(temporary_paths["daily"], daily, first=first_output)
        _append_csv(temporary_paths["weekly"], weekly, first=first_output)
        _append_csv(temporary_paths["tier_weekly"], tier, first=first_output)
        suffix = f"part-{record.index:05d}.parquet"
        daily.to_parquet(part_roots["daily"] / suffix, index=False)
        weekly.to_parquet(part_roots["weekly"] / suffix, index=False)
        tier.to_parquet(part_roots["tier_weekly"] / suffix, index=False)
        weekly_total += float(weekly["restored_natural_sales"].sum())
        tier_rows += len(tier)
        unknown_days += int(weekly["unknown_exposure_days"].sum())
        stockout_days += int(weekly["stockout_days"].sum())
        nondisplay_days += int(weekly["nondisplay_days"].sum())
        event_rows += int((weekly["period_factor_status"] == "estimated").sum())
        _emit(
            progress,
            "final_batch_written",
            batch=record.index,
            batches=record.count,
            daily_rows=len(daily),
            weekly_rows=len(weekly),
            tier_weekly_rows=len(tier),
            elapsed_seconds=round(monotonic() - started, 3),
        )
        first_output = False

    if first_output:
        raise DataContractViolation("M2 produced no output batches")
    for key, temporary in temporary_paths.items():
        temporary.replace(paths[key])
    for key, temporary_root in part_roots.items():
        final_root = Path(str(temporary_root).removesuffix(".tmp"))
        if final_root.exists():
            shutil.rmtree(final_root)
        temporary_root.replace(final_root)

    model_report = dict(fitted.report)
    model_report["abnormal_order_policy"] = {
        "quantile": config.abnormal_order_quantile,
        "strictly_above_threshold": True,
        "thresholds": {
            key: None if not np.isfinite(value) else value
            for key, value in sorted(thresholds.items())
        },
    }
    quality_report = _combine_quality_reports(
        quality_parts,
        requested_fit_skus=fit_skus,
        model_report=model_report,
    )
    order_stats = {
        key: sum(record.order_stats[key] for record in records)
        for key in (
            "input_order_rows",
            "latest_order_lines",
            "exact_duplicate_rows",
            "signed_aggregation_extra_rows",
        )
    }
    stats = RestorationStats(
        input_order_rows=order_stats["input_order_rows"],
        latest_order_lines=order_stats["latest_order_lines"],
        exact_duplicate_rows=order_stats["exact_duplicate_rows"],
        signed_aggregation_extra_rows=order_stats["signed_aggregation_extra_rows"],
        abnormal_order_lines=sum(int(item["abnormal_order_lines"]) for item in manifests),
        daily_panel_rows=sum(int(item["daily_rows"]) for item in manifests),
        weekly_rows=sum(int(item["weekly_rows"]) for item in manifests),
        tier_weekly_rows=tier_rows,
        unknown_exposure_days=unknown_days,
        stockout_days=stockout_days,
        nondisplay_days=nondisplay_days,
        event_adjusted_rows=event_rows,
    )
    version_payload = {
        "spec_version": "3.2",
        "normalization_version": "source-flags-v2",
        "cutoff_time": cutoff_time.isoformat(),
        "window_days": window_days,
        "config": asdict(config),
        "skus": sorted(str(value) for value in selected["sku_code"]),
        "weekly_rows": stats.weekly_rows,
        "weekly_total": weekly_total,
    }
    digest = hashlib.sha256(
        json.dumps(version_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    output = BatchedRestorationOutput(
        paths=paths,
        stats=stats,
        model_report=model_report,
        quality_report=quality_report,
        dataset_version=f"m2-v3.2-{cutoff_time.date().isoformat()}-{digest[:16]}",
        fit_weight_mean=fit_weight_mean,
        reused_pre_model_batches=reused_pre,
        computed_pre_model_batches=computed_pre,
    )
    return output, source_bundle
