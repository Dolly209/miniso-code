from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config.settings import TableSpec
from data.domestic import (
    INVENTORY_SOURCE,
    PRODUCT_SOURCE,
    SALES_SOURCE,
    InventoryDataService,
    ProductDataService,
    SalesDataService,
)
from data.parquet_repository import ParquetDataRepository
from data.point_in_time import DecisionType, PointInTimeContext


class ValidationEvidenceError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationEvidenceError(f"required probe artifact not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationEvidenceError(f"invalid probe JSON {path}: {exc}") from exc


def validate_probe_evidence(
    *, probe_dir: Path, table_specs: Sequence[TableSpec]
) -> dict[str, Any]:
    tables_payload = _read_json(probe_dir / "tables.json")
    schema_payload = _read_json(probe_dir / "schema_report.json")
    partition_payload = _read_json(probe_dir / "partition_report.json")
    quality_payload = _read_json(probe_dir / "data_quality_summary.json")
    expected = {spec.table_name for spec in table_specs}

    tables = {row["table_name"]: row for row in tables_payload.get("tables", [])}
    schemas = {row["table_name"]: row for row in schema_payload.get("tables", [])}
    partitions = {
        row["table_name"]: row for row in partition_payload.get("tables", [])
    }
    quality = {
        row["table_name"]: row for row in quality_payload.get("tables", [])
    }
    errors: list[str] = []
    warnings: list[str] = []
    for name, payload in (
        ("tables.json", tables_payload),
        ("schema_report.json", schema_payload),
        ("partition_report.json", partition_payload),
        ("data_quality_summary.json", quality_payload),
    ):
        if payload.get("spec_version") != "3.2":
            errors.append(f"{name} spec_version is not 3.2")
    for source_name in sorted(expected):
        table = tables.get(source_name)
        schema = schemas.get(source_name)
        partition = partitions.get(source_name)
        quality_row = quality.get(source_name)
        if not table or not schema or not partition or not quality_row:
            errors.append(f"incomplete probe evidence for {source_name}")
            continue
        if not table.get("exists"):
            errors.append(f"source directory is missing: {source_name}")
        if table.get("detected_format") != "parquet":
            errors.append(
                f"source format is not confirmed Parquet: {source_name}="
                f"{table.get('detected_format')}"
            )
        missing = schema.get("missing_required_fields") or []
        if missing:
            errors.append(f"missing fields in {source_name}: {', '.join(missing)}")
        if schema.get("errors"):
            warnings.append(f"schema inspection reported errors for {source_name}")
        if not partition.get("max_ds"):
            errors.append(f"no usable ds partition in {source_name}")
        if quality_row.get("status") != "ok":
            warnings.append(
                f"bounded quality sample status for {source_name}: "
                f"{quality_row.get('status')}"
            )
        all_null = sorted(
            name
            for name, rate in (quality_row.get("critical_field_missing_rate") or {}).items()
            if rate == 1.0
        )
        if all_null:
            warnings.append(
                f"all-null fields in bounded sample for {source_name}: "
                + ", ".join(all_null)
            )

    extras = sorted(set(tables) - expected)
    if extras:
        warnings.append("unconfigured probe tables: " + ", ".join(extras))
    return {
        "status": "ok" if not errors else "failed",
        "spec_version": tables_payload.get("spec_version"),
        "expected_source_count": len(expected),
        "errors": errors,
        "warnings": warnings,
    }


def validate_parquet_reads(
    *,
    repository: ParquetDataRepository,
    table_specs: Sequence[TableSpec],
    cutoff_time: datetime,
    sample_rows: int,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for spec in table_specs:
        partitions = repository.available_partitions(spec.table_name)
        selected = repository.latest_partition(spec.table_name, cutoff_time)
        projection = tuple(
            dict.fromkeys(
                [
                    spec.partition_column,
                    *spec.grain_columns,
                    spec.business_date_column,
                    *(
                        [spec.sku_column]
                        if spec.sku_column is not None
                        else []
                    ),
                    *(
                        [spec.store_column]
                        if spec.store_column is not None
                        else []
                    ),
                ]
            )
        )
        sampled = repository.sample_latest_partition(
            source_name=spec.table_name,
            cutoff_time=cutoff_time,
            columns=projection,
            sample_rows=sample_rows,
        )
        reports.append(
            {
                "table_name": spec.table_name,
                "read_mode": spec.read_mode,
                "selected_partition": selected.isoformat(),
                "partition_min": partitions[0].isoformat(),
                "partition_max": partitions[-1].isoformat(),
                "freshness_days_at_cutoff": (cutoff_time.date() - selected).days,
                "projected_columns": list(sampled.projected_columns),
                "sample_rows": sampled.row_count,
                "status": "ok" if sampled.row_count else "empty_latest_partition",
                "bounded_sample": True,
            }
        )
    return reports


def _sample_sku(
    repository: ParquetDataRepository,
    *,
    source_name: str,
    cutoff_time: datetime,
    sample_rows: int,
) -> tuple[str | None, date]:
    sampled = repository.sample_latest_partition(
        source_name=source_name,
        cutoff_time=cutoff_time,
        columns=("sku_code",),
        sample_rows=sample_rows,
    )
    selected = sampled.selected_partitions[0]
    values = sampled.frame["sku_code"].dropna().astype(str)
    return (None if values.empty else values.iloc[0], selected)


def validate_domain_contracts(
    *,
    repository: ParquetDataRepository,
    cutoff_time: datetime,
    sample_rows: int,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []

    product_sku, _ = _sample_sku(
        repository,
        source_name=PRODUCT_SOURCE,
        cutoff_time=cutoff_time,
        sample_rows=sample_rows,
    )
    if product_sku is None:
        reports.append({"contract": "product_multi_ip", "status": "skipped_no_sku"})
    else:
        context = PointInTimeContext(
            run_id="ROUND2-VALIDATION-PRODUCT",
            sku_code=product_sku,
            decision_type=DecisionType.FIRST_ORDER,
            decision_cutoff_time=cutoff_time,
        )
        snapshot = ProductDataService(repository).read_snapshot(context)
        membership = ProductDataService.ip_membership(snapshot)
        attributes = ProductDataService.one_row_per_sku_attributes(snapshot)
        reports.append(
            {
                "contract": "product_multi_ip",
                "status": "ok",
                "sample_sku": product_sku,
                "source_rows": snapshot.row_count,
                "ip_membership_rows": len(membership),
                "sku_attribute_rows": len(attributes),
            }
        )

    sales_sku, sales_day = _sample_sku(
        repository,
        source_name=SALES_SOURCE,
        cutoff_time=cutoff_time,
        sample_rows=sample_rows,
    )
    if sales_sku is None:
        reports.append({"contract": "normal_sales", "status": "skipped_no_sku"})
    else:
        context = PointInTimeContext(
            run_id="ROUND2-VALIDATION-SALES",
            sku_code=sales_sku,
            decision_type=DecisionType.FIRST_ORDER,
            decision_cutoff_time=cutoff_time,
        )
        service = SalesDataService(repository)
        lines = service.read_order_lines(context, start_date=sales_day)
        daily = service.aggregate_sku_store_day(lines)
        reports.append(
            {
                "contract": "normal_sales",
                "status": "ok",
                "sample_sku": sales_sku,
                "deduplicated_order_lines": lines.row_count,
                "sku_store_day_rows": len(daily),
                "actual_first_sale_date_in_sample": (
                    service.actual_first_sale_date(lines).isoformat()
                    if service.actual_first_sale_date(lines)
                    else None
                ),
                "refund_subtracted": False,
            }
        )

    inventory_sku, inventory_day = _sample_sku(
        repository,
        source_name=INVENTORY_SOURCE,
        cutoff_time=cutoff_time,
        sample_rows=sample_rows,
    )
    if inventory_sku is None:
        reports.append({"contract": "inventory_snapshot", "status": "skipped_no_sku"})
    else:
        context = PointInTimeContext(
            run_id="ROUND2-VALIDATION-INVENTORY",
            sku_code=inventory_sku,
            decision_type=DecisionType.FIRST_ORDER,
            decision_cutoff_time=cutoff_time,
        )
        snapshots = InventoryDataService(repository).read_daily_snapshots(
            context, start_date=inventory_day
        )
        reports.append(
            {
                "contract": "inventory_snapshot",
                "status": "ok",
                "sample_sku": inventory_sku,
                "sku_store_day_rows": snapshots.row_count,
                "total_plus_transit_computed": False,
            }
        )
    return reports


def build_validation_report(
    *,
    repository: ParquetDataRepository,
    table_specs: Sequence[TableSpec],
    probe_dir: Path,
    cutoff_time: datetime,
    sample_rows: int,
) -> dict[str, Any]:
    if cutoff_time.tzinfo is None or cutoff_time.utcoffset() is None:
        raise ValidationEvidenceError("cutoff_time must be timezone-aware")
    if sample_rows <= 0:
        raise ValidationEvidenceError("sample_rows must be positive")
    evidence = validate_probe_evidence(probe_dir=probe_dir, table_specs=table_specs)
    not_validated_here = [
        "1178/955/223 cohort reproduction",
        "B1-B5 rolling blocks",
        "long-tail sales bands",
        "M1-M7 and N1-N6 outputs",
        "MySQL connectivity or table creation",
    ]
    if evidence["status"] != "ok":
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "spec_version": "3.2",
            "status": "failed",
            "scope": "ROUND 2 bounded read validation; not cohort/model validation",
            "decision_cutoff_time": cutoff_time.isoformat(),
            "point_in_time_policy": {
                "partition_rule": "ds <= decision_cutoff_date",
                "availability_resolution": (
                    "day-level proxy because publish timestamp is absent"
                ),
                "future_event_rule": (
                    "event timestamps are filtered again by domain services"
                ),
            },
            "probe_evidence": evidence,
            "source_reads": [],
            "domain_contracts": [],
            "warnings": [
                "Probe evidence is invalid; source reads were not attempted. "
                "Rerun server_probe.py in the same Python environment used for validation."
            ],
            "not_validated_here": not_validated_here,
        }
    source_reads = validate_parquet_reads(
        repository=repository,
        table_specs=table_specs,
        cutoff_time=cutoff_time,
        sample_rows=sample_rows,
    )
    configured_sources = {spec.table_name for spec in table_specs}
    required_domain_sources = {PRODUCT_SOURCE, SALES_SOURCE, INVENTORY_SOURCE}
    if required_domain_sources <= configured_sources:
        domain_contracts = validate_domain_contracts(
            repository=repository,
            cutoff_time=cutoff_time,
            sample_rows=sample_rows,
        )
    else:
        domain_contracts = [
            {
                "contract": "domestic_core_contracts",
                "status": "skipped_incomplete_source_catalog",
            }
        ]
    read_errors = [row for row in source_reads if row["status"] not in {"ok", "empty_latest_partition"}]
    warnings = [
        f"{row['table_name']}: latest eligible partition is empty"
        for row in source_reads
        if row["status"] == "empty_latest_partition"
    ]
    status = "ok" if evidence["status"] == "ok" and not read_errors else "failed"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec_version": "3.2",
        "status": status,
        "scope": "ROUND 2 bounded read validation; not cohort/model validation",
        "decision_cutoff_time": cutoff_time.isoformat(),
        "point_in_time_policy": {
            "partition_rule": "ds <= decision_cutoff_date",
            "availability_resolution": "day-level proxy because publish timestamp is absent",
            "future_event_rule": "event timestamps are filtered again by domain services",
        },
        "probe_evidence": evidence,
        "source_reads": source_reads,
        "domain_contracts": domain_contracts,
        "warnings": warnings,
        "not_validated_here": not_validated_here,
    }


def write_validation_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
