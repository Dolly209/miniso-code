from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config.settings import AppConfig, load_table_specs
from data.decision_adapter import SOURCES
from data.domain import require_columns
from data.parquet_repository import DataLayerError, ParquetDataRepository


DEFAULT_AUDIT_SOURCE_KEYS = (
    "store",
    "sales",
    "inventory",
    "placement",
    "pack",
    "member",
    "supplier",
    "social",
)

SOURCE_ROLES = {
    "product": "snapshot",
    "store": "snapshot",
    "sales": "history",
    "inventory": "history",
    "placement": "history",
    "pack": "snapshot",
    "member": "snapshot",
    "supplier": "snapshot",
    "social": "history",
}


@dataclass(frozen=True)
class PrelaunchPartitionAudit:
    schedule: pd.DataFrame
    source_detail: pd.DataFrame
    source_summary: pd.DataFrame


def _iso_or_blank(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def audit_prelaunch_source_partitions(
    repository: ParquetDataRepository,
    schedule: pd.DataFrame,
    *,
    source_keys: Sequence[str] = DEFAULT_AUDIT_SOURCE_KEYS,
) -> PrelaunchPartitionAudit:
    """Audit source visibility for every independent forecast checkpoint.

    This intentionally reads directory metadata only.  It confirms that a
    partition exists at or before each cutoff; row-level SKU/IP coverage is a
    separate replay check and is not inferred from a directory existing.
    """

    require_columns(
        schedule,
        (
            "prediction_checkpoint_key",
            "sku_code",
            "split",
            "lead_months",
            "decision_date",
            "decision_cutoff_time",
        ),
        "prelaunch decision schedule",
    )
    normalized_keys = tuple(dict.fromkeys(str(value) for value in source_keys))
    unknown = sorted(set(normalized_keys) - set(SOURCES))
    if unknown:
        raise ValueError(f"unknown audit source keys: {', '.join(unknown)}")
    decision_dates = pd.to_datetime(
        schedule["decision_date"], errors="raise"
    ).dt.date
    decision_ordinals = np.asarray([value.toordinal() for value in decision_dates])
    details: list[pd.DataFrame] = []

    for source_key in normalized_keys:
        source_name = SOURCES[source_key]
        source_error = ""
        try:
            partitions = repository.available_partitions(source_name)
        except DataLayerError as exc:
            partitions = ()
            source_error = f"{type(exc).__name__}: {exc}"
        detail = schedule[
            [
                "prediction_checkpoint_key",
                "sku_code",
                "split",
                "lead_months",
                "decision_date",
                "decision_cutoff_time",
            ]
        ].copy()
        detail["source_key"] = source_key
        detail["source_table"] = source_name
        detail["source_role"] = SOURCE_ROLES.get(source_key, "unknown")
        detail["source_error"] = source_error

        if not partitions:
            detail["partition_available"] = False
            detail["selected_partition"] = ""
            detail["source_first_partition"] = ""
            detail["source_last_partition"] = ""
            detail["available_partition_count"] = 0
            detail["history_span_days"] = np.nan
            detail["selected_partition_staleness_days"] = np.nan
            details.append(detail)
            continue

        partition_ordinals = np.asarray(
            [value.toordinal() for value in partitions], dtype=np.int64
        )
        positions = np.searchsorted(
            partition_ordinals, decision_ordinals, side="right"
        ) - 1
        available = positions >= 0
        safe_positions = np.maximum(positions, 0)
        selected_ordinals = partition_ordinals[safe_positions]
        selected_values = [
            date.fromordinal(int(value)).isoformat() if is_available else ""
            for value, is_available in zip(selected_ordinals, available)
        ]
        detail["partition_available"] = available
        detail["selected_partition"] = selected_values
        detail["source_first_partition"] = partitions[0].isoformat()
        detail["source_last_partition"] = partitions[-1].isoformat()
        detail["available_partition_count"] = np.where(
            available, positions + 1, 0
        ).astype(int)
        detail["history_span_days"] = np.where(
            available,
            decision_ordinals - partition_ordinals[0],
            np.nan,
        )
        detail["selected_partition_staleness_days"] = np.where(
            available,
            decision_ordinals - selected_ordinals,
            np.nan,
        )
        details.append(detail)

    source_detail = pd.concat(details, ignore_index=True)

    def joined_missing(group: pd.DataFrame) -> str:
        return "|".join(
            group.loc[~group["partition_available"], "source_key"].astype(str)
        )

    checkpoint_coverage = (
        source_detail.groupby("prediction_checkpoint_key", sort=False)
        .agg(
            audited_source_count=("source_key", "size"),
            available_source_count=("partition_available", "sum"),
            missing_sources=("source_key", lambda _: ""),
            minimum_history_span_days=("history_span_days", "min"),
            maximum_partition_staleness_days=(
                "selected_partition_staleness_days",
                "max",
            ),
        )
        .reset_index()
    )
    missing = (
        source_detail.groupby("prediction_checkpoint_key", sort=False)
        .apply(joined_missing)
        .rename("missing_sources")
        .reset_index()
    )
    checkpoint_coverage = checkpoint_coverage.drop(columns="missing_sources").merge(
        missing,
        on="prediction_checkpoint_key",
        how="left",
        validate="one_to_one",
    )
    checkpoint_coverage["all_audited_sources_available"] = checkpoint_coverage[
        "available_source_count"
    ].eq(checkpoint_coverage["audited_source_count"])
    audited_schedule = schedule.merge(
        checkpoint_coverage,
        on="prediction_checkpoint_key",
        how="left",
        validate="one_to_one",
    )
    audited_schedule["partition_audit_status"] = np.where(
        audited_schedule["all_audited_sources_available"],
        "all_audited_sources_have_visible_partition",
        "one_or_more_sources_have_no_visible_partition",
    )

    source_summary = (
        source_detail.groupby(
            ["lead_months", "split", "source_key", "source_table", "source_role"],
            dropna=False,
            as_index=False,
        )
        .agg(
            prediction_checkpoints=("prediction_checkpoint_key", "size"),
            checkpoints_with_visible_partition=("partition_available", "sum"),
            earliest_decision_date=("decision_date", "min"),
            latest_decision_date=("decision_date", "max"),
            source_first_partition=("source_first_partition", "first"),
            source_last_partition=("source_last_partition", "first"),
            median_history_span_days=("history_span_days", "median"),
            median_partition_staleness_days=(
                "selected_partition_staleness_days",
                "median",
            ),
            source_error=("source_error", "first"),
        )
        .sort_values(["lead_months", "split", "source_key"], kind="stable")
        .reset_index(drop=True)
    )
    source_summary["partition_coverage_rate"] = (
        source_summary["checkpoints_with_visible_partition"]
        / source_summary["prediction_checkpoints"]
    )
    return PrelaunchPartitionAudit(
        schedule=audited_schedule,
        source_detail=source_detail,
        source_summary=source_summary,
    )


class PrelaunchTrainingAuditAdapter:
    """Data-layer entry point for prelaunch historical training audits."""

    def __init__(self, repository: ParquetDataRepository) -> None:
        self.repository = repository

    @classmethod
    def from_project(
        cls, *, project_root: Path, data_root: Path, config: AppConfig
    ) -> "PrelaunchTrainingAuditAdapter":
        return cls(
            ParquetDataRepository(
                data_root=data_root,
                table_specs=load_table_specs(config.probe.table_specs_path),
                max_rows_per_read=config.data_reader.max_rows_per_read,
            )
        )

    def audit(
        self,
        schedule: pd.DataFrame,
        *,
        source_keys: Sequence[str] = DEFAULT_AUDIT_SOURCE_KEYS,
    ) -> PrelaunchPartitionAudit:
        return audit_prelaunch_source_partitions(
            self.repository,
            schedule,
            source_keys=source_keys,
        )
