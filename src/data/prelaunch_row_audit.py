from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import AppConfig, load_table_specs
from data.decision_adapter import SOURCES
from data.domain import require_columns
from data.parquet_repository import DataLayerError, ParquetDataRepository
from data.social_ip_aliases import (
    build_sku_social_aliases,
    normalize_social_ip_name,
)


@dataclass(frozen=True)
class RowSourcePolicy:
    source_key: str
    coverage_grain: str
    expectation: str
    required_for_core_replay: bool


ROW_SOURCE_POLICIES = (
    RowSourcePolicy("store", "global_snapshot_last_value", "required", True),
    RowSourcePolicy("supplier", "target_sku_last_value", "optional", False),
    RowSourcePolicy("member", "target_sku_last_value", "optional", False),
    RowSourcePolicy(
        "sales", "target_sku_predecision_history", "diagnostic_zero_expected", False
    ),
    RowSourcePolicy(
        "inventory", "target_sku_predecision_history", "diagnostic_zero_expected", False
    ),
    RowSourcePolicy(
        "placement", "target_sku_predecision_history", "diagnostic_zero_expected", False
    ),
    RowSourcePolicy("social", "target_ip_recent_history", "optional", False),
)


SOURCE_MINIMUM_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "product": ("ds", "sku_code"),
    "store": ("ds", "store_code"),
    "supplier": ("ds", "sku_code", "supplier_code"),
    "member": ("ds", "sku_code", "stat_month"),
    "sales": ("ds", "sku_code"),
    "inventory": ("ds", "sku_code"),
    "placement": ("ds", "sku_code"),
    "social": ("ds", "date", "ip_name", "platform_code"),
}


@dataclass(frozen=True)
class PrelaunchRowAudit:
    checkpoint_summary: pd.DataFrame
    source_detail: pd.DataFrame
    source_summary: pd.DataFrame
    common_core_skus: pd.DataFrame
    formal_eligible_skus: pd.DataFrame


@dataclass(frozen=True)
class PrelaunchPartitionCandidates:
    sku_audit: pd.DataFrame
    schedule: pd.DataFrame


@dataclass(frozen=True)
class _DailyIndex:
    ordinals: tuple[int, ...]
    counts: tuple[int, ...]
    cumulative: tuple[int, ...]

    def last_value(self, cutoff: date) -> tuple[int, int, str, str]:
        position = bisect_right(self.ordinals, cutoff.toordinal()) - 1
        if position < 0:
            return 0, 0, "", ""
        observed = date.fromordinal(self.ordinals[position]).isoformat()
        return self.counts[position], 1, observed, observed

    def range_value(self, start: date, end: date) -> tuple[int, int, str, str]:
        left = bisect_left(self.ordinals, start.toordinal())
        right = bisect_right(self.ordinals, end.toordinal())
        if left >= right:
            return 0, 0, "", ""
        previous = self.cumulative[left - 1] if left else 0
        rows = self.cumulative[right - 1] - previous
        first = date.fromordinal(self.ordinals[left]).isoformat()
        last = date.fromordinal(self.ordinals[right - 1]).isoformat()
        return int(rows), right - left, first, last


def _build_daily_index(
    counts: Mapping[tuple[str, int], int]
) -> dict[str, _DailyIndex]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for (entity, ordinal), value in counts.items():
        grouped.setdefault(entity, []).append((ordinal, int(value)))
    result: dict[str, _DailyIndex] = {}
    for entity, values in grouped.items():
        ordered = sorted(values)
        ordinals = tuple(item[0] for item in ordered)
        daily = tuple(item[1] for item in ordered)
        cumulative = tuple(np.cumsum(daily).astype(int).tolist())
        result[entity] = _DailyIndex(ordinals, daily, cumulative)
    return result


def _status(
    *, expectation: str, row_count: int, alias_count: int | None = None
) -> tuple[bool, str]:
    if expectation == "required":
        return row_count > 0, (
            "required_rows_available" if row_count > 0 else "required_rows_missing"
        )
    if expectation == "diagnostic_zero_expected":
        return True, (
            "target_rows_before_checkpoint_verify_first_sale_time"
            if row_count > 0
            else "expected_no_target_rows_before_checkpoint"
        )
    if alias_count is not None and alias_count == 0:
        return True, "optional_no_reviewed_standard_ip_alias"
    return True, (
        "optional_rows_available" if row_count > 0 else "optional_rows_missing_neutral"
    )


def _checkpoint_columns(schedule: pd.DataFrame) -> list[str]:
    return [
        column
        for column in (
            "prediction_checkpoint_key",
            "sku_code",
            "split",
            "release_dt",
            "first_actual_sale_date",
            "sales_26w_qty",
            "lead_months",
            "prediction_checkpoint",
            "decision_date",
            "decision_cutoff_time",
            "within_available_through",
            "product_snapshot_partition",
            "product_snapshot_status",
        )
        if column in schedule
    ]


def select_partition_replay_candidates(
    schedule: pd.DataFrame,
    *,
    expected_lead_months: Sequence[int],
    label_as_of_date: date,
    label_window_days: int = 182,
) -> PrelaunchPartitionCandidates:
    """Select SKU schedules that can be replayed at every prelaunch checkpoint.

    This is a cheap, partition-level gate applied before raw row reads.  It
    deliberately does not claim that SKU-level rows exist; the row audit makes
    that final determination.  The mature 26-week label includes the first sale
    day, so its last required day is ``first_sale + 181 days``.
    """

    require_columns(
        schedule,
        (
            "prediction_checkpoint_key",
            "sku_code",
            "first_actual_sale_date",
            "lead_months",
            "decision_date",
            "within_available_through",
            "all_audited_sources_available",
            "sales_26w_qty",
        ),
        "prelaunch partition-audited schedule",
    )
    leads = tuple(sorted({int(value) for value in expected_lead_months}))
    if not leads:
        raise ValueError("expected_lead_months must not be empty")
    if label_window_days <= 0:
        raise ValueError("label_window_days must be positive")

    work = schedule.copy()
    work["sku_code"] = work["sku_code"].astype(str).str.strip()
    work["_decision_date"] = pd.to_datetime(
        work["decision_date"], errors="raise"
    ).dt.normalize()
    work["_first_sale_date"] = pd.to_datetime(
        work["first_actual_sale_date"], errors="raise"
    ).dt.normalize()
    work["_decision_before_first_sale"] = work["_decision_date"].lt(
        work["_first_sale_date"]
    )
    work["_partition_available"] = work[
        "all_audited_sources_available"
    ].fillna(False).astype(bool)
    work["_within_available_through"] = work[
        "within_available_through"
    ].fillna(False).astype(bool)
    work["_label_value_available"] = pd.to_numeric(
        work["sales_26w_qty"], errors="coerce"
    ).notna()

    grouped = work.groupby("sku_code", sort=False)
    audit = grouped.agg(
        audited_checkpoint_count=("prediction_checkpoint_key", "size"),
        observed_lead_months=(
            "lead_months",
            lambda value: "|".join(
                str(item) for item in sorted({int(item) for item in value})
            ),
        ),
        all_source_partitions_available=("_partition_available", "all"),
        all_decisions_before_first_sale=(
            "_decision_before_first_sale",
            "all",
        ),
        all_checkpoints_within_available_through=(
            "_within_available_through",
            "all",
        ),
        label_value_available=("_label_value_available", "all"),
        first_actual_sale_date=("_first_sale_date", "first"),
    ).reset_index()
    expected_lead_text = "|".join(str(value) for value in leads)
    audit["expected_lead_months"] = expected_lead_text
    audit["complete_checkpoint_set"] = (
        audit["audited_checkpoint_count"].eq(len(leads))
        & audit["observed_lead_months"].eq(expected_lead_text)
    )
    audit["label_maturity_date"] = audit["first_actual_sale_date"] + pd.Timedelta(
        days=label_window_days - 1
    )
    audit["label_mature_at_cutoff"] = audit["label_maturity_date"].dt.date.le(
        label_as_of_date
    )
    audit["partition_replay_candidate"] = (
        audit["complete_checkpoint_set"]
        & audit["all_source_partitions_available"]
        & audit["all_decisions_before_first_sale"]
        & audit["all_checkpoints_within_available_through"]
        & audit["label_value_available"]
        & audit["label_mature_at_cutoff"]
    )

    def reasons(row: pd.Series) -> str:
        values: list[str] = []
        if not bool(row["complete_checkpoint_set"]):
            values.append("incomplete_M5_M4_M3_checkpoint_set")
        if not bool(row["all_source_partitions_available"]):
            values.append("one_or_more_source_partitions_unavailable")
        if not bool(row["all_decisions_before_first_sale"]):
            values.append("decision_not_strictly_before_first_sale")
        if not bool(row["all_checkpoints_within_available_through"]):
            values.append("checkpoint_after_global_cutoff")
        if not bool(row["label_value_available"]):
            values.append("sales_26w_label_missing")
        if not bool(row["label_mature_at_cutoff"]):
            values.append("sales_26w_label_not_mature")
        return "|".join(values)

    audit["partition_ineligibility_reasons"] = audit.apply(reasons, axis=1)
    audit["first_actual_sale_date"] = audit[
        "first_actual_sale_date"
    ].dt.date.astype(str)
    audit["label_maturity_date"] = audit["label_maturity_date"].dt.date.astype(
        str
    )
    audit = audit.sort_values(
        ["partition_replay_candidate", "first_actual_sale_date", "sku_code"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    candidate_skus = set(
        audit.loc[audit["partition_replay_candidate"], "sku_code"]
    )
    candidate_schedule = schedule.loc[
        schedule["sku_code"].astype(str).isin(candidate_skus)
    ].copy()
    maturity = audit.loc[
        audit["partition_replay_candidate"],
        ["sku_code", "label_maturity_date"],
    ]
    candidate_schedule = candidate_schedule.merge(
        maturity,
        on="sku_code",
        how="left",
        validate="many_to_one",
    )
    candidate_schedule["partition_replay_candidate"] = True
    candidate_schedule = candidate_schedule.sort_values(
        ["first_actual_sale_date", "sku_code", "lead_months"],
        kind="stable",
    ).reset_index(drop=True)
    return PrelaunchPartitionCandidates(
        sku_audit=audit,
        schedule=candidate_schedule,
    )


def _temporal_resplit(
    checkpoint_summary: pd.DataFrame,
    *,
    expected_lead_months: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    leads = tuple(sorted({int(value) for value in expected_lead_months}))
    work = checkpoint_summary.copy()
    work["_decision_before_first_sale"] = pd.to_datetime(
        work["decision_date"], errors="raise"
    ).lt(pd.to_datetime(work["first_actual_sale_date"], errors="raise"))
    work["_partition_available"] = (
        work["all_audited_sources_available"].fillna(False).astype(bool)
        if "all_audited_sources_available" in work
        else True
    )
    work["_within_available_through"] = (
        work["within_available_through"].fillna(False).astype(bool)
        if "within_available_through" in work
        else True
    )
    grouped = work.groupby("sku_code", sort=False)
    eligibility = grouped.agg(
        audited_checkpoint_count=("lead_months", "nunique"),
        observed_lead_months=(
            "lead_months",
            lambda value: "|".join(
                str(item) for item in sorted({int(item) for item in value})
            ),
        ),
        all_core_rows_ready=("core_row_ready_for_feature_replay", "all"),
        all_labels_available=("label_available", "all"),
        all_source_partitions_available=("_partition_available", "all"),
        all_decisions_before_first_sale=(
            "_decision_before_first_sale",
            "all",
        ),
        all_checkpoints_within_available_through=(
            "_within_available_through",
            "all",
        ),
    )
    expected_lead_text = "|".join(str(value) for value in leads)
    eligibility["complete_checkpoint_set"] = (
        eligibility["audited_checkpoint_count"].eq(len(leads))
        & eligibility["observed_lead_months"].eq(expected_lead_text)
    )
    eligibility["common_core_row_ready"] = (
        eligibility["complete_checkpoint_set"]
        & eligibility["all_core_rows_ready"]
        & eligibility["all_labels_available"]
    )
    eligibility["formal_replay_eligible"] = (
        eligibility["common_core_row_ready"]
        & eligibility["all_source_partitions_available"]
        & eligibility["all_decisions_before_first_sale"]
        & eligibility["all_checkpoints_within_available_through"]
    )

    def formal_reasons(row: pd.Series) -> str:
        values: list[str] = []
        if not bool(row["complete_checkpoint_set"]):
            values.append("incomplete_M5_M4_M3_checkpoint_set")
        if not bool(row["all_core_rows_ready"]):
            values.append("missing_required_product_or_store_rows")
        if not bool(row["all_labels_available"]):
            values.append("sales_26w_label_missing")
        if not bool(row["all_source_partitions_available"]):
            values.append("one_or_more_source_partitions_unavailable")
        if not bool(row["all_decisions_before_first_sale"]):
            values.append("decision_not_strictly_before_first_sale")
        if not bool(row["all_checkpoints_within_available_through"]):
            values.append("checkpoint_after_global_cutoff")
        return "|".join(values)

    eligibility["formal_ineligibility_reasons"] = eligibility.apply(
        formal_reasons, axis=1
    )
    common = eligibility.reset_index()
    base = work.sort_values(
        ["first_actual_sale_date", "sku_code", "lead_months"], kind="stable"
    ).drop_duplicates("sku_code")
    common = base[
        [
            column
            for column in (
                "sku_code",
                "first_actual_sale_date",
                "split",
                "sales_26w_qty",
            )
            if column in base
        ]
    ].merge(common, on="sku_code", how="left", validate="one_to_one")
    common = common.sort_values(
        ["formal_replay_eligible", "first_actual_sale_date", "sku_code"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    eligible = common.loc[common["formal_replay_eligible"]].copy()
    if len(eligible) < 2:
        return common, pd.DataFrame()
    eligible["first_actual_sale_date"] = pd.to_datetime(
        eligible["first_actual_sale_date"], errors="raise"
    )
    eligible = eligible.sort_values(
        ["first_actual_sale_date", "sku_code"], kind="stable"
    ).reset_index(drop=True)
    from training.split import SkuSample, chronological_ratio_split

    if eligible["first_actual_sale_date"].nunique() < 2:
        return common, pd.DataFrame()
    _, _, boundary_day = chronological_ratio_split(
        [SkuSample(row.first_actual_sale_date.date(), str(row.sku_code)) for row in eligible.itertuples(index=False)],
        train_ratio=0.80,
    )
    boundary = pd.Timestamp(boundary_day)
    eligible = eligible.rename(columns={"split": "original_split"})
    eligible["split"] = np.where(
        eligible["first_actual_sale_date"].le(boundary), "Train", "Test"
    )
    eligible["first_actual_sale_date"] = eligible[
        "first_actual_sale_date"
    ].dt.date.astype(str)
    eligible["temporal_split_boundary_date"] = boundary.date().isoformat()
    eligible["split_rule"] = (
        "first_actual_sale_date ascending; approximately 80/20; same launch date "
        "never crosses Train/Test"
    )
    eligible["eligibility_scope"] = "formal_for_historical_feature_replay"
    eligible["training_ready"] = False
    eligible["requires_full_feature_replay_before_training"] = True
    eligible["requires_train_only_weight_refit"] = True
    return common, eligible


class PrelaunchRowAuditAdapter:
    """Row-grain point-in-time source audit for M5/M4/M3 replay candidates.

    The audit deliberately distinguishes mandatory target/global rows from
    optional factors and from prelaunch target-history diagnostics.  A target
    SKU is not rejected merely because it has no sales, inventory or placement
    rows months before launch.
    """

    def __init__(
        self,
        repository: ParquetDataRepository,
        *,
        batch_size: int = 50_000,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.repository = repository
        self.batch_size = batch_size

    @classmethod
    def from_project(
        cls,
        *,
        project_root: Path,
        data_root: Path,
        config: AppConfig,
        batch_size: int = 50_000,
    ) -> "PrelaunchRowAuditAdapter":
        return cls(
            ParquetDataRepository(
                data_root=data_root,
                table_specs=load_table_specs(config.probe.table_specs_path),
                max_rows_per_read=config.data_reader.max_rows_per_read,
            ),
            batch_size=batch_size,
        )

    def _daily_counts(
        self,
        *,
        source_key: str,
        schedule: pd.DataFrame,
        sku_codes: Sequence[str],
        social_start: date,
        maximum_cutoff: datetime,
        progress: Callable[[str, Mapping[str, Any]], None] | None,
    ) -> tuple[dict[str, _DailyIndex], str]:
        source_name = SOURCES[source_key]
        try:
            partitions = self.repository.available_partitions(source_name)
        except DataLayerError as exc:
            return {}, f"{type(exc).__name__}: {exc}"
        end_date = min(maximum_cutoff.date(), partitions[-1])
        start_date = max(
            partitions[0], social_start if source_key == "social" else partitions[0]
        )
        if start_date > end_date:
            return {}, ""
        if progress is not None:
            progress(
                "source_row_read_start",
                {
                    "source_key": source_key,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "sku_count": len(sku_codes) if source_key != "social" else 0,
                },
            )
        counts: dict[tuple[str, int], int] = {}
        total_rows = 0
        batch_count = 0
        try:
            iterator = self.repository.iter_partition_batches(
                source_name=source_name,
                start_date=start_date,
                end_date=end_date,
                cutoff_time=maximum_cutoff,
                columns=SOURCE_MINIMUM_COLUMNS[source_key],
                batch_size=self.batch_size,
                sku_codes=(
                    sku_codes
                    if source_key not in {"store", "social"}
                    else None
                ),
            )
            for batch_count, batch in enumerate(iterator, start=1):
                frame = batch.to_pandas()
                if frame.empty:
                    continue
                days = pd.to_datetime(frame["ds"], errors="coerce").dt.date
                valid = days.notna()
                if source_key == "store":
                    entities = pd.Series("__GLOBAL__", index=frame.index)
                elif source_key == "social":
                    entities = frame["ip_name"].map(normalize_social_ip_name)
                    valid &= entities.ne("")
                else:
                    entities = frame["sku_code"].fillna("").astype(str).str.strip()
                    valid &= entities.ne("")
                compact = pd.DataFrame(
                    {"entity": entities.loc[valid], "day": days.loc[valid]}
                )
                daily = compact.groupby(["entity", "day"], sort=False).size()
                for (entity, day), value in daily.items():
                    key = (str(entity), day.toordinal())
                    counts[key] = counts.get(key, 0) + int(value)
                total_rows += len(frame)
                if progress is not None and (
                    batch_count == 1 or batch_count % 20 == 0
                ):
                    progress(
                        "source_row_read_progress",
                        {
                            "source_key": source_key,
                            "arrow_batches": batch_count,
                            "rows": total_rows,
                        },
                    )
        except DataLayerError as exc:
            return {}, f"{type(exc).__name__}: {exc}"
        if progress is not None:
            progress(
                "source_row_read_complete",
                {
                    "source_key": source_key,
                    "arrow_batches": batch_count,
                    "rows": total_rows,
                    "entity_days": len(counts),
                },
            )
        return _build_daily_index(counts), ""

    def audit(
        self,
        schedule: pd.DataFrame,
        *,
        standard_ip_mapping: pd.DataFrame,
        social_ip_alias_mapping: pd.DataFrame | None,
        social_window_days: int,
        progress: Callable[[str, Mapping[str, Any]], None] | None = None,
        emit_formal_eligibility: bool = False,
    ) -> PrelaunchRowAudit:
        require_columns(
            schedule,
            (
                "prediction_checkpoint_key",
                "sku_code",
                "release_dt",
                "first_actual_sale_date",
                "lead_months",
                "decision_date",
                "decision_cutoff_time",
                "product_snapshot_partition",
                "product_snapshot_status",
            ),
            "prelaunch decision schedule",
        )
        if social_window_days <= 0:
            raise ValueError("social_window_days must be positive")
        base = schedule.copy()
        base["sku_code"] = base["sku_code"].astype(str).str.strip()
        base["decision_date"] = pd.to_datetime(
            base["decision_date"], errors="raise"
        ).dt.date
        cutoff_values = pd.to_datetime(
            base["decision_cutoff_time"], errors="raise", utc=True
        )
        if cutoff_values.isna().any():
            raise ValueError("decision_cutoff_time contains invalid values")
        maximum_cutoff = cutoff_values.max().to_pydatetime()
        minimum_decision = min(base["decision_date"])
        social_start = minimum_decision - timedelta(days=social_window_days - 1)
        sku_codes = base["sku_code"].drop_duplicates().tolist()
        aliases, provenance = build_sku_social_aliases(
            standard_ip_mapping, social_ip_alias_mapping
        )

        detail_rows: list[dict[str, Any]] = []
        checkpoint_columns = _checkpoint_columns(base)
        checkpoints = base[checkpoint_columns].to_dict(orient="records")
        for policy in ROW_SOURCE_POLICIES:
            index, source_error = self._daily_counts(
                source_key=policy.source_key,
                schedule=base,
                sku_codes=sku_codes,
                social_start=social_start,
                maximum_cutoff=maximum_cutoff,
                progress=progress,
            )
            for checkpoint in checkpoints:
                sku_code = str(checkpoint["sku_code"])
                decision_day = pd.Timestamp(checkpoint["decision_date"]).date()
                matched_aliases = ""
                alias_count: int | None = None
                if policy.coverage_grain == "global_snapshot_last_value":
                    series = index.get("__GLOBAL__")
                    result = (0, 0, "", "") if series is None else series.last_value(decision_day)
                elif policy.coverage_grain == "target_sku_last_value":
                    series = index.get(sku_code)
                    result = (0, 0, "", "") if series is None else series.last_value(decision_day)
                elif policy.coverage_grain == "target_sku_predecision_history":
                    series = index.get(sku_code)
                    result = (
                        (0, 0, "", "")
                        if series is None
                        else series.range_value(date.min, decision_day)
                    )
                else:
                    sku_aliases = sorted(aliases.get(sku_code, set()))
                    alias_count = len(sku_aliases)
                    matched_aliases = "|".join(sku_aliases)
                    start_day = decision_day - timedelta(days=social_window_days - 1)
                    parts = [
                        index[alias].range_value(start_day, decision_day)
                        for alias in sku_aliases
                        if alias in index
                    ]
                    rows = sum(value[0] for value in parts)
                    observed_days = {
                        ordinal
                        for alias in sku_aliases
                        if alias in index
                        for ordinal in index[alias].ordinals
                        if start_day.toordinal() <= ordinal <= decision_day.toordinal()
                    }
                    first = (
                        date.fromordinal(min(observed_days)).isoformat()
                        if observed_days
                        else ""
                    )
                    last = (
                        date.fromordinal(max(observed_days)).isoformat()
                        if observed_days
                        else ""
                    )
                    result = (rows, len(observed_days), first, last)
                row_count, distinct_days, first_date, last_date = result
                available, status = _status(
                    expectation=policy.expectation,
                    row_count=row_count,
                    alias_count=alias_count,
                )
                if source_error:
                    available = False if policy.required_for_core_replay else True
                    status = "source_read_error"
                row = dict(checkpoint)
                row.update(
                    {
                        "source_key": policy.source_key,
                        "source_table": SOURCES[policy.source_key],
                        "coverage_grain": policy.coverage_grain,
                        "expectation": policy.expectation,
                        "required_for_core_replay": policy.required_for_core_replay,
                        "row_count": int(row_count),
                        "distinct_row_days": int(distinct_days),
                        "first_observed_date": first_date,
                        "last_observed_date": last_date,
                        "reviewed_social_alias_count": alias_count,
                        "reviewed_social_aliases": matched_aliases,
                        "social_alias_sources": "|".join(
                            sorted(
                                {
                                    item
                                    for alias in aliases.get(sku_code, set())
                                    for item in provenance.get(sku_code, {}).get(alias, set())
                                }
                            )
                        ),
                        "row_availability_ok": bool(available),
                        "row_audit_status": status,
                        "source_error": source_error,
                    }
                )
                detail_rows.append(row)

        # Product attributes and release_dt are intentionally fixed from the
        # exact global product snapshot selected when the schedule is built.
        # They are not re-read from each M5/M4/M3 historical partition: many
        # products legitimately enter the delivered daily dimension only close
        # to launch, while the business decision itself occurs months earlier.
        for checkpoint in checkpoints:
            snapshot_partition = str(
                checkpoint.get("product_snapshot_partition") or ""
            )
            snapshot_status = str(
                checkpoint.get("product_snapshot_status") or ""
            )
            release_dt = str(checkpoint.get("release_dt") or "")
            available = bool(
                snapshot_partition
                and snapshot_status == "latest_partition"
                and release_dt
            )
            row = dict(checkpoint)
            row.update(
                {
                    "source_key": "product",
                    "source_table": SOURCES["product"],
                    "coverage_grain": "target_sku_fixed_cutoff_snapshot",
                    "expectation": "required",
                    "required_for_core_replay": True,
                    "row_count": 1 if available else 0,
                    "distinct_row_days": 1 if available else 0,
                    "first_observed_date": (
                        snapshot_partition if available else ""
                    ),
                    "last_observed_date": (
                        snapshot_partition if available else ""
                    ),
                    "reviewed_social_alias_count": None,
                    "reviewed_social_aliases": "",
                    "social_alias_sources": "",
                    "row_availability_ok": available,
                    "row_audit_status": (
                        "fixed_product_snapshot_available"
                        if available
                        else "fixed_product_snapshot_missing"
                    ),
                    "source_error": "",
                }
            )
            detail_rows.append(row)

        # Standard-IP mapping is a reviewed, frozen artifact rather than a raw
        # server source. Keep its row-grain coverage visible as a separate row.
        mapping = standard_ip_mapping.copy()
        mapping["sku_code"] = mapping["sku_code"].astype(str).str.strip()
        if "path1_eligible" in mapping:
            eligible = pd.to_numeric(
                mapping["path1_eligible"], errors="coerce"
            ).fillna(0).eq(1)
            mapping = mapping.loc[eligible]
        mapping_counts = mapping.groupby("sku_code").size().to_dict()
        for checkpoint in checkpoints:
            row_count = int(mapping_counts.get(str(checkpoint["sku_code"]), 0))
            row = dict(checkpoint)
            row.update(
                {
                    "source_key": "standard_ip_mapping",
                    "source_table": "frozen reviewed CSV artifact",
                    "coverage_grain": "target_sku_reviewed_mapping",
                    "expectation": "optional",
                    "required_for_core_replay": False,
                    "row_count": row_count,
                    "distinct_row_days": 0,
                    "first_observed_date": "",
                    "last_observed_date": "",
                    "reviewed_social_alias_count": len(
                        aliases.get(str(checkpoint["sku_code"]), set())
                    ),
                    "reviewed_social_aliases": "|".join(
                        sorted(aliases.get(str(checkpoint["sku_code"]), set()))
                    ),
                    "social_alias_sources": "|".join(
                        sorted(
                            {
                                item
                                for alias in aliases.get(
                                    str(checkpoint["sku_code"]), set()
                                )
                                for item in provenance.get(
                                    str(checkpoint["sku_code"]), {}
                                ).get(alias, set())
                            }
                        )
                    ),
                    "row_availability_ok": True,
                    "row_audit_status": (
                        "optional_rows_available"
                        if row_count > 0
                        else "optional_rows_missing_path1_disabled"
                    ),
                    "source_error": "",
                }
            )
            detail_rows.append(row)

        detail = pd.DataFrame(detail_rows)
        required = detail.loc[detail["required_for_core_replay"]].copy()

        def joined_missing(group: pd.DataFrame) -> str:
            return "|".join(
                group.loc[~group["row_availability_ok"], "source_key"].astype(str)
            )

        required_summary = (
            required.groupby("prediction_checkpoint_key", sort=False)
            .agg(
                required_source_count=("source_key", "size"),
                required_source_available_count=("row_availability_ok", "sum"),
            )
            .reset_index()
        )
        missing = (
            required.groupby("prediction_checkpoint_key", sort=False)
            .apply(joined_missing)
            .rename("missing_required_row_sources")
            .reset_index()
        )
        required_summary = required_summary.merge(
            missing,
            on="prediction_checkpoint_key",
            how="left",
            validate="one_to_one",
        )
        optional = detail.loc[detail["expectation"].eq("optional")]
        optional_counts = (
            optional.groupby("prediction_checkpoint_key", as_index=False)
            .agg(
                optional_source_count=("source_key", "size"),
                optional_sources_with_rows=("row_count", lambda value: int((value > 0).sum())),
            )
        )
        diagnostics = detail.loc[
            detail["expectation"].eq("diagnostic_zero_expected")
        ]
        diagnostic_counts = (
            diagnostics.groupby("prediction_checkpoint_key", as_index=False)
            .agg(
                predecision_target_history_rows=("row_count", "sum"),
                predecision_target_history_sources=(
                    "row_count",
                    lambda value: int((value > 0).sum()),
                ),
            )
        )
        checkpoint = base.merge(
            required_summary,
            on="prediction_checkpoint_key",
            how="left",
            validate="one_to_one",
        ).merge(
            optional_counts,
            on="prediction_checkpoint_key",
            how="left",
            validate="one_to_one",
        ).merge(
            diagnostic_counts,
            on="prediction_checkpoint_key",
            how="left",
            validate="one_to_one",
        )
        checkpoint["core_row_ready_for_feature_replay"] = checkpoint[
            "required_source_available_count"
        ].eq(checkpoint["required_source_count"])
        if "within_available_through" in checkpoint:
            checkpoint["core_row_ready_for_feature_replay"] &= checkpoint[
                "within_available_through"
            ].fillna(False).astype(bool)
        checkpoint["label_available"] = (
            pd.to_numeric(checkpoint["sales_26w_qty"], errors="coerce").notna()
            if "sales_26w_qty" in checkpoint
            else False
        )
        checkpoint["row_audit_status"] = np.where(
            checkpoint["core_row_ready_for_feature_replay"],
            "core_rows_ready_requires_historical_feature_replay",
            "missing_required_core_rows",
        )
        checkpoint["decision_date"] = checkpoint["decision_date"].map(
            date.isoformat
        )
        detail["decision_date"] = pd.to_datetime(
            detail["decision_date"], errors="raise"
        ).dt.date.astype(str)

        source_summary = (
            detail.groupby(
                [
                    "lead_months",
                    "source_key",
                    "coverage_grain",
                    "expectation",
                    "required_for_core_replay",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                prediction_checkpoints=("prediction_checkpoint_key", "size"),
                checkpoints_with_rows=("row_count", lambda value: int((value > 0).sum())),
                total_rows=("row_count", "sum"),
                source_errors=("source_error", lambda value: int(value.fillna("").ne("").sum())),
            )
            .sort_values(["lead_months", "source_key"], kind="stable")
            .reset_index(drop=True)
        )
        source_summary["row_coverage_rate"] = (
            source_summary["checkpoints_with_rows"]
            / source_summary["prediction_checkpoints"]
        )

        common, formal_eligible = _temporal_resplit(
            checkpoint,
            expected_lead_months=sorted(checkpoint["lead_months"].unique()),
        )
        if not emit_formal_eligibility:
            formal_eligible = pd.DataFrame()
        return PrelaunchRowAudit(
            checkpoint_summary=checkpoint,
            source_detail=detail,
            source_summary=source_summary,
            common_core_skus=common,
            formal_eligible_skus=formal_eligible,
        )
