from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import AppConfig, load_table_specs
from data.domain import filter_skus, normalize_sku_column, require_columns
from data.domestic import DataContractViolation
from data.launch_contract import validate_launch_origin_contract
from data.parquet_repository import DataLayerError, ParquetDataRepository


# Physical names and physical-to-canonical mappings live here, never in an
# algorithm module. A server schema change should normally be handled in this file
# and config/source_tables.json only.
SOURCES = {
    "product": "scm.dim_goods_info_ip_forecast",
    "store": "scm.dim_store_info_ip_forecast",
    "sales": "scm.dwd_trd_order_all_flow_di_ip_forecast",
    "inventory": "scm.dwd_zhgyl_store_sale_inv_ip_forecast",
    "placement": "scm.dws_rpt_store_goods_sale_stock_query_df_ip_forecast",
    "purchase": "scm.sap_ztmm135_ip_forecast",
    "supplier": "scm.sap_ztmm135_supplier_sku_profile_ip_forecast",
    "pack": "scm.dim_goods_info_pack_spec_ip_forecast",
    "member": "scm.dws_trd_offline_order_goods_1d_member_profile_ip_forecast",
    "social": "ods.ods_miniso_cn_qianchuan_ip_platform_note",
}

PRODUCT_COLUMNS = (
    "ds", "sku_code", "sku_name", "spu_code", "spu_name", "ip_code", "ip_name",
    "copyright_code", "copyright_name", "copyright_category", "copyright_exp_date",
    "cate_level1_code", "cate_level1_name", "cate_level2_code", "cate_level2_name",
    "cate_level3_code", "cate_level3_name", "cate_level4_code", "cate_level4_name",
    "manage_cate_name", "release_dt", "first_sale_date", "lifecycle", "retail_price",
    "comb_price", "box_unit", "goods_unit", "pkg_unit",
)
STORE_COLUMNS = (
    "ds", "store_code", "store_name", "store_business_state",
    "store_business_state_name", "store_level", "area_class1_name",
    "area_class2_name", "area_class3_name", "province_name", "city_name",
    "district_name", "city_level", "secbiz_area_name", "operate_size", "meter_sum",
    "open_date", "close_date", "sts_state", "mainwarehouse", "mainwarehouse_name",
)
SUPPLIER_COLUMNS = (
    "ds", "supplier_code", "sku_code", "target_qty", "received_qty",
    "receipt_fulfillment_rate", "purchase_order_count", "multi_supplier_flag",
)
PURCHASE_ORDER_COLUMNS = (
    "ds",
    "purchase_order_no",
    "purchase_order_line_no",
    "order_type",
    "order_type_desc",
    "demand_type",
    "demand_type_desc",
    "changed_date",
    "approval_date",
    "confirm_date",
    "sku_code",
    "sku_name",
    "supplier_code",
    "supplier_name",
    "target_qty",
    "received_qty",
    "unreceived_qty",
    "purchase_unit",
    "arrival_date",
    "document_status",
    "approval_status_desc",
    "purchase_suggestion_no",
    "deletion_flag",
)
PACK_COLUMNS = ("ds", "sku_code", "goods_unit", "pkg_unit", "box_unit")
MEMBER_COLUMNS = (
    "ds", "stat_month", "sku_code", "age_under_18_rate", "age_18_24_rate",
    "age_25_34_rate", "age_35_44_rate", "age_45_plus_rate", "age_unknown_rate",
    "male_rate", "female_rate", "member_count",
)
PLACEMENT_LAUNCH_COLUMNS = (
    "ds",
    "sku_code",
    "store_code",
    "first_store_arrival_time",
)
SOCIAL_MEDIA_COLUMNS = (
    "ds", "date", "ip_name", "platform_code", "platform_name", "tracked_post_count",
    "post_count", "unique_author_count", "video_post_count", "pic_post_count",
    "like_count", "comment_count", "share_count", "collect_count", "view_count",
    "interaction_count", "max_post_interaction_count", "avg_post_interaction_count",
    "interaction_rate", "view_coverage_rate", "text_coverage_rate",
    "ordinary_author_count", "koc_author_count", "kol_author_count",
    "max_author_follower_count", "avg_author_follower_count",
    "positive_post_count", "negative_post_count", "positive_post_share",
    "negative_post_share", "create_time", "update_time", "data_load_time",
)
SOCIAL_COUNT_COLUMNS = (
    "tracked_post_count", "post_count", "unique_author_count", "video_post_count",
    "pic_post_count", "like_count", "comment_count", "share_count", "collect_count",
    "view_count", "interaction_count", "max_post_interaction_count",
    "avg_post_interaction_count", "ordinary_author_count", "koc_author_count",
    "kol_author_count", "max_author_follower_count", "avg_author_follower_count",
    "positive_post_count", "negative_post_count",
)
SOCIAL_RATE_COLUMNS = (
    "view_coverage_rate", "text_coverage_rate",
    "positive_post_share", "negative_post_share",
)
SOCIAL_UNBOUNDED_RATE_COLUMNS = ("interaction_rate",)


@dataclass(frozen=True)
class ProductDomain:
    products: pd.DataFrame
    ip_membership: pd.DataFrame
    source_partition: str
    snapshot_audit: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class PlacementEvidenceDomain:
    evidence: pd.DataFrame
    source_partitions: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PlacementLaunchSourceAuditDomain:
    """Read-only audit of the physical first-arrival launch window."""

    summary: pd.DataFrame
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ExternalMediaDomain:
    """Canonical IP x platform x snapshot-day social-media history."""

    history: pd.DataFrame
    source_partitions: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PurchaseOrderHistoryDomain:
    """Standardized purchase-order history used to locate historical decisions."""

    history: pd.DataFrame
    source_partitions: tuple[str, ...]


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


class DecisionDataAdapter:
    """Read-only adapter from delivered Parquet schemas to canonical domain frames."""

    def __init__(self, repository: ParquetDataRepository) -> None:
        self.repository = repository

    def restoration_clock(self, history: pd.DataFrame, products: pd.DataFrame,
                          *, verified_origins: pd.DataFrame | None = None) -> pd.DataFrame:
        from data.restoration_audit import annotate_lifecycle_clock
        partitions = self.repository.available_partitions(SOURCES["sales"])
        return annotate_lifecycle_clock(history, products,
                                        source_boundary=min(partitions) if partitions else None,
                                        verified_origins=verified_origins)

    @classmethod
    def from_project(
        cls, *, project_root: Path, data_root: Path, config: AppConfig
    ) -> "DecisionDataAdapter":
        return cls(
            ParquetDataRepository(
                data_root=data_root,
                table_specs=load_table_specs(config.probe.table_specs_path),
                max_rows_per_read=config.data_reader.max_rows_per_read,
            )
        )

    def _latest(
        self,
        source_key: str,
        cutoff_time: datetime,
        columns: Sequence[str],
        *,
        sku_codes: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, str]:
        source = SOURCES[source_key]
        selected = self.repository.latest_partition(source, cutoff_time)
        if sku_codes:
            frames = [
                self.repository.read_latest_partition(
                    source_name=source,
                    cutoff_time=cutoff_time,
                    columns=columns,
                    sku_code=str(sku),
                ).frame
                for sku in dict.fromkeys(str(value) for value in sku_codes)
            ]
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        else:
            frame = self.repository.read_latest_partition(
                source_name=source,
                cutoff_time=cutoff_time,
                columns=columns,
            ).frame
        return frame, selected.isoformat()

    def products(
        self,
        cutoff_time: datetime,
        *,
        sku_codes: Sequence[str] | None = None,
        allow_missing: bool = False,
        fallback_to_historical: bool = True,
        progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> ProductDomain:
        """Resolve each requested SKU to its latest snapshot at/before cutoff.

        A daily product partition need not contain every historical SKU. By
        default, requested SKUs absent from the latest partition are resolved
        from older partitions, keeping all IP memberships from the single
        selected date per SKU.  Callers that need one exact reproducible
        product snapshot can disable ``fallback_to_historical``. Unfiltered
        callers retain the latest-partition universe.
        """

        def emit(event: str, **details: Any) -> None:
            if progress is not None:
                progress(event, details)

        requested = (
            None
            if sku_codes is None
            else list(dict.fromkeys(str(value).strip() for value in sku_codes))
        )
        if requested is not None and (not requested or any(not sku for sku in requested)):
            raise ValueError("product sku_codes must contain non-empty SKU codes")
        source = SOURCES["product"]
        latest_day = self.repository.latest_partition(source, cutoff_time)
        partition = latest_day.isoformat()
        emit("product_snapshot_read_start", partition=partition, requested_skus=(
            len(requested) if requested is not None else "all_latest"
        ))
        raw = self.repository.read_latest_partition(
            source_name=source,
            cutoff_time=cutoff_time,
            columns=PRODUCT_COLUMNS,
            sku_codes=requested,
        ).frame
        require_columns(raw, PRODUCT_COLUMNS, "product source")
        raw = normalize_sku_column(raw)
        present = set(raw["sku_code"])
        source_days = {sku: partition for sku in present}
        pending = set(requested or ()) - present
        emit("product_snapshot_read_complete", partition=partition, rows=len(raw),
             missing_skus=sorted(pending))
        frames = [raw]
        if pending and fallback_to_historical:
            # Read bounded date blocks, newest first. Resolved SKUs leave the
            # predicate immediately; business rows are never collected across
            # the whole history merely to find MAX(ds).
            older_days = sorted(
                (day for day in self.repository.available_partitions(source)
                 if day < latest_day),
                reverse=True,
            )
            emit("product_snapshot_fallback_start", missing_skus=sorted(pending),
                 historical_partitions=len(older_days))
            for offset in range(0, len(older_days), 30):
                if not pending:
                    break
                block = older_days[offset:offset + 30]
                emit("product_snapshot_fallback_read_start",
                     start_date=block[-1].isoformat(), end_date=block[0].isoformat(),
                     remaining_skus=len(pending))
                historical = self.repository.read_partition_range(
                    source_name=source,
                    start_date=block[-1],
                    end_date=block[0],
                    cutoff_time=cutoff_time,
                    columns=PRODUCT_COLUMNS,
                    sku_codes=sorted(pending),
                ).frame
                require_columns(historical, PRODUCT_COLUMNS, "product history")
                historical = normalize_sku_column(historical)
                resolved: dict[str, str] = {}
                if not historical.empty:
                    days = pd.to_datetime(historical["ds"], errors="coerce").dt.date
                    if days.isna().any() or not days.between(block[-1], block[0]).all():
                        raise DataContractViolation("product history contains invalid/out-of-range ds")
                    newest = days.groupby(historical["sku_code"]).transform("max")
                    historical = historical.loc[days.eq(newest)].copy()
                    historical["ds"] = days.loc[historical.index].map(date.isoformat)
                    resolved = historical.groupby("sku_code")["ds"].first().to_dict()
                    frames.append(historical)
                    source_days.update(resolved)
                    pending.difference_update(resolved)
                emit("product_snapshot_fallback_progress", scanned_partitions=offset + len(block),
                     historical_partitions=len(older_days), resolved_partitions=resolved,
                     remaining_skus=len(pending))
            emit("product_snapshot_fallback_complete",
                 fallback_partitions={sku: day for sku, day in source_days.items() if day != partition},
                 missing_skus=sorted(pending))
        audit_skus = requested if requested is not None else sorted(source_days)
        audit = pd.DataFrame(
            [{"sku_code": sku, "source_product_partition": source_days.get(sku),
              "latest_product_partition": partition,
              "product_snapshot_status": (
                  (
                      "missing_before_cutoff"
                      if fallback_to_historical
                      else "missing_in_latest_partition"
                  ) if sku in pending else
                  "latest_partition" if source_days[sku] == partition else
                  "historical_fallback"
              )} for sku in audit_skus],
            columns=["sku_code", "source_product_partition", "latest_product_partition",
                     "product_snapshot_status"],
        )
        if pending and not allow_missing:
            raise DataContractViolation(
                f"product snapshots on/before {partition} are missing requested SKUs: {sorted(pending)}"
            )
        raw = pd.concat(frames, ignore_index=True)
        ip = raw[["sku_code", "ip_code", "ip_name"]].copy()
        ip["ip_code"] = ip["ip_code"].fillna("").astype(str).str.strip()
        ip["ip_name"] = ip["ip_name"].fillna("").astype(str).str.strip()
        ip = ip.drop_duplicates().reset_index(drop=True)
        attribute_columns = [name for name in PRODUCT_COLUMNS if name not in {"ip_code", "ip_name"}]
        attributes = raw[attribute_columns].copy()
        for column in attribute_columns:
            if column in {"ds", "sku_code"}:
                continue
            conflicts = attributes.groupby("sku_code", dropna=False)[column].nunique(dropna=False)
            if (conflicts > 1).any():
                examples = conflicts.loc[conflicts > 1].head(5).index.tolist()
                raise DataContractViolation(
                    f"product canonical attribute {column} is not SKU-invariant: {examples}"
                )
        attributes = attributes.drop_duplicates("sku_code").reset_index(drop=True)
        attributes = _to_numeric(
            attributes, ("retail_price", "comb_price", "box_unit", "goods_unit", "pkg_unit")
        )
        attributes["product_type"] = attributes["manage_cate_name"].fillna("__UNKNOWN__").astype(str)
        names = attributes["sku_name"].fillna("").astype(str)
        attributes["sales_unit_type"] = np.select(
            [names.str.contains("端盒"), names.str.contains("明盒"), names.str.contains("盲盒")],
            ["end_box", "open_box", "blind_box"],
            default=attributes["product_type"],
        )
        attributes["market_code"] = "CN"
        return ProductDomain(attributes, ip, partition, audit)

    def stores(self, cutoff_time: datetime) -> tuple[pd.DataFrame, str]:
        raw, partition = self._latest("store", cutoff_time, STORE_COLUMNS)
        require_columns(raw, STORE_COLUMNS, "store source")
        raw = raw.copy()
        raw["store_code"] = raw["store_code"].astype(str).str.strip()
        if raw["store_code"].eq("").any() or raw["store_code"].duplicated().any():
            raise DataContractViolation("store snapshot is not unique and non-empty by store_code")
        raw["store_level"] = raw["store_level"].fillna("__UNKNOWN__").astype(str)
        raw["store_business_state"] = raw["store_business_state"].fillna("__UNKNOWN__").astype(str)
        raw["store_tier"] = raw["store_business_state"] + "|" + raw["store_level"]
        raw["eligible"] = (
            raw["sts_state"].fillna("").astype(str).str.upper().eq("3A")
            & pd.to_datetime(raw["close_date"], errors="coerce").isna()
        ).astype(int)
        return _to_numeric(raw, ("operate_size", "meter_sum")), partition

    def purchase_order_history(
        self,
        cutoff_time: datetime,
        *,
        sku_codes: Sequence[str],
        start_date: date | None = None,
        end_date: date | None = None,
        batch_size: int = 50_000,
        progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> PurchaseOrderHistoryDomain:
        """Read standardized purchase rows for historical decision-time audits.

        Table 18 is partitioned by arrival date, not by order or approval date,
        so the bounded partition range is scanned once with a pushed-down SKU
        predicate.  This adapter deliberately does not decide which demand type
        means "first order"; that business meaning is audited upstream.
        """

        if cutoff_time.tzinfo is None or cutoff_time.utcoffset() is None:
            raise ValueError("cutoff_time must be timezone-aware")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        normalized_skus = tuple(
            dict.fromkeys(str(value).strip() for value in sku_codes)
        )
        if not normalized_skus or any(not value for value in normalized_skus):
            raise ValueError("sku_codes must contain non-empty values")
        source = SOURCES["purchase"]
        visible = tuple(
            value
            for value in self.repository.available_partitions(source)
            if value <= cutoff_time.date()
        )
        if not visible:
            return PurchaseOrderHistoryDomain(
                pd.DataFrame(columns=PURCHASE_ORDER_COLUMNS), ()
            )
        selected_start = max(start_date or visible[0], visible[0])
        selected_end = min(end_date or visible[-1], cutoff_time.date())
        partitions = tuple(
            value for value in visible if selected_start <= value <= selected_end
        )
        if selected_start > selected_end or not partitions:
            return PurchaseOrderHistoryDomain(
                pd.DataFrame(columns=PURCHASE_ORDER_COLUMNS), ()
            )
        if progress is not None:
            progress(
                "purchase_history_read_start",
                {
                    "start_date": selected_start.isoformat(),
                    "end_date": selected_end.isoformat(),
                    "partition_count": len(partitions),
                    "sku_count": len(normalized_skus),
                },
            )
        frames: list[pd.DataFrame] = []
        row_count = 0
        for index, batch in enumerate(
            self.repository.iter_partition_batches(
                source_name=source,
                start_date=selected_start,
                end_date=selected_end,
                cutoff_time=cutoff_time,
                columns=PURCHASE_ORDER_COLUMNS,
                batch_size=batch_size,
                sku_codes=normalized_skus,
            ),
            start=1,
        ):
            frame = batch.to_pandas()
            frames.append(frame)
            row_count += len(frame)
            if progress is not None and (index == 1 or index % 20 == 0):
                progress(
                    "purchase_history_read_progress",
                    {"arrow_batches": index, "rows": row_count},
                )
        history = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=PURCHASE_ORDER_COLUMNS)
        )
        require_columns(history, PURCHASE_ORDER_COLUMNS, "purchase-order source")
        if history.empty:
            return PurchaseOrderHistoryDomain(
                history,
                tuple(value.isoformat() for value in partitions),
            )
        history = normalize_sku_column(history)
        history["ds"] = pd.to_datetime(history["ds"], errors="coerce").dt.date
        if history["ds"].isna().any():
            raise DataContractViolation("purchase-order source contains invalid ds")
        for column in (
            "changed_date",
            "approval_date",
            "confirm_date",
            "arrival_date",
        ):
            history[column] = pd.to_datetime(
                history[column], errors="coerce", format="mixed"
            )
        for column in (
            "order_type",
            "order_type_desc",
            "demand_type",
            "demand_type_desc",
            "document_status",
            "approval_status_desc",
            "deletion_flag",
        ):
            history[column] = history[column].fillna("").astype(str).str.strip()
        history = _to_numeric(
            history, ("target_qty", "received_qty", "unreceived_qty")
        )
        keys = ["purchase_order_no", "purchase_order_line_no", "sku_code"]
        if history[keys].isna().any(axis=None):
            raise DataContractViolation(
                "purchase-order source contains null document/line/SKU keys"
            )
        history["purchase_order_no"] = (
            history["purchase_order_no"].astype(str).str.strip()
        )
        history["purchase_order_line_no"] = (
            history["purchase_order_line_no"].astype(str).str.strip()
        )
        if history[keys].eq("").any(axis=None):
            raise DataContractViolation(
                "purchase-order source contains blank document/line/SKU keys"
            )
        history = (
            history.sort_values([*keys, "ds"], kind="stable")
            .drop_duplicates(keys, keep="last")
            .reset_index(drop=True)
        )
        if progress is not None:
            progress(
                "purchase_history_read_complete",
                {
                    "rows": len(history),
                    "sku_count": int(history["sku_code"].nunique()),
                    "partition_count": len(partitions),
                },
            )
        return PurchaseOrderHistoryDomain(
            history,
            tuple(value.isoformat() for value in partitions),
        )

    def external_media_history(
        self,
        cutoff_time: datetime,
        *,
        start_date: date,
        end_date: date | None = None,
        batch_size: int = 50_000,
        progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> ExternalMediaDomain:
        """Read social snapshots without leaking data after the decision cutoff.

        The physical ``ds`` partition bounds reads; the delivered ``date``
        column is the business statistic date and is standardized to
        ``stat_date``.  Their equality is enforced because a mismatch would
        shift the auditable pre-launch window.  Algorithms never access raw
        table fields directly.
        """

        if cutoff_time.tzinfo is None or cutoff_time.utcoffset() is None:
            raise ValueError("cutoff_time must be timezone-aware")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected_end = min(end_date or cutoff_time.date(), cutoff_time.date())
        if start_date > selected_end:
            return ExternalMediaDomain(
                pd.DataFrame(columns=[*SOCIAL_MEDIA_COLUMNS, "stat_date"]), (), ()
            )
        source = SOURCES["social"]
        partitions = tuple(
            value
            for value in self.repository.available_partitions(source)
            if start_date <= value <= selected_end
        )
        if progress is not None:
            progress(
                "social_history_read_start",
                {
                    "start_date": start_date.isoformat(),
                    "end_date": selected_end.isoformat(),
                    "partition_count": len(partitions),
                },
            )
        frames: list[pd.DataFrame] = []
        for index, batch in enumerate(
            self.repository.iter_partition_batches(
                source_name=source,
                start_date=start_date,
                end_date=selected_end,
                cutoff_time=cutoff_time,
                columns=SOCIAL_MEDIA_COLUMNS,
                batch_size=batch_size,
            ),
            start=1,
        ):
            frames.append(batch.to_pandas())
            if progress is not None and (index == 1 or index % 20 == 0):
                progress(
                    "social_history_read_progress",
                    {"arrow_batches": index, "rows": sum(len(item) for item in frames)},
                )
        raw = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=SOCIAL_MEDIA_COLUMNS)
        )
        require_columns(raw, SOCIAL_MEDIA_COLUMNS, "social media source")
        warnings: list[str] = []
        if raw.empty:
            warnings.append("no social-media rows exist in the requested point-in-time window")
            return ExternalMediaDomain(
                raw.assign(stat_date=pd.Series(dtype="object")),
                tuple(value.isoformat() for value in partitions),
                tuple(warnings),
            )
        frame = raw.copy()
        partition_dates = pd.to_datetime(frame["ds"], errors="coerce").dt.date
        frame["stat_date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
        invalid_dates = partition_dates.isna() | frame["stat_date"].isna()
        if invalid_dates.any():
            examples = frame.loc[invalid_dates, ["ds", "date"]].head(5).to_dict("records")
            raise DataContractViolation(
                f"social media source contains invalid ds/date values: {examples}"
            )
        mismatched_dates = frame["stat_date"].ne(partition_dates)
        if mismatched_dates.any():
            examples = frame.loc[mismatched_dates, ["ds", "date"]].head(5).to_dict("records")
            raise DataContractViolation(
                "social media business date differs from its ds partition; "
                f"examples={examples}"
            )
        for column in ("ip_name", "platform_code", "platform_name"):
            frame[column] = frame[column].fillna("").astype(str).str.strip()
        if frame["ip_name"].eq("").any() or frame["platform_code"].eq("").any():
            raise DataContractViolation(
                "social media source requires non-empty ip_name and platform_code"
            )
        frame = _to_numeric(
            frame,
            (
                *SOCIAL_COUNT_COLUMNS,
                *SOCIAL_RATE_COLUMNS,
                *SOCIAL_UNBOUNDED_RATE_COLUMNS,
            ),
        )
        for column in SOCIAL_COUNT_COLUMNS:
            negative = frame[column].notna() & frame[column].lt(0)
            if negative.any():
                examples = frame.loc[negative, ["ip_name", "platform_code", column]].head(5)
                raise DataContractViolation(
                    f"social media {column} contains negative values: "
                    f"{examples.to_dict(orient='records')}"
                )
        for column in SOCIAL_RATE_COLUMNS:
            values = frame[column]
            outside_percent = values.notna() & ((values < 0) | (values > 100))
            if outside_percent.any():
                examples = frame.loc[
                    outside_percent, ["ip_name", "platform_code", column]
                ].head(5)
                raise DataContractViolation(
                    f"social media {column} is outside both 0-1 and 0-100 ranges: "
                    f"{examples.to_dict(orient='records')}"
                )
            percent = values.notna() & values.gt(1)
            if percent.any():
                frame.loc[percent, column] = values.loc[percent] / 100.0
                warnings.append(
                    f"{column}: {int(percent.sum())} values were normalized from percent to ratio"
                )
        for column in SOCIAL_UNBOUNDED_RATE_COLUMNS:
            negative = frame[column].notna() & frame[column].lt(0)
            if negative.any():
                examples = frame.loc[
                    negative, ["ip_name", "platform_code", column]
                ].head(5)
                raise DataContractViolation(
                    f"social media {column} contains negative values: "
                    f"{examples.to_dict(orient='records')}"
                )
        # A partition can contain repeated physical versions.  Keep the latest
        # source update for each documented grain instead of adding snapshots.
        order_columns: list[str] = []
        for column in ("update_time", "data_load_time", "create_time"):
            parsed = pd.to_datetime(frame[column], errors="coerce", utc=True, format="mixed")
            sort_column = f"__{column}_parsed"
            frame[sort_column] = parsed
            order_columns.append(sort_column)
        frame = frame.sort_values(
            ["stat_date", "ip_name", "platform_code", *order_columns],
            kind="stable",
            na_position="first",
        ).drop_duplicates(["stat_date", "ip_name", "platform_code"], keep="last")
        frame = frame.drop(columns=order_columns).reset_index(drop=True)
        if progress is not None:
            progress(
                "social_history_read_complete",
                {
                    "rows": len(frame),
                    "ips": int(frame["ip_name"].nunique()),
                    "platforms": int(frame["platform_code"].nunique()),
                    "partition_count": len(partitions),
                },
            )
        return ExternalMediaDomain(
            frame,
            tuple(value.isoformat() for value in partitions),
            tuple(dict.fromkeys(warnings)),
        )

    def placement_launch_evidence(
        self,
        cutoff_time: datetime,
        *,
        candidate_products: pd.DataFrame,
        eligible_stores: pd.DataFrame,
        restored_store_weekly: pd.DataFrame,
        launch_window_days: int,
        candidate_launch_origins: pd.DataFrame | None = None,
        stock_pressure_recent_weeks: int = 4,
        progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> PlacementEvidenceDomain:
        """Build standardized M3 launch-window evidence from physical placement rows.

        Raw schema access stays in this adapter.  The placement engine receives only
        one canonical row per comparable SKU and eligible store.  The current store
        snapshot is the best available eligibility universe; this limitation is
        carried as a status instead of being hidden.
        """

        if launch_window_days <= 0:
            raise ValueError("launch_window_days must be positive")
        if stock_pressure_recent_weeks <= 0:
            raise ValueError("stock_pressure_recent_weeks must be positive")
        require_columns(
            candidate_products,
            ("sku_code", "first_sale_date", "release_dt"),
            "candidate products",
        )
        require_columns(
            eligible_stores,
            ("store_code", "store_tier"),
            "eligible stores",
        )
        require_columns(
            restored_store_weekly,
            (
                "sku_code",
                "store_code",
                "store_tier",
                "v_neutral",
                "stockout_days",
                "source_day_rows",
            ),
            "restored store weekly",
        )

        products = candidate_products.copy()
        products["sku_code"] = products["sku_code"].astype(str)
        products = products.drop_duplicates("sku_code")
        launch_origins: dict[str, pd.Timestamp] | None = None
        if candidate_launch_origins is not None:
            require_columns(candidate_launch_origins, ("launch_universe_version",), "candidate launch origins")
            validate_launch_origin_contract(candidate_launch_origins, name="candidate launch origins")
            origins = normalize_sku_column(candidate_launch_origins)
            if origins["sku_code"].duplicated().any():
                raise DataContractViolation("candidate launch origins must contain one row per SKU")
            missing_origins = set(products["sku_code"]) - set(origins["sku_code"])
            if missing_origins:
                raise DataContractViolation(f"missing canonical launch origins for candidates: {sorted(missing_origins)[:5]}")
            launch_origins = dict(zip(origins["sku_code"], pd.to_datetime(origins["first_actual_sale_date"], errors="raise", format="mixed")))
        store_columns = [
            column
            for column in (
                "store_code",
                "store_tier",
                "store_level",
                "store_business_state",
                "city_name",
                "province_name",
                "secbiz_area_name",
                "area_class1_name",
                "open_date",
                "close_date",
                "eligible",
            )
            if column in eligible_stores
        ]
        stores = eligible_stores[store_columns].copy()
        stores["store_code"] = stores["store_code"].astype(str)
        stores["store_tier"] = stores["store_tier"].astype(str)
        stores = stores.drop_duplicates("store_code")

        restored = restored_store_weekly.copy()
        for column in ("sku_code", "store_code", "store_tier"):
            restored[column] = restored[column].astype(str)
        restored["v_neutral"] = pd.to_numeric(
            restored["v_neutral"], errors="coerce"
        )
        for column in (
            "lifecycle_week",
            "ending_available_stock_qty",
            "normal_sales_for_pressure",
            "normal_sales_days_for_pressure",
        ):
            if column not in restored:
                restored[column] = np.nan
            restored[column] = pd.to_numeric(restored[column], errors="coerce")

        evidence_frames: list[pd.DataFrame] = []
        partitions: set[str] = set()
        warnings: list[str] = []
        product_rows = list(products.itertuples(index=False))
        for index, product in enumerate(product_rows, start=1):
            first_sale = pd.to_datetime(
                getattr(product, "first_sale_date", None), errors="coerce"
            )
            release = pd.to_datetime(
                getattr(product, "release_dt", None), errors="coerce"
            )
            if launch_origins is not None:
                launch_day = launch_origins[str(product.sku_code)].date()
                launch_source = "cohort.sales_observed_first_positive"
            elif pd.notna(first_sale):
                launch_day = first_sale.date()
                launch_source = "product.first_sale_date"
            elif pd.notna(release):
                launch_day = release.date()
                launch_source = "product.release_dt_fallback"
            else:
                warnings.append(
                    f"{product.sku_code}: no first_sale_date/release_dt; "
                    "launch-window placement evidence remains missing; "
                    "later restoration exposure must not be used as a launch proxy"
                )
                continue
            if launch_day > cutoff_time.date():
                warnings.append(
                    f"{product.sku_code}: launch date {launch_day} is after cutoff"
                )
                continue
            window_end = min(
                launch_day + timedelta(days=launch_window_days - 1),
                cutoff_time.date(),
            )
            if progress is not None:
                progress(
                    "launch_window_read_start",
                    {
                        "candidate_index": index,
                        "candidate_count": len(product_rows),
                        "sku_code": str(product.sku_code),
                        "start_date": launch_day.isoformat(),
                        "end_date": window_end.isoformat(),
                    },
                )
            read = self.repository.read_partition_range(
                source_name=SOURCES["placement"],
                start_date=launch_day,
                end_date=window_end,
                cutoff_time=cutoff_time,
                columns=PLACEMENT_LAUNCH_COLUMNS,
                sku_code=str(product.sku_code),
                max_rows=self.repository.max_rows_per_read,
            )
            if progress is not None:
                progress(
                    "launch_window_read_complete",
                    {
                        "candidate_index": index,
                        "candidate_count": len(product_rows),
                        "sku_code": str(product.sku_code),
                        "source_rows": len(read.frame),
                        "partition_count": len(read.selected_partitions),
                    },
                )
            partitions.update(value.isoformat() for value in read.selected_partitions)
            raw = read.frame.copy()
            if raw.empty:
                warnings.append(
                    f"{product.sku_code}: no launch-window source rows; "
                    "launch-window placement evidence remains missing for this SKU; "
                    "later restoration exposure must not be used as a launch proxy"
                )
                continue
            if not raw.empty:
                raw["store_code"] = raw["store_code"].astype(str).str.strip()
                raw["arrival_time"] = pd.to_datetime(
                    raw["first_store_arrival_time"], errors="coerce"
                )
                arrivals = (
                    raw.loc[raw["arrival_time"].notna()]
                    .groupby("store_code", as_index=False)["arrival_time"]
                    .min()
                )
                if arrivals.empty:
                    warnings.append(
                        f"{product.sku_code}: launch-window arrival dates are all missing/invalid; "
                        "launch-window placement evidence remains missing for this SKU; "
                        "later restoration exposure must not be used as a launch proxy"
                    )
                    continue
            else:
                arrivals = pd.DataFrame(columns=["store_code", "arrival_time"])

            universe = stores.copy()
            open_date = pd.to_datetime(
                universe.get("open_date", pd.Series(pd.NaT, index=universe.index)),
                errors="coerce",
            )
            close_date = pd.to_datetime(
                universe.get("close_date", pd.Series(pd.NaT, index=universe.index)),
                errors="coerce",
            )
            has_open_close = open_date.notna() | close_date.notna()
            launch_timestamp_for_eligibility = pd.Timestamp(launch_day)
            launch_eligible = (
                (
                    open_date.isna()
                    | open_date.le(launch_timestamp_for_eligibility)
                )
                & (
                    close_date.isna()
                    | close_date.ge(launch_timestamp_for_eligibility)
                )
            )
            if "eligible" in universe:
                fallback_eligible = pd.to_numeric(
                    universe["eligible"], errors="coerce"
                ).fillna(0).eq(1)
            else:
                fallback_eligible = pd.Series(True, index=universe.index)
            universe = universe.loc[
                np.where(has_open_close, launch_eligible, fallback_eligible)
            ].copy()
            universe["sku_code"] = str(product.sku_code)
            universe = universe.merge(
                arrivals, on="store_code", how="left", validate="one_to_one"
            )
            arrival_values = pd.to_datetime(universe["arrival_time"], errors="coerce")
            launch_start = pd.Timestamp(launch_day)
            launch_end_exclusive = pd.Timestamp(window_end) + pd.Timedelta(days=1)
            arrival_timezone = getattr(arrival_values.dt, "tz", None)
            if arrival_timezone is not None:
                launch_start = launch_start.tz_localize(arrival_timezone)
                launch_end_exclusive = launch_end_exclusive.tz_localize(
                    arrival_timezone
                )
            universe["eligible_at_launch"] = 1
            universe["arrived_in_launch_window"] = (
                arrival_values.notna()
                & arrival_values.ge(launch_start)
                & arrival_values.lt(launch_end_exclusive)
            ).astype(int)
            universe["launch_date"] = launch_day.isoformat()
            universe["launch_window_end"] = window_end.isoformat()
            universe["launch_date_source"] = launch_source
            universe["eligibility_universe_status"] = (
                "launch_date_open_close_eligibility_from_cutoff_dimension_snapshot"
            )
            universe["placement_source_row_count"] = len(raw)
            launch_weeks = int(ceil(launch_window_days / 7.0))
            candidate_performance = restored.loc[
                restored["sku_code"].eq(str(product.sku_code))
                & restored["lifecycle_week"].between(1, launch_weeks)
            ].copy()
            if not candidate_performance.empty:
                candidate_performance = candidate_performance.sort_values(
                    ["store_code", "lifecycle_week"], kind="stable"
                )
                max_week = candidate_performance.groupby("store_code")[
                    "lifecycle_week"
                ].transform("max")
                recent = candidate_performance.loc[
                    candidate_performance["lifecycle_week"].ge(
                        max_week - stock_pressure_recent_weeks + 1
                    )
                ].copy()
                recent_sales = recent.groupby("store_code", as_index=False).agg(
                    recent_normal_sales_qty=("normal_sales_for_pressure", "sum"),
                    recent_normal_sales_days=("normal_sales_days_for_pressure", "sum"),
                )
                performance = candidate_performance.groupby(
                    ["sku_code", "store_code"], as_index=False
                ).agg(
                    v_neutral=("v_neutral", "mean"),
                    ending_available_stock_qty=(
                        "ending_available_stock_qty",
                        "last",
                    ),
                ).merge(
                    recent_sales,
                    on="store_code",
                    how="left",
                    validate="one_to_one",
                )
                latest_inventory = candidate_performance.drop_duplicates(["sku_code", "store_code"], keep="last")[["sku_code", "store_code", "ending_available_stock_qty"]]
                performance = performance.drop(columns="ending_available_stock_qty").merge(
                    latest_inventory, on=["sku_code", "store_code"], how="left", validate="one_to_one")
                performance["recent_normal_daily_sales"] = (
                    performance["recent_normal_sales_qty"]
                    / performance["recent_normal_sales_days"].replace(0, np.nan)
                )
                performance["stock_pressure"] = (
                    performance["ending_available_stock_qty"]
                    / performance["recent_normal_daily_sales"].replace(0, np.nan)
                )
                performance["stock_pressure_unit"] = "days_of_supply"
                performance["stock_pressure_status"] = np.select(
                    [
                        performance["ending_available_stock_qty"].isna(),
                        performance["recent_normal_sales_days"].fillna(0).le(0),
                        performance["recent_normal_daily_sales"].fillna(0).le(0),
                    ],
                    [
                        "missing_ending_available_inventory",
                        "missing_recent_normal_exposure_days",
                        "zero_recent_normal_daily_sales_backlog_state",
                    ],
                    default="computed_inventory_div_recent_normal_daily_sales",
                )
                universe = universe.merge(
                    performance,
                    on=["sku_code", "store_code"],
                    how="left",
                    validate="one_to_one",
                )
            else:
                universe["v_neutral"] = np.nan
                universe["ending_available_stock_qty"] = np.nan
                universe["recent_normal_sales_qty"] = np.nan
                universe["recent_normal_sales_days"] = np.nan
                universe["recent_normal_daily_sales"] = np.nan
                universe["stock_pressure"] = np.nan
                universe["stock_pressure_unit"] = "days_of_supply"
                universe["stock_pressure_status"] = (
                    "missing_launch_window_restoration_inputs"
                )
            universe["stock_pressure_status"] = universe[
                "stock_pressure_status"
            ].fillna("missing_store_launch_window_restoration_inputs")
            evidence_frames.append(universe)

        if not evidence_frames:
            return PlacementEvidenceDomain(
                pd.DataFrame(), tuple(sorted(partitions)), tuple(warnings)
            )
        evidence = pd.concat(evidence_frames, ignore_index=True)
        evidence["evidence_weight"] = 1.0
        output_columns = [
            "sku_code",
            "store_code",
            "store_tier",
            *[
                column for column in (
                    "store_business_state",
                    "store_level",
                    "province_name",
                    "city_name",
                    "secbiz_area_name",
                    "area_class1_name",
                ) if column in evidence
            ],
            "eligible_at_launch",
            "arrived_in_launch_window",
            "v_neutral",
            "ending_available_stock_qty",
            "recent_normal_sales_qty",
            "recent_normal_sales_days",
            "recent_normal_daily_sales",
            "stock_pressure",
            "stock_pressure_unit",
            "stock_pressure_status",
            "evidence_weight",
            "arrival_time",
            "launch_date",
            "launch_window_end",
            "launch_date_source",
            "eligibility_universe_status",
            "placement_source_row_count",
        ]
        return PlacementEvidenceDomain(
            evidence[output_columns],
            tuple(sorted(partitions)),
            tuple(warnings),
        )

    def audit_placement_launch_source(
        self,
        cutoff_time: datetime,
        *,
        candidate_products: pd.DataFrame,
        launch_window_days: int,
        candidate_cohort: pd.DataFrame | None = None,
        progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> PlacementLaunchSourceAuditDomain:
        """Audit physical launch coverage and optional cohort lifecycle origin.

        This method deliberately reports only what the delivered first-arrival
        source can prove.  It never substitutes restored sales/exposure for a
        launch-window arrival, so callers can distinguish an empty physical
        window from an algorithmic fallback.
        """

        if launch_window_days <= 0:
            raise ValueError("launch_window_days must be positive")
        require_columns(
            candidate_products,
            ("sku_code", "first_sale_date", "release_dt"),
            "candidate products",
        )
        products = candidate_products.copy()
        products["sku_code"] = products["sku_code"].astype(str)
        products = products.drop_duplicates("sku_code")
        available = tuple(
            day
            for day in self.repository.available_partitions(SOURCES["placement"])
            if day <= cutoff_time.date()
        )
        source_min = min(available).isoformat() if available else None
        source_max = max(available).isoformat() if available else None
        rows: list[dict[str, Any]] = []
        warnings: list[str] = []

        for index, product in enumerate(products.itertuples(index=False), start=1):
            sku_code = str(product.sku_code)
            first_sale = pd.to_datetime(
                getattr(product, "first_sale_date", None), errors="coerce"
            )
            release = pd.to_datetime(
                getattr(product, "release_dt", None), errors="coerce"
            )
            if pd.notna(first_sale):
                launch_day = first_sale.date()
                launch_source = "product.first_sale_date"
            elif pd.notna(release):
                launch_day = release.date()
                launch_source = "product.release_dt_fallback"
            else:
                rows.append(
                    {
                        "sku_code": sku_code,
                        "product_first_sale_date": None,
                        "product_release_dt": None,
                        "launch_date": None,
                        "launch_window_end": None,
                        "launch_date_source": "missing",
                        "source_partition_min": source_min,
                        "source_partition_max_at_cutoff": source_max,
                        "source_partition_count_in_window": 0,
                        "source_rows": 0,
                        "source_store_count": 0,
                        "valid_first_arrival_store_count": 0,
                        "arrived_in_launch_window_store_count": 0,
                        "arrival_before_window_store_count": 0,
                        "arrival_after_window_store_count": 0,
                        "audit_status": "missing_product_launch_date",
                    }
                )
                warnings.append(f"{sku_code}: no first_sale_date/release_dt")
                continue

            window_end = min(
                launch_day + timedelta(days=launch_window_days - 1),
                cutoff_time.date(),
            )
            window_partitions = tuple(
                day for day in available if launch_day <= day <= window_end
            )
            if progress is not None:
                progress(
                    "placement_launch_source_audit_start",
                    {
                        "candidate_index": index,
                        "candidate_count": len(products),
                        "sku_code": sku_code,
                        "start_date": launch_day.isoformat(),
                        "end_date": window_end.isoformat(),
                        "source_partition_count": len(window_partitions),
                    },
                )

            if window_partitions:
                read = self.repository.read_partition_range(
                    source_name=SOURCES["placement"],
                    start_date=launch_day,
                    end_date=window_end,
                    cutoff_time=cutoff_time,
                    columns=PLACEMENT_LAUNCH_COLUMNS,
                    sku_code=sku_code,
                    max_rows=self.repository.max_rows_per_read,
                )
                raw = read.frame.copy()
                selected_partition_count = len(read.selected_partitions)
            else:
                raw = pd.DataFrame(columns=PLACEMENT_LAUNCH_COLUMNS)
                selected_partition_count = 0

            if raw.empty:
                source_store_count = 0
                valid_store_count = 0
                in_window_count = 0
                before_count = 0
                after_count = 0
            else:
                raw["store_code"] = raw["store_code"].astype(str).str.strip()
                raw["arrival_time"] = pd.to_datetime(
                    raw["first_store_arrival_time"], errors="coerce"
                )
                source_store_count = int(raw["store_code"].nunique())
                arrivals = (
                    raw.loc[raw["arrival_time"].notna() & raw["store_code"].ne("")]
                    .groupby("store_code", as_index=False)["arrival_time"]
                    .min()
                )
                valid_store_count = len(arrivals)
                start_ts = pd.Timestamp(launch_day)
                end_exclusive = pd.Timestamp(window_end) + pd.Timedelta(days=1)
                arrival_timezone = getattr(arrivals["arrival_time"].dt, "tz", None)
                if arrival_timezone is not None:
                    start_ts = start_ts.tz_localize(arrival_timezone)
                    end_exclusive = end_exclusive.tz_localize(arrival_timezone)
                in_window_count = int(
                    (
                        arrivals["arrival_time"].ge(start_ts)
                        & arrivals["arrival_time"].lt(end_exclusive)
                    ).sum()
                )
                before_count = int(arrivals["arrival_time"].lt(start_ts).sum())
                after_count = int(
                    arrivals["arrival_time"].ge(end_exclusive).sum()
                )

            if not window_partitions:
                status = "no_source_partitions_in_launch_window"
            elif raw.empty:
                status = "source_partitions_present_but_no_sku_rows"
            elif valid_store_count == 0:
                status = "sku_rows_present_but_arrival_time_missing"
            elif in_window_count == 0:
                status = "valid_arrivals_present_but_none_in_launch_window"
            else:
                status = "usable_first_arrival_evidence"
            rows.append(
                {
                    "sku_code": sku_code,
                    "product_first_sale_date": (
                        first_sale.date().isoformat() if pd.notna(first_sale) else None
                    ),
                    "product_release_dt": (
                        release.date().isoformat() if pd.notna(release) else None
                    ),
                    "launch_date": launch_day.isoformat(),
                    "launch_window_end": window_end.isoformat(),
                    "launch_date_source": launch_source,
                    "source_partition_min": source_min,
                    "source_partition_max_at_cutoff": source_max,
                    "source_partition_count_in_window": selected_partition_count,
                    "source_rows": len(raw),
                    "source_store_count": source_store_count,
                    "valid_first_arrival_store_count": valid_store_count,
                    "arrived_in_launch_window_store_count": in_window_count,
                    "arrival_before_window_store_count": before_count,
                    "arrival_after_window_store_count": after_count,
                    "audit_status": status,
                }
            )
            if status != "usable_first_arrival_evidence":
                warnings.append(f"{sku_code}: {status}")
            if progress is not None:
                progress(
                    "placement_launch_source_audit_complete",
                    {
                        "sku_code": sku_code,
                        "source_rows": len(raw),
                        "source_store_count": source_store_count,
                        "arrived_in_launch_window_store_count": in_window_count,
                        "audit_status": status,
                    },
                )

        summary = pd.DataFrame(rows)
        if candidate_cohort is not None:
            require_columns(
                candidate_cohort,
                ("sku_code", "first_actual_sale_date"),
                "candidate cohort lifecycle origin",
            )
            cohort = candidate_cohort[["sku_code", "first_actual_sale_date"]].copy()
            cohort["sku_code"] = cohort["sku_code"].astype(str)
            cohort = cohort.loc[cohort["sku_code"].isin(products["sku_code"])]
            cohort["cohort_first_actual_sale_date"] = pd.to_datetime(
                cohort.pop("first_actual_sale_date"), errors="raise"
            ).dt.normalize()
            if cohort.groupby("sku_code")["cohort_first_actual_sale_date"].nunique().gt(1).any():
                raise DataContractViolation("candidate cohort has conflicting lifecycle origins per SKU")
            summary = summary.merge(
                cohort.drop_duplicates("sku_code"), on="sku_code", how="left",
                validate="one_to_one",
            )
            launch = pd.to_datetime(summary["launch_date"], errors="coerce")
            cohort_start = summary["cohort_first_actual_sale_date"]
            offset = (cohort_start - launch).dt.days
            summary["cohort_origin_offset_days"] = offset.astype("Int64")
            summary["product_lifecycle_week_at_cohort_start"] = (
                offset.floordiv(7).add(1).where(offset.ge(0)).astype("Int64")
            )
            summary["lifecycle_origin_status"] = np.select(
                [offset.eq(0), offset.gt(0), offset.lt(0)],
                ["aligned", "cohort_starts_after_product_launch", "cohort_starts_before_product_launch"],
                default="missing_origin_for_comparison",
            )
            for item in summary.loc[offset.notna() & offset.ne(0)].itertuples(index=False):
                warnings.append(
                    f"{item.sku_code}: cohort lifecycle origin differs from {item.launch_date_source} "
                    f"by {item.cohort_origin_offset_days} days; audit only, no date override applied"
                )
            summary["cohort_first_actual_sale_date"] = cohort_start.dt.strftime("%Y-%m-%d")
        return PlacementLaunchSourceAuditDomain(summary, tuple(warnings))

    def factor_features(
        self,
        cutoff_time: datetime,
        *,
        sku_codes: Sequence[str] | None = None,
        restored_weekly: pd.DataFrame | None = None,
        standard_ip_membership: pd.DataFrame | None = None,
        product_snapshot_time: datetime | None = None,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        product = self.products(
            product_snapshot_time or cutoff_time,
            sku_codes=sku_codes,
        )
        supplier_raw, supplier_partition = self._latest(
            "supplier", cutoff_time, SUPPLIER_COLUMNS, sku_codes=sku_codes
        )
        member_raw, member_partition = self._latest(
            "member", cutoff_time, MEMBER_COLUMNS, sku_codes=sku_codes
        )
        product_frame = product.products.copy()
        feature_columns = [
            "sku_code",
            "retail_price",
            "lifecycle",
            "copyright_category",
            "copyright_code",
            "cate_level1_code",
            "cate_level2_code",
            "cate_level3_code",
            "cate_level4_code",
            "manage_cate_name",
            "release_dt",
            "first_sale_date",
            "product_type",
            "sales_unit_type",
            "goods_unit",
            "pkg_unit",
            "box_unit",
        ]
        # Unit tests and older frozen product snapshots may predate optional
        # canonical columns. Missing optional attributes stay missing; they are
        # never interpreted as zero or inferred from another physical table.
        for column in feature_columns:
            if column not in product_frame:
                product_frame[column] = np.nan
        features = product_frame[feature_columns].copy()
        # Packaging attributes are canonical static SKU attributes in the
        # product master.  The delivered pack table repeats the same three
        # fields and does not add a business-confirmed series_pack_count or
        # unit conversion, so algorithms must not create a second source of
        # truth from that table.
        features["pack_size"] = pd.to_numeric(features["pkg_unit"], errors="coerce")
        features["box_size"] = pd.to_numeric(features["box_unit"], errors="coerce")
        # The source field is a national merchandise status (for example
        # 补货期/淘汰期), not V3.2 PF-02's numeric lifecycle-persistence factor.
        features = features.rename(columns={"lifecycle": "lifecycle_status"})
        # M5 must use the reviewed, versioned standard-IP membership used by M1.
        # The raw product ``ip_code`` is only a coarse display/theme field and is
        # deliberately not a fallback for a missing standard-IP mapping.
        if standard_ip_membership is not None:
            require_columns(
                standard_ip_membership,
                ("sku_code", "ip_code"),
                "standard_ip_membership",
            )
            standard_ip = normalize_sku_column(standard_ip_membership)
            standard_ip["ip_code"] = (
                standard_ip["ip_code"].fillna("").astype(str).str.strip()
            )
            ip_count = (
                standard_ip.loc[standard_ip["ip_code"].ne("")]
                .drop_duplicates(["sku_code", "ip_code"])
                .groupby("sku_code")["ip_code"]
                .nunique()
                .rename("ip_breadth")
            )
            features = features.merge(ip_count, on="sku_code", how="left")
        else:
            features["ip_breadth"] = np.nan
        features["ip_breadth_status"] = np.where(
            features["ip_breadth"].notna(),
            "resolved_standard_ip",
            "missing_standard_ip",
        )
        if not supplier_raw.empty:
            supplier_raw = normalize_sku_column(supplier_raw)
            supplier_raw = _to_numeric(
                supplier_raw,
                (
                    "target_qty",
                    "received_qty",
                    "receipt_fulfillment_rate",
                    "purchase_order_count",
                    "multi_supplier_flag",
                ),
            )
            supplier = supplier_raw.groupby("sku_code", as_index=False).agg(
                supplier_fulfillment=("receipt_fulfillment_rate", "mean"),
                multi_supplier=("multi_supplier_flag", "max"),
            )
            features = features.merge(supplier, on="sku_code", how="left")
        if not member_raw.empty:
            member_raw = normalize_sku_column(member_raw)
            member_raw = _to_numeric(member_raw, MEMBER_COLUMNS[3:])
            member_raw = member_raw.sort_values(["sku_code", "ds"]).drop_duplicates("sku_code", keep="last")
            member_raw["member_young_rate"] = member_raw[
                ["age_under_18_rate", "age_18_24_rate", "age_25_34_rate"]
            ].sum(axis=1, min_count=1)
            members = member_raw[["sku_code", "member_count", "female_rate", "member_young_rate"]].rename(
                columns={"female_rate": "member_female_rate"}
            )
            features = features.merge(members, on="sku_code", how="left")
        if restored_weekly is not None and not restored_weekly.empty:
            require_columns(
                restored_weekly,
                (
                    "sku_code",
                    "lifecycle_week",
                    "v_neutral",
                    "n_eff",
                    "restored_exposure_days",
                ),
                "restored_weekly",
            )
            restored = normalize_sku_column(restored_weekly)
            restored = _to_numeric(
                restored,
                ("lifecycle_week", "v_neutral", "n_eff", "restored_exposure_days"),
            )
            history = restored.groupby("sku_code", as_index=False).agg(
                historical_velocity=("v_neutral", "mean"),
                exposure_quality=("n_eff", lambda value: float(np.clip(value.sum() / (len(value) * 7.0), 0, 1))),
            )
            # V3.2 PR-04 / PF-02: lifecycle persistence is based on the
            # neutral-sales curve, exemplified by weeks 5-10 divided by weeks
            # 1-10. Aggregate tiers to an all-store weekly velocity using their
            # restored exposure days; incomplete 10-week histories remain NA.
            restored["weighted_v_neutral"] = (
                restored["v_neutral"] * restored["restored_exposure_days"].clip(lower=0)
            )
            weekly = (
                restored.loc[restored["lifecycle_week"].between(1, 10)]
                .groupby(["sku_code", "lifecycle_week"], as_index=False)
                .agg(
                    weighted_v_neutral=("weighted_v_neutral", "sum"),
                    exposure_days=("restored_exposure_days", "sum"),
                    mean_v_neutral=("v_neutral", "mean"),
                )
            )
            weekly["all_store_v_neutral"] = np.where(
                weekly["exposure_days"] > 0,
                weekly["weighted_v_neutral"] / weekly["exposure_days"],
                weekly["mean_v_neutral"],
            )
            persistence_rows: list[dict[str, Any]] = []
            required_weeks = set(range(1, 11))
            for sku_code, group in weekly.groupby("sku_code"):
                valid = group.loc[group["all_store_v_neutral"].notna()].copy()
                observed_weeks = set(valid["lifecycle_week"].astype(int))
                denominator = float(valid["all_store_v_neutral"].sum())
                if required_weeks.issubset(observed_weeks) and denominator > 0:
                    numerator = float(
                        valid.loc[
                            valid["lifecycle_week"].between(5, 10),
                            "all_store_v_neutral",
                        ].sum()
                    )
                    value = numerator / denominator
                    status = "v3_2_pr04_weeks_5_10_over_1_10"
                else:
                    value = np.nan
                    status = (
                        "missing_nonpositive_weeks_1_10"
                        if required_weeks.issubset(observed_weeks)
                        else "missing_incomplete_weeks_1_10"
                    )
                persistence_rows.append(
                    {
                        "sku_code": sku_code,
                        "lifecycle_persistence": value,
                        "lifecycle_persistence_status": status,
                    }
                )
            persistence = pd.DataFrame(
                persistence_rows,
                columns=(
                    "sku_code",
                    "lifecycle_persistence",
                    "lifecycle_persistence_status",
                ),
            )
            history = history.merge(persistence, on="sku_code", how="left")
            features = features.merge(history, on="sku_code", how="left")
        if "lifecycle_persistence" not in features:
            features["lifecycle_persistence"] = np.nan
            features["lifecycle_persistence_status"] = "missing_restore_history"
        else:
            features["lifecycle_persistence_status"] = features[
                "lifecycle_persistence_status"
            ].fillna("missing_restore_history")
        # Unknown is not zero: an unmapped IP does not mean "no IP", and an
        # absent/null supplier record does not prove that a SKU has one supplier.
        if "multi_supplier" not in features:
            features["multi_supplier"] = np.nan
        features["multi_supplier_status"] = np.where(
            features["multi_supplier"].notna(),
            "observed_supplier_flag",
            "missing_supplier_flag",
        )
        return features, {
            "product": product.source_partition,
            "supplier": supplier_partition,
            "packaging_attributes": f"product_master:{product.source_partition}",
            "member": member_partition,
        }


def build_repository(*, data_root: Path, config: AppConfig) -> ParquetDataRepository:
    """Central factory used by legacy dataset/probe orchestration scripts."""

    return ParquetDataRepository(
        data_root=data_root,
        table_specs=load_table_specs(config.probe.table_specs_path),
        max_rows_per_read=config.data_reader.max_rows_per_read,
    )


def select_target_skus(
    cohort: pd.DataFrame,
    *,
    sku_codes: Sequence[str],
    limit_skus: int | None,
) -> pd.DataFrame:
    require_columns(cohort, ("sku_code", "first_actual_sale_date", "split"), "cohort")
    selected = filter_skus(cohort, sku_codes, limit_skus=limit_skus)
    if not sku_codes and limit_skus is not None:
        selected = (
            normalize_sku_column(cohort)
            .sort_values(["first_actual_sale_date", "sku_code"], ascending=[False, True])
        )
        keep = selected["sku_code"].drop_duplicates().head(limit_skus)
        selected = selected.loc[selected["sku_code"].isin(set(keep))]
    return selected.reset_index(drop=True)


def standardize_reorder_actuals(frame: pd.DataFrame) -> pd.DataFrame:
    """Map a standardized M2 daily audit to the canonical N1 actual-demand input."""

    require_columns(
        frame,
        ("sku_code", "store_code", "store_tier", "business_day", "e_stock", "e_display"),
        "reorder actual source",
    )
    result = frame.copy()
    if "actual_sales" not in result:
        if "clean_sale_qty" not in result:
            raise DataContractViolation(
                "reorder actual source needs actual_sales or M2 clean_sale_qty"
            )
        result["actual_sales"] = pd.to_numeric(
            result["clean_sale_qty"], errors="coerce"
        ).fillna(0.0)
    result["sku_code"] = result["sku_code"].astype(str)
    result["store_code"] = result["store_code"].astype(str)
    result["business_day"] = pd.to_datetime(
        result["business_day"], errors="raise"
    ).dt.date
    keep = [
        "sku_code", "store_code", "store_tier", "business_day", "actual_sales",
        "e_stock", "e_display",
    ]
    if "available_stock_qty" in result:
        keep.append("available_stock_qty")
    return result[keep].reset_index(drop=True)
