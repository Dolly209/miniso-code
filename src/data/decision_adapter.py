from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import AppConfig, load_table_specs
from data.domain import filter_skus, normalize_sku_column, require_columns
from data.domestic import DataContractViolation
from data.parquet_repository import ParquetDataRepository


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


@dataclass(frozen=True)
class ProductDomain:
    products: pd.DataFrame
    ip_membership: pd.DataFrame
    source_partition: str


@dataclass(frozen=True)
class PlacementEvidenceDomain:
    evidence: pd.DataFrame
    source_partitions: tuple[str, ...]
    warnings: tuple[str, ...]


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
        self, cutoff_time: datetime, *, sku_codes: Sequence[str] | None = None
    ) -> ProductDomain:
        raw, partition = self._latest(
            "product", cutoff_time, PRODUCT_COLUMNS, sku_codes=sku_codes
        )
        require_columns(raw, PRODUCT_COLUMNS, "product source")
        raw = normalize_sku_column(raw)
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
        return ProductDomain(attributes, ip, partition)

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

    def placement_launch_evidence(
        self,
        cutoff_time: datetime,
        *,
        candidate_products: pd.DataFrame,
        eligible_stores: pd.DataFrame,
        restored_store_weekly: pd.DataFrame,
        launch_window_days: int,
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
        stores = eligible_stores[["store_code", "store_tier"]].copy()
        stores["store_code"] = stores["store_code"].astype(str)
        stores["store_tier"] = stores["store_tier"].astype(str)
        stores = stores.drop_duplicates("store_code")

        restored = restored_store_weekly.copy()
        for column in ("sku_code", "store_code", "store_tier"):
            restored[column] = restored[column].astype(str)
        restored["v_neutral"] = pd.to_numeric(
            restored["v_neutral"], errors="coerce"
        )
        restored["stockout_days"] = pd.to_numeric(
            restored["stockout_days"], errors="coerce"
        )
        restored["source_day_rows"] = pd.to_numeric(
            restored["source_day_rows"], errors="coerce"
        )
        performance = (
            restored.groupby(["sku_code", "store_code"], as_index=False)
            .agg(
                store_tier_history=("store_tier", "last"),
                v_neutral=("v_neutral", "mean"),
                stockout_days=("stockout_days", "sum"),
                source_day_rows=("source_day_rows", "sum"),
            )
        )
        performance["stock_pressure"] = performance["stockout_days"] / performance[
            "source_day_rows"
        ].replace(0, np.nan)

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
            if pd.notna(first_sale):
                launch_day = first_sale.date()
                launch_source = "product.first_sale_date"
            elif pd.notna(release):
                launch_day = release.date()
                launch_source = "product.release_dt_fallback"
            else:
                warnings.append(
                    f"{product.sku_code}: no first_sale_date/release_dt; "
                    "restore-derived arrival proxy will be used"
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
            else:
                arrivals = pd.DataFrame(columns=["store_code", "arrival_time"])

            universe = stores.copy()
            universe["sku_code"] = str(product.sku_code)
            universe = universe.merge(
                arrivals, on="store_code", how="left", validate="one_to_one"
            )
            arrival_values = universe["arrival_time"]
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
                "current_cutoff_store_universe_proxy"
            )
            universe["placement_source_row_count"] = len(raw)
            evidence_frames.append(universe)

        if not evidence_frames:
            return PlacementEvidenceDomain(
                pd.DataFrame(), tuple(sorted(partitions)), tuple(warnings)
            )
        evidence = pd.concat(evidence_frames, ignore_index=True)
        evidence = evidence.merge(
            performance.drop(columns="store_tier_history"),
            on=["sku_code", "store_code"],
            how="left",
            validate="one_to_one",
        )
        evidence["evidence_weight"] = 1.0
        return PlacementEvidenceDomain(
            evidence[
                [
                    "sku_code",
                    "store_code",
                    "store_tier",
                    "eligible_at_launch",
                    "arrived_in_launch_window",
                    "v_neutral",
                    "stock_pressure",
                    "evidence_weight",
                    "arrival_time",
                    "launch_date",
                    "launch_window_end",
                    "launch_date_source",
                    "eligibility_universe_status",
                    "placement_source_row_count",
                ]
            ],
            tuple(sorted(partitions)),
            tuple(warnings),
        )

    def factor_features(
        self,
        cutoff_time: datetime,
        *,
        sku_codes: Sequence[str] | None = None,
        restored_weekly: pd.DataFrame | None = None,
        standard_ip_membership: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        product = self.products(cutoff_time, sku_codes=sku_codes)
        supplier_raw, supplier_partition = self._latest(
            "supplier", cutoff_time, SUPPLIER_COLUMNS, sku_codes=sku_codes
        )
        pack_raw, pack_partition = self._latest(
            "pack", cutoff_time, PACK_COLUMNS, sku_codes=sku_codes
        )
        member_raw, member_partition = self._latest(
            "member", cutoff_time, MEMBER_COLUMNS, sku_codes=sku_codes
        )
        features = product.products[[
            "sku_code", "retail_price", "lifecycle", "copyright_category", "cate_level4_code"
        ]].copy()
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
        if not pack_raw.empty:
            pack_raw = normalize_sku_column(pack_raw)
            pack_raw = _to_numeric(pack_raw, ("goods_unit", "pkg_unit", "box_unit"))
            pack = pack_raw.groupby("sku_code", as_index=False).agg(
                pack_size=("pkg_unit", "max"), box_size=("box_unit", "max")
            )
            features = features.merge(pack, on="sku_code", how="left")
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
            "pack": pack_partition,
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
