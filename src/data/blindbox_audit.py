from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
import unicodedata
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from data.decision_adapter import SOURCES
from data.domain import normalize_sku_column, require_columns
from data.domestic import DataContractViolation
from data.parquet_repository import ParquetDataRepository
from data.restoration_adapter import INVENTORY_COLUMNS, SALES_COLUMNS
from restore.historical import normalize_order_lines


ProgressCallback = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class BlindBoxClassification:
    variants: pd.DataFrame
    family_summary: pd.DataFrame


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


_MISSING_MASTER_TOKENS = {
    "",
    "-",
    "--",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "unknown",
    "无",
    "未知",
}


def _optional_master_text(value: object) -> str:
    """Normalize source placeholders without inventing a master-data value."""

    text = _text(value)
    if text.casefold() in _MISSING_MASTER_TOKENS:
        return ""
    return text


def _spu_code(value: object) -> str:
    """Return only usable SPU identifiers.

    The product source uses all-zero values when no SPU has been assigned.  An
    all-zero value must never create a relationship between otherwise unrelated
    products.
    """

    text = _optional_master_text(value)
    if text and re.fullmatch(r"0+", text):
        return ""
    return text


def _normalized_family(value: object) -> str:
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    # Keep the series/IP words but remove sales-unit and display words.  This
    # key is only a secondary heuristic; shared SPU is always preferred.
    text = re.sub(
        r"(盲盒|盲袋|盲抽|随机款|端盒|整盒|明盒|隐藏款?|普通款|常规款|手办|摆件|公仔|挂件|icon\s*box)",
        "",
        text,
    )
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _series_prefix(value: object) -> str:
    raw = unicodedata.normalize("NFKC", _text(value))
    parts = re.split(r"[-—–_（(【\[]", raw, maxsplit=1)
    prefix = parts[0] if parts else raw
    return _normalized_family(prefix)


def classify_blindbox_variants(
    products: pd.DataFrame,
    *,
    sku_spu_overrides: Mapping[str, str] | None = None,
    allow_name_parent_match: bool = True,
) -> BlindBoxClassification:
    """Classify explicit and strongly-linked blind-box variant SKUs.

    Direct name tokens are reported separately from inferred child variants.
    The inference is accepted only when the SKU shares an SPU with a normal
    blind-box parent or its normalized series prefix matches one.  It is never
    silently treated as a confirmed business master-data relationship.
    """

    require_columns(
        products,
        ("sku_code", "sku_name", "spu_code", "spu_name"),
        "product domain",
    )
    frame = normalize_sku_column(products).drop_duplicates("sku_code").copy()
    frame["sku_name"] = frame["sku_name"].map(_text)
    frame["spu_name"] = frame["spu_name"].map(_optional_master_text)
    frame["spu_code"] = frame["spu_code"].map(_spu_code)
    frame["relationship_spu_code"] = frame["spu_code"]
    frame["spu_resolution_source"] = np.where(
        frame["spu_code"].ne(""), "product_master", "missing"
    )
    overrides = {
        str(sku): _spu_code(value)
        for sku, value in dict(sku_spu_overrides or {}).items()
        if _spu_code(value)
    }
    override_values = frame["sku_code"].map(overrides).fillna("")
    use_override = frame["relationship_spu_code"].eq("") & override_values.ne("")
    frame.loc[use_override, "relationship_spu_code"] = override_values.loc[use_override]
    frame.loc[use_override, "spu_resolution_source"] = "observed_source_spu_id"
    frame["combined_name"] = frame["sku_name"] + "|" + frame["spu_name"]
    frame["has_blindbox_base_cue"] = frame["combined_name"].str.contains(
        "盲盒|盲袋|盲抽|随机款", regex=True
    )
    frame["is_end_box"] = (
        frame["combined_name"].str.contains("端盒", regex=False)
        | (
            frame["combined_name"].str.contains("整盒", regex=False)
            & frame["has_blindbox_base_cue"]
        )
    )
    frame["is_hidden"] = (
        frame["combined_name"].str.contains("隐藏款?|隐藏", regex=True)
        & frame["has_blindbox_base_cue"]
    )
    frame["is_open_box"] = frame["combined_name"].str.contains("明盒", regex=False)
    frame["has_blindbox_cue"] = frame["combined_name"].str.contains(
        "盲盒|盲袋|盲抽|随机款|端盒|整盒|明盒|隐藏", regex=True
    )
    frame["has_variant_suffix"] = frame["sku_name"].str.contains(
        r"(?:盲盒|盲袋|盲抽|随机款).*[-—–_（(【\[]", regex=True
    )
    frame["is_parent_candidate"] = (
        frame["has_blindbox_base_cue"]
        & ~frame[["is_end_box", "is_hidden", "is_open_box"]].any(axis=1)
        & ~frame["has_variant_suffix"]
    )
    frame["sku_family_key"] = frame["sku_name"].map(_series_prefix)
    frame["spu_family_key"] = frame["spu_name"].map(_series_prefix)
    frame["family_key"] = frame["spu_family_key"].where(
        frame["spu_family_key"].str.len().ge(4), frame["sku_family_key"]
    )

    parents = frame.loc[frame["is_parent_candidate"]].copy()
    parent_by_spu: dict[str, str] = {}
    for spu, group in parents.loc[
        parents["relationship_spu_code"].ne("")
    ].groupby("relationship_spu_code"):
        candidates = sorted(set(group["sku_code"].astype(str)))
        if len(candidates) == 1:
            parent_by_spu[str(spu)] = candidates[0]
    parent_by_family: dict[str, str] = {}
    parent_family_candidates = pd.concat(
        [
            parents[["sku_code", "sku_family_key"]].rename(
                columns={"sku_family_key": "family_key"}
            ),
            parents[["sku_code", "spu_family_key"]].rename(
                columns={"spu_family_key": "family_key"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    parent_family_candidates = parent_family_candidates.loc[
        parent_family_candidates["family_key"].str.len().ge(4)
    ]
    for family, group in parent_family_candidates.groupby("family_key"):
        candidates = sorted(set(group["sku_code"].astype(str)))
        if len(candidates) == 1:
            parent_by_family[str(family)] = candidates[0]

    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        if row.is_parent_candidate:
            continue
        variant_type = None
        method = None
        confidence = None
        if row.is_end_box:
            variant_type, method, confidence = "end_box", "direct_name_token", "confirmed_token"
        elif row.is_hidden:
            variant_type, method, confidence = "hidden_variant", "direct_name_token", "confirmed_token"
        elif row.is_open_box:
            variant_type, method, confidence = "open_box", "direct_name_token", "confirmed_token"
        parent_sku = None
        parent_method = None
        if row.relationship_spu_code and row.relationship_spu_code in parent_by_spu:
            candidate = parent_by_spu[row.relationship_spu_code]
            if candidate != row.sku_code:
                parent_sku = candidate
                parent_method = (
                    "shared_product_spu_code"
                    if row.spu_resolution_source == "product_master"
                    else "shared_observed_spu_id"
                )
        if parent_sku is None and allow_name_parent_match:
            family_keys = []
            for family_key in (row.spu_family_key, row.sku_family_key):
                if len(family_key) >= 4 and family_key not in family_keys:
                    family_keys.append(family_key)
            matched_parents = {
                parent_by_family[family_key]
                for family_key in family_keys
                if family_key in parent_by_family
                and parent_by_family[family_key] != row.sku_code
            }
            if len(matched_parents) == 1:
                parent_sku = next(iter(matched_parents))
                parent_method = "normalized_series_prefix"
        if variant_type is None and parent_sku is not None and row.has_blindbox_cue:
            variant_type = "child_variant"
            method = parent_method
            confidence = "inferred_strong_link"
        if variant_type is None:
            continue
        rows.append(
            {
                "sku_code": str(row.sku_code),
                "sku_name": row.sku_name,
                "spu_code": row.spu_code,
                "spu_name": row.spu_name,
                "relationship_spu_code": row.relationship_spu_code,
                "spu_resolution_source": row.spu_resolution_source,
                "variant_type": variant_type,
                "classification_method": method,
                "classification_confidence": confidence,
                "family_key": row.family_key,
                "parent_sku_code": parent_sku,
                "parent_match_method": parent_method,
            }
        )
    variants = pd.DataFrame(rows)
    if variants.empty:
        variants = pd.DataFrame(
            columns=[
                "sku_code", "sku_name", "spu_code", "spu_name",
                "relationship_spu_code", "spu_resolution_source", "variant_type",
                "classification_method", "classification_confidence", "family_key",
                "parent_sku_code", "parent_match_method",
            ]
        )
        return BlindBoxClassification(variants, pd.DataFrame())
    family = (
        variants.assign(
            family_identifier=np.where(
                variants["parent_sku_code"].notna(),
                variants["parent_sku_code"],
                variants["family_key"],
            )
        )
        .groupby("family_identifier", as_index=False, dropna=False)
        .agg(
            variant_sku_count=("sku_code", "nunique"),
            variant_types=("variant_type", lambda value: "|".join(sorted(set(value)))),
            parent_sku_code=("parent_sku_code", "first"),
            example_variant_sku=("sku_code", "first"),
            example_variant_name=("sku_name", "first"),
        )
    )
    return BlindBoxClassification(variants, family)


def _filter_sku_windows(
    frame: pd.DataFrame,
    *,
    sku_column: str,
    date_values: pd.Series,
    windows: Mapping[str, tuple[date, date]],
) -> pd.DataFrame:
    skus = frame[sku_column].astype("string").str.strip()
    starts = skus.map({key: value[0] for key, value in windows.items()})
    ends = skus.map({key: value[1] for key, value in windows.items()})
    keep = starts.notna() & ends.notna() & date_values.ge(starts) & date_values.le(ends)
    return frame.loc[keep].copy()


class BlindBoxAuditDataAdapter:
    """Bounded raw-source access for the blind-box grain audit only."""

    def __init__(self, repository: ParquetDataRepository) -> None:
        self.repository = repository

    def resolve_spu_ids(
        self,
        *,
        entities: pd.DataFrame,
        cutoff_time: datetime,
        batch_size: int = 50_000,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        """Resolve one unambiguous mother-code value from sales/inventory evidence."""

        require_columns(entities, ("sku_code", "window_start", "window_end"), "SPU entities")
        output_columns = [
            "sku_code",
            "observed_sales_spu_ids",
            "observed_inventory_spu_ids",
            "observed_spu_candidate_count",
            "resolved_observed_spu_id",
            "observed_spu_resolution_status",
        ]
        if entities.empty:
            return pd.DataFrame(columns=output_columns)
        windows = {
            str(row.sku_code): (row.window_start, row.window_end)
            for row in entities.itertuples(index=False)
        }
        start_date = min(value[0] for value in windows.values())
        end_date = max(value[1] for value in windows.values())
        skus = list(windows)
        candidates: dict[str, dict[str, set[str]]] = {
            sku: {"sales": set(), "inventory": set()} for sku in skus
        }
        source_specs = (
            (
                "sales",
                SOURCES["sales"],
                ("ds", "sku_code", "spu_id"),
                "ds",
            ),
            (
                "inventory",
                SOURCES["inventory"],
                ("ds", "stock_date", "sku_code", "spu_id"),
                "stock_date",
            ),
        )
        for source_label, source_name, columns, date_column in source_specs:
            if progress:
                progress(
                    "spu_read_start",
                    {
                        "source": source_label,
                        "skus": len(skus),
                        "start_date": str(start_date),
                        "end_date": str(end_date),
                    },
                )
            rows = 0
            arrow_batches = 0
            for arrow_batches, arrow_batch in enumerate(
                self.repository.iter_partition_batches(
                    source_name=source_name,
                    start_date=start_date,
                    end_date=end_date,
                    cutoff_time=cutoff_time,
                    columns=columns,
                    sku_codes=skus,
                    batch_size=batch_size,
                ),
                start=1,
            ):
                frame = arrow_batch.to_pandas()
                if frame.empty:
                    continue
                frame["sku_code"] = frame["sku_code"].astype("string").str.strip()
                dates = pd.to_datetime(frame[date_column], errors="coerce").dt.date
                if dates.isna().any():
                    raise DataContractViolation(
                        f"blind-box audit {source_label} contains invalid {date_column}"
                    )
                frame = _filter_sku_windows(
                    frame,
                    sku_column="sku_code",
                    date_values=dates,
                    windows=windows,
                )
                rows += len(frame)
                frame["__spu_id"] = frame["spu_id"].map(_spu_code)
                frame = frame.loc[frame["__spu_id"].ne("")]
                for sku, group in frame.groupby("sku_code", sort=False):
                    candidates[str(sku)][source_label].update(
                        group["__spu_id"].astype(str)
                    )
                if progress and (arrow_batches == 1 or arrow_batches % 1000 == 0):
                    progress(
                        "spu_read_progress",
                        {
                            "source": source_label,
                            "arrow_batches": arrow_batches,
                            "rows": rows,
                        },
                    )
            if progress:
                progress(
                    "spu_read_complete",
                    {
                        "source": source_label,
                        "arrow_batches": arrow_batches,
                        "rows": rows,
                    },
                )

        rows: list[dict[str, Any]] = []
        for sku in skus:
            sales_values = candidates[sku]["sales"]
            inventory_values = candidates[sku]["inventory"]
            combined = sales_values | inventory_values
            if len(combined) == 1:
                resolved = next(iter(combined))
                status = "unique_observed_spu_id"
            elif combined:
                resolved = ""
                status = "conflicting_observed_spu_ids"
            else:
                resolved = ""
                status = "no_valid_observed_spu_id"
            rows.append(
                {
                    "sku_code": sku,
                    "observed_sales_spu_ids": "|".join(sorted(sales_values)),
                    "observed_inventory_spu_ids": "|".join(sorted(inventory_values)),
                    "observed_spu_candidate_count": len(combined),
                    "resolved_observed_spu_id": resolved,
                    "observed_spu_resolution_status": status,
                }
            )
        return pd.DataFrame(rows, columns=output_columns)

    def entity_metrics(
        self,
        *,
        entities: pd.DataFrame,
        cutoff_time: datetime,
        batch_size: int = 50_000,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        require_columns(entities, ("sku_code", "window_start", "window_end"), "audit entities")
        if entities.empty:
            return pd.DataFrame()
        windows = {
            str(row.sku_code): (row.window_start, row.window_end)
            for row in entities.itertuples(index=False)
        }
        start_date = min(value[0] for value in windows.values())
        end_date = max(value[1] for value in windows.values())
        skus = list(windows)
        row_limit = self.repository.max_rows_per_read * max(len(skus), 1)
        if progress:
            progress(
                "sales_read_start",
                {"skus": len(skus), "start_date": str(start_date), "end_date": str(end_date)},
            )
        sales_read = self.repository.read_partition_range(
            source_name=SOURCES["sales"],
            start_date=start_date,
            end_date=end_date,
            cutoff_time=cutoff_time,
            columns=SALES_COLUMNS,
            sku_codes=skus,
            max_rows=row_limit,
        )
        raw_sales = sales_read.frame
        if not raw_sales.empty:
            raw_days = pd.to_datetime(raw_sales["ds"], errors="coerce").dt.date
            if raw_days.isna().any():
                raise DataContractViolation("blind-box audit sales source contains invalid ds")
            raw_sales = _filter_sku_windows(
                raw_sales,
                sku_column="sku_code",
                date_values=raw_days,
                windows=windows,
            )
        normalized_sales, sales_stats = normalize_order_lines(
            raw_sales, cutoff_time=cutoff_time
        )
        if not normalized_sales.empty:
            normalized_sales = _filter_sku_windows(
                normalized_sales,
                sku_column="sku_code",
                date_values=normalized_sales["business_day"],
                windows=windows,
            )
        sales_metrics = (
            normalized_sales.groupby("sku_code", as_index=False).agg(
                net_sale_qty=("sale_qty", "sum"),
                positive_sale_qty=("sale_qty", lambda value: float(value.clip(lower=0).sum())),
                negative_sale_qty=("sale_qty", lambda value: float(value.clip(upper=0).sum())),
                latest_order_line_count=("order_detail_id", "size"),
                selling_store_count=("store_code", "nunique"),
            )
            if not normalized_sales.empty
            else pd.DataFrame(columns=[
                "sku_code", "net_sale_qty", "positive_sale_qty", "negative_sale_qty",
                "latest_order_line_count", "selling_store_count",
            ])
        )
        if progress:
            progress(
                "sales_read_complete",
                {
                    "raw_rows": len(raw_sales),
                    "latest_rows": len(normalized_sales),
                    **sales_stats,
                },
            )

        inventory_totals: dict[str, dict[str, float]] = {
            sku: {
                "inventory_row_count": 0,
                "positive_available_row_count": 0,
                "zero_available_row_count": 0,
                "null_available_row_count": 0,
                "available_qty_store_day_sum": 0.0,
                "max_available_qty": np.nan,
            }
            for sku in skus
        }
        if progress:
            progress(
                "inventory_read_start",
                {"skus": len(skus), "start_date": str(start_date), "end_date": str(end_date)},
            )
        inventory_rows = 0
        arrow_batches = 0
        for arrow_batches, arrow_batch in enumerate(
            self.repository.iter_partition_batches(
                source_name=SOURCES["inventory"],
                start_date=start_date,
                end_date=end_date,
                cutoff_time=cutoff_time,
                columns=INVENTORY_COLUMNS,
                sku_codes=skus,
                batch_size=batch_size,
            ),
            start=1,
        ):
            frame = arrow_batch.to_pandas()
            if frame.empty:
                continue
            frame["sku_code"] = frame["sku_code"].astype("string").str.strip()
            stock_days = pd.to_datetime(frame["stock_date"], errors="coerce").dt.date
            if stock_days.isna().any():
                raise DataContractViolation("blind-box audit inventory contains invalid stock_date")
            frame = _filter_sku_windows(
                frame,
                sku_column="sku_code",
                date_values=stock_days,
                windows=windows,
            )
            available = pd.to_numeric(frame["available_stock_qty"], errors="coerce")
            invalid = frame["available_stock_qty"].notna() & available.isna()
            if invalid.any():
                raise DataContractViolation(
                    "blind-box audit inventory contains non-numeric available_stock_qty"
                )
            frame = frame.assign(__available=available)
            inventory_rows += len(frame)
            for sku, group in frame.groupby("sku_code", sort=False):
                values = group["__available"]
                target = inventory_totals[str(sku)]
                target["inventory_row_count"] += int(len(group))
                target["positive_available_row_count"] += int(values.gt(0).sum())
                target["zero_available_row_count"] += int(values.eq(0).sum())
                target["null_available_row_count"] += int(values.isna().sum())
                batch_sum = values.sum(min_count=1)
                if pd.notna(batch_sum):
                    target["available_qty_store_day_sum"] += float(batch_sum)
                batch_max = values.max()
                if pd.notna(batch_max):
                    target["max_available_qty"] = float(
                        batch_max
                        if pd.isna(target["max_available_qty"])
                        else max(float(target["max_available_qty"]), float(batch_max))
                    )
            if progress and (arrow_batches == 1 or arrow_batches % 1000 == 0):
                progress(
                    "inventory_read_progress",
                    {"arrow_batches": arrow_batches, "rows": inventory_rows},
                )
        inventory_metrics = pd.DataFrame(
            [{"sku_code": sku, **values} for sku, values in inventory_totals.items()]
        )
        if progress:
            progress(
                "inventory_read_complete",
                {"arrow_batches": arrow_batches, "rows": inventory_rows},
            )
        result = entities[["sku_code", "window_start", "window_end"]].copy()
        result["sku_code"] = result["sku_code"].astype(str)
        result = result.merge(sales_metrics, on="sku_code", how="left", validate="one_to_one")
        result = result.merge(inventory_metrics, on="sku_code", how="left", validate="one_to_one")
        for column in (
            "net_sale_qty", "positive_sale_qty", "negative_sale_qty",
            "latest_order_line_count", "selling_store_count", "inventory_row_count",
            "positive_available_row_count", "zero_available_row_count",
            "null_available_row_count", "available_qty_store_day_sum",
        ):
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
        result["positive_inventory_row_rate"] = np.where(
            result["inventory_row_count"].gt(0),
            result["positive_available_row_count"] / result["inventory_row_count"],
            np.nan,
        )
        result["inventory_evidence_status"] = np.select(
            [
                result["inventory_row_count"].eq(0),
                result["positive_available_row_count"].eq(0),
            ],
            ["no_inventory_rows", "inventory_rows_all_nonpositive_or_null"],
            default="positive_inventory_observed",
        )
        result["sales_inventory_issue"] = np.select(
            [
                result["net_sale_qty"].gt(0) & result["inventory_row_count"].eq(0),
                result["net_sale_qty"].gt(0) & result["inventory_row_count"].gt(0)
                & result["positive_available_row_count"].eq(0),
            ],
            ["sales_positive_inventory_missing", "sales_positive_inventory_all_nonpositive"],
            default="not_observed",
        )
        return result


def audit_windows(
    cohort: pd.DataFrame,
    *,
    sku_codes: Sequence[str],
    cutoff_time: datetime,
    window_days: int,
) -> pd.DataFrame:
    require_columns(cohort, ("sku_code", "first_actual_sale_date"), "cohort")
    selected = normalize_sku_column(cohort)
    selected = selected.loc[selected["sku_code"].isin(set(map(str, sku_codes)))].copy()
    selected["window_start"] = pd.to_datetime(
        selected["first_actual_sale_date"], errors="coerce"
    ).dt.date
    if selected["window_start"].isna().any():
        examples = selected.loc[selected["window_start"].isna(), "sku_code"].head(5).tolist()
        raise DataContractViolation(f"audit cohort has invalid first sale dates: {examples}")
    selected["window_end"] = selected["window_start"].map(
        lambda value: min(value + timedelta(days=window_days - 1), cutoff_time.date())
    )
    return selected[["sku_code", "window_start", "window_end"]].drop_duplicates("sku_code")
