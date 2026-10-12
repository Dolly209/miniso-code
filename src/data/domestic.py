from __future__ import annotations

from datetime import date
from typing import Sequence

from data.parquet_repository import ParquetDataRepository, ReadResult
from data.point_in_time import PointInTimeContext


PRODUCT_SOURCE = "scm.dim_goods_info_ip_forecast"
SALES_SOURCE = "scm.dwd_trd_order_all_flow_di_ip_forecast"
INVENTORY_SOURCE = "scm.dwd_zhgyl_store_sale_inv_ip_forecast"
NON_GIFT_FLAG_VALUES = frozenset({"0", "0.0", "false", "n", "no", "否"})


class DataContractViolation(ValueError):
    """Raised when delivered rows violate a documented V3.2 physical contract."""


def _parse_local_timestamp(frame: object, column: str, context: PointInTimeContext) -> object:
    import pandas as pd

    values = frame[column]
    parsed = pd.to_datetime(values, errors="coerce")
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).head(5).tolist()
        raise DataContractViolation(f"unparseable {column} values: {examples}")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(context.decision_cutoff_time.tzinfo)
    else:
        parsed = parsed.dt.tz_convert(context.decision_cutoff_time.tzinfo)
    return parsed


def _normal_zero_flag(values: object) -> object:
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"0", "0.0", "false"})


def _non_gift_flag(values: object) -> object:
    """Recognize documented/observed encodings for a non-gift sales row."""

    normalized = values.astype(str).str.strip().str.lower()
    return values.isna() | normalized.isin(NON_GIFT_FLAG_VALUES)


def validate_delivered_sales_flags(frame: object) -> None:
    if not _normal_zero_flag(frame["is_return"]).all():
        bad = frame.loc[~_normal_zero_flag(frame["is_return"]), "is_return"]
        raise DataContractViolation(
            "sales source contains return rows although the delivered contract requires "
            f"is_return='0'; examples={bad.astype(str).head(5).tolist()}"
        )
    non_gift = _non_gift_flag(frame["gift_flag"])
    if not non_gift.all():
        bad = frame.loc[~non_gift, "gift_flag"]
        raise DataContractViolation(
            "sales source contains gift rows although the delivered contract excludes them; "
            f"examples={bad.astype(str).head(5).tolist()}"
        )


class SalesDataService:
    """Physical sales implementation for the delivered normal-sales-only table."""

    REQUIRED_COLUMNS = (
        "ds",
        "order_id",
        "order_detail_id",
        "order_time",
        "sku_code",
        "store_code",
        "sale_qty",
        "order_amt",
        "paid_amt",
        "achievement_amt",
        "is_return",
        "gift_flag",
    )

    def __init__(self, repository: ParquetDataRepository) -> None:
        self.repository = repository

    def read_order_lines(
        self,
        context: PointInTimeContext,
        *,
        start_date: date,
        extra_columns: Sequence[str] = (),
        max_rows: int | None = None,
    ) -> ReadResult:
        columns = tuple(dict.fromkeys(self.REQUIRED_COLUMNS + tuple(extra_columns)))
        raw = self.repository.read_history(
            context,
            source_name=SALES_SOURCE,
            start_date=start_date,
            columns=columns,
            max_rows=max_rows,
        )
        frame = raw.frame.copy()
        if frame.empty:
            frame["business_day"] = []
            return ReadResult(
                **{**raw.__dict__, "frame": frame}
            )

        validate_delivered_sales_flags(frame)

        key = ["order_id", "order_detail_id"]
        same_version_duplicate = frame.duplicated(subset=["ds", *key], keep=False)
        if same_version_duplicate.any():
            examples = frame.loc[same_version_duplicate, ["ds", *key]].head(5)
            raise DataContractViolation(
                "duplicate order lines exist within one ds version; examples="
                + examples.to_dict(orient="records").__repr__()
            )
        frame = frame.sort_values(["ds", *key]).drop_duplicates(subset=key, keep="last")
        order_time = _parse_local_timestamp(frame, "order_time", context)
        visible = order_time <= context.decision_cutoff_time
        frame = frame.loc[visible].copy()
        order_time = order_time.loc[visible]
        frame["order_time"] = order_time
        frame["business_day"] = order_time.dt.date
        numeric = ("sale_qty", "order_amt", "paid_amt", "achievement_amt")
        import pandas as pd

        for column in numeric:
            converted = pd.to_numeric(frame[column], errors="coerce")
            invalid = frame[column].notna() & converted.isna()
            if invalid.any():
                raise DataContractViolation(f"non-numeric sales measure: {column}")
            frame[column] = converted
        return ReadResult(**{**raw.__dict__, "frame": frame.reset_index(drop=True)})

    @staticmethod
    def aggregate_sku_store_day(lines: ReadResult) -> object:
        measures = ["sale_qty", "order_amt", "paid_amt", "achievement_amt"]
        frame = lines.frame
        if frame.empty:
            return frame[["sku_code", "store_code", "business_day", *measures]].copy()
        return (
            frame.groupby(
                ["sku_code", "store_code", "business_day"],
                as_index=False,
                dropna=False,
            )[measures]
            .sum(min_count=1)
            .sort_values(["business_day", "store_code"])
            .reset_index(drop=True)
        )

    @staticmethod
    def actual_first_sale_date(lines: ReadResult) -> date | None:
        positive = lines.frame.loc[lines.frame["sale_qty"] > 0, "business_day"]
        return None if positive.empty else min(positive)


class InventoryDataService:
    """Store-SKU-day inventory snapshots without transit double counting."""

    REQUIRED_COLUMNS = (
        "ds",
        "stock_date",
        "store_code",
        "sku_code",
        "available_stock_qty",
        "available_and_direct_transit_qty",
        "total_stock_qty",
        "transit_stock_qty",
        "out_of_stock_flag",
    )

    def __init__(self, repository: ParquetDataRepository) -> None:
        self.repository = repository

    def read_daily_snapshots(
        self,
        context: PointInTimeContext,
        *,
        start_date: date,
        max_rows: int | None = None,
    ) -> ReadResult:
        raw = self.repository.read_history(
            context,
            source_name=INVENTORY_SOURCE,
            start_date=start_date,
            columns=self.REQUIRED_COLUMNS,
            max_rows=max_rows,
        )
        frame = raw.frame.copy()
        if frame.empty:
            frame["business_day"] = []
            return ReadResult(**{**raw.__dict__, "frame": frame})
        stock_time = _parse_local_timestamp(frame, "stock_date", context)
        if (stock_time > context.decision_cutoff_time).any():
            raise DataContractViolation("inventory stock_date exceeds decision cutoff")
        frame["stock_date"] = stock_time
        frame["business_day"] = stock_time.dt.date
        key = ["sku_code", "store_code", "business_day"]
        duplicated = frame.duplicated(subset=key, keep=False)
        if duplicated.any():
            examples = frame.loc[duplicated, key].head(5).to_dict(orient="records")
            raise DataContractViolation(
                "inventory source is not unique at SKU x store x day; "
                f"examples={examples}"
            )
        import pandas as pd

        for column in (
            "available_stock_qty",
            "available_and_direct_transit_qty",
            "total_stock_qty",
            "transit_stock_qty",
        ):
            converted = pd.to_numeric(frame[column], errors="coerce")
            invalid = frame[column].notna() & converted.isna()
            if invalid.any():
                raise DataContractViolation(f"non-numeric inventory measure: {column}")
            frame[column] = converted
        return ReadResult(**{**raw.__dict__, "frame": frame.reset_index(drop=True)})


class ProductDataService:
    """Separates multi-IP membership from one-row-per-SKU join attributes."""

    def __init__(self, repository: ParquetDataRepository) -> None:
        self.repository = repository

    def read_snapshot(self, context: PointInTimeContext) -> ReadResult:
        return self.repository.read_latest(context, source_name=PRODUCT_SOURCE)

    @staticmethod
    def ip_membership(snapshot: ReadResult) -> object:
        fields = ["sku_code", "ip_code", "ip_name"]
        return snapshot.frame[fields].drop_duplicates().reset_index(drop=True)

    @staticmethod
    def one_row_per_sku_attributes(snapshot: ReadResult) -> object:
        frame = snapshot.frame
        excluded = {"ip_code", "ip_name"}
        columns = [name for name in frame.columns if name not in excluded]
        for column in columns:
            if column in {"sku_code", "ds"}:
                continue
            conflicts = frame.groupby("sku_code", dropna=False)[column].nunique(dropna=False)
            if (conflicts > 1).any():
                skus = conflicts[conflicts > 1].index.astype(str).tolist()[:5]
                raise DataContractViolation(
                    f"product attribute {column} is not SKU-invariant; examples={skus}"
                )
        result = frame[columns].drop_duplicates(subset=["sku_code"])
        if result["sku_code"].duplicated().any():
            raise DataContractViolation("product attributes are not unique by SKU")
        return result.reset_index(drop=True)


def safe_join_sku_attributes(facts: object, sku_attributes: object) -> object:
    if sku_attributes["sku_code"].duplicated().any():
        raise DataContractViolation(
            "refusing a fact join because SKU attributes are not unique by sku_code"
        )
    return facts.merge(sku_attributes, on="sku_code", how="left", validate="many_to_one")
