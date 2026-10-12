from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from data.domestic import SALES_SOURCE, DataContractViolation, validate_delivered_sales_flags
from data.parquet_repository import ParquetDataRepository


@dataclass(frozen=True)
class CohortRow:
    sku_code: str
    first_actual_sale_date: date
    sales_26w_qty: float


@dataclass(frozen=True)
class CohortBuildStats:
    source_partition_min: date
    label_as_of_date: date
    window_days: int
    partitions_scanned: int
    physical_rows_scanned: int
    visible_rows_scanned: int
    order_line_versions: int
    negative_sale_qty_versions: int
    negative_sale_qty_sum: float
    negative_sale_qty_min: float | None
    distinct_latest_order_lines: int
    latest_negative_order_lines: int
    latest_negative_sale_qty_sum: float
    latest_negative_sale_qty_min: float | None
    same_ds_duplicate_rows: int
    same_ds_exact_duplicate_rows: int
    same_ds_signed_aggregation_extra_rows: int
    same_ds_signed_aggregation_groups: int
    same_ds_zero_net_groups: int
    same_ds_nonzero_net_groups: int
    sku_with_positive_sales: int
    mature_sku_count: int
    immature_sku_count: int
    first_sale_on_source_min_count: int


@dataclass(frozen=True)
class CohortBuildResult:
    rows: tuple[CohortRow, ...]
    stats: CohortBuildStats
    dataset_version: str


class _LatestSalesLineStore:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-262144")
        self.connection.execute(
            """
            CREATE TABLE latest_sales_lines (
                order_id TEXT NOT NULL,
                order_detail_id TEXT NOT NULL,
                ds TEXT NOT NULL,
                event_epoch_us INTEGER NOT NULL,
                business_day TEXT NOT NULL,
                sku_code TEXT NOT NULL,
                sale_qty REAL NOT NULL,
                PRIMARY KEY (order_id, order_detail_id)
            ) WITHOUT ROWID
            """
        )
        self.connection.execute(
            """
            CREATE TABLE partition_sales_lines (
                order_id TEXT NOT NULL,
                order_detail_id TEXT NOT NULL,
                ds TEXT NOT NULL,
                event_epoch_us INTEGER NOT NULL,
                business_day TEXT NOT NULL,
                sku_code TEXT NOT NULL,
                sale_qty REAL NOT NULL,
                PRIMARY KEY (
                    order_id, order_detail_id, ds, event_epoch_us,
                    business_day, sku_code, sale_qty
                )
            ) WITHOUT ROWID
            """
        )

    def insert_rows(
        self, rows: list[tuple[str, str, str, int, str, str, float]]
    ) -> int:
        before = self.connection.total_changes
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO partition_sales_lines
                (order_id, order_detail_id, ds, event_epoch_us, business_day,
                 sku_code, sale_qty)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        inserted = self.connection.total_changes - before
        return len(rows) - inserted

    def finish_partition(self, partition: date) -> dict[str, int]:
        distinct_ds = self.connection.execute(
            "SELECT DISTINCT ds FROM partition_sales_lines"
        ).fetchall()
        if distinct_ds and distinct_ds != [(partition.isoformat(),)]:
            raise DataContractViolation(
                f"partition {partition} yielded unexpected ds values: {distinct_ds[:5]}"
            )
        invalid_groups = self.connection.execute(
            """
            SELECT order_id, order_detail_id, ds,
                   COUNT(*) AS event_rows,
                   COUNT(DISTINCT sku_code) AS sku_count,
                   COUNT(DISTINCT business_day) AS business_day_count,
                   MIN(sale_qty) AS min_qty,
                   MAX(sale_qty) AS max_qty,
                   SUM(sale_qty) AS net_qty
            FROM partition_sales_lines
            GROUP BY order_id, order_detail_id, ds
            HAVING COUNT(*) > 1
               AND (
                    COUNT(DISTINCT sku_code) <> 1
                    OR COUNT(DISTINCT business_day) <> 1
               )
            ORDER BY order_id, order_detail_id
            LIMIT 5
            """
        ).fetchall()
        if invalid_groups:
            raise DataContractViolation(
                "same-ds order-line rows cannot be net-aggregated because a group "
                "contains different SKUs or different business days; "
                f"examples={invalid_groups}"
            )
        group_summary = self.connection.execute(
            """
            WITH grouped AS (
                SELECT order_id, order_detail_id, ds,
                       COUNT(*) AS event_rows,
                       SUM(sale_qty) AS net_qty
                FROM partition_sales_lines
                GROUP BY order_id, order_detail_id, ds
            )
            SELECT
                COALESCE(SUM(CASE WHEN event_rows > 1 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN event_rows > 1 THEN event_rows - 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN event_rows > 1 AND ABS(net_qty) < 1e-12
                                  THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN event_rows > 1 AND ABS(net_qty) >= 1e-12
                                  THEN 1 ELSE 0 END), 0)
            FROM grouped
            """
        ).fetchone()
        self.connection.execute(
            """
            INSERT OR IGNORE INTO latest_sales_lines
                (order_id, order_detail_id, ds, event_epoch_us, business_day,
                 sku_code, sale_qty)
            SELECT
                order_id,
                order_detail_id,
                MIN(ds),
                MAX(event_epoch_us),
                MIN(business_day),
                MIN(sku_code),
                SUM(sale_qty)
            FROM partition_sales_lines
            GROUP BY order_id, order_detail_id
            """
        )
        self.connection.execute("DELETE FROM partition_sales_lines")
        self.connection.commit()
        return {
            "signed_aggregation_groups": int(group_summary[0]),
            "signed_aggregation_extra_rows": int(group_summary[1]),
            "zero_net_groups": int(group_summary[2]),
            "nonzero_net_groups": int(group_summary[3]),
        }

    def summarize(
        self,
        *,
        cutoff_time: datetime,
        label_as_of_date: date,
        window_days: int,
        source_min: date,
    ) -> tuple[tuple[CohortRow, ...], dict[str, int | float | None]]:
        parameters = {
            "label_as_of": label_as_of_date.isoformat(),
            "window_end_offset": f"+{window_days - 1} days",
            "source_min": source_min.isoformat(),
            "cutoff_epoch_us": int(cutoff_time.timestamp() * 1_000_000),
        }
        base_cte = """
            WITH latest AS (
                SELECT business_day, sku_code, sale_qty
                FROM latest_sales_lines
                WHERE event_epoch_us <= :cutoff_epoch_us
            ),
            first_sales AS (
                SELECT sku_code, MIN(business_day) AS first_actual_sale_date
                FROM latest
                WHERE sale_qty > 0
                GROUP BY sku_code
            )
        """
        cursor = self.connection.execute(
            base_cte
            + """
            SELECT
                f.sku_code,
                f.first_actual_sale_date,
                SUM(
                    CASE
                        WHEN l.business_day BETWEEN f.first_actual_sale_date
                            AND date(f.first_actual_sale_date, :window_end_offset)
                        THEN l.sale_qty
                        ELSE 0
                    END
                ) AS sales_26w_qty
            FROM first_sales AS f
            JOIN latest AS l ON l.sku_code = f.sku_code
            WHERE date(f.first_actual_sale_date, :window_end_offset) <= :label_as_of
            GROUP BY f.sku_code, f.first_actual_sale_date
            ORDER BY f.first_actual_sale_date, f.sku_code
            """,
            parameters,
        )
        rows = tuple(
            CohortRow(
                sku_code=str(sku),
                first_actual_sale_date=date.fromisoformat(str(first_sale)),
                sales_26w_qty=float(quantity),
            )
            for sku, first_sale, quantity in cursor
        )
        sku_with_positive_sales = int(
            self.connection.execute(
                base_cte + "SELECT COUNT(*) FROM first_sales", parameters
            ).fetchone()[0]
        )
        latest_order_lines = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM latest_sales_lines"
            ).fetchone()[0]
        )
        negative_summary = self.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(sale_qty), 0), MIN(sale_qty)
            FROM latest_sales_lines
            WHERE event_epoch_us <= :cutoff_epoch_us
              AND sale_qty < 0
            """,
            parameters,
        ).fetchone()
        first_sale_on_min = int(
            self.connection.execute(
                base_cte
                + "SELECT COUNT(*) FROM first_sales WHERE first_actual_sale_date = :source_min",
                parameters,
            ).fetchone()[0]
        )
        return rows, {
            "sku_with_positive_sales": sku_with_positive_sales,
            "distinct_latest_order_lines": latest_order_lines,
            "latest_negative_order_lines": int(negative_summary[0]),
            "latest_negative_sale_qty_sum": float(negative_summary[1]),
            "latest_negative_sale_qty_min": (
                None if negative_summary[2] is None else float(negative_summary[2])
            ),
            "first_sale_on_source_min_count": first_sale_on_min,
        }

    def close(self) -> None:
        self.connection.close()


def _parse_order_time(values: Any, cutoff_time: datetime) -> Any:
    import pandas as pd

    # The delivered string column contains more than one timestamp rendering.
    # pandas >= 2 supports explicit per-element parsing for this physical case.
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).head(5).tolist()
        raise DataContractViolation(f"unparseable order_time values: {examples}")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(cutoff_time.tzinfo)
    return parsed.dt.tz_convert(cutoff_time.tzinfo)


def _dataset_version(rows: tuple[CohortRow, ...], label_as_of_date: date) -> str:
    payload = {
        "spec_version": "3.2",
        "label_as_of_date": label_as_of_date.isoformat(),
        "rows": [
            {
                **asdict(row),
                "first_actual_sale_date": row.first_actual_sale_date.isoformat(),
            }
            for row in rows
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"v3.2-{label_as_of_date.isoformat()}-{digest[:16]}"


def build_first_order_cohort(
    *,
    repository: ParquetDataRepository,
    cutoff_time: datetime,
    cache_path: Path,
    window_weeks: int = 26,
    batch_size: int = 50_000,
    progress: Callable[[int, int, date], None] | None = None,
) -> CohortBuildResult:
    if cutoff_time.tzinfo is None or cutoff_time.utcoffset() is None:
        raise ValueError("cutoff_time must be timezone-aware")
    if window_weeks <= 0:
        raise ValueError("window_weeks must be positive")
    partitions = tuple(
        value
        for value in repository.available_partitions(SALES_SOURCE)
        if value <= cutoff_time.date()
    )
    if not partitions:
        raise ValueError("no sales partitions are visible at the cutoff")
    source_min = partitions[0]
    label_as_of = partitions[-1]
    window_days = window_weeks * 7
    # Partitions are scanned newest to oldest. The store therefore only needs
    # one row per order line: an older version can never replace the first row.
    store = _LatestSalesLineStore(cache_path)
    physical_rows = 0
    visible_rows = 0
    order_line_versions = 0
    negative_versions = 0
    negative_quantity_sum = 0.0
    negative_quantity_min: float | None = None
    exact_duplicate_rows = 0
    signed_aggregation_extra_rows = 0
    signed_aggregation_groups = 0
    zero_net_groups = 0
    nonzero_net_groups = 0
    try:
        for index, partition in enumerate(reversed(partitions), start=1):
            if progress is not None:
                progress(index, len(partitions), partition)
            for batch in repository.iter_partition_batches(
                source_name=SALES_SOURCE,
                start_date=partition,
                end_date=partition,
                cutoff_time=cutoff_time,
                columns=(
                    "ds",
                    "order_id",
                    "order_detail_id",
                    "order_time",
                    "sku_code",
                    "sale_qty",
                    "is_return",
                    "gift_flag",
                ),
                batch_size=batch_size,
            ):
                import pandas as pd

                frame = batch.to_pandas()
                physical_rows += len(frame)
                if frame.empty:
                    continue
                validate_delivered_sales_flags(frame)
                order_time = _parse_order_time(frame["order_time"], cutoff_time)
                visible_rows += int((order_time <= cutoff_time).sum())
                required_keys = ["order_id", "order_detail_id", "sku_code"]
                if frame[required_keys].isna().any(axis=None):
                    raise DataContractViolation("sales order-line keys contain null values")
                quantity = pd.to_numeric(frame["sale_qty"], errors="coerce")
                invalid_quantity = frame["sale_qty"].notna() & quantity.isna()
                if invalid_quantity.any() or quantity.isna().any():
                    raise DataContractViolation("sale_qty contains null or non-numeric values")
                negative = quantity < 0
                if negative.any():
                    negative_values = quantity.loc[negative]
                    negative_versions += int(negative.sum())
                    negative_quantity_sum += float(negative_values.sum())
                    batch_min = float(negative_values.min())
                    negative_quantity_min = (
                        batch_min
                        if negative_quantity_min is None
                        else min(negative_quantity_min, batch_min)
                    )
                frame["sale_qty"] = quantity.astype(float)
                frame["event_epoch_us"] = order_time.astype("int64") // 1_000
                frame["business_day"] = order_time.dt.date.astype(str)
                records = list(
                    frame[
                        [
                            "order_id",
                            "order_detail_id",
                            "ds",
                            "event_epoch_us",
                            "business_day",
                            "sku_code",
                            "sale_qty",
                        ]
                    ].itertuples(index=False, name=None)
                )
                normalized = [
                    (str(a), str(b), str(c), int(d), str(e), str(f), float(g))
                    for a, b, c, d, e, f, g in records
                ]
                order_line_versions += len(normalized)
                exact_duplicate_rows += store.insert_rows(normalized)
            partition_summary = store.finish_partition(partition)
            signed_aggregation_groups += partition_summary[
                "signed_aggregation_groups"
            ]
            signed_aggregation_extra_rows += partition_summary[
                "signed_aggregation_extra_rows"
            ]
            zero_net_groups += partition_summary["zero_net_groups"]
            nonzero_net_groups += partition_summary["nonzero_net_groups"]
        rows, summary = store.summarize(
            cutoff_time=cutoff_time,
            label_as_of_date=label_as_of,
            window_days=window_days,
            source_min=source_min,
        )
        negative_labels = [row for row in rows if row.sales_26w_qty < 0]
        if negative_labels:
            examples = [
                (row.sku_code, row.sales_26w_qty) for row in negative_labels[:5]
            ]
            raise DataContractViolation(
                "signed delivered sale_qty produces negative mature 26-week labels; "
                f"count={len(negative_labels)} examples={examples}"
            )
    finally:
        store.close()
    stats = CohortBuildStats(
        source_partition_min=source_min,
        label_as_of_date=label_as_of,
        window_days=window_days,
        partitions_scanned=len(partitions),
        physical_rows_scanned=physical_rows,
        visible_rows_scanned=visible_rows,
        order_line_versions=order_line_versions,
        negative_sale_qty_versions=negative_versions,
        negative_sale_qty_sum=negative_quantity_sum,
        negative_sale_qty_min=negative_quantity_min,
        distinct_latest_order_lines=int(summary["distinct_latest_order_lines"]),
        latest_negative_order_lines=int(summary["latest_negative_order_lines"]),
        latest_negative_sale_qty_sum=float(
            summary["latest_negative_sale_qty_sum"]
        ),
        latest_negative_sale_qty_min=(
            None
            if summary["latest_negative_sale_qty_min"] is None
            else float(summary["latest_negative_sale_qty_min"])
        ),
        same_ds_duplicate_rows=exact_duplicate_rows + signed_aggregation_extra_rows,
        same_ds_exact_duplicate_rows=exact_duplicate_rows,
        same_ds_signed_aggregation_extra_rows=signed_aggregation_extra_rows,
        same_ds_signed_aggregation_groups=signed_aggregation_groups,
        same_ds_zero_net_groups=zero_net_groups,
        same_ds_nonzero_net_groups=nonzero_net_groups,
        sku_with_positive_sales=int(summary["sku_with_positive_sales"]),
        mature_sku_count=len(rows),
        immature_sku_count=int(summary["sku_with_positive_sales"]) - len(rows),
        first_sale_on_source_min_count=int(
            summary["first_sale_on_source_min_count"]
        ),
    )
    return CohortBuildResult(
        rows=rows,
        stats=stats,
        dataset_version=_dataset_version(rows, label_as_of),
    )
