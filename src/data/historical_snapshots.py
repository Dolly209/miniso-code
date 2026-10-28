from __future__ import annotations

from datetime import date, datetime
from typing import Any, Sequence

from data.decision_adapter import SOURCES
from data.domestic import DataContractViolation
from data.parquet_repository import ParquetDataRepository


def read_latest_nonempty_snapshot(
    *,
    repository: ParquetDataRepository,
    source_name: str,
    cutoff_time: datetime,
    columns: Sequence[str],
) -> tuple[Any, date]:
    eligible = [
        partition
        for partition in repository.available_partitions(source_name)
        if partition <= cutoff_time.date()
    ]
    for partition in reversed(eligible):
        result = repository.read_partition_range(
            source_name=source_name,
            start_date=partition,
            end_date=partition,
            cutoff_time=cutoff_time,
            columns=columns,
        )
        if not result.frame.empty:
            return result.frame, partition
    raise DataContractViolation(
        f"no non-empty snapshot for {source_name} on or before {cutoff_time.date()}"
    )


def read_historical_snapshots(
    *, repository: ParquetDataRepository, cutoff_time: datetime
) -> tuple[Any, date, Any, date]:
    product, product_partition = read_latest_nonempty_snapshot(
        repository=repository,
        source_name=SOURCES["product"],
        cutoff_time=cutoff_time,
        columns=(
            "ds", "sku_code", "sku_name", "first_sale_date",
            "cate_level3_name", "cate_level4_name", "pkg_unit",
        ),
    )
    # The delivered packaging table repeats goods_unit/pkg_unit/box_unit from
    # the product master.  Keep the historical-reproduction API stable while
    # deriving its pack snapshot from the same authoritative product snapshot.
    pack = product[["ds", "sku_code", "pkg_unit"]].copy()
    return product, product_partition, pack, product_partition
