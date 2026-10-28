from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from config.settings import AppConfig, load_table_specs
from data.decision_adapter import SOURCES
from data.domestic import DataContractViolation
from data.parquet_repository import ParquetDataRepository


RESTORATION_SOURCE_KEYS = (
    "product",
    "store",
    "sales",
    "inventory",
    "placement",
)
RESTORATION_SOURCE_NAMES = tuple(SOURCES[key] for key in RESTORATION_SOURCE_KEYS)

SALES_COLUMNS = (
    "ds", "order_id", "order_detail_id", "order_time", "sku_code", "store_code",
    "sale_qty", "order_amt", "paid_amt", "unit_paid_price", "achievement_amt",
    "is_return", "gift_flag",
)
INVENTORY_COLUMNS = (
    "ds", "stock_date", "store_code", "sku_code", "available_stock_qty", "out_of_stock_flag",
)
PLACEMENT_COLUMNS = (
    "ds", "sku_code", "store_code", "first_store_arrival_time", "stock_available_itm_cnt",
    "out_of_stock_flag", "in_stock_not_displayed_flag",
)
STORE_COLUMNS = (
    "ds", "store_code", "store_business_state", "store_level", "open_date", "close_date", "sts_state",
    "city_level", "city_name", "secbiz_area_name", "area_class1_name", "operate_size", "meter_sum",
)
PRODUCT_COLUMNS = ("ds", "sku_code", "cate_level4_code", "retail_price")

CHECKPOINT_VERSION = "m2-source-batches-v2"
SHARED_SOURCE_CACHE_VERSION = "m2-cross-cutoff-source-cache-v1"
ProgressCallback = Callable[[str, Mapping[str, Any]], None]
BatchCallback = Callable[["RestorationSourceBatch"], None]


@dataclass(frozen=True)
class RestorationSourceBundle:
    raw_sales: pd.DataFrame
    inventory: pd.DataFrame
    placement: pd.DataFrame
    stores: pd.DataFrame
    product_attributes: pd.DataFrame
    first_sale_dates: dict[str, Any]
    fit_skus: set[str]
    training_weights: dict[str, float]
    source_windows: list[dict[str, str]]
    checkpoint_run_dir: Path | None = None
    reused_batch_count: int = 0
    computed_batch_count: int = 0


@dataclass(frozen=True)
class RestorationSourceBatch:
    batch_index: int
    batch_count: int
    selected: pd.DataFrame
    frames: dict[str, pd.DataFrame]
    stores: pd.DataFrame
    windows: list[dict[str, str]]
    checkpoint_run_dir: Path | None
    checkpoint_batch_dir: Path | None
    reused: bool
    deferred_source_frames: bool = False


def _concat(frames: list[pd.DataFrame], columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns))


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


def _source_windows(
    selected: pd.DataFrame, cutoff_time: datetime, window_days: int
) -> list[dict[str, str]]:
    windows: list[dict[str, str]] = []
    for row in selected.itertuples(index=False):
        start = row.first_actual_sale_date
        end = min(start + timedelta(days=window_days - 1), cutoff_time.date())
        if start > end:
            raise DataContractViolation(
                f"SKU {row.sku_code} first sale date {start} is after cutoff {cutoff_time.date()}"
            )
        windows.append(
            {
                "sku_code": str(row.sku_code),
                "split": str(row.split),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            }
        )
    return windows


def _filter_to_sku_windows(
    frame: pd.DataFrame, windows: Sequence[Mapping[str, str]]
) -> pd.DataFrame:
    """Undo the extra dates introduced by a batch union read."""

    if frame.empty:
        return frame.reset_index(drop=True)
    missing = sorted({"ds", "sku_code"} - set(frame.columns))
    if missing:
        raise DataContractViolation(
            f"batched restoration source is missing columns: {', '.join(missing)}"
        )
    starts = {
        str(item["sku_code"]): date.fromisoformat(str(item["start_date"]))
        for item in windows
    }
    ends = {
        str(item["sku_code"]): date.fromisoformat(str(item["end_date"]))
        for item in windows
    }
    sku_keys = frame["sku_code"].astype("string").str.strip()
    partition_days = pd.to_datetime(frame["ds"], errors="coerce").dt.date
    if partition_days.isna().any():
        examples = frame.loc[partition_days.isna(), "ds"].head(5).tolist()
        raise DataContractViolation(
            f"batched restoration source has invalid ds values: {examples}"
        )
    start_values = sku_keys.map(starts)
    end_values = sku_keys.map(ends)
    keep = (
        start_values.notna()
        & end_values.notna()
        & (partition_days >= start_values)
        & (partition_days <= end_values)
    )
    return frame.loc[keep].reset_index(drop=True)


class RestorationDataAdapter:
    """All physical reads needed by M2, isolated from the restore algorithm and CLI."""

    def __init__(self, repository: ParquetDataRepository) -> None:
        self.repository = repository

    def sales_partition_dates(self, cutoff_time: datetime) -> set[date] | None:
        # Small directory metadata, cached by the repository, never fact rows.
        if not hasattr(self.repository, "available_partitions"):
            return None  # legacy injected repositories: explicitly unaudited
        return {day for day in self.repository.available_partitions(SOURCES["sales"]) if day <= cutoff_time.date()}

    @classmethod
    def from_project(
        cls, *, project_root: Path, data_root: Path, config: AppConfig
    ) -> "RestorationDataAdapter":
        return cls(
            ParquetDataRepository(
                data_root=data_root,
                table_specs=load_table_specs(config.probe.table_specs_path),
                max_rows_per_read=config.data_reader.max_rows_per_read,
            )
        )

    def _run_fingerprint(
        self,
        *,
        cutoff_time: datetime,
        window_days: int,
        product_snapshot_time: datetime | None = None,
    ) -> str:
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "data_root": str(self.repository.data_root.resolve()),
            "cutoff_time": cutoff_time.isoformat(),
            "product_snapshot_time": (
                product_snapshot_time or cutoff_time
            ).isoformat(),
            "window_days": window_days,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _batch_fingerprint(
        selected: pd.DataFrame,
        *,
        run_fingerprint: str,
    ) -> str:
        records = [
            {
                "dataset_version": str(row.dataset_version),
                "sku_code": str(row.sku_code),
                "first_actual_sale_date": row.first_actual_sale_date.isoformat(),
                "split": str(row.split),
                "long_tail_weight": float(row.long_tail_weight),
            }
            for row in selected.itertuples(index=False)
        ]
        return hashlib.sha256(
            json.dumps(
                {
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "run_fingerprint": run_fingerprint,
                    "selected": records,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _validate_products(product: pd.DataFrame, skus: Sequence[str]) -> pd.DataFrame:
        business_columns = ["sku_code", "cate_level4_code", "retail_price"]
        conflicts = product.groupby("sku_code", dropna=False)[
            ["cate_level4_code", "retail_price"]
        ].nunique(dropna=False)
        if (conflicts > 1).any(axis=None):
            bad = conflicts.loc[(conflicts > 1).any(axis=1)].head(5).index.tolist()
            raise DataContractViolation(
                f"product category/retail price is not SKU-invariant: {bad}"
            )
        result = product[business_columns].drop_duplicates("sku_code")
        result_keys = set(result["sku_code"].astype("string").str.strip())
        missing = sorted(set(skus) - result_keys)
        if missing:
            raise DataContractViolation(f"selected SKUs have no product snapshot: {missing}")
        return result.reset_index(drop=True)

    def _read_products(
        self,
        skus: Sequence[str],
        cutoff_time: datetime,
        *,
        progress: ProgressCallback | None = None,
        batch_index: int | None = None,
        batch_count: int | None = None,
    ) -> pd.DataFrame:
        latest = self.repository.read_latest_partition(
            source_name=SOURCES["product"],
            cutoff_time=cutoff_time,
            columns=PRODUCT_COLUMNS,
            sku_codes=skus,
            max_rows=self.repository.max_rows_per_read * max(len(skus), 1),
        ).frame
        latest_keys = set(
            latest["sku_code"].astype("string").str.strip().dropna()
        )
        missing = sorted(set(str(value) for value in skus) - latest_keys)
        if not missing:
            return self._validate_products(latest, skus)

        # The delivered product dimension is not guaranteed to repeat every
        # historical SKU in every daily snapshot.  For absent SKUs, use the
        # most recent row on or before the decision cutoff.  This remains a
        # strict point-in-time lookup and never consults a future partition.
        eligible = tuple(
            value
            for value in self.repository.available_partitions(SOURCES["product"])
            if value <= cutoff_time.date()
        )
        if not eligible:
            raise DataContractViolation(
                "product source has no partition on or before the decision cutoff"
            )
        _emit(
            progress,
            "product_snapshot_fallback_start",
            batch=batch_index,
            batches=batch_count,
            missing_skus=",".join(missing),
            start_date=eligible[0].isoformat(),
            end_date=eligible[-1].isoformat(),
        )
        historical = self.repository.read_partition_range(
            source_name=SOURCES["product"],
            start_date=eligible[0],
            end_date=eligible[-1],
            cutoff_time=cutoff_time,
            columns=PRODUCT_COLUMNS,
            sku_codes=missing,
            max_rows=self.repository.max_rows_per_read * max(len(missing), 1),
        ).frame
        if not historical.empty:
            partition_day = pd.to_datetime(
                historical["ds"], errors="coerce"
            ).dt.date
            if partition_day.isna().any():
                examples = historical.loc[partition_day.isna(), "ds"].head(5).tolist()
                raise DataContractViolation(
                    f"product history contains invalid ds values: {examples}"
                )
            historical = historical.copy()
            historical["__partition_day"] = partition_day
            sku_keys = historical["sku_code"].astype("string").str.strip()
            latest_day = historical.groupby(sku_keys, dropna=False)[
                "__partition_day"
            ].transform("max")
            historical = historical.loc[
                historical["__partition_day"].eq(latest_day)
            ].drop(columns="__partition_day")
        combined = pd.concat([latest, historical], ignore_index=True)
        validated = self._validate_products(combined, skus)
        fallback_days = (
            historical.groupby(
                historical["sku_code"].astype("string").str.strip()
            )["ds"]
            .first()
            .astype(str)
            .to_dict()
            if not historical.empty
            else {}
        )
        _emit(
            progress,
            "product_snapshot_fallback_complete",
            batch=batch_index,
            batches=batch_count,
            resolved=len(fallback_days),
            fallback_partitions=json.dumps(
                fallback_days, ensure_ascii=False, sort_keys=True
            ),
        )
        return validated

    @staticmethod
    def _checkpoint_frames(
        directory: Path, manifest: Mapping[str, Any], *, defer_sources: bool = False,
    ) -> dict[str, pd.DataFrame] | None:
        if manifest.get("status") != "complete":
            return None
        frames: dict[str, pd.DataFrame] = {}
        try:
            for name in ("sales", "inventory", "placement", "product"):
                path = directory / f"{name}.parquet"
                expected = int(manifest["row_counts"][name])
                if defer_sources and name != "product":
                    import pyarrow.parquet as pq
                    if pq.read_metadata(path).num_rows != expected:
                        return None
                    frame = pd.DataFrame(columns={"sales": SALES_COLUMNS, "inventory": INVENTORY_COLUMNS, "placement": PLACEMENT_COLUMNS}[name])
                else:
                    frame = pd.read_parquet(path)
                    if len(frame) != expected:
                        return None
                frames[name] = frame
        except (KeyError, OSError, ValueError):
            return None
        return frames

    def _read_source_batch(
        self,
        *,
        selected: pd.DataFrame,
        cutoff_time: datetime,
        product_snapshot_time: datetime | None,
        window_days: int,
        batch_index: int,
        batch_count: int,
        progress: ProgressCallback | None,
        shared_source_cache_dir: Path | None = None,
    ) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
        windows = _source_windows(selected, cutoff_time, window_days)
        skus = [str(value) for value in selected["sku_code"]]
        union_start = min(date.fromisoformat(item["start_date"]) for item in windows)
        union_end = max(date.fromisoformat(item["end_date"]) for item in windows)
        row_limit = self.repository.max_rows_per_read * max(len(skus), 1)
        frames: dict[str, pd.DataFrame] = {}
        for domain_name, source_name, columns in (
            ("sales", SOURCES["sales"], SALES_COLUMNS),
            ("inventory", SOURCES["inventory"], INVENTORY_COLUMNS),
            ("placement", SOURCES["placement"], PLACEMENT_COLUMNS),
        ):
            if shared_source_cache_dir is None:
                _emit(
                    progress,
                    "source_read_start",
                    batch=batch_index,
                    batches=batch_count,
                    source=domain_name,
                    sku_count=len(skus),
                    start_date=union_start.isoformat(),
                    end_date=union_end.isoformat(),
                    row_limit=row_limit,
                )
                started = monotonic()
                result = self.repository.read_partition_range(
                    source_name=source_name,
                    start_date=union_start,
                    end_date=union_end,
                    cutoff_time=cutoff_time,
                    columns=columns,
                    sku_codes=skus,
                    max_rows=row_limit,
                )
                frame = _filter_to_sku_windows(result.frame, windows)
                frames[domain_name] = frame
                _emit(
                    progress,
                    "source_read_complete",
                    batch=batch_index,
                    batches=batch_count,
                    source=domain_name,
                    rows=len(frame),
                    partitions=len(result.selected_partitions),
                    elapsed_seconds=round(monotonic() - started, 3),
                )
            else:
                frames[domain_name] = self._read_shared_cached_fact_batch(
                    domain_name=domain_name,
                    source_name=source_name,
                    columns=columns,
                    windows=windows,
                    cutoff_time=cutoff_time,
                    cache_dir=shared_source_cache_dir,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    progress=progress,
                )
        _emit(
            progress,
            "source_read_start",
            batch=batch_index,
            batches=batch_count,
            source="product",
            sku_count=len(skus),
        )
        started = monotonic()
        if shared_source_cache_dir is None:
            frames["product"] = self._read_products(
                skus,
                product_snapshot_time or cutoff_time,
                progress=progress,
                batch_index=batch_index,
                batch_count=batch_count,
            )
        else:
            frames["product"] = self._read_shared_cached_products(
                skus=skus,
                snapshot_time=product_snapshot_time or cutoff_time,
                cache_dir=shared_source_cache_dir,
                batch_index=batch_index,
                batch_count=batch_count,
                progress=progress,
            )
        _emit(
            progress,
            "source_read_complete",
            batch=batch_index,
            batches=batch_count,
            source="product",
            rows=len(frames["product"]),
            elapsed_seconds=round(monotonic() - started, 3),
        )
        return frames, windows

    def _shared_cache_fingerprint(
        self, *, source_name: str, columns: Sequence[str]
    ) -> str:
        payload = {
            "version": SHARED_SOURCE_CACHE_VERSION,
            "data_root": str(self.repository.data_root.resolve()),
            "source_name": source_name,
            "columns": list(columns),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _shared_cache_paths(
        cache_dir: Path, *, domain_name: str, sku_code: str
    ) -> tuple[Path, Path]:
        digest = hashlib.sha256(str(sku_code).encode("utf-8")).hexdigest()[:24]
        root = cache_dir / domain_name
        return root / f"{digest}.parquet", root / f"{digest}.json"

    def _read_shared_cached_fact_batch(
        self,
        *,
        domain_name: str,
        source_name: str,
        columns: Sequence[str],
        windows: Sequence[Mapping[str, str]],
        cutoff_time: datetime,
        cache_dir: Path,
        batch_index: int,
        batch_count: int,
        progress: ProgressCallback | None,
    ) -> pd.DataFrame:
        """Read every SKU/date from the physical source at most once.

        Replay groups are processed in cutoff order.  A SKU cache therefore
        grows monotonically from its first sale date through the latest date
        requested so far.  Earlier-cutoff reruns remain safe because the final
        frame is always sliced back to that group's explicit SKU window.
        """

        fingerprint = self._shared_cache_fingerprint(
            source_name=source_name, columns=columns
        )
        window_by_sku = {str(item["sku_code"]): dict(item) for item in windows}
        cached_frames: dict[str, pd.DataFrame] = {}
        missing_windows: list[dict[str, str]] = []
        cache_hits = 0
        for sku_code, item in window_by_sku.items():
            data_path, manifest_path = self._shared_cache_paths(
                cache_dir, domain_name=domain_name, sku_code=sku_code
            )
            desired_start = date.fromisoformat(str(item["start_date"]))
            desired_end = date.fromisoformat(str(item["end_date"]))
            existing = pd.DataFrame(columns=list(columns))
            cached_through: date | None = None
            try:
                manifest = _read_json(manifest_path)
                if (
                    manifest.get("status") == "complete"
                    and manifest.get("fingerprint") == fingerprint
                    and manifest.get("sku_code") == sku_code
                    and manifest.get("start_date") == desired_start.isoformat()
                    and data_path.is_file()
                ):
                    existing = pd.read_parquet(data_path)
                    cached_through = date.fromisoformat(
                        str(manifest["cached_through"])
                    )
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                existing = pd.DataFrame(columns=list(columns))
                cached_through = None
            cached_frames[sku_code] = existing
            if cached_through is not None and cached_through >= desired_end:
                cache_hits += 1
                continue
            missing_start = (
                desired_start
                if cached_through is None
                else max(desired_start, cached_through + timedelta(days=1))
            )
            if missing_start <= desired_end:
                missing_windows.append(
                    {
                        "sku_code": sku_code,
                        "start_date": missing_start.isoformat(),
                        "end_date": desired_end.isoformat(),
                    }
                )

        _emit(
            progress,
            "shared_source_cache_plan",
            batch=batch_index,
            batches=batch_count,
            source=domain_name,
            requested_skus=len(window_by_sku),
            cache_hit_skus=cache_hits,
            extend_skus=len(missing_windows),
        )
        if missing_windows:
            started = monotonic()
            missing_skus = [str(item["sku_code"]) for item in missing_windows]
            read_start = min(
                date.fromisoformat(str(item["start_date"]))
                for item in missing_windows
            )
            read_end = max(
                date.fromisoformat(str(item["end_date"]))
                for item in missing_windows
            )
            result = self.repository.read_partition_range(
                source_name=source_name,
                start_date=read_start,
                end_date=read_end,
                cutoff_time=cutoff_time,
                columns=columns,
                sku_codes=missing_skus,
                max_rows=self.repository.max_rows_per_read
                * max(len(missing_skus), 1),
            )
            additions = _filter_to_sku_windows(result.frame, missing_windows)
            additions_by_sku = {
                str(sku): group.reset_index(drop=True)
                for sku, group in additions.groupby(
                    additions["sku_code"].astype("string").str.strip(),
                    sort=False,
                )
            } if not additions.empty else {}
            for item in missing_windows:
                sku_code = str(item["sku_code"])
                existing = cached_frames[sku_code]
                addition = additions_by_sku.get(
                    sku_code, pd.DataFrame(columns=list(columns))
                )
                combined = pd.concat([existing, addition], ignore_index=True)
                if not combined.empty:
                    combined = combined.drop_duplicates().reset_index(drop=True)
                data_path, manifest_path = self._shared_cache_paths(
                    cache_dir, domain_name=domain_name, sku_code=sku_code
                )
                _write_parquet_atomic(data_path, combined)
                _write_json_atomic(
                    manifest_path,
                    {
                        "version": SHARED_SOURCE_CACHE_VERSION,
                        "status": "complete",
                        "fingerprint": fingerprint,
                        "source_name": source_name,
                        "sku_code": sku_code,
                        "start_date": window_by_sku[sku_code]["start_date"],
                        "cached_through": item["end_date"],
                        "rows": len(combined),
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                cached_frames[sku_code] = combined
            _emit(
                progress,
                "shared_source_cache_extended",
                batch=batch_index,
                batches=batch_count,
                source=domain_name,
                skus=len(missing_windows),
                rows=len(additions),
                partitions=len(result.selected_partitions),
                elapsed_seconds=round(monotonic() - started, 3),
            )
        combined = _concat(list(cached_frames.values()), tuple(columns))
        return _filter_to_sku_windows(combined, windows)

    def _read_shared_cached_products(
        self,
        *,
        skus: Sequence[str],
        snapshot_time: datetime,
        cache_dir: Path,
        batch_index: int,
        batch_count: int,
        progress: ProgressCallback | None,
    ) -> pd.DataFrame:
        fingerprint = self._shared_cache_fingerprint(
            source_name=SOURCES["product"], columns=PRODUCT_COLUMNS
        )
        frames: list[pd.DataFrame] = []
        missing: list[str] = []
        for sku_code in skus:
            data_path, manifest_path = self._shared_cache_paths(
                cache_dir, domain_name="product", sku_code=str(sku_code)
            )
            try:
                manifest = _read_json(manifest_path)
                if (
                    manifest.get("status") == "complete"
                    and manifest.get("fingerprint") == fingerprint
                    and manifest.get("sku_code") == str(sku_code)
                    and manifest.get("snapshot_time") == snapshot_time.isoformat()
                    and data_path.is_file()
                ):
                    frames.append(pd.read_parquet(data_path))
                    continue
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                pass
            missing.append(str(sku_code))
        if missing:
            resolved = self._read_products(
                missing,
                snapshot_time,
                progress=progress,
                batch_index=batch_index,
                batch_count=batch_count,
            )
            for sku_code, group in resolved.groupby("sku_code", sort=False):
                sku_code = str(sku_code)
                data_path, manifest_path = self._shared_cache_paths(
                    cache_dir, domain_name="product", sku_code=sku_code
                )
                _write_parquet_atomic(data_path, group.reset_index(drop=True))
                _write_json_atomic(
                    manifest_path,
                    {
                        "version": SHARED_SOURCE_CACHE_VERSION,
                        "status": "complete",
                        "fingerprint": fingerprint,
                        "sku_code": sku_code,
                        "snapshot_time": snapshot_time.isoformat(),
                        "rows": len(group),
                    },
                )
                frames.append(group.reset_index(drop=True))
        _emit(
            progress,
            "shared_product_cache_complete",
            batch=batch_index,
            batches=batch_count,
            requested_skus=len(skus),
            cache_hit_skus=len(skus) - len(missing),
            computed_skus=len(missing),
        )
        return self._validate_products(
            _concat(frames, PRODUCT_COLUMNS), [str(value) for value in skus]
        )

    def _load_stores(
        self,
        *,
        cutoff_time: datetime,
        run_dir: Path | None,
        run_fingerprint: str,
        resume: bool,
        progress: ProgressCallback | None,
    ) -> pd.DataFrame:
        manifest_path = None if run_dir is None else run_dir / "stores.json"
        store_path = None if run_dir is None else run_dir / "stores.parquet"
        if resume and manifest_path is not None and store_path is not None:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("status") == "complete"
                    and manifest.get("run_fingerprint") == run_fingerprint
                ):
                    stores = pd.read_parquet(store_path)
                    if len(stores) == int(manifest["rows"]) and set(STORE_COLUMNS).issubset(stores.columns):
                        _emit(progress, "store_checkpoint_reused", rows=len(stores))
                        return stores
                    _emit(progress, "store_attributes_cache_upgrade", missing_columns=sorted(set(STORE_COLUMNS) - set(stores.columns)))
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                _emit(progress, "store_checkpoint_missing_or_invalid")
        _emit(progress, "store_read_start")
        started = monotonic()
        stores = self.repository.read_latest_partition(
            source_name=SOURCES["store"],
            cutoff_time=cutoff_time,
            columns=STORE_COLUMNS,
        ).frame
        _emit(
            progress,
            "store_read_complete",
            rows=len(stores),
            elapsed_seconds=round(monotonic() - started, 3),
        )
        if manifest_path is not None and store_path is not None:
            _write_json_atomic(
                manifest_path,
                {
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "run_fingerprint": run_fingerprint,
                    "status": "writing",
                },
            )
            _write_parquet_atomic(store_path, stores)
            _write_json_atomic(
                manifest_path,
                {
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "run_fingerprint": run_fingerprint,
                    "status": "complete",
                    "rows": len(stores),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        return stores

    def load_batched(
        self,
        *,
        selected: pd.DataFrame,
        cutoff_time: datetime,
        window_days: int,
        product_snapshot_time: datetime | None = None,
        batch_size: int = 5,
        checkpoint_dir: Path | None = None,
        resume: bool = True,
        progress: ProgressCallback | None = None,
        batch_callback: BatchCallback | None = None,
        materialize_sources: bool = True,
        shared_source_cache_dir: Path | None = None,
        defer_cached_sources: bool = False,
    ) -> RestorationSourceBundle:
        if selected.empty:
            raise DataContractViolation("no SKUs selected for restoration")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        run_fingerprint = self._run_fingerprint(
            cutoff_time=cutoff_time,
            window_days=window_days,
            product_snapshot_time=product_snapshot_time,
        )
        run_dir = (
            None
            if checkpoint_dir is None
            else checkpoint_dir / f"m2-{run_fingerprint[:20]}"
        )
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
        batch_count = (len(selected) + batch_size - 1) // batch_size
        _emit(
            progress,
            "checkpoint_plan",
            checkpoint_dir=str(run_dir) if run_dir is not None else "disabled",
            resume=resume,
            batch_size=batch_size,
            batches=batch_count,
            skus=len(selected),
        )
        stores = self._load_stores(
            cutoff_time=cutoff_time,
            run_dir=run_dir,
            run_fingerprint=run_fingerprint,
            resume=resume,
            progress=progress,
        )
        sales_frames: list[pd.DataFrame] = []
        inventory_frames: list[pd.DataFrame] = []
        placement_frames: list[pd.DataFrame] = []
        product_frames: list[pd.DataFrame] = []
        all_windows: list[dict[str, str]] = []
        reused = 0
        computed = 0
        for zero_index in range(batch_count):
            batch_index = zero_index + 1
            batch = selected.iloc[
                zero_index * batch_size : (zero_index + 1) * batch_size
            ].copy()
            skus = [str(value) for value in batch["sku_code"]]
            batch_fingerprint = self._batch_fingerprint(
                batch, run_fingerprint=run_fingerprint
            )
            batch_dir = (
                None
                if run_dir is None
                else run_dir / f"batch-{batch_fingerprint[:20]}"
            )
            manifest_path = None if batch_dir is None else batch_dir / "manifest.json"
            frames: dict[str, pd.DataFrame] | None = None
            batch_reused = False
            windows = _source_windows(batch, cutoff_time, window_days)
            if resume and manifest_path is not None:
                _emit(
                    progress,
                    "batch_checkpoint_load_start",
                    batch=batch_index,
                    batches=batch_count,
                    skus=",".join(skus),
                )
                started = monotonic()
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if (
                        manifest.get("run_fingerprint") == run_fingerprint
                        and manifest.get("batch_fingerprint") == batch_fingerprint
                        and manifest.get("skus") == skus
                    ):
                        frames = self._checkpoint_frames(batch_dir, manifest, defer_sources=defer_cached_sources and not materialize_sources)
                except (OSError, ValueError, json.JSONDecodeError):
                    frames = None
                if frames is not None:
                    reused += 1
                    batch_reused = True
                    _emit(
                        progress,
                        "batch_checkpoint_reused",
                        batch=batch_index,
                        batches=batch_count,
                        skus=",".join(skus),
                        elapsed_seconds=round(monotonic() - started, 3),
                    )
                else:
                    _emit(
                        progress,
                        "batch_checkpoint_missing_or_invalid",
                        batch=batch_index,
                        batches=batch_count,
                    )
            if frames is None:
                computed += 1
                _emit(
                    progress,
                    "batch_start",
                    batch=batch_index,
                    batches=batch_count,
                    skus=",".join(skus),
                )
                if manifest_path is not None:
                    _write_json_atomic(
                        manifest_path,
                        {
                            "checkpoint_version": CHECKPOINT_VERSION,
                            "run_fingerprint": run_fingerprint,
                            "batch_fingerprint": batch_fingerprint,
                            "status": "writing",
                            "batch": batch_index,
                            "skus": skus,
                        },
                    )
                frames, windows = self._read_source_batch(
                    selected=batch,
                    cutoff_time=cutoff_time,
                    product_snapshot_time=product_snapshot_time,
                    window_days=window_days,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    progress=progress,
                    shared_source_cache_dir=shared_source_cache_dir,
                )
                if batch_dir is not None and manifest_path is not None:
                    _emit(
                        progress,
                        "batch_checkpoint_write_start",
                        batch=batch_index,
                        batches=batch_count,
                    )
                    started = monotonic()
                    for name, frame in frames.items():
                        _write_parquet_atomic(batch_dir / f"{name}.parquet", frame)
                    _write_json_atomic(
                        manifest_path,
                        {
                            "checkpoint_version": CHECKPOINT_VERSION,
                            "run_fingerprint": run_fingerprint,
                            "batch_fingerprint": batch_fingerprint,
                            "status": "complete",
                            "batch": batch_index,
                            "skus": skus,
                            "windows": windows,
                            "row_counts": {
                                name: len(frame) for name, frame in frames.items()
                            },
                            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    _emit(
                        progress,
                        "batch_checkpoint_write_complete",
                        batch=batch_index,
                        batches=batch_count,
                        elapsed_seconds=round(monotonic() - started, 3),
                    )
                _emit(
                    progress,
                    "batch_complete",
                    batch=batch_index,
                    batches=batch_count,
                    sales_rows=len(frames["sales"]),
                    inventory_rows=len(frames["inventory"]),
                    placement_rows=len(frames["placement"]),
                )
            if batch_callback is not None:
                batch_callback(
                    RestorationSourceBatch(
                        batch_index=batch_index,
                        batch_count=batch_count,
                        selected=batch,
                        frames=frames,
                        stores=stores,
                        windows=windows,
                        checkpoint_run_dir=run_dir,
                        checkpoint_batch_dir=batch_dir,
                        reused=batch_reused,
                        deferred_source_frames=batch_reused and defer_cached_sources and not materialize_sources,
                    )
                )
            if materialize_sources:
                sales_frames.append(frames["sales"])
                inventory_frames.append(frames["inventory"])
                placement_frames.append(frames["placement"])
            product_frames.append(frames["product"])
            all_windows.extend(windows)
        products = self._validate_products(
            _concat(product_frames, PRODUCT_COLUMNS),
            [str(value) for value in selected["sku_code"]],
        )
        _emit(
            progress,
            "all_source_batches_complete",
            batches=batch_count,
            reused_batches=reused,
            computed_batches=computed,
        )
        return RestorationSourceBundle(
            raw_sales=_concat(sales_frames, SALES_COLUMNS),
            inventory=_concat(inventory_frames, INVENTORY_COLUMNS),
            placement=_concat(placement_frames, PLACEMENT_COLUMNS),
            stores=stores,
            product_attributes=products,
            first_sale_dates=dict(zip(selected["sku_code"], selected["first_actual_sale_date"])),
            fit_skus=set(selected.loc[selected["split"] == "Train", "sku_code"]),
            training_weights=dict(zip(selected["sku_code"], selected["long_tail_weight"])),
            source_windows=all_windows,
            checkpoint_run_dir=run_dir,
            reused_batch_count=reused,
            computed_batch_count=computed,
        )

    def load(
        self,
        *,
        selected: pd.DataFrame,
        cutoff_time: datetime,
        window_days: int,
        product_snapshot_time: datetime | None = None,
    ) -> RestorationSourceBundle:
        """Compatibility wrapper for callers that do not need checkpoints."""

        return self.load_batched(
            selected=selected,
            cutoff_time=cutoff_time,
            window_days=window_days,
            product_snapshot_time=product_snapshot_time,
            batch_size=len(selected),
            checkpoint_dir=None,
            resume=False,
        )
