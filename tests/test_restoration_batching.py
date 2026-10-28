from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from data.decision_adapter import SOURCES
from data.parquet_repository import ReadResult
from data.restoration_adapter import (
    PRODUCT_COLUMNS,
    STORE_COLUMNS,
    RestorationDataAdapter,
)


class FakeRepository:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.max_rows_per_read = 100
        self.range_calls = 0
        self.store_calls = 0
        self.product_calls = 0

    def read_latest_partition(
        self,
        *,
        source_name: str,
        cutoff_time: datetime,
        columns: tuple[str, ...],
        sku_codes: list[str] | None = None,
        max_rows: int | None = None,
        **_: object,
    ) -> ReadResult:
        if source_name == SOURCES["store"]:
            self.store_calls += 1
            row = {name: None for name in STORE_COLUMNS}
            row.update(
                {
                    "ds": cutoff_time.date().isoformat(),
                    "store_code": "STORE1",
                    "store_business_state": "normal",
                    "store_level": "A",
                    "sts_state": "active",
                }
            )
            frame = pd.DataFrame([row], columns=columns)
        else:
            assert source_name == SOURCES["product"]
            self.product_calls += 1
            frame = pd.DataFrame(
                [
                    {
                        "ds": cutoff_time.date().isoformat(),
                        "sku_code": sku,
                        "cate_level4_code": "C1",
                        "retail_price": 10.0,
                    }
                    for sku in (sku_codes or [])
                ],
                columns=PRODUCT_COLUMNS,
            )
        return ReadResult(
            source_name=source_name,
            frame=frame,
            selected_partitions=(cutoff_time.date(),),
            projected_columns=tuple(columns),
            sku_filter=None,
            sku_filters=tuple(sku_codes or ()),
        )

    def read_partition_range(
        self,
        *,
        source_name: str,
        start_date: date,
        end_date: date,
        columns: tuple[str, ...],
        sku_codes: list[str],
        **_: object,
    ) -> ReadResult:
        self.range_calls += 1
        rows = []
        current = start_date
        while current <= end_date:
            for sku in sku_codes:
                row = {name: None for name in columns}
                row.update(
                    {
                        "ds": current.isoformat(),
                        "sku_code": sku,
                        "store_code": "STORE1",
                    }
                )
                rows.append(row)
            current += timedelta(days=1)
        return ReadResult(
            source_name=source_name,
            frame=pd.DataFrame(rows, columns=columns),
            selected_partitions=tuple(),
            projected_columns=tuple(columns),
            sku_filter=None,
            sku_filters=tuple(sku_codes),
        )


def _selected() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_version": "round3-test",
                "sku_code": "SKU1",
                "first_actual_sale_date": date(2025, 1, 2),
                "split": "Train",
                "long_tail_weight": 1.0,
            },
            {
                "dataset_version": "round3-test",
                "sku_code": "SKU2",
                "first_actual_sale_date": date(2025, 1, 3),
                "split": "Train",
                "long_tail_weight": 1.0,
            },
            {
                "dataset_version": "round3-test",
                "sku_code": "SKU3",
                "first_actual_sale_date": date(2025, 1, 4),
                "split": "Test",
                "long_tail_weight": 0.5,
            },
        ]
    )


def test_batched_restore_reads_each_source_once_per_batch_and_resumes(
    tmp_path: Path,
) -> None:
    repository = FakeRepository(tmp_path / "data")
    adapter = RestorationDataAdapter(repository)  # type: ignore[arg-type]
    events: list[str] = []
    kwargs = {
        "selected": _selected(),
        "cutoff_time": datetime.fromisoformat("2025-01-10T23:59:59+08:00"),
        "window_days": 2,
        "batch_size": 2,
        "checkpoint_dir": tmp_path / "checkpoints",
        "progress": lambda event, details: events.append(event),
    }

    first = adapter.load_batched(**kwargs)

    assert repository.range_calls == 6  # 2 sequential batches x 3 history sources
    assert repository.store_calls == 1
    assert repository.product_calls == 2
    assert len(first.raw_sales) == 6  # 3 SKUs x their exact 2-day windows
    assert first.computed_batch_count == 2
    assert first.reused_batch_count == 0
    assert "source_read_complete" in events

    calls_before_resume = (
        repository.range_calls,
        repository.store_calls,
        repository.product_calls,
    )
    second = adapter.load_batched(**kwargs)

    assert (
        repository.range_calls,
        repository.store_calls,
        repository.product_calls,
    ) == calls_before_resume
    assert second.computed_batch_count == 0
    assert second.reused_batch_count == 2
    assert len(second.raw_sales) == len(first.raw_sales)


def test_batched_restore_reuses_unchanged_batches_when_sku_scope_expands(
    tmp_path: Path,
) -> None:
    repository = FakeRepository(tmp_path / "data")
    adapter = RestorationDataAdapter(repository)  # type: ignore[arg-type]
    common = {
        "cutoff_time": datetime.fromisoformat("2025-01-10T23:59:59+08:00"),
        "window_days": 2,
        "batch_size": 2,
        "checkpoint_dir": tmp_path / "checkpoints",
    }

    first = adapter.load_batched(selected=_selected().iloc[:2].copy(), **common)
    assert first.computed_batch_count == 1
    assert repository.range_calls == 3
    assert repository.product_calls == 1

    expanded = adapter.load_batched(selected=_selected(), **common)

    assert expanded.reused_batch_count == 1
    assert expanded.computed_batch_count == 1
    assert repository.range_calls == 6
    assert repository.product_calls == 2


def test_shared_source_cache_extends_once_and_never_leaks_future_days(
    tmp_path: Path,
) -> None:
    repository = FakeRepository(tmp_path / "data")
    adapter = RestorationDataAdapter(repository)  # type: ignore[arg-type]
    cache_dir = tmp_path / "shared-cache"
    common = {
        "domain_name": "inventory",
        "source_name": SOURCES["inventory"],
        "columns": (
            "ds",
            "stock_date",
            "store_code",
            "sku_code",
            "available_stock_qty",
            "out_of_stock_flag",
        ),
        "cache_dir": cache_dir,
        "batch_index": 1,
        "batch_count": 1,
        "progress": None,
    }

    first = adapter._read_shared_cached_fact_batch(  # noqa: SLF001
        windows=[
            {
                "sku_code": "SKU1",
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",
            }
        ],
        cutoff_time=datetime.fromisoformat("2025-01-02T23:59:59+08:00"),
        **common,
    )
    assert repository.range_calls == 1
    assert first["ds"].tolist() == ["2025-01-01", "2025-01-02"]

    extended = adapter._read_shared_cached_fact_batch(  # noqa: SLF001
        windows=[
            {
                "sku_code": "SKU1",
                "start_date": "2025-01-01",
                "end_date": "2025-01-04",
            }
        ],
        cutoff_time=datetime.fromisoformat("2025-01-04T23:59:59+08:00"),
        **common,
    )
    assert repository.range_calls == 2
    assert extended["ds"].tolist() == [
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
        "2025-01-04",
    ]

    earlier_cutoff = adapter._read_shared_cached_fact_batch(  # noqa: SLF001
        windows=[
            {
                "sku_code": "SKU1",
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",
            }
        ],
        cutoff_time=datetime.fromisoformat("2025-01-02T23:59:59+08:00"),
        **common,
    )
    assert repository.range_calls == 2
    assert earlier_cutoff["ds"].tolist() == ["2025-01-01", "2025-01-02"]


def test_deferred_resume_uses_footer_not_fact_deserialization(tmp_path, monkeypatch):
    repository = FakeRepository(tmp_path / "data")
    adapter = RestorationDataAdapter(repository)
    common = dict(selected=_selected(), cutoff_time=datetime.fromisoformat("2025-01-10T23:59:59+08:00"),
                  window_days=2, batch_size=2, checkpoint_dir=tmp_path / "checkpoints")
    adapter.load_batched(**common)
    original = pd.read_parquet
    def guarded(path, *args, **kwargs):
        assert Path(path).name not in {"sales.parquet", "inventory.parquet", "placement.parquet"}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", guarded)
    batches = []
    resumed = adapter.load_batched(**common, materialize_sources=False, defer_cached_sources=True, batch_callback=batches.append)
    assert resumed.reused_batch_count == 2
    assert all(batch.deferred_source_frames for batch in batches)
    assert all(batch.frames["sales"].empty for batch in batches)
    # Empty placeholders must never be passed off as materialized facts.
    monkeypatch.setattr(pd, "read_parquet", original)
    materialized = adapter.load_batched(**common, materialize_sources=True, defer_cached_sources=True)
    assert len(materialized.raw_sales) == 6


def test_old_store_cache_upgrades_only_real_dimension_not_fact_sources(tmp_path):
    repository = FakeRepository(tmp_path / "data")
    adapter = RestorationDataAdapter(repository)
    common = dict(selected=_selected(), cutoff_time=datetime.fromisoformat("2025-01-10T23:59:59+08:00"),
                  window_days=2, batch_size=2, checkpoint_dir=tmp_path / "checkpoints")
    bundle = adapter.load_batched(**common)
    store_path = bundle.checkpoint_run_dir / "stores.parquet"
    old = pd.read_parquet(store_path).drop(columns=["city_name", "operate_size"])
    old.to_parquet(store_path, index=False)
    fact_calls = repository.range_calls
    upgraded = adapter.load_batched(**common)
    assert {"city_name", "operate_size"}.issubset(upgraded.stores.columns)
    assert repository.store_calls == 2
    assert repository.range_calls == fact_calls


def test_deferred_corrupt_footer_is_not_silently_treated_as_empty(tmp_path):
    repository = FakeRepository(tmp_path / "data")
    adapter = RestorationDataAdapter(repository)
    common = dict(selected=_selected(), cutoff_time=datetime.fromisoformat("2025-01-10T23:59:59+08:00"),
                  window_days=2, batch_size=2, checkpoint_dir=tmp_path / "checkpoints")
    bundle = adapter.load_batched(**common)
    path = next(bundle.checkpoint_run_dir.glob("batch-*/inventory.parquet"))
    frame = pd.read_parquet(path).iloc[:0]
    frame.to_parquet(path, index=False)
    resumed = adapter.load_batched(**common, materialize_sources=False, defer_cached_sources=True)
    assert resumed.computed_batch_count == 1
    assert resumed.reused_batch_count == 1


def test_full_batched_fit_keeps_features_and_reuses_old_pre_model_cache(tmp_path, monkeypatch):
    from config.settings import AppConfig
    from restore.batched import run_batched_restoration

    class NormalRepository(FakeRepository):
        def read_latest_partition(self, **kwargs):
            result = super().read_latest_partition(**kwargs)
            if kwargs["source_name"] == SOURCES["store"]:
                result.frame["sts_state"] = "1"
                result.frame["store_business_state"] = "7"
                result.frame["city_name"] = "Shanghai"
                result.frame["operate_size"] = 120.
            return result

        def read_partition_range(self, **kwargs):
            result = super().read_partition_range(**kwargs)
            frame = result.frame
            source = kwargs["source_name"]
            if source == SOURCES["sales"]:
                frame["order_id"] = frame["sku_code"] + frame["ds"]
                frame["order_detail_id"] = "1"
                frame["order_time"] = frame["ds"] + " 10:00:00"
                frame["sale_qty"] = 1.
                for field in ("order_amt", "paid_amt", "unit_paid_price", "achievement_amt"):
                    frame[field] = 10.
                frame["is_return"] = "0"
                frame["gift_flag"] = "N"
            elif source == SOURCES["inventory"]:
                frame["stock_date"] = frame["ds"]
                frame["available_stock_qty"] = 5.
                frame["out_of_stock_flag"] = 0
            elif source == SOURCES["placement"]:
                frame["first_store_arrival_time"] = "2025-01-02"
                frame["stock_available_itm_cnt"] = 5.
                frame["out_of_stock_flag"] = 0
                frame["in_stock_not_displayed_flag"] = "--"
            return result

    repository = NormalRepository(tmp_path / "data")
    adapter = RestorationDataAdapter(repository)
    events = []
    common = dict(adapter=adapter, selected=_selected(), cutoff_time=datetime.fromisoformat("2025-01-21T23:59:59+08:00"),
                  window_days=14, batch_size=2, checkpoint_dir=tmp_path / "checkpoints",
                  config=AppConfig.load(Path(__file__).resolve().parents[1]).restoration,
                  events=None, resume=True, write_daily_audit=False, write_partition_parts=False,
                  progress=lambda event, details: events.append(event))
    first, bundle = run_batched_restoration(**common, output_dir=tmp_path / "baseline")
    assert {"store_tier", "lifecycle_week", "city_name", "operate_size"}.issubset(first.model_report["fit_input_columns"])
    assert first.model_report["category_models"]["__GLOBAL__"]["feature_input_audit"]["city_name"]["observed_rows"] > 0
    baseline = pd.read_csv(first.paths["weekly"])
    # Simulate old placeholders; enrichment must recover attributes without
    # invalidating facts/exposure or relying on newly persisted weekly inputs.
    for path in bundle.checkpoint_run_dir.glob("batch-*/processed/weekly_pre_model.parquet"):
        frame = pd.read_parquet(path)
        frame["city_name"] = None
        frame["operate_size"] = float("nan")
        frame.to_parquet(path, index=False)
    original = pd.read_parquet
    def guarded(path, *args, **kwargs):
        assert Path(path).name not in {"sales.parquet", "inventory.parquet", "placement.parquet"}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", guarded)
    events.clear()
    second, resumed = run_batched_restoration(**common, output_dir=tmp_path / "new_output")
    assert resumed.reused_batch_count == 2 and resumed.computed_batch_count == 0
    assert events.count("pre_model_checkpoint_reused") == 2
    assert "pre_model_checkpoint_written" not in events
    pd.testing.assert_frame_equal(baseline, pd.read_csv(second.paths["weekly"]))
    assert first.paths["weekly"].is_file()  # baseline not overwritten
    events.clear()
    increased, _ = run_batched_restoration(**common, output_dir=tmp_path / "increased_budget", huber_max_iter=1000)
    assert events.count("pre_model_checkpoint_reused") == 2
    assert "pre_model_checkpoint_written" not in events
    assert increased.model_report["solver_controls"]["huber_max_iter"] == 1000
    assert common["config"].huber_max_iter == 100
    assert increased.model_report["model_id"] != second.model_report["model_id"]
    pd.testing.assert_frame_equal(baseline, pd.read_csv(increased.paths["weekly"]))
