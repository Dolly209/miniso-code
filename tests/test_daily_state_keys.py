from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest

from config.settings import AppConfig
from data.daily_state_keys import DailyStateConflict
from restore.batched import BatchRecord, _prepare_batch
from restore.historical import _prepare_inventory, _prepare_placement, build_daily_panel, normalize_order_lines, prepare_sales_daily
from test_restore import CUTOFF, _sales_row


def placement(**changes):
    return dict(dict(ds="2025-01-01", sku_code="SKU1", store_code="S1",
                     first_store_arrival_time="2025-01-01 09:00:00", stock_available_itm_cnt=3,
                     out_of_stock_flag=0, in_stock_not_displayed_flag=0), **changes)


def inventory(**changes):
    return dict(dict(stock_date="2025-01-01", sku_code="SKU1", store_code="S1",
                     available_stock_qty=3, out_of_stock_flag=0), **changes)


def test_exact_daily_placement_duplicates_are_one_snapshot_not_summed_stock():
    result = _prepare_placement(pd.DataFrame([placement(), placement(), placement()]))
    assert len(result) == 1
    assert result.iloc[0].stock_available_itm_cnt == 3
    assert result.attrs["daily_key_audit"]["equivalent_extra_rows"] == 2
    assert result.attrs["daily_key_audit"]["duplicate_key_count"] == 1


def test_equivalent_raw_spellings_and_day_level_arrival_aliases_are_collapsed():
    frame = pd.DataFrame([
        placement(),
        placement(ds="2025/1/1", sku_code=" SKU1 ", store_code=" S1 ",
                  first_store_arrival_time="2025-01-01 18:00:00", stock_available_itm_cnt="3.0",
                  out_of_stock_flag="有货", in_stock_not_displayed_flag="正常陈列"),
    ])
    result = _prepare_placement(frame)
    assert len(result) == 1
    assert result.iloc[0].sku_code == "SKU1"
    assert result.attrs["daily_key_audit"]["equivalent_extra_rows"] == 1


@pytest.mark.parametrize("change", [
    dict(stock_available_itm_cnt=4), dict(stock_available_itm_cnt=None),
    dict(out_of_stock_flag=1), dict(in_stock_not_displayed_flag=1),
    dict(first_store_arrival_time="2024-12-31"),
])
def test_true_daily_conflicts_never_choose_first_max_or_sum(change):
    with pytest.raises(DailyStateConflict) as caught:
        _prepare_placement(pd.DataFrame([placement(), placement(**change)]))
    assert caught.value.audit["conflicting_key_count"] == 1
    assert caught.value.audit["conflict_examples"][0]["sku_code"] == "SKU1"
    json.dumps(caught.value.audit)


def test_distinct_unrecognized_flags_do_not_become_equal_just_because_both_are_unknown():
    with pytest.raises(DailyStateConflict):
        _prepare_placement(pd.DataFrame([placement(out_of_stock_flag="new_code_1"), placement(out_of_stock_flag="new_code_2")]))
    result = _prepare_placement(pd.DataFrame([placement(out_of_stock_flag="new_code_1"), placement(out_of_stock_flag="new_code_1")]))
    assert len(result) == 1 and result.iloc[0].placement_stockout_flag_unrecognized


def test_inventory_snapshot_duplicates_follow_the_same_safe_policy():
    result = _prepare_inventory(pd.DataFrame([inventory(), inventory(available_stock_qty="3.0", out_of_stock_flag="有货")]))
    assert len(result) == 1 and result.iloc[0].available_stock_qty == 3
    with pytest.raises(DailyStateConflict):
        _prepare_inventory(pd.DataFrame([inventory(), inventory(available_stock_qty=4)]))


def test_equivalent_duplicates_do_not_multiply_sales_or_exposure_in_outer_joins():
    latest, _ = normalize_order_lines(pd.DataFrame([_sales_row("A", 2)]), cutoff_time=CUTOFF)
    products = pd.DataFrame([dict(sku_code="SKU1", cate_level4_code="C1", retail_price=10.)])
    sales, _ = prepare_sales_daily(latest, products, abnormal_order_quantile=.995, threshold_fit_skus={"SKU1"})
    stores = pd.DataFrame([dict(store_code="S1", store_business_state="normal", store_level="A", open_date=None, close_date=None, sts_state="active")])
    daily = build_daily_panel(
        sales_daily=sales, inventory=pd.DataFrame([inventory()] * 2), placement=pd.DataFrame([placement()] * 3),
        stores=stores, first_sale_dates={"SKU1": date(2025, 1, 1)}, cutoff_date=date(2025, 1, 1),
        window_days=1, promotion_price_ratio_threshold=.95,
    )
    assert len(daily) == 1
    assert daily.iloc[0].clean_sale_qty == 2
    assert daily.iloc[0].stock_available_itm_cnt == 3
    assert daily.iloc[0].e_total == 1
    assert daily.attrs["source_daily_key_audits"]["placement"]["equivalent_extra_rows"] == 2


def test_no_duplicates_and_empty_state_frames_remain_supported():
    frame = pd.DataFrame([placement(), placement(store_code="S2")])
    normalized = _prepare_placement(frame)
    assert len(normalized) == 2
    assert normalized.attrs["daily_key_audit"]["equivalent_extra_rows"] == 0
    assert _prepare_placement(frame.iloc[:0]).empty
    assert _prepare_inventory(pd.DataFrame([inventory()]).iloc[:0]).empty


def test_pre_model_failure_writes_bounded_diagnostic_without_touching_source_checkpoints(tmp_path):
    source = tmp_path / "source"
    processed = tmp_path / "processed"
    source.mkdir()
    processed.mkdir()
    pd.DataFrame([inventory()]).to_parquet(source / "inventory.parquet", index=False)
    pd.DataFrame([placement(), placement(stock_available_itm_cnt=4)]).to_parquet(source / "placement.parquet", index=False)
    before = (source / "placement.parquet").read_bytes()
    latest, _ = normalize_order_lines(pd.DataFrame([_sales_row("A", 1)]), cutoff_time=CUTOFF)
    latest.to_parquet(processed / "latest_sales.parquet", index=False)
    selected = pd.DataFrame([dict(sku_code="SKU1", first_actual_sale_date=date(2025, 1, 1), split="Train", long_tail_weight=1.)])
    record = BatchRecord(index=1, count=59, selected=selected, source_dir=source, processed_dir=processed, order_stats={})
    root = Path(__file__).resolve().parents[1]
    events = []
    with pytest.raises(DailyStateConflict):
        _prepare_batch(
            record, products=pd.DataFrame([dict(sku_code="SKU1", cate_level4_code="C1", retail_price=10.)]),
            stores=pd.DataFrame([dict(store_code="S1", store_business_state="normal", store_level="A", open_date=None, close_date=None, sts_state="active")]),
            fit_skus={"SKU1"}, training_weights={"SKU1": 1.}, cutoff_time=CUTOFF, window_days=7,
            config=AppConfig.load(root).restoration, thresholds={"__GLOBAL__": float("inf")},
            fingerprint="fixture", resume=True, persist_daily_checkpoint=False,
            progress=lambda event, details: events.append((event, details)),
        )
    diagnostic = json.loads((processed / "daily_state_conflict.json").read_text())
    assert diagnostic["batch"] == 1 and diagnostic["batches"] == 59
    assert diagnostic["conflicting_key_count"] == 1
    assert len(diagnostic["conflict_examples"]) == 2
    assert (source / "placement.parquet").read_bytes() == before
    assert events[-1][0] == "daily_state_conflict"
