from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from config.settings import AppConfig
from restore.batched import _write_parquet_atomic
from restore.historical import _aggregate_weekly
from test_step4_verification import panel_for


@pytest.mark.parametrize("observed_store,fallback_store", [("A", "B"), ("B", "A")])
def test_mixed_observed_and_arrival_fallback_dates_have_stable_parquet_schema(
    tmp_path, observed_store, fallback_store,
):
    observed = panel_for(days=(5,), flag=0, stocked=True, arrival=3)
    fallback = panel_for(days=(5,), flag=None, stocked=False, arrival=3)
    observed["store_code"] = observed_store
    fallback["store_code"] = fallback_store
    panel = pd.concat([observed, fallback], ignore_index=True)
    panel.attrs["cutoff_date"] = date(2025, 1, 21)
    config = AppConfig.load(Path(__file__).resolve().parents[1]).restoration
    weekly = _aggregate_weekly(panel, config)
    path = tmp_path / "weekly_pre_model.parquet"
    # The real checkpoint writer must succeed regardless of which date type
    # was encountered first; keep exposure/sales accounting unchanged.
    _write_parquet_atomic(path, weekly)
    loaded = pd.read_parquet(path)
    assert pd.api.types.is_datetime64_any_dtype(loaded.sellable_proxy_date)
    for frame in (weekly, loaded):
        good = frame.loc[frame.store_code.eq(observed_store)]
        bad = frame.loc[frame.store_code.eq(fallback_store)]
        assert pd.to_datetime(good.sellable_proxy_date).eq(pd.Timestamp("2025-01-05")).all()
        assert pd.to_datetime(bad.sellable_proxy_date).eq(pd.Timestamp("2025-01-03")).all()
        assert good.sellable_proxy_status.eq("first_observed_open_stocked_display_flag_zero").all()
        assert bad.sellable_proxy_status.eq("arrival_only_low_confidence").all()
        assert good.observed_exposure_days.sum() == 1
        assert bad.observed_exposure_days.sum() == 0
        assert bad.unknown_exposure_days.sum() == 19
        assert good.clean_sales_qty.sum() == bad.clean_sales_qty.sum() == 2
    assert not path.with_suffix(".parquet.tmp").exists()


@pytest.mark.parametrize("arrival_missing", [False, True])
def test_no_observed_sellable_dates_still_has_typed_fallback(tmp_path, arrival_missing):
    panel = panel_for(days=(5,), flag=None, stocked=False, arrival=3)
    if arrival_missing:
        panel["arrival_date"] = pd.NaT
    config = AppConfig.load(Path(__file__).resolve().parents[1]).restoration
    weekly = _aggregate_weekly(panel, config)
    _write_parquet_atomic(tmp_path / "weekly.parquet", weekly)
    assert pd.api.types.is_datetime64_any_dtype(weekly.sellable_proxy_date)
    if arrival_missing:
        assert weekly.sellable_proxy_date.isna().all()
    else:
        assert weekly.sellable_proxy_date.eq(pd.Timestamp("2025-01-03")).all()
    assert weekly.sellable_proxy_status.eq("arrival_only_low_confidence").all()
