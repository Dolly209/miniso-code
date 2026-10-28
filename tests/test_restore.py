from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from config.settings import AppConfig
from restore.historical import (
    FittedRestorationModel,
    _aggregate_weekly,
    assign_event_codes,
    build_daily_panel,
    build_restoration_quality_report,
    build_restoration_quality_report_from_summary,
    build_restoration_result,
    fit_and_restore_weekly,
    fit_restoration_model,
    neutralize_period_effects,
    normalize_order_lines,
    summarize_daily_quality_inputs,
    write_restoration_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUTOFF = datetime.fromisoformat("2025-01-21T23:59:59+08:00")


def test_restoration_model_artifact_round_trip() -> None:
    original = FittedRestorationModel(
        models={
            "__GLOBAL__": (
                {"intercept": 1.25, "price_elasticity": -0.2},
                {"SKU1": 0.1},
                {"STORE1": -0.05},
            )
        },
        category_p99={"C1||*": 9.0},
        global_p99=10.0,
        report={
            "model_id": "vmodel-test",
            "train_end_time": "2025-01-01T23:59:59+08:00",
            "abnormal_order_thresholds": {"__GLOBAL__": 3.0},
        },
    )

    restored = FittedRestorationModel.from_dict(original.as_dict())

    assert restored == original


def _sales_row(
    order_id: str,
    quantity: float,
    *,
    order_time: str = "2025-01-01 10:00:00",
    ds: str = "2025-01-01",
) -> dict[str, object]:
    return {
        "ds": ds,
        "order_id": order_id,
        "order_detail_id": "1",
        "order_time": order_time,
        "sku_code": "SKU1",
        "store_code": "S1",
        "sale_qty": quantity,
        "order_amt": quantity * 10,
        "paid_amt": quantity * 10,
        "unit_paid_price": 10.0,
        "achievement_amt": quantity * 10,
        "is_return": "0",
        "gift_flag": "N",
    }


def test_order_normalization_collapses_exact_rows_and_nets_signed_events() -> None:
    exact = _sales_row("A", 2)
    raw = pd.DataFrame(
        [
            exact,
            exact,
            _sales_row("B", 1, order_time="2025-01-01 11:00:00"),
            _sales_row("B", -1, order_time="2025-01-01 11:05:00"),
        ]
    )

    normalized, stats = normalize_order_lines(raw, cutoff_time=CUTOFF)

    assert stats["exact_duplicate_rows"] == 1
    assert stats["signed_aggregation_extra_rows"] == 1
    assert dict(zip(normalized["order_id"], normalized["sale_qty"])) == {
        "A": 2.0,
        "B": 0.0,
    }


def test_order_normalization_does_not_fall_back_after_future_latest_version() -> None:
    raw = pd.DataFrame(
        [
            _sales_row("A", 2),
            _sales_row(
                "A",
                3,
                order_time="2025-01-21 23:59:59",
                ds="2025-01-21",
            ),
        ]
    )
    cutoff = datetime.fromisoformat("2025-01-21T12:00:00+08:00")

    normalized, _ = normalize_order_lines(raw, cutoff_time=cutoff)

    assert normalized.empty


def test_weekly_aggregation_keeps_nullable_exposure_out_of_boolean_masks() -> None:
    config = AppConfig.load(PROJECT_ROOT).restoration
    panel = pd.DataFrame(
        [
            {
                "sku_code": "SKU1",
                "cate_level4_code": "TOY",
                "store_code": "S1",
                "store_business_state": "NORMAL",
                "store_level": "A",
                "store_tier": "NORMAL|A",
                "lifecycle_week": 1,
                "week_start": date(2025, 1, 1),
                "week_end": date(2025, 1, 7),
                "complete_week": True,
                "business_day": date(2025, 1, 1),
                "e_open": 1.0,
                "e_stock": 1.0,
                "e_display": 1.0,
                "e_total": 1.0,
                "anomaly_confidence": 1.0,
                "reported_sale_qty": 2.0,
                "clean_sale_qty": 2.0,
                "abnormal_sale_qty": 0.0,
                "abnormal_order_count": 0,
                "price_log_ratio": 0.0,
                "promotion_price_proxy": False,
            },
            {
                "sku_code": "SKU1",
                "cate_level4_code": "TOY",
                "store_code": "S1",
                "store_business_state": "NORMAL",
                "store_level": "A",
                "store_tier": "NORMAL|A",
                "lifecycle_week": 1,
                "week_start": date(2025, 1, 1),
                "week_end": date(2025, 1, 7),
                "complete_week": True,
                "business_day": date(2025, 1, 2),
                "e_open": 1.0,
                "e_stock": pd.NA,
                "e_display": pd.NA,
                "e_total": pd.NA,
                "anomaly_confidence": 1.0,
                "reported_sale_qty": 0.0,
                "clean_sale_qty": 0.0,
                "abnormal_sale_qty": 0.0,
                "abnormal_order_count": 0,
                "price_log_ratio": float("nan"),
                "promotion_price_proxy": False,
            },
        ]
    )
    for column in ("e_open", "e_stock", "e_display", "e_total"):
        panel[column] = panel[column].astype("Float64")

    weekly = _aggregate_weekly(panel, config)

    assert weekly.loc[0, "observed_exposure_days"] == 1
    assert weekly.loc[0, "known_censored_days"] == 0
    assert weekly.loc[0, "unknown_exposure_days"] == 1
    assert weekly.loc[0, "n_eff"] == 1.0


def test_unknown_exposure_sales_are_preserved_but_not_used_for_speed_fit() -> None:
    config = AppConfig.load(PROJECT_ROOT).restoration
    panel = pd.DataFrame(
        [
            {
                "sku_code": "SKU1",
                "cate_level4_code": "TOY",
                "store_code": "S1",
                "store_business_state": "NORMAL",
                "store_level": "A",
                "store_tier": "NORMAL|A",
                "lifecycle_week": 1,
                "week_start": date(2025, 1, 1),
                "week_end": date(2025, 1, 7),
                "complete_week": True,
                "business_day": date(2025, 1, 1),
                "e_open": 1.0,
                "e_stock": 1.0,
                "e_display": 1.0,
                "e_total": 1.0,
                "anomaly_confidence": 1.0,
                "reported_sale_qty": 2.0,
                "clean_sale_qty": 2.0,
                "abnormal_sale_qty": 0.0,
                "abnormal_order_count": 0,
                "price_log_ratio": 0.0,
                "promotion_price_proxy": False,
            },
            {
                "sku_code": "SKU1",
                "cate_level4_code": "TOY",
                "store_code": "S1",
                "store_business_state": "NORMAL",
                "store_level": "A",
                "store_tier": "NORMAL|A",
                "lifecycle_week": 1,
                "week_start": date(2025, 1, 1),
                "week_end": date(2025, 1, 7),
                "complete_week": True,
                "business_day": date(2025, 1, 2),
                "e_open": 1.0,
                "e_stock": pd.NA,
                "e_display": pd.NA,
                "e_total": pd.NA,
                "anomaly_confidence": 1.0,
                "reported_sale_qty": 3.0,
                "clean_sale_qty": 3.0,
                "abnormal_sale_qty": 0.0,
                "abnormal_order_count": 0,
                "price_log_ratio": 0.0,
                "promotion_price_proxy": False,
            },
        ]
    )
    for column in ("e_open", "e_stock", "e_display", "e_total"):
        panel[column] = panel[column].astype("Float64")

    weekly, _ = fit_and_restore_weekly(
        panel,
        config=config,
        fit_skus={"SKU1"},
        training_weights={"SKU1": 1.0},
    )
    row = weekly.iloc[0]

    assert row["unknown_exposure_clean_sales_qty"] == 3.0
    assert row["unmodeled_observed_sales_qty"] == 3.0
    assert row["price_neutral_observed_sales"] == 5.0
    assert row["restored_natural_sales"] == 5.0
    assert row["model_natural_sales"] == 2.0
    assert row["v_obs"] == 14.0


def test_negative_signed_store_week_is_preserved_but_demand_is_floored() -> None:
    config = AppConfig.load(PROJECT_ROOT).restoration
    rows: list[dict[str, object]] = []
    for sku_code, quantity in (("FIT", 2.0), ("RETURN_ONLY", -1.0)):
        rows.append(
            {
                "sku_code": sku_code,
                "cate_level4_code": "TOY",
                "store_code": "S1",
                "store_business_state": "NORMAL",
                "store_level": "A",
                "store_tier": "NORMAL|A",
                "lifecycle_week": 1,
                "week_start": date(2025, 1, 1),
                "week_end": date(2025, 1, 7),
                "complete_week": True,
                "business_day": date(2025, 1, 1),
                "e_open": 1.0,
                "e_stock": 1.0,
                "e_display": 1.0,
                "e_total": 1.0,
                "anomaly_confidence": 1.0,
                "reported_sale_qty": quantity,
                "clean_sale_qty": quantity,
                "abnormal_sale_qty": 0.0,
                "abnormal_order_count": 0,
                "price_log_ratio": 0.0,
                "promotion_price_proxy": False,
            }
        )
    panel = pd.DataFrame(rows)
    for column in ("e_open", "e_stock", "e_display", "e_total"):
        panel[column] = panel[column].astype("Float64")

    weekly, _ = fit_and_restore_weekly(
        panel,
        config=config,
        fit_skus={"FIT", "RETURN_ONLY"},
        training_weights={"FIT": 1.0, "RETURN_ONLY": 1.0},
    )
    returned = weekly.loc[weekly["sku_code"] == "RETURN_ONLY"].iloc[0]

    assert returned["clean_sales_qty"] == -1.0
    assert returned["price_neutral_observed_sales"] == -1.0
    assert returned["raw_restored_natural_sales"] == -1.0
    assert returned["negative_net_sales_floor_adjustment"] == 1.0
    assert bool(returned["negative_net_sales_floor_applied"])
    assert returned["restored_natural_sales"] == 0.0
    assert returned["model_natural_sales"] == 0.0
    assert returned["v_obs"] == 0.0
    assert returned["v_star"] >= 0.0


def test_persistent_stockout_with_sales_uses_audited_lower_bound() -> None:
    config = AppConfig.load(PROJECT_ROOT).restoration
    rows: list[dict[str, object]] = []
    for sku_code in ("GOOD", "BAD"):
        for index in range(28):
            day = date(2025, 1, 1) + timedelta(days=index)
            exposed = sku_code == "GOOD"
            sale = 1.0 if exposed or index < 3 else 0.0
            rows.append(
                {
                    "sku_code": sku_code,
                    "cate_level4_code": "TOY",
                    "store_code": "S1",
                    "store_business_state": "NORMAL",
                    "store_level": "A",
                    "store_tier": "NORMAL|A",
                    "lifecycle_week": index // 7 + 1,
                    "week_start": date(2025, 1, 1)
                    + timedelta(days=(index // 7) * 7),
                    "week_end": date(2025, 1, 7)
                    + timedelta(days=(index // 7) * 7),
                    "complete_week": True,
                    "business_day": day,
                    "e_open": 1.0,
                    "e_stock": 1.0 if exposed else 0.0,
                    "e_display": 1.0,
                    "e_total": 1.0 if exposed else 0.0,
                    "anomaly_confidence": 1.0,
                    "reported_sale_qty": sale,
                    "clean_sale_qty": sale,
                    "abnormal_sale_qty": 0.0,
                    "abnormal_order_count": 0,
                    "price_log_ratio": 0.0,
                    "promotion_price_proxy": False,
                }
            )
    panel = pd.DataFrame(rows)

    weekly, model_report = fit_and_restore_weekly(
        panel,
        config=config,
        fit_skus={"GOOD", "BAD"},
        training_weights={"GOOD": 1.0, "BAD": 1.0},
    )
    bad = weekly.loc[weekly["sku_code"] == "BAD"]

    assert model_report["fit_sku_count"] == 1
    assert bad["stock_grain_unreliable"].all()
    assert not bad["model_fit_eligible"].any()
    assert (bad["restored_lost_sales"] == 0).all()
    assert (bad["v_star"] == bad["observed_lower_bound_velocity"]).all()
    assert (
        bad["restoration_status"]
        == "observed_lower_bound_stock_grain_unreliable"
    ).all()


def test_nonconverged_huber_scopes_use_explicit_fallback() -> None:
    config = replace(
        AppConfig.load(PROJECT_ROOT).restoration,
        huber_max_iter=1,
        huber_convergence_tolerance=1e-12,
    )
    fit_rows = pd.DataFrame(
        [
            {
                "sku_code": f"SKU{index % 3}",
                "store_code": f"S{index % 5}",
                "cate_level4_code": "TOY",
                "v_observed_raw": float((index % 7) + 1),
                "mean_price_log_ratio": float((index % 4) - 2) / 10.0,
                "fit_weight": 1.0 + index / 100.0,
            }
            for index in range(40)
        ]
    )

    fitted = fit_restoration_model(fit_rows, config=config)

    assert fitted.report["status"] == "warning"
    assert "__GLOBAL__" in fitted.report["nonconverged_scopes"]
    assert "TOY" in fitted.report["nonconverged_scopes"]
    assert fitted.report["fallback_scopes"]["TOY"] == "__GLOBAL__"
    assert "TOY" not in fitted.models
    assert fitted.report["category_models"]["TOY"]["used_for_prediction"] is False


def test_quality_report_treats_nullable_stock_state_as_unknown_not_stockout() -> None:
    daily = pd.DataFrame(
        {
            "sku_code": ["SKU1"],
            "out_of_stock_flag_x": [pd.NA],
            "inventory_stockout_flag": pd.Series([pd.NA], dtype="Float64"),
            "inventory_stockout_flag_unrecognized": [False],
            "out_of_stock_flag_y": [pd.NA],
            "placement_stockout_flag": pd.Series([pd.NA], dtype="Float64"),
            "placement_stockout_flag_unrecognized": [False],
            "in_stock_not_displayed_flag": [pd.NA],
            "nondisplay_flag": pd.Series([pd.NA], dtype="Float64"),
            "nondisplay_flag_unrecognized": [False],
            "e_open": pd.Series([1.0], dtype="Float64"),
            "e_stock": pd.Series([pd.NA], dtype="Float64"),
            "e_display": pd.Series([pd.NA], dtype="Float64"),
            "e_total": pd.Series([pd.NA], dtype="Float64"),
            "clean_sale_qty": [0.0],
            "stock_state_conflict": [False],
        }
    )
    weekly = pd.DataFrame(
        {
            "sku_code": ["SKU1"],
            "model_fit_eligible": [False],
            "v_model": [1.0],
            "restored_lost_sales": [0.0],
        }
    )

    report = build_restoration_quality_report(
        daily,
        weekly,
        requested_fit_skus={"SKU1"},
        model_report={"fit_sku_count": 0},
    )

    assert report["exposure"]["unknown_exposure_days"] == 1
    assert report["exposure"]["stockout_days"] == 0

    compact_report = build_restoration_quality_report_from_summary(
        summarize_daily_quality_inputs(daily),
        weekly,
        requested_fit_skus={"SKU1"},
        model_report={"fit_sku_count": 0},
    )
    assert compact_report["flag_normalization"] == report["flag_normalization"]
    assert compact_report["exposure"] == report["exposure"]
    assert compact_report["model_gate"] == report["model_gate"]
    assert compact_report["signed_sales"] == report["signed_sales"]


def test_m2_restores_stockout_and_nondisplay_with_auditable_shrinkage(
    tmp_path: Path,
) -> None:
    config = replace(
        AppConfig.load(PROJECT_ROOT).restoration,
        minimum_price_observations=100,
    )
    raw_sales = pd.DataFrame(
        [_sales_row(f"N{index:02d}", 1) for index in range(25)]
        + [_sales_row("ABNORMAL", 100)]
    )
    days = [date(2025, 1, 1) + timedelta(days=index) for index in range(7)]
    inventory = pd.DataFrame(
        [
            {
                "ds": day.isoformat(),
                "stock_date": day.isoformat(),
                "store_code": "S1",
                "sku_code": "SKU1",
                "available_stock_qty": 0 if index == 1 else 5,
                "out_of_stock_flag": 1 if index == 1 else 0,
            }
            for index, day in enumerate(days)
        ]
    )
    placement = pd.DataFrame(
        [
            {
                "ds": day.isoformat(),
                "sku_code": "SKU1",
                "store_code": "S1",
                "first_store_arrival_time": "2025-01-01",
                "stock_available_itm_cnt": 0 if index == 1 else 1,
                "out_of_stock_flag": 1 if index == 1 else 0,
                "in_stock_not_displayed_flag": 1 if index == 2 else 0,
            }
            for index, day in enumerate(days)
        ]
    )
    stores = pd.DataFrame(
        [
            {
                "store_code": "S1",
                "store_business_state": "NORMAL",
                "store_level": "A",
                "open_date": "2020-01-01",
                "close_date": None,
                "sts_state": "1",
                "city_level": "T1",
                "city_name": "Shanghai",
                "secbiz_area_name": "CBD",
                "area_class1_name": "MALL",
                "operate_size": 120.0,
                "meter_sum": 80.0,
            }
        ]
    )
    products = pd.DataFrame(
        [{"sku_code": "SKU1", "cate_level4_code": "TOY", "retail_price": 10.0}]
    )

    result = build_restoration_result(
        raw_sales=raw_sales,
        inventory=inventory,
        placement=placement,
        stores=stores,
        product_attributes=products,
        first_sale_dates={"SKU1": date(2025, 1, 1)},
        fit_skus={"SKU1"},
        training_weights={"SKU1": 1.0},
        cutoff_time=CUTOFF,
        window_days=7,
        config=config,
    )

    week = result.weekly.iloc[0]
    assert result.stats.abnormal_order_lines == 1
    # V3.2 retains real P99.5 bursts in the business total and robustly
    # downweights them for model fitting instead of deleting them.
    assert week["clean_sales_qty"] == 125
    assert week["abnormal_sales_qty"] == 100
    assert week["observed_exposure_days"] == 5
    assert week["known_censored_days"] == 2
    assert week["stockout_days"] == 1
    assert week["nondisplay_days"] == 1
    assert week["restored_lost_sales"] > 0
    assert week["restored_natural_sales"] > 25
    assert 4.0 < week["n_eff"] < 5.0
    assert week["reliability_r"] == pytest.approx(
        week["n_eff"] / (week["n_eff"] + config.shrinkage_k_exposure_days)
    )
    assert week["v_star"] >= 0
    assert week["period_factor_h"] == 1
    assert result.tier_weekly.iloc[0]["store_tier"] == "NORMAL|A"
    global_model = result.model_report["category_models"]["__GLOBAL__"]
    assert set(global_model["numeric_covariates"]) == {
        "operate_size",
        "meter_sum",
        "lifecycle_week",
    }
    assert set(global_model["categorical_covariates"]) == {
        "store_tier",
        "city_level",
        "city_name",
        "secbiz_area_name",
        "area_class1_name",
    }
    paths = write_restoration_artifacts(
        output_dir=tmp_path / "round4", result=result, report={"status": "ok"}
    )
    assert paths["report"].is_file()
    assert paths["daily"].is_file()
    assert paths["weekly"].is_file()
    assert paths["tier_weekly"].is_file()


def test_period_neutralization_uses_supplied_event_and_adjacent_weeks() -> None:
    weekly = pd.DataFrame(
        [
            {
                "sku_code": "SKU1",
                "cate_level4_code": "TOY",
                "store_code": "S1",
                "lifecycle_week": week,
                "week_start": date(2025, 1, 1) + timedelta(days=(week - 1) * 7),
                "week_end": date(2025, 1, 7) + timedelta(days=(week - 1) * 7),
                "v_star": value,
                "n_eff": 7.0,
            }
            for week, value in [(1, 10.0), (2, 20.0), (3, 10.0)]
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_code": "E1",
                "start_date": "2025-01-08",
                "end_date": "2025-01-14",
                "cate_level4_code": "TOY",
            }
        ]
    )

    assigned = assign_event_codes(weekly, events)
    neutral = neutralize_period_effects(assigned)

    event_week = neutral.loc[neutral["lifecycle_week"] == 2].iloc[0]
    assert event_week["event_code"] == "E1"
    assert event_week["period_factor_h"] == pytest.approx(2.0)
    assert event_week["v_neutral"] == pytest.approx(10.0)
    assert event_week["period_factor_status"] == "estimated"


def test_missing_first_store_arrival_uses_earliest_observed_business_day(
    tmp_path: Path,
) -> None:
    config = replace(
        AppConfig.load(PROJECT_ROOT).restoration,
        minimum_price_observations=100,
    )
    inventory = pd.DataFrame(
        [
            {
                "ds": "2025-01-01",
                "stock_date": "2025-01-01",
                "store_code": "S1",
                "sku_code": "SKU1",
                "available_stock_qty": 5,
                "out_of_stock_flag": 0,
            }
        ]
    )
    placement = pd.DataFrame(
        [
            {
                "ds": "2025-01-01",
                "sku_code": "SKU1",
                "store_code": "S1",
                "first_store_arrival_time": None,
                "stock_available_itm_cnt": 1,
                "out_of_stock_flag": 0,
                "in_stock_not_displayed_flag": 0,
            }
        ]
    )
    result = build_restoration_result(
        raw_sales=pd.DataFrame([_sales_row("A", 1)]),
        inventory=inventory,
        placement=placement,
        stores=pd.DataFrame(
            [
                {
                    "store_code": "S1",
                    "store_business_state": "NORMAL",
                    "store_level": "A",
                    "open_date": "2020-01-01",
                    "close_date": None,
                    "sts_state": "1",
                }
            ]
        ),
        product_attributes=pd.DataFrame(
            [
                {
                    "sku_code": "SKU1",
                    "cate_level4_code": "TOY",
                    "retail_price": 10.0,
                }
            ]
        ),
        first_sale_dates={"SKU1": date(2025, 1, 1)},
        fit_skus={"SKU1"},
        training_weights={"SKU1": 1.0},
        cutoff_time=CUTOFF,
        window_days=7,
        config=config,
    )

    assert result.daily["arrival_inferred"].all()
    assert result.daily.iloc[0]["arrival_date"] == pd.Timestamp("2025-01-01")


def _store_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "store_code": "S1",
                "store_business_state": "NORMAL",
                "store_level": "A",
                "open_date": "2020-01-01",
                "close_date": None,
                "sts_state": "1",
            }
        ]
    )


def test_source_display_aliases_preserve_strict_three_state_exposure() -> None:
    days = [date(2025, 1, 1) + timedelta(days=index) for index in range(3)]
    inventory = pd.DataFrame(
        [
            {
                "ds": day.isoformat(),
                "stock_date": day.isoformat(),
                "store_code": "S1",
                "sku_code": "SKU1",
                "available_stock_qty": 5,
                "out_of_stock_flag": 0,
            }
            for day in days
        ]
    )
    placement_values = ["有库存未陈列商品", "--", "新出现的来源值"]
    placement = pd.DataFrame(
        [
            {
                "ds": day.isoformat(),
                "sku_code": "SKU1",
                "store_code": "S1",
                "first_store_arrival_time": "2025-01-01",
                "stock_available_itm_cnt": 5,
                "out_of_stock_flag": 0,
                "in_stock_not_displayed_flag": value,
            }
            for day, value in zip(days, placement_values)
        ]
    )

    panel = build_daily_panel(
        sales_daily=pd.DataFrame(
            columns=["sku_code", "store_code", "business_day"]
        ),
        inventory=inventory,
        placement=placement,
        stores=_store_snapshot(),
        first_sale_dates={"SKU1": date(2025, 1, 1)},
        cutoff_date=date(2025, 1, 3),
        window_days=3,
        promotion_price_ratio_threshold=0.98,
    )

    assert panel["e_display"].tolist()[:2] == [0.0, 1.0]
    assert pd.isna(panel.iloc[2]["e_display"])
    assert panel["e_total"].tolist()[:2] == [0.0, 1.0]
    assert pd.isna(panel.iloc[2]["e_total"])
    assert panel["nondisplay_flag_unrecognized"].tolist() == [False, False, True]


def test_contradictory_stock_sources_remain_unknown() -> None:
    panel = build_daily_panel(
        sales_daily=pd.DataFrame(
            columns=["sku_code", "store_code", "business_day"]
        ),
        inventory=pd.DataFrame(
            [
                {
                    "ds": "2025-01-01",
                    "stock_date": "2025-01-01",
                    "store_code": "S1",
                    "sku_code": "SKU1",
                    "available_stock_qty": 5,
                    "out_of_stock_flag": 0,
                }
            ]
        ),
        placement=pd.DataFrame(
            [
                {
                    "ds": "2025-01-01",
                    "sku_code": "SKU1",
                    "store_code": "S1",
                    "first_store_arrival_time": "2025-01-01",
                    "stock_available_itm_cnt": 0,
                    "out_of_stock_flag": 1,
                    "in_stock_not_displayed_flag": "--",
                }
            ]
        ),
        stores=_store_snapshot(),
        first_sale_dates={"SKU1": date(2025, 1, 1)},
        cutoff_date=date(2025, 1, 1),
        window_days=1,
        promotion_price_ratio_threshold=0.98,
    )

    row = panel.iloc[0]
    assert row["inventory_stock_state"] == 1
    assert row["placement_stock_state"] == 0
    assert bool(row["stock_state_conflict"])
    assert pd.isna(row["e_stock"])
    assert pd.isna(row["e_total"])
