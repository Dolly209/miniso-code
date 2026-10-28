from datetime import date
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from config.settings import AppConfig
from data.restoration_audit import annotate_lifecycle_clock
from first_order import calculate_first_order
from placement.engine import _normalize_curve_profile
from placement.lifecycle import all_store_lifecycle, mix_comparable_curves
from restore.historical import _aggregate_weekly, build_daily_panel, apply_restoration_model, FittedRestorationModel
from restore.step4_audit import exposure_audit, lifecycle_contributions

ROOT = Path(__file__).resolve().parents[1]
PARAMS = json.loads((ROOT / "config/algorithm_v3_2.json").read_text())


def panel_for(days=(3, 5), *, flag=0, stocked=True, arrival=3, close=None, sales_available_dates=None):
    sales = pd.DataFrame([dict(sku_code="X", store_code="S", business_day=date(2025, 1, d),
                               clean_sale_qty=2., reported_sale_qty=2., cate_level4_code="C") for d in days])
    inv = pd.DataFrame([dict(ds=f"2025-01-{d:02}", stock_date=f"2025-01-{d:02}", sku_code="X", store_code="S",
                            available_stock_qty=5. if stocked else np.nan, out_of_stock_flag=0 if stocked else None) for d in days])
    placement = pd.DataFrame([dict(ds=f"2025-01-{d:02}", sku_code="X", store_code="S", first_store_arrival_time=f"2025-01-{arrival:02}",
                                  stock_available_itm_cnt=5. if stocked else np.nan, out_of_stock_flag=0 if stocked else None,
                                  in_stock_not_displayed_flag=flag) for d in days])
    stores = pd.DataFrame([dict(store_code="S", store_business_state="1", store_level="A", sts_state="0" if close else "1",
                                open_date="2020-01-01", close_date=close)])
    return build_daily_panel(sales_daily=sales, inventory=inv, placement=placement, stores=stores,
                             first_sale_dates={"X": date(2025, 1, 1)}, cutoff_date=date(2025, 1, 21),
                             window_days=21, promotion_price_ratio_threshold=.9, sales_available_dates=sales_available_dates)


def test_positive_sales_never_impute_missing_stock_or_display():
    panel = panel_for(flag=None, stocked=False)
    assert panel.e_stock.isna().all()
    assert panel.e_display.isna().all()
    assert panel.e_total.isna().all()
    weekly = _aggregate_weekly(panel, AppConfig.load(ROOT).restoration)
    assert weekly.observed_exposure_days.sum() == 0
    assert weekly.known_censored_days.sum() == 0
    assert weekly.clean_sales_qty.sum() == 4
    assert weekly.unknown_exposure_days.sum() == 19  # Jan3..21, not just two source rows


def test_weekly_calendar_tracks_missing_whole_weeks_without_resetting_lifecycle():
    panel = panel_for()
    weekly = _aggregate_weekly(panel, AppConfig.load(ROOT).restoration).set_index("lifecycle_week")
    assert weekly.index.tolist() == [1, 2, 3]
    assert weekly.loc[1, "observed_exposure_days"] == 2
    assert weekly.loc[1, "unknown_exposure_days"] == 3
    assert weekly.loc[2, "unknown_exposure_days"] == 7
    assert weekly.loc[2, "source_day_rows"] == 0
    assert weekly.loc[2, "calendar_row_status"] == "missing_whole_week_not_zero_sales"
    assert weekly.loc[1, "sellable_proxy_date"] == pd.Timestamp("2025-01-03")
    assert panel.lifecycle_week.eq(1).all()
    assert len(panel) == 2  # no SKU/store/day Cartesian materialization


def test_historical_closed_snapshot_does_not_close_days_before_close_date():
    panel = panel_for(close="2025-01-10")
    assert panel.e_open.eq(1).all()
    weekly = _aggregate_weekly(panel, AppConfig.load(ROOT).restoration)
    assert weekly.missing_unknown_days.sum() == 6  # Jan3..10, less two observed days


def test_unknown_only_week_is_not_a_model_lifecycle_observation():
    panel = panel_for()
    config = AppConfig.load(ROOT).restoration
    weekly = _aggregate_weekly(panel, config)
    fitted = FittedRestorationModel(models={"__GLOBAL__": ({"intercept": np.log1p(7), "price_elasticity": 0.}, {}, {})},
                                    category_p99={}, global_p99=100., report={})
    result = apply_restoration_model(weekly, fitted=fitted, config=config)
    assert result.loc[result.source_day_rows.eq(0), "v_star"].isna().all()
    ledger, summary = exposure_audit(result)
    assert summary.unknown_only_velocity_rows.eq(0).all()
    assert ledger.denominator_error.eq(0).all()
    assert ledger.velocity_formula_error.dropna().lt(1e-10).all()


def test_missing_sales_partition_does_not_become_a_normal_zero_sales_day():
    panel = panel_for(sales_available_dates={date(2025, 1, 3)})
    assert panel.loc[panel.business_day.eq(date(2025, 1, 5)), "e_total"].isna().all()
    weekly = _aggregate_weekly(panel, AppConfig.load(ROOT).restoration)
    assert weekly.observed_exposure_days.sum() == 1
    assert weekly.missing_sales_partition_days.sum() == 1


def test_missing_week_end_inventory_is_not_carried_from_an_earlier_day():
    weekly = _aggregate_weekly(panel_for(), AppConfig.load(ROOT).restoration)
    assert weekly.ending_available_stock_qty.isna().all()
    assert weekly.ending_inventory_status.eq("last_calendar_day_missing_not_carried_forward").all()


def test_signed_sales_before_recorded_arrival_are_preserved_but_not_normal_exposure():
    panel = panel_for(arrival=8)
    assert len(panel) == 2
    assert panel.clean_sale_qty.sum() == 4.
    assert panel.e_total.isna().all()
    weekly = _aggregate_weekly(panel, AppConfig.load(ROOT).restoration)
    assert weekly.clean_sales_qty.sum() == 4.
    assert weekly.arrival_conflict_days.sum() == 2
    assert weekly.observed_exposure_days.sum() == 0


def test_driver_dry_run_plans_bounded_stages_without_creating_outputs(tmp_path, monkeypatch, capsys):
    from scripts.verify_restoration_lifecycle import main
    from data.artifacts import StageArtifactStore as Store
    baseline = tmp_path / "baseline"
    group = baseline / "checkpoint_groups/g1"
    Store.write_frame(group / "target_schedule.csv", pd.DataFrame(dict(sku_code=["T"])))
    model = tmp_path / "model.json"
    model.write_text("{}")
    Store.write_json(group / "replay_manifest.json", dict(decision_cutoff_time="2025-01-31T23:59:59+08:00",
        product_snapshot_time="2026-09-30T23:59:59+08:00", restoration_model_file=str(model)))
    Store.write_frame(group / "round5_preselection/selected_candidate_cohort.csv", pd.DataFrame(dict(sku_code=["A", "B", "C"])))
    for path in ("round5/candidate_detail.csv", "round5/comparable_summary.csv", "round7/demand_adjustments.csv", "round8/first_order_decisions.csv"):
        Store.write_frame(group / path, pd.DataFrame(dict(sku_code=["T"])))
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", ["verify_restoration_lifecycle.py", "--data-root", "/absent-server-data",
        "--baseline-replay-dir", str(baseline), "--output-dir", str(output), "--sku-code", "T",
        "--decision-cutoff-time", "2026-09-30T23:59:59+08:00", "--dry-run"])
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("dry-run must not start a subprocess"))
    assert main() == 0
    assert not output.exists()
    text = capsys.readouterr().out
    assert "--sku-code A --sku-code B --sku-code C" in text
    assert "2025-01-31T23:59:59+08:00" in text
    assert "--skip-daily-exposure-audit" in text


def test_lifecycle_normalizes_each_comparable_then_averages_not_the_reverse():
    rows = [dict(sku_code=sku, store_tier=tier, lifecycle_week=w, v_neutral=v,
                 restored_exposure_days=7.)
            for sku, speeds in (("A", [1., 4., 3., 2.]), ("B", [100., 100., 100., 100.]))
            for w, v in enumerate(speeds, 1) for tier in ("1|A", "1|B")]
    all_store = all_store_lifecycle(pd.DataFrame(rows))
    factors, components, _, _ = mix_comparable_curves(pd.DataFrame(dict(candidate_sku_code=["A", "B"], alpha_weight=[.5, .5])),
        all_store, _normalize_curve_profile, lifecycle_weeks=5, min_week1_velocity=.001, max_factor=10., min_observed_weeks=4)
    assert factors[1] == 1.
    assert factors[2] == 2.5
    assert factors[2] != pytest.approx(104 / 101)
    assert components[5]["edge_carried"] == factors[5]


def test_known_left_boundary_does_not_become_fake_week_one():
    history = pd.DataFrame(dict(sku_code=["OLD", "NEW"], lifecycle_week=[1, 1], week_start=["2025-01-01", "2025-02-01"]))
    products = pd.DataFrame(dict(sku_code=["OLD", "NEW"], first_sale_date=["2024-09-21", "2025-01-20"]))
    audited = annotate_lifecycle_clock(history, products, source_boundary=date(2025, 1, 1))
    assert audited.usable_for_launch_curve.tolist() == [False, True]
    assert audited.lifecycle_origin_date.dt.date.tolist() == [date(2025, 1, 1), date(2025, 2, 1)]
    assert audited.loc[0, "lifecycle_clock_status"] == "known_pre_source_sale_left_truncated_not_week1"


def first_order_args():
    return dict(targets=pd.DataFrame(dict(sku_code=["T"])),
        placement_plan=pd.DataFrame(dict(sku_code=["T"] * 2, store_code=["S1", "S2"], store_tier=["1|A"] * 2,
                                         store_level=["A"] * 2, selected=[1, 1], v_anchor_week1=[2., 8.])),
        demand_curve=pd.DataFrame(dict(sku_code=["T"] * 2, store_tier=["1|A"] * 2, lifecycle_week=[1, 2],
                                       lifecycle_factor=[1., 1.], period_factor_h=[1., 1.], curve_source=["test"] * 2,
                                       lifecycle_component_observed=[1., 1.], lifecycle_component_interpolated=[0., 0.],
                                       lifecycle_component_edge_carried=[0., 0.], lifecycle_component_flat_default=[0., 0.])),
        adjustments=pd.DataFrame(dict(sku_code=["T"], C_demand=[1.], input_completeness=[100.])),
        comparable_summary=pd.DataFrame(dict(sku_code=["T"], path_reliability=[60.])),
        config=PARAMS["first_order"], factor_config=PARAMS["factors"],
        constraints=pd.DataFrame(dict(sku_code=["T"], decision_weeks=[1], buffer_weeks=[0], lead_weeks=[1])))


def test_default_launch_and_confirmed_staged_plan_have_exact_weekly_ledgers():
    default = calculate_first_order(**first_order_args())
    assert default.decisions.iloc[0].D_demand_base == 20.
    assert default.weekly_demand.weekly_sales_demand.sum() == 20.
    args = first_order_args()
    args["arrival_plan"] = pd.DataFrame(dict(sku_code=["T", "T"], store_code=["S1", "S2"], arrival_week=[1, 2]))
    staged = calculate_first_order(**args)
    assert staged.weekly_demand.weekly_sales_demand.tolist() == [5., 10.]
    assert staged.store_targets.sellable_demand.tolist() == [7., 8.]
    assert staged.decisions.iloc[0].D_demand_base == 15.
    sources = lifecycle_contributions(staged.weekly_demand)
    assert sources.sales_demand_qty.sum() == 15.
    assert sources.loc[sources.component.eq("flat_default"), "sales_demand_qty"].sum() == 0.


def test_partial_or_invalid_confirmed_plan_does_not_silently_default_to_week1():
    with pytest.raises(ValueError, match="every selected store"):
        calculate_first_order(**first_order_args(), arrival_plan=pd.DataFrame(dict(sku_code=["T"], store_code=["S1"], arrival_week=[1])))
    with pytest.raises(ValueError, match="integer"):
        calculate_first_order(**first_order_args(), arrival_plan=pd.DataFrame(dict(sku_code=["T"], store_code=["S1"], arrival_week=[1.5])))
