from __future__ import annotations

import json
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from comparable import select_comparables
from db.result_writer import ResultWriter
from factors import calculate_demand_adjustments
from first_order import calculate_first_order
from placement import plan_placement
from reorder import calculate_reorder
from training import replay_test, rolling_oof_train


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = json.loads((ROOT / "config" / "algorithm_v3_2.json").read_text())


def test_first_order_default_cover_matches_v3_2_section_6_1_example() -> None:
    config = PARAMETERS["first_order"]
    assert config["decision_weeks"] == 2
    assert config["buffer_weeks"] == 2
    assert config["lead_weeks"] == 6
    assert config["lifecycle_weeks"] == 26
    assert config["decision_weeks"] + config["buffer_weeks"] + config["lead_weeks"] == 10


def test_algorithm_layers_do_not_reference_physical_sources() -> None:
    for directory in (
        "restore", "comparable", "placement", "factors", "first_order", "reorder", "training"
    ):
        for path in (ROOT / "src" / directory).glob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "scm." not in text, path
            assert "data.parquet_repository" not in text, path
    for path in (ROOT / "scripts").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "data.parquet_repository" not in text, path


def _m1_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    products = pd.DataFrame(
        [
            {"sku_code": "T", "market_code": "CN", "sales_unit_type": "blind_box", "cate_level2_code": "2", "cate_level3_code": "3", "cate_level4_code": "4", "retail_price": 20, "pkg_unit": 6, "release_dt": "2026-01-01"},
            {"sku_code": "C1", "market_code": "CN", "sales_unit_type": "blind_box", "cate_level2_code": "2", "cate_level3_code": "3", "cate_level4_code": "4", "retail_price": 22, "pkg_unit": 6, "release_dt": "2025-10-01"},
            {"sku_code": "C2", "market_code": "CN", "sales_unit_type": "end_box", "cate_level2_code": "2", "cate_level3_code": "3", "cate_level4_code": "4", "retail_price": 20, "pkg_unit": 6, "release_dt": "2025-09-01"},
        ]
    )
    ips = pd.DataFrame(
        [{"sku_code": "T", "ip_code": "IP1"}, {"sku_code": "C1", "ip_code": "IP1"}, {"sku_code": "C2", "ip_code": "IP1"}]
    )
    restored = pd.DataFrame(
        [
            {"sku_code": "C1", "history_quality": 0.9, "historical_velocity": 5.0, "mature_week_count": 26},
            {"sku_code": "C2", "history_quality": 0.9, "historical_velocity": 8.0, "mature_week_count": 26},
        ]
    )
    return pd.DataFrame({"sku_code": ["T"]}), products, ips, restored


def test_m1_path_priority_hard_filter_and_weight_sum() -> None:
    targets, products, ips, restored = _m1_inputs()
    result = select_comparables(
        targets=targets,
        products=products,
        ip_membership=ips,
        restored_summary=restored,
        config=PARAMETERS["comparable"],
    )
    selected = result.candidates.loc[result.candidates["selected"] == 1]
    assert selected["candidate_sku_code"].tolist() == ["C1"]
    assert selected.iloc[0]["path_code"] == "PATH1"
    assert selected["alpha_weight"].sum() == 1.0
    c2 = result.candidates.loc[result.candidates["candidate_sku_code"] == "C2"].iloc[0]
    assert c2["hard_filter_pass"] == 0


def _placement_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = pd.DataFrame({"sku_code": ["T"], "cate_level4_code": ["4"]})
    comparable = pd.DataFrame(
        [{"target_sku_code": "T", "candidate_sku_code": "C1", "selected": 1, "alpha_weight": 1.0, "historical_velocity": 5.0}]
    )
    stores = pd.DataFrame(
        [
            {"store_code": "S1", "store_tier": "1|S", "store_level": "S", "store_business_state": "1", "eligible": 1, "operate_size": 100},
            {"store_code": "S2", "store_tier": "1|A", "store_level": "A", "store_business_state": "1", "eligible": 1, "operate_size": 80},
            {"store_code": "S3", "store_tier": "1|B", "store_level": "B", "store_business_state": "1", "eligible": 1, "operate_size": 50},
        ]
    )
    rows = []
    for store, tier, velocity in (("S1", "1|S", 8.0), ("S2", "1|A", 5.0)):
        for week in (1, 2):
            rows.append({"sku_code": "C1", "cate_level4_code": "4", "store_code": store, "store_tier": tier, "lifecycle_week": week, "v_star": velocity * (1.0 if week == 1 else 0.8), "v_neutral": velocity * (1.0 if week == 1 else 0.8), "stockout_days": 0, "source_day_rows": 7, "restored_exposure_days": 7})
    restored_store = pd.DataFrame(rows)
    restored_tier = (
        restored_store.groupby(["sku_code", "store_tier", "lifecycle_week"], as_index=False)
        .agg(v_neutral=("v_star", "mean"))
    )
    return targets, comparable, stores, restored_store, restored_tier


def test_m3_to_m7_and_reorder_formula_chain() -> None:
    targets, comparables, stores, restored_store, restored_tier = _placement_inputs()
    restored_tier = restored_tier.merge(
        pd.DataFrame({"sku_code": ["C1"], "cate_level4_code": ["4"]}),
        on="sku_code",
        how="left",
    )
    policy = pd.DataFrame(
        [
            {"store_tier": tier, "v_min": 1.0, "inv_max": 0.4, "tau_tier": 0.5}
            for tier in stores["store_tier"]
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "sku_code": "C1",
                "store_code": row.store_code,
                "store_tier": row.store_tier,
                "eligible_at_launch": 1,
                "arrived_in_launch_window": int(row.store_code != "S3"),
                "v_neutral": {"S1": 8.0, "S2": 5.0}.get(row.store_code, np.nan),
                "stock_pressure": 0.0 if row.store_code != "S3" else np.nan,
                "evidence_weight": 1.0,
                "prior_probability": 0.8 if row.store_code != "S3" else 0.2,
                "performance_probability": 0.9 if row.store_code != "S3" else 0.3,
                "reliability_r": 0.7,
            }
            for row in stores.itertuples(index=False)
        ]
    )
    features = pd.DataFrame(
        [
            {
                "target_sku_code": "T",
                "store_code": row.store_code,
                "category_fit": 0.8,
                "execution": 0.9,
                "category_fit_status": "v3_2_complete_90d",
                "execution_status": "v3_2_complete_execution_universe",
            }
            for row in stores.itertuples(index=False)
        ]
    )
    placement_config = deepcopy(PARAMETERS["placement"])
    placement_config["lifecycle_curve_weeks"] = 2
    placement = plan_placement(
        targets=targets,
        comparables=comparables,
        stores=stores,
        restored_store_weekly=restored_store,
        restored_tier_weekly=restored_tier,
        config=placement_config,
        placement_policy=policy,
        placement_evidence=evidence,
        target_store_features=features,
    )
    assert placement.store_plan["selected"].sum() >= 1
    assert placement.summary.iloc[0]["shelf_capacity_status"] == "missing_not_hard_filtered"
    assert set(placement.demand_curve["lifecycle_week"]) == {1, 2}
    assert placement.compliance["status"].eq("DECLARED_FALLBACK").sum() == 0

    features = pd.DataFrame(
        [
            {"sku_code": "T", "retail_price": 20, "ip_breadth": 2, "member_count": 100, "member_female_rate": 0.7, "member_young_rate": 0.8, "pack_size": 6, "lifecycle_persistence": 0.6, "historical_velocity": 6, "supplier_fulfillment": 0.9, "multi_supplier": 1, "exposure_quality": 0.9},
            {"sku_code": "C1", "retail_price": 22, "ip_breadth": 1, "member_count": 80, "member_female_rate": 0.6, "member_young_rate": 0.7, "pack_size": 6, "lifecycle_persistence": 0.5, "historical_velocity": 5, "supplier_fulfillment": 0.8, "multi_supplier": 0, "exposure_quality": 0.8},
        ]
    )
    comparable_summary = pd.DataFrame(
        [{"sku_code": "T", "external_direction_coefficient": 1.0, "path_reliability": 80.0}]
    )
    factors = calculate_demand_adjustments(
        targets=targets,
        features=features,
        comparables=comparables,
        comparable_summary=comparable_summary,
        config=PARAMETERS["factors"],
    )
    assert 0.5 <= factors.adjustments.iloc[0]["C_demand"] <= 2.0
    product_target = pd.DataFrame([{"sku_code": "T", "copyright_category": "自有IP"}])
    constraints = pd.DataFrame(
        [{"sku_code": "T", "moq": 10, "mold_redline": 10, "case_pack": 2, "lead_weeks": 1, "decision_weeks": 1, "buffer_weeks": 0, "strategy": "normal"}]
    )
    first = calculate_first_order(
        targets=product_target,
        placement_plan=placement.store_plan,
        demand_curve=placement.demand_curve,
        adjustments=factors.adjustments,
        comparable_summary=comparable_summary,
        config=PARAMETERS["first_order"],
        factor_config=PARAMETERS["factors"],
        constraints=constraints,
    )
    assert first.decisions.iloc[0]["Q_feas"] >= 10
    assert first.decisions.iloc[0]["Q_feas"] % 2 == 0

    cutoff = datetime(2026, 1, 7, 23, 59, tzinfo=timezone(timedelta(hours=8)))
    actual_rows = []
    for store in first.store_targets.itertuples(index=False):
        for day in range(7):
            actual_rows.append(
                {"sku_code": "T", "store_code": store.store_code, "store_tier": store.store_tier, "business_day": (cutoff.date() - timedelta(days=day)), "actual_sales": 1.0, "e_stock": 1, "e_display": 1}
            )
    reorder_constraints = pd.DataFrame(
        [{"sku_code": "T", "launch_date": "2026-01-01", "reorder_moq": 1, "case_pack": 1, "reorder_lead_days": 1, "store_inventory": 0, "warehouse_inventory": 0, "hard_end_date": "2026-02-28"}]
    )
    reorder = calculate_reorder(
        decision_cutoff_time=cutoff,
        first_order_decisions=first.decisions,
        first_order_store_targets=first.store_targets,
        actual_daily=pd.DataFrame(actual_rows),
        demand_curve=placement.demand_curve,
        config=PARAMETERS["reorder"],
        constraints=reorder_constraints,
    )
    assert reorder.decisions.iloc[0]["M_dev"] > 0
    assert reorder.decisions.iloc[0]["QR_feas"] >= 0
    assert reorder.decisions.iloc[0]["action"] in {"freeze", "observe", "reorder", "remold"}


def test_placement_runs_with_versioned_initial_parameters_by_default() -> None:
    targets, comparables, stores, restored_store, restored_tier = _placement_inputs()
    result = plan_placement(
        targets=targets,
        comparables=comparables,
        stores=stores,
        restored_store_weekly=restored_store,
        restored_tier_weekly=restored_tier,
        config=PARAMETERS["placement"],
    )
    assert result.store_plan["selected"].sum() >= 1
    assert "VERSIONED_INITIAL_PARAMETER" in set(result.compliance["status"])
    assert "EMPIRICAL_INITIAL_MODEL" in set(result.compliance["status"])


def test_placement_missing_execution_is_not_replaced_by_half_score() -> None:
    targets, comparables, stores, restored_store, restored_tier = _placement_inputs()
    config = deepcopy(PARAMETERS["placement"])
    config["lifecycle_curve_weeks"] = 2
    result = plan_placement(
        targets=targets,
        comparables=comparables,
        stores=stores,
        restored_store_weekly=restored_store,
        restored_tier_weekly=restored_tier,
        config=config,
    )
    assert result.store_plan["execution"].isna().all()
    assert result.store_plan["category_fit"].isna().all()
    statuses = set(result.compliance["status"])
    assert "VERSIONED_INITIAL_PARAMETER" in statuses
    assert "EMPIRICAL_INITIAL_MODEL" in statuses
    assert "MISSING_RENORMALIZED" in statuses


def test_path4_uses_category_then_higher_level_then_global_history() -> None:
    targets, comparables, stores, restored_store, restored_tier = _placement_inputs()
    targets = pd.DataFrame(
        {
            "sku_code": ["T"],
            "cate_level1_code": ["1"],
            "cate_level2_code": ["2"],
            "cate_level3_code": ["3"],
            "cate_level4_code": ["TARGET-L4"],
        }
    )
    comparables = pd.DataFrame(
        [
            {
                "target_sku_code": "T",
                "candidate_sku_code": "__STATIC__T",
                "selected": 1,
                "alpha_weight": 1.0,
                "historical_velocity": 1.0,
            }
        ]
    )
    restored_tier = restored_tier.assign(
        cate_level1_code="1",
        cate_level2_code="2",
        cate_level3_code="3",
        cate_level4_code="OTHER-L4",
    )
    config = deepcopy(PARAMETERS["placement"])
    config["lifecycle_curve_weeks"] = 2
    config["initial_defaults"]["path4_min_week1_samples"] = 1
    config["initial_defaults"]["category_curve_min_week1_samples"] = 1
    result = plan_placement(
        targets=targets,
        comparables=comparables,
        stores=stores,
        restored_store_weekly=restored_store,
        restored_tier_weekly=restored_tier,
        config=config,
    )
    path4 = result.compliance.loc[
        result.compliance["concept"] == "path4_static_base"
    ].iloc[0]
    assert path4["source"] == "category_l3"
    assert result.store_plan["v_anchor_source"].eq("category_l3").all()


def test_lifecycle_curve_rejects_near_zero_week1_instead_of_exploding() -> None:
    targets, comparables, stores, restored_store, restored_tier = _placement_inputs()
    restored_tier = restored_tier.copy()
    restored_tier.loc[restored_tier["lifecycle_week"].eq(1), "v_neutral"] = 1e-10
    restored_tier.loc[restored_tier["lifecycle_week"].eq(2), "v_neutral"] = 5.0
    config = deepcopy(PARAMETERS["placement"])
    config["lifecycle_curve_weeks"] = 2
    config["initial_defaults"]["category_curve_min_week1_samples"] = 1

    result = plan_placement(
        targets=targets,
        comparables=comparables,
        stores=stores,
        restored_store_weekly=restored_store,
        restored_tier_weekly=restored_tier,
        config=config,
    )

    factors = pd.to_numeric(result.demand_curve["lifecycle_factor"], errors="coerce")
    assert np.isfinite(factors).all()
    assert factors.max() <= config["initial_defaults"]["curve_max_factor"]
    assert result.demand_curve.loc[
        result.demand_curve["lifecycle_week"].eq(1), "lifecycle_factor"
    ].eq(1.0).all()
    assert result.demand_curve["curve_source"].eq("versioned_flat_default").all()


def test_path4_zero_history_falls_back_to_positive_static_anchor() -> None:
    targets, _, stores, restored_store, restored_tier = _placement_inputs()
    comparables = pd.DataFrame(
        [
            {
                "target_sku_code": "T",
                "candidate_sku_code": "__STATIC__T",
                "selected": 1,
                "alpha_weight": 1.0,
                "historical_velocity": 1.0,
            }
        ]
    )
    restored_tier = restored_tier.copy()
    restored_tier["cate_level4_code"] = "4"
    restored_tier["v_neutral"] = 0.0
    config = deepcopy(PARAMETERS["placement"])
    config["lifecycle_curve_weeks"] = 2
    config["initial_defaults"]["path4_min_week1_samples"] = 1
    config["initial_defaults"]["category_curve_min_week1_samples"] = 1

    result = plan_placement(
        targets=targets,
        comparables=comparables,
        stores=stores,
        restored_store_weekly=restored_store,
        restored_tier_weekly=restored_tier,
        config=config,
    )

    path4 = result.compliance.loc[
        result.compliance["concept"] == "path4_static_base"
    ].iloc[0]
    expected_anchor = config["initial_defaults"]["path4_anchor_velocity"]
    assert path4["source"] == "versioned_static_base"
    assert result.store_plan["v_anchor_week1"].eq(expected_anchor).all()
    assert result.demand_curve["lifecycle_factor"].eq(1.0).all()


def test_train_uses_oof_and_test_replay_is_separate() -> None:
    rows = []
    start = pd.Timestamp("2025-01-01")
    for index in range(15):
        rows.append(
            {
                "sku_code": f"S{index:02d}",
                "first_actual_sale_date": (start + pd.Timedelta(days=index)).date(),
                "split": "Train" if index < 12 else "Test",
                "label": float(10 + index * 2),
                "feature__x": float(index),
                "feature__z": float(index % 3),
            }
        )
    frame = pd.DataFrame(rows)
    result = rolling_oof_train(
        frame,
        feature_columns=["feature__x", "feature__z"],
        n_time_blocks=5,
        ridge_grid=[0.0, 0.1],
        half_life_grid=[26, "infinity"],
    )
    assert result.report["test_rows_consumed"] == 0
    assert set(result.oof_predictions["sku_code"]).isdisjoint({"S12", "S13", "S14"})
    test_predictions, report = replay_test(
        frame,
        result.model,
        interval_log_quantiles=result.report["oof_log_residual_quantiles"],
    )
    assert len(test_predictions) == 3
    assert report["test_did_not_select_model"] is True


def test_result_writer_uses_one_append_only_transaction() -> None:
    class Cursor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def execute(self, sql: str, params: tuple[object, ...]) -> None:
            self.calls.append((sql, params))

    class Database:
        def __init__(self) -> None:
            self.cursor = Cursor()
            self.transactions = 0

        @contextmanager
        def transaction(self):
            self.transactions += 1
            yield self.cursor

    database = Database()
    writer = ResultWriter(database)  # type: ignore[arg-type]
    decisions = pd.DataFrame(
        [{"sku_code": "T", "D_first": 20.0, "Q_nat": 25.0, "Q_plan": 30.0, "Q_feas": 32.0, "Q_opp": 32.0, "grade": "C", "reliability_score": 60.0}]
    )
    calculations = pd.DataFrame(
        [{"sku_code": "T", "calc_item_code": "Q_FEAS", "calc_sequence": 1, "calc_value": 32.0}]
    )
    run_ids = writer.write_first_order(
        cutoff_time=datetime(2026, 9, 30, 23, 59, tzinfo=timezone(timedelta(hours=8))),
        decisions=decisions,
        calculations=calculations,
        metadata={"parameter_version": "test"},
    )
    assert database.transactions == 1
    assert len(run_ids) == 1
    assert len(database.cursor.calls) == 3
    assert all("UPDATE" not in sql.upper() and "DELETE" not in sql.upper() for sql, _ in database.cursor.calls)
