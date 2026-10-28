from __future__ import annotations

from datetime import date, timedelta

from training.split import (
    SkuSample,
    build_time_blocks,
    chronological_ratio_split,
    expanding_window_folds,
    fixed_train_test_split,
)
from data.modeling_dataset import assemble_modeling_dataset, assess_modeling_readiness
import pandas as pd


def synthetic_samples() -> list[SkuSample]:
    start = date(2025, 1, 1)
    return [
        SkuSample(start + timedelta(days=index // 2), f"SKU{index:03d}")
        for index in range(20)
    ]


def test_fixed_split_keeps_same_day_together() -> None:
    cutoff = date(2025, 1, 5)
    train, test = fixed_train_test_split(synthetic_samples(), cutoff=cutoff)
    assert all(sample.first_actual_sale_date <= cutoff for sample in train)
    assert all(sample.first_actual_sale_date > cutoff for sample in test)
    assert {sample.first_actual_sale_date for sample in train}.isdisjoint(
        {sample.first_actual_sale_date for sample in test}
    )


def test_ratio_split_selects_closest_date_boundary() -> None:
    samples = synthetic_samples()
    train, test, cutoff = chronological_ratio_split(samples, train_ratio=0.8)
    assert cutoff == date(2025, 1, 8)
    assert len(train) == 16
    assert len(test) == 4
    assert all(sample.first_actual_sale_date <= cutoff for sample in train)
    assert all(sample.first_actual_sale_date > cutoff for sample in test)


def test_ratio_split_never_splits_a_large_same_day_group() -> None:
    samples = [
        SkuSample(date(2025, 1, 1), "EARLY"),
        *(
            SkuSample(date(2025, 1, 2), f"GROUP{index:02d}")
            for index in range(8)
        ),
        SkuSample(date(2025, 1, 3), "LATE"),
    ]
    train, test, cutoff = chronological_ratio_split(samples, train_ratio=0.8)
    assert cutoff == date(2025, 1, 2)
    assert len(train) == 9
    assert len(test) == 1
    assert {sample.first_actual_sale_date for sample in train}.isdisjoint(
        {sample.first_actual_sale_date for sample in test}
    )


def test_expanding_fold_count_is_not_hard_coded() -> None:
    blocks = build_time_blocks(synthetic_samples(), n_time_blocks=5)
    folds = expanding_window_folds(blocks)
    assert len(blocks) == 5
    assert len(folds) == 4
    for block in blocks:
        dates = {sample.first_actual_sale_date for sample in block}
        for other in blocks:
            if other is block:
                continue
            assert dates.isdisjoint(
                {sample.first_actual_sale_date for sample in other}
            )


def test_modeling_readiness_blocks_all_test_and_future_features() -> None:
    cohort = pd.DataFrame(
        [
            {"sku_code": "A", "first_actual_sale_date": "2026-01-01", "split": "Test", "sales_26w_qty": 10},
            {"sku_code": "B", "first_actual_sale_date": "2026-01-02", "split": "Test", "sales_26w_qty": 20},
        ]
    )
    features = pd.DataFrame(
        [
            {"sku_code": "A", "feature_cutoff_time": "2026-09-30", "value": 1},
            {"sku_code": "B", "feature_cutoff_time": "2026-09-30", "value": 2},
        ]
    )

    report = assess_modeling_readiness(
        cohort=cohort,
        feature_frames=[("factors", features)],
        label_frame=pd.DataFrame(
            [{"sku_code": "A", "D_first_ref": 5.0}, {"sku_code": "B", "D_first_ref": 6.0}]
        ),
    )

    assert report["status"] == "blocked"
    assert report["split_sku_counts"] == {"Train": 0, "Test": 2}
    assert report["feature_artifacts"]["factors"]["future_leakage_rows"] == 2


def test_prelaunch_checkpoint_dataset_requires_checkpoint_keyed_features() -> None:
    cohort = pd.DataFrame(
        [
            {
                "sku_code": sku,
                "first_actual_sale_date": first_sale,
                "split": split,
                "sales_26w_qty": label,
                "prediction_checkpoint_key": f"{sku}@{checkpoint}",
                "prediction_checkpoint": checkpoint,
                "lead_months": int(checkpoint[1:]),
                "decision_cutoff_time": cutoff,
                "decision_date": cutoff[:10],
                "release_dt": release,
                "forecast_scope": "complete_first_order_demand",
                "checkpoint_weight_policy": (
                    "no_w_i_div_m_i; evaluate_each_prelaunch_checkpoint_separately"
                ),
            }
            for sku, first_sale, release, split, label in (
                ("A", "2025-09-01", "2025-08-15", "Train", 100.0),
                ("B", "2026-09-01", "2026-08-15", "Test", 200.0),
            )
            for checkpoint, cutoff in (
                ("M5", "2025-03-15T23:59:59+08:00")
                if sku == "A"
                else ("M5", "2026-03-15T23:59:59+08:00"),
                ("M4", "2025-04-15T23:59:59+08:00")
                if sku == "A"
                else ("M4", "2026-04-15T23:59:59+08:00"),
                ("M3", "2025-05-15T23:59:59+08:00")
                if sku == "A"
                else ("M3", "2026-05-15T23:59:59+08:00"),
            )
        ]
    )
    checkpoint_features = cohort[
        ["sku_code", "prediction_checkpoint_key", "decision_cutoff_time"]
    ].copy()
    checkpoint_features["value"] = range(len(checkpoint_features))
    labels = pd.DataFrame(
        [{"sku_code": "A", "D_first_ref": 100.0}, {"sku_code": "B", "D_first_ref": 200.0}]
    )

    report = assess_modeling_readiness(
        cohort=cohort,
        feature_frames=[("factors", checkpoint_features)],
        label_frame=labels,
    )
    assert report["status"] == "ready_for_train_oof_and_locked_test"
    assert report["independent_checkpoint_rows"] is True
    assert report["feature_artifacts"]["factors"]["covered_cohort_rows"] == 6
    assert report["feature_artifacts"]["factors"]["covered_cohort_skus"] == 2

    dataset = assemble_modeling_dataset(
        cohort=cohort,
        feature_frames=[("factors", checkpoint_features)],
        label_frame=labels,
    )
    assert len(dataset) == 6
    assert dataset["prediction_checkpoint_key"].is_unique
    assert dataset.groupby("sku_code")["label"].nunique().eq(1).all()

    sku_level = checkpoint_features.drop(
        columns=["prediction_checkpoint_key"]
    ).drop_duplicates("sku_code")
    blocked = assess_modeling_readiness(
        cohort=cohort,
        feature_frames=[("factors", sku_level)],
        label_frame=labels,
    )
    assert blocked["status"] == "blocked"
    assert any("not keyed" in reason for reason in blocked["blocking_reasons"])
