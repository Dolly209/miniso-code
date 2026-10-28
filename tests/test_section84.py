from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from training.section84 import (
    DemandResidualModel,
    _metrics,
    fix_section84_dataset,
    replay_section84_locked_test,
    train_section84_first_order,
)


def test_band_mae_is_equal_average_of_quantity_mae_by_actual_sales_band() -> None:
    metrics = _metrics(
        np.asarray([100.0, 200.0, 2_000.0]),
        np.asarray([110.0, 230.0, 2_100.0]),
        np.asarray([1.0, 1.0, 1.0]),
        grade_thresholds=(15_000, 30_000, 50_000),
        sales_band_edges=(0, 500, 1_000, 9_000),
    )
    # [0,500) MAE=(10+30)/2=20; [1000,9000) MAE=100; equal-band mean=60.
    assert metrics["BandMAE"] == 60.0


def _dataset() -> pd.DataFrame:
    start = date(2025, 1, 1)
    rows = []
    for index in range(25):
        base = 100.0 + 5.0 * index
        positive = (index % 5) / 10.0
        price = (index % 3) / 10.0
        multiplier = np.exp(0.4 * positive - 0.2 * price)
        first_sale = start + timedelta(days=index)
        rows.append(
            {
                "prediction_checkpoint_key": f"S{index:02d}@M3",
                "prediction_checkpoint": "M3",
                "sku_code": f"S{index:02d}",
                "dataset_version": "dataset-v1",
                "first_actual_sale_date": first_sale.isoformat(),
                "decision_cutoff_time": (
                    first_sale - timedelta(days=90)
                ).isoformat()
                + "T23:59:59+08:00",
                "split": "Train" if index < 20 else "Test",
                "label": base * multiplier,
                "base": base,
                "positive": positive,
                "price": price,
                "point_in_time_verified": True,
            }
        )
    return pd.DataFrame(rows)


def _candidate_detail(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    components = ("emotion", "ip", "form", "category", "price", "pack", "recent")
    for target_index, target in enumerate(frame.itertuples(index=False)):
        for candidate_index, velocity in enumerate((80.0, 100.0, 120.0)):
            row = {
                "prediction_checkpoint_key": target.prediction_checkpoint_key,
                "target_sku_code": target.sku_code,
                "candidate_sku_code": f"C{target_index:02d}_{candidate_index}",
                "selected": 1,
                "alpha_weight": 1.0 / 3.0,
                "historical_velocity": velocity,
                "quality_score": 1.0,
                "path_priority": 1,
            }
            row.update(
                {
                    f"similarity_{name}": 0.9 - 0.1 * candidate_index
                    for name in components
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _train_kwargs(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "oof_candidate_detail": _candidate_detail(
            frame.loc[frame["split"].eq("Train")]
        ),
        "final_candidate_detail": _candidate_detail(frame),
        "initial_similarity_weights": {
            "emotion": 0.30,
            "ip": 0.30,
            "form": 0.12,
            "category": 0.08,
            "price": 0.08,
            "pack": 0.06,
            "recent": 0.06,
        },
        "comparable_temperature_grid": (0.10, 0.15),
        "comparable_top_k_grid": (1, 2, 3),
        "similarity_regularization_grid": (0.0, 0.1),
    }


def test_section84_freezes_blocks_and_keeps_test_out_of_oof() -> None:
    frame = _dataset()
    fixed = fix_section84_dataset(frame, n_time_blocks=5)

    assert fixed.manifest["train_skus"] == 20
    assert fixed.manifest["test_skus"] == 5
    assert fixed.time_blocks.groupby("time_block")["sku_code"].nunique().to_dict() == {
        "B1": 4,
        "B2": 4,
        "B3": 4,
        "B4": 4,
        "B5": 4,
    }
    assert fixed.folds["validation_block"].tolist() == ["B2", "B3", "B4", "B5"]

    result = train_section84_first_order(
        fixed=fixed,
        oof_feature_rows=frame.loc[frame["split"].eq("Train")],
        final_feature_rows=frame,
        **_train_kwargs(frame),
        feature_columns=("positive", "price"),
        feature_signs={"positive": "nonnegative", "price": "nonpositive"},
        base_demand_column="base",
        ridge_grid=(0.01, 0.1),
        l1_ratio_grid=(0.0,),
        half_life_grid=("infinity",),
        demand_bounds=(0.5, 2.0),
        grade_thresholds=(15000, 30000, 50000),
    )

    assert result.report["test_rows_consumed"] == 0
    assert result.report["oof_blocks"] == ["B2", "B3", "B4", "B5"]
    assert set(result.oof_predictions["time_block"]) == {"B2", "B3", "B4", "B5"}
    assert len(result.oof_predictions) == 16
    assert result.model.coefficients[0] >= 0
    assert result.model.coefficients[1] <= 0
    assert result.fold_models["test_rows_consumed"].eq(0).all()
    assert result.report["final_comparable_model_id"]


def test_section84_locked_test_uses_frozen_model() -> None:
    frame = _dataset()
    fixed = fix_section84_dataset(frame, n_time_blocks=5)
    trained = train_section84_first_order(
        fixed=fixed,
        oof_feature_rows=frame.loc[frame["split"].eq("Train")],
        final_feature_rows=frame,
        **_train_kwargs(frame),
        feature_columns=("positive", "price"),
        feature_signs={"positive": "nonnegative", "price": "nonpositive"},
        base_demand_column="base",
        ridge_grid=(0.01,),
        l1_ratio_grid=(0.0,),
        half_life_grid=("infinity",),
        demand_bounds=(0.5, 2.0),
        grade_thresholds=(15000, 30000, 50000),
    )
    restored = DemandResidualModel.from_dict(trained.model.as_dict())
    predictions, report = replay_section84_locked_test(
        frame,
        restored,
        candidate_detail=_candidate_detail(frame),
        comparable_model=trained.comparable_model,
        interval_log_quantiles=trained.report["oof_log_residual_quantiles"],
        grade_thresholds=(15000, 30000, 50000),
    )

    assert len(predictions) == 5
    assert predictions["split"].eq("Test").all()
    assert report["model_refit_during_test"] is False
    assert report["model_selection_reopened"] is False


def test_section84_records_invalid_rows_in_exclusion_artifact() -> None:
    frame = _dataset()
    bad = frame.iloc[[0]].copy()
    bad["prediction_checkpoint_key"] = "BAD@M3"
    bad["sku_code"] = "BAD"
    bad["label"] = np.nan
    fixed = fix_section84_dataset(pd.concat([frame, bad], ignore_index=True))

    assert len(fixed.exclusions) == 1
    assert fixed.exclusions.iloc[0]["exclusion_reason"] == "immature_or_missing_label"
