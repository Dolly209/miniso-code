"""Train-only paired version comparisons; absent baseline evidence stays absent."""
from __future__ import annotations
import numpy as np
import pandas as pd
from training.section84 import _metrics


def train_ablation(*, oof: pd.DataFrame, key: str, base_column: str,
                   grade_thresholds, sales_band_edges,
                   baseline: pd.DataFrame | None = None,
                   placement: pd.DataFrame | None = None) -> pd.DataFrame:
    if "split" in oof and oof.split.astype(str).str.lower().ne("train").any():
        raise ValueError("ablation must contain only Train OOF predictions")
    rows = []
    previous = None
    for name, source, column in (
        ("01_correct_quantity_baseline", baseline, base_column),
        ("02_placement_correction", placement, base_column),
        ("03_current_full_chain_before_small_factors", oof, "base_demand"),
        ("04_fitted_small_factors", oof, "prediction"),
    ):
        record = {"stage": name, "evaluation_scope": "Train_B2_B5_out_of_fold_only",
                  "test_rows_consumed": 0, "rows": len(oof)}
        if name.startswith("03_"):
            record["change_scope"] = "current restoration/lifecycle plus fitted comparable parameters; not attributed to lifecycle alone"
        if source is None:
            record.update(status="not_available", reason="paired frozen baseline version not supplied; do not invent an improvement")
            previous = None
        else:
            if column not in source or key not in source or source[key].duplicated().any():
                raise ValueError(f"invalid paired ablation source: {name}")
            compare = oof[[key, "label"]].merge(source[[key, column, "label"]], on=key,
                how="left", validate="one_to_one", suffixes=("", "__version"))
            if compare[column].isna().any() or not np.allclose(compare.label, compare.label__version):
                raise ValueError(f"ablation keys/labels differ; cannot attribute version changes: {name}")
            metrics = _metrics(compare.label.to_numpy(float), compare[column].to_numpy(float),
                np.ones(len(compare)), grade_thresholds=grade_thresholds, sales_band_edges=sales_band_edges)
            record.update(status="evaluated", **metrics)
            if previous:
                record.update(WAPE_change=metrics["WAPE"] - previous["WAPE"],
                              BandMAE_change=metrics["BandMAE"] - previous["BandMAE"])
            previous = metrics
        rows.append(record)
    return pd.DataFrame(rows)
