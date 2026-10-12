from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from data.artifacts import StageArtifactStore
from data.domain import require_columns


def assemble_modeling_dataset(
    *,
    cohort: pd.DataFrame,
    feature_frames: Sequence[tuple[str, pd.DataFrame]],
    label_frame: pd.DataFrame | None = None,
    label_column: str = "sales_26w_qty",
) -> pd.DataFrame:
    """Join one-row-per-SKU standardized artifacts without exposing raw schemas."""

    require_columns(cohort, ("sku_code", "first_actual_sale_date", "split"), "cohort")
    base_columns = ["sku_code", "first_actual_sale_date", "split"]
    for optional in ("time_block", "long_tail_weight", label_column):
        if optional in cohort and optional not in base_columns:
            base_columns.append(optional)
    result = cohort[base_columns].copy()
    labels = label_frame if label_frame is not None else cohort
    require_columns(labels, ("sku_code", label_column), "training labels")
    if label_column not in result:
        label_rows = labels[["sku_code", label_column]].copy()
        if label_rows["sku_code"].duplicated().any():
            raise ValueError("training labels are not unique by sku_code")
        result = result.merge(label_rows, on="sku_code", how="left", validate="one_to_one")
    result = result.rename(columns={label_column: "label"})
    for namespace, frame in feature_frames:
        require_columns(frame, ("sku_code",), f"feature frame {namespace}")
        if frame["sku_code"].duplicated().any():
            raise ValueError(f"feature frame {namespace} is not unique by sku_code")
        columns = [name for name in frame.columns if name != "sku_code"]
        selected = frame[["sku_code", *columns]].copy()
        selected = selected.rename(
            columns={name: f"feature__{namespace}__{name}" for name in columns}
        )
        result = result.merge(selected, on="sku_code", how="left", validate="one_to_one")
    result["label"] = pd.to_numeric(result["label"], errors="coerce")
    result["label_mature"] = result["label"].notna()
    return result.loc[result["label_mature"]].reset_index(drop=True)


def read_feature_artifacts(paths: Sequence[tuple[str, Path]]) -> list[tuple[str, pd.DataFrame]]:
    store = StageArtifactStore()
    return [(namespace, store.read_frame(path)) for namespace, path in paths]
