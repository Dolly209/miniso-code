"""Retain fixed identities/OOF blocks while auditing unusable Train forecasts."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from data.forecast_eligibility import forecast_fit_population
from training.section84 import FixedDataset, _stable_frame_hash


def prepare_fit_population(
    fixed: FixedDataset, oof: pd.DataFrame, final: pd.DataFrame
) -> tuple[FixedDataset, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = str(fixed.manifest["primary_key"])
    original_train = fixed.rows.loc[fixed.rows["split"].eq("train")]
    for role, expected, actual in (("OOF", original_train, oof), ("final", fixed.rows, final)):
        if actual[key].duplicated().any() or set(actual[key]) != set(expected[key]):
            raise ValueError(f"{role} features do not exactly cover the original fixed population")
        columns = [key, "sku_code", "split", "label"]
        checked = expected[columns].merge(actual[columns], on=key, suffixes=("_fixed", "_feature"), validate="one_to_one")
        if (checked["sku_code_fixed"].astype(str).ne(checked["sku_code_feature"].astype(str)).any()
            or checked["split_fixed"].astype(str).str.lower().ne(checked["split_feature"].astype(str).str.lower()).any()
            or not np.allclose(pd.to_numeric(checked["label_fixed"]), pd.to_numeric(checked["label_feature"]), equal_nan=True)):
            raise ValueError(f"{role} identity/label differs from the original fixed population")
    prefix = "feature__first_order__"
    oof_population = forecast_fit_population(original_train, oof, prefix=prefix)
    final_population = forecast_fit_population(fixed.rows, final, prefix=prefix)
    excluded = final_population.loc[~final_population["forecast_available"]].copy()
    excluded_keys = set(excluded[key])
    if excluded_keys != set(oof_population.loc[~oof_population["forecast_available"], key]):
        raise ValueError("OOF/final unavailable Train forecasts differ; review before changing the fit population")
    excluded = excluded.rename(columns={"fit_exclusion_reason": "exclusion_reason"}).drop(columns="forecast_available")
    excluded["exclusion_scope"] = "parameter_fit_only_original_identity_and_blocks_retained"
    usable = fixed.rows.loc[~fixed.rows[key].isin(excluded_keys)].copy()
    usable_train = usable.loc[usable["split"].eq("train")]
    folds = fixed.folds.copy()
    for index, fold in folds.iterrows():
        fit = usable_train.loc[usable_train["time_block"].isin(str(fold.fit_blocks).split("+"))]
        validation = usable_train.loc[usable_train["time_block"].eq(str(fold.validation_block))]
        if fit.empty or validation.empty:
            raise ValueError(f"unavailable forecasts leave fold {fold.fold} empty; fixed boundaries must not be silently rebuilt")
        folds.loc[index, "fit_skus"] = fit["sku_code"].nunique()
        folds.loc[index, "validation_skus"] = validation["sku_code"].nunique()
    manifest = {**fixed.manifest,
        "original_fixed_train_skus": fixed.manifest["train_skus"],
        "train_skus": int(usable_train["sku_code"].nunique()),
        "eligible_rows": len(usable),
        "parameter_fit_excluded_rows": len(excluded),
        "parameter_fit_excluded_keys": sorted(excluded_keys),
        "fit_population_hash": _stable_frame_hash(usable, [key, "sku_code", "split", "label"]),
        "fixed_blocks_policy": "original blocks retained; only audited unavailable Train rows excluded from fitting",
    }
    fit_fixed = replace(fixed, rows=usable, folds=folds, manifest=manifest)
    return (fit_fixed, oof.loc[~oof[key].isin(excluded_keys)].copy(),
            final.loc[~final[key].isin(excluded_keys)].copy(), excluded)
