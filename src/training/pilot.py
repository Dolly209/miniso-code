"""Fixed, time-spread development targets; never split or select by Test error."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from data.domain import require_columns, normalize_sku_column
from data.domestic import DataContractViolation
from data.launch_contract import require_verified_training_schedule
from training.split import SkuSample, build_time_blocks
from training.weights import long_tail_weights


@dataclass(frozen=True)
class FixedPilot:
    schedule: pd.DataFrame
    skus: pd.DataFrame
    time_blocks: pd.DataFrame
    report: dict[str, Any]


def evaluation_scope_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    versions = sorted(frame["pilot_version"].dropna().astype(str).unique()) if "pilot_version" in frame else []
    health = sorted(frame["label_source_model_health"].dropna().astype(str).unique()) if "label_source_model_health" in frame else ["not_verified"]
    return {
        "pilot_versions": versions,
        "evaluation_scope": "development_pilot_not_final_acceptance" if versions else "full_fixed_schedule",
        "label_source_model_health": health,
        "label_model_validation_complete": health == ["converged"],
        "final_acceptance_claim_allowed": not versions and health == ["converged"],
    }


def freeze_development_pilot(
    schedule: pd.DataFrame, *, train_skus: int = 20, test_skus: int = 5,
    n_time_blocks: int = 5, checkpoint: str = "M3",
) -> FixedPilot:
    """Preserve final source split; choose quantile positions by time/SKU only.

    Labels are validated, but neither their magnitudes nor errors determine
    selection. Dates are never reassigned to the other side to reach a quota.
    The pilot remains a development subset, not a new independent final Test.
    """
    require_columns(schedule, ("sku_code", "split", "first_actual_sale_date", "dataset_version",
                              "prediction_checkpoint", "prediction_checkpoint_key", "decision_cutoff_time",
                              "sales_26w_qty", "D_first_ref", "W_cover", "label_quality_status",
                              "training_ready_for_feature_replay", "temporal_split_boundary_date"), "label-qualified pilot schedule")
    require_verified_training_schedule(schedule)
    frame = normalize_sku_column(schedule)
    if frame.empty or train_skus <= 0 or test_skus <= 0:
        raise ValueError("pilot requires positive Train/Test counts and a nonempty schedule")
    if frame["dataset_version"].nunique(dropna=False) != 1:
        raise DataContractViolation("pilot requires one source dataset_version")
    ready = frame["training_ready_for_feature_replay"].astype(str).str.lower().isin(("true", "1"))
    if not ready.all() or not frame["label_quality_status"].astype(str).str.startswith("eligible_").all():
        raise DataContractViolation("pilot input must contain only label-qualified rows; no automatic label-quality relaxation")
    if frame["prediction_checkpoint_key"].duplicated().any():
        raise DataContractViolation("duplicate pilot checkpoint key")
    identity_columns = ["split", "first_actual_sale_date", "dataset_version", "sales_26w_qty", "D_first_ref", "W_cover"]
    if frame.groupby("sku_code")[identity_columns].nunique(dropna=False).gt(1).any().any():
        raise DataContractViolation("conflicting pilot SKU identity/label across checkpoints")
    identities = frame.loc[frame["prediction_checkpoint"].eq(checkpoint)].copy()
    if set(identities["sku_code"]) != set(frame["sku_code"]):
        raise DataContractViolation(f"every pilot candidate must have {checkpoint}")
    dates = pd.to_datetime(identities["first_actual_sale_date"], errors="raise").dt.normalize()
    boundaries = pd.to_datetime(frame["temporal_split_boundary_date"], errors="coerce").dt.normalize()
    if boundaries.isna().any() or boundaries.nunique() != 1:
        raise DataContractViolation("pilot requires one existing Train/Test boundary")
    expected = np.where(dates.le(boundaries.iloc[0]), "Train", "Test")
    if not identities["split"].eq(expected).all():
        raise DataContractViolation("source Train/Test violates the fixed chronological boundary")
    for column in ("D_first_ref", "sales_26w_qty", "W_cover"):
        values = pd.to_numeric(identities[column], errors="coerce")
        if not np.isfinite(values).all() or values.lt(0).any() or (column == "W_cover" and values.eq(0).any()):
            raise DataContractViolation(f"invalid pilot {column}; missing is not zero")
    identities = identities.assign(__first_sale=dates).sort_values(["__first_sale", "sku_code"], kind="stable")
    selected: list[str] = []
    for split, count in (("Train", train_skus), ("Test", test_skus)):
        pool = identities.loc[identities["split"].eq(split)]
        if len(pool) < count:
            raise DataContractViolation(f"requested {count} {split} targets, only {len(pool)} label-qualified SKUs exist")
        positions = [len(pool) // 2] if count == 1 else [i * (len(pool) - 1) // (count - 1) for i in range(count)]
        selected.extend(pool.iloc[positions]["sku_code"].astype(str))
    selected_set = set(selected)
    pilot = frame.loc[frame["sku_code"].isin(selected_set)].copy()
    pilot = pilot.drop(columns=[column for column in pilot if column.startswith("__")])
    pilot = pilot.sort_values(["first_actual_sale_date", "sku_code", "prediction_checkpoint_key"], kind="stable").reset_index(drop=True)
    train = identities.loc[identities["sku_code"].isin(selected_set) & identities["split"].eq("Train")]
    blocks = build_time_blocks(
        [SkuSample(pd.Timestamp(day).date(), str(sku)) for sku, day in zip(train["sku_code"], train["__first_sale"])],
        n_time_blocks=n_time_blocks,
    )
    block_rows = [{"sku_code": row.sku_code, "first_actual_sale_date": row.first_actual_sale_date.isoformat(), "time_block": i}
                  for i, block in enumerate(blocks, start=1) for row in block]
    train_sales = dict(zip(train["sku_code"].astype(str), train["sales_26w_qty"].astype(float)))
    weights = {row.sku_code: row.weight for row in long_tail_weights(train_sales)}
    if "long_tail_weight" in pilot:
        pilot["source_full_cohort_long_tail_weight"] = pilot["long_tail_weight"]
    pilot["long_tail_weight"] = pilot["sku_code"].map(weights)
    digest = hashlib.sha256(pilot.reindex(sorted(pilot.columns), axis=1).to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    version = f"development-m3-{digest[:20]}"
    pilot["pilot_version"] = version
    pilot["evaluation_scope"] = "development_pilot_not_final_acceptance"
    skus = pilot.loc[pilot["prediction_checkpoint"].eq(checkpoint)].copy()
    report = {
        "status": "fixed_development_pilot", "pilot_version": version, "content_sha256": digest,
        "source_dataset_version": str(frame["dataset_version"].iloc[0]),
        "prediction_checkpoint": checkpoint, "train_skus": train_skus, "test_skus": test_skus,
        "source_train_skus": int(identities["split"].eq("Train").sum()),
        "source_test_skus": int(identities["split"].eq("Test").sum()),
        "temporal_split_boundary_date": boundaries.iloc[0].date().isoformat(),
        "selection_policy": "fixed deterministic evenly spaced timeline positions; label magnitude, Test error and forecast error never used",
        "source_split_preserved": True, "training_started": False, "test_consumed": False,
        "time_blocks": n_time_blocks, "rolling_folds": n_time_blocks - 1,
        "train_weight_policy": "recomputed SKU/sales-band mixed weights using only selected Train; Test has no fitting weight",
        "label_source_model_health": sorted(pilot["label_source_model_health"].astype(str).unique()) if "label_source_model_health" in pilot else ["not_verified"],
        "evaluation_scope": "development_pilot_not_final_acceptance",
        "known_limit": "this pilot is for diagnosis; repeated inspection of these Test predictions cannot support untouched final-Test claims",
    }
    return FixedPilot(pilot, skus, pd.DataFrame(block_rows), report)
