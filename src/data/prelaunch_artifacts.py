"""Canonical audit metadata and provenance checks for offline label artifacts."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import pandas as pd

from data.domestic import DataContractViolation
from data.domain import require_columns
from data.launch_contract import validate_launch_origin_contract


AUDIT_STATE_FIELDS = ("training_ready", "temporal_split_boundary_date", "eligibility_scope")


def restoration_model_health(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Describe label-source fitting without reclassifying SKU quality or sales."""
    model = (report or {}).get("model", {})
    scopes = model.get("category_models", {})
    global_model = scopes.get("__GLOBAL__", {})
    failed = list(model.get("nonconverged_scopes", []))
    verified = bool(global_model) and all(bool(row.get("converged", False)) for row in scopes.values())
    return {
        "status": "converged" if verified else "fit_warning" if failed or scopes else "not_verified",
        "all_scopes_converged": verified,
        "global_model_converged": bool(global_model.get("converged", False)),
        "model_id": model.get("model_id"),
        "nonconverged_scopes": failed,
        "policy": "model-health evidence is separate from per-SKU label eligibility; warning labels may support a disclosed development pilot, not a claim of final model acceptance",
    }


def normalize_prelaunch_audit_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Read existing v2 outputs without forcing the expensive audit to run again.

    Only these known stage-owned fields are reconciled. Preserve earlier-stage
    values under explicit provenance names, never guess at arbitrary _x/_y fields.
    A new unsuffixed field (for example the final label split) is authoritative.
    """
    result = frame.copy()
    for key in AUDIT_STATE_FIELDS:
        left, right = key + "_x", key + "_y"
        if left not in result and right not in result:
            continue
        for source, prefix in ((left, "preflight_"), (right, "row_audit_")):
            if source in result:
                destination = prefix + key
                if destination in result and not result[destination].fillna("").astype(str).eq(result[source].fillna("").astype(str)).all():
                    raise DataContractViolation(f"conflicting audit provenance: {destination}")
                result[destination] = result[source]
        if key not in result:
            result[key] = result[right] if right in result else result[left]
        result = result.drop(columns=[name for name in (left, right) if name in result])
    return result


def merge_row_audit_schedule(schedule: pd.DataFrame, split_mapping: pd.DataFrame) -> pd.DataFrame:
    """Retain preflight lineage while giving row-audit fields canonical names."""
    base = normalize_prelaunch_audit_metadata(schedule)
    overlapping = (set(base) & set(split_mapping)) - {"sku_code"}
    for key in overlapping:
        provenance = "preflight_" + key
        if provenance not in base:
            base[provenance] = base[key]
    if "split" in base and "original_split" not in base:
        base["original_split"] = base["split"]
    return base.drop(columns=sorted(overlapping)).merge(
        split_mapping, on="sku_code", how="inner", validate="many_to_one",
    ).sort_values(["first_actual_sale_date", "sku_code", "lead_months"], kind="stable").reset_index(drop=True)


def validate_label_restoration_provenance(
    *, schedule: pd.DataFrame, windows: pd.DataFrame,
    report: Mapping[str, Any] | None, available_through: datetime,
) -> dict[str, Any]:
    """Validate version, cutoff and every SKU's actual restored lifecycle span."""
    if "launch_universe_version" not in schedule:
        return {"status": "legacy_unversioned_schedule", "verified_universe": False}
    validate_launch_origin_contract(schedule, name="W_cover label schedule")
    if report is None:
        raise DataContractViolation("verified label schedule requires round4_report.json; pass --restore-report")
    versions = set(schedule["dataset_version"].astype(str))
    if len(versions) != 1 or str(report.get("round3_dataset_version")) not in versions:
        raise DataContractViolation("label schedule and Round 4 report have different source dataset_version")
    raw_cutoff = report.get("decision_cutoff_time")
    cutoff = pd.to_datetime(raw_cutoff, errors="coerce", utc=True)
    if pd.isna(cutoff) or cutoff != pd.Timestamp(available_through).tz_convert("UTC"):
        raise DataContractViolation("label Round 4 cutoff must match the global label available-through time")
    if "product_snapshot_partition" in schedule:
        snapshot = pd.to_datetime(report.get("product_snapshot_time"), errors="coerce")
        expected_snapshots = set(schedule["product_snapshot_partition"].astype(str))
        if pd.isna(snapshot) or expected_snapshots != {snapshot.date().isoformat()}:
            raise DataContractViolation("label Round 4 product snapshot differs from the fixed schedule snapshot")
    if report.get("model_execution_mode") == "fit_train_only":
        raise DataContractViolation("model-fit-only Round 4 has no restored demand for labels")
    sources = pd.DataFrame(report.get("selected_skus", []))
    require_columns(sources, ("sku_code", "start_date", "end_date"), "Round 4 per-SKU windows")
    sources["sku_code"] = sources["sku_code"].astype(str)
    if sources["sku_code"].duplicated().any():
        raise DataContractViolation("Round 4 report has duplicate per-SKU windows")
    checks = windows[["sku_code", "label_start_date", "label_end_date"]].merge(
        sources[["sku_code", "start_date", "end_date"]], on="sku_code", how="left", validate="one_to_one",
    )
    parsed = {key: pd.to_datetime(checks[key], errors="coerce").dt.normalize()
              for key in ("label_start_date", "label_end_date", "start_date", "end_date")}
    invalid = ~parsed["start_date"].eq(parsed["label_start_date"]) | ~parsed["end_date"].ge(parsed["label_end_date"])
    if invalid.any():
        raise DataContractViolation(f"Round 4 origin/coverage does not match W_cover: {checks.loc[invalid, 'sku_code'].head(5).tolist()}")
    return {"status": "verified", "verified_universe": True,
            "source_dataset_version": next(iter(versions)),
            "restoration_dataset_version": report.get("dataset_version"),
            "validated_skus": len(checks), "label_outcomes_only": True}
