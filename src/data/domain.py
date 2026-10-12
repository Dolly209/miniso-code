from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from data.domestic import DataContractViolation
from data.normalization import normalize_identifier


CANONICAL_KEYS: dict[str, tuple[str, ...]] = {
    "products": ("sku_code",),
    "ip_membership": ("sku_code", "ip_code"),
    "stores": ("store_code",),
    "restored_store_weekly": ("sku_code", "store_code", "lifecycle_week"),
    "restored_tier_weekly": ("sku_code", "store_tier", "lifecycle_week"),
    "comparables": ("target_sku_code", "candidate_sku_code"),
    "placement": ("sku_code", "store_code"),
    "first_order": ("sku_code",),
    "reorder_actuals": ("sku_code", "store_code", "business_day"),
}


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataContractViolation(f"{name} is missing canonical columns: {', '.join(missing)}")


def normalize_sku_column(frame: pd.DataFrame, column: str = "sku_code") -> pd.DataFrame:
    result = frame.copy()
    if column not in result:
        raise DataContractViolation(f"missing SKU column: {column}")
    try:
        result[column] = normalize_identifier(result[column], field_name=column)
    except ValueError as exc:
        raise DataContractViolation(str(exc)) from exc
    return result


def filter_skus(
    frame: pd.DataFrame,
    sku_codes: Sequence[str] | None,
    *,
    column: str = "sku_code",
    limit_skus: int | None = None,
) -> pd.DataFrame:
    result = normalize_sku_column(frame, column)
    requested = [str(value).strip() for value in (sku_codes or []) if str(value).strip()]
    if requested:
        order = {sku: index for index, sku in enumerate(dict.fromkeys(requested))}
        result = result.loc[result[column].isin(order)].copy()
        missing = sorted(set(order) - set(result[column]))
        if missing:
            raise DataContractViolation(f"requested SKUs are absent: {missing}")
        result["__sku_order"] = result[column].map(order)
        result = result.sort_values("__sku_order").drop(columns="__sku_order")
    elif limit_skus is not None:
        if limit_skus <= 0:
            raise ValueError("limit_skus must be positive")
        keep = result[column].drop_duplicates().head(limit_skus)
        result = result.loc[result[column].isin(set(keep))]
    return result.reset_index(drop=True)


def frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.sort_index(axis=1).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_algorithm_parameters(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataContractViolation(f"algorithm parameter file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataContractViolation(f"invalid algorithm parameter JSON: {path}: {exc}") from exc
    if str(payload.get("spec_version")) != "3.2":
        raise DataContractViolation("algorithm parameters must declare spec_version=3.2")
    return payload


@dataclass(frozen=True)
class DecisionMetadata:
    run_id: str
    decision_cutoff_time: datetime
    dataset_version: str
    model_version: str
    parameter_version: str
    rule_version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "decision_cutoff_time": self.decision_cutoff_time.isoformat(),
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
            "parameter_version": self.parameter_version,
            "rule_version": self.rule_version,
        }
