from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

from data.domestic import DataContractViolation
from training.cohort import CohortBuildResult
from training.round3 import V3_2_DISTRIBUTION_AUDITS
from training.split import SkuSample, fixed_train_test_split


BLIND_LEVEL3 = frozenset({"盲盒", "盲盒系列", "盲盒系列-TT", "迷你粒"})


@dataclass(frozen=True)
class HistoricalReproductionRow:
    sku_code: str
    product_first_sale_date: date
    first_actual_sale_date: date
    product_type: str
    product_rule: str
    pkg_unit: float | None
    sales_26w_sku: float
    sales_26w_business: float
    split: str


def _parse_date_series(values: Any, *, name: str) -> Any:
    import pandas as pd

    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).head(5).tolist()
        raise DataContractViolation(f"unparseable {name} values: {examples}")
    return parsed.dt.date


def _normalize_sku_codes(values: Any) -> Any:
    import pandas as pd

    if values.isna().any():
        raise DataContractViolation("snapshot sku_code contains null values")

    def normalize(value: object) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    result = values.map(normalize)
    if (result == "").any():
        raise DataContractViolation("snapshot sku_code contains blank values")
    return result.astype(pd.StringDtype())


def _collapse_product_snapshot(frame: Any) -> Any:
    required = (
        "sku_code",
        "sku_name",
        "first_sale_date",
        "cate_level3_name",
        "cate_level4_name",
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataContractViolation(
            "product snapshot is missing reproduction fields: " + ", ".join(missing)
        )
    product = frame[list(required)].copy()
    product["sku_code"] = _normalize_sku_codes(product["sku_code"])
    product["first_sale_date"] = _parse_date_series(
        product["first_sale_date"], name="first_sale_date"
    )
    for column in ("sku_name", "cate_level3_name", "cate_level4_name"):
        product[column] = product[column].map(
            lambda value: None if _is_missing(value) else str(value).strip()
        )
    for column in required[1:]:
        conflicts = product.groupby("sku_code", dropna=False)[column].nunique(
            dropna=True
        )
        if (conflicts > 1).any():
            examples = conflicts.loc[conflicts > 1].index.astype(str).tolist()[:5]
            raise DataContractViolation(
                f"product snapshot field {column} conflicts within SKU; "
                f"examples={examples}"
            )
    return product.drop_duplicates(subset=["sku_code"], keep="first").reset_index(
        drop=True
    )


def _collapse_pack_snapshot(frame: Any) -> Any:
    import pandas as pd

    required = ("sku_code", "pkg_unit")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataContractViolation(
            "pack snapshot is missing reproduction fields: " + ", ".join(missing)
        )
    pack = frame[list(required)].copy()
    pack["sku_code"] = _normalize_sku_codes(pack["sku_code"])
    raw = pack["pkg_unit"]
    pack["pkg_unit"] = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & pack["pkg_unit"].isna()
    if invalid.any():
        examples = raw.loc[invalid].astype(str).head(5).tolist()
        raise DataContractViolation(f"non-numeric pkg_unit values: {examples}")
    conflicts = pack.groupby("sku_code", dropna=False)["pkg_unit"].nunique(
        dropna=True
    )
    if (conflicts > 1).any():
        examples = conflicts.loc[conflicts > 1].index.astype(str).tolist()[:5]
        raise DataContractViolation(
            "pack snapshot pkg_unit conflicts within SKU; " f"examples={examples}"
        )
    return pack.drop_duplicates(subset=["sku_code"], keep="first").reset_index(
        drop=True
    )


def _is_missing(value: object) -> bool:
    import pandas as pd

    return bool(pd.isna(value))


def _safe_text(value: object) -> str:
    return "" if _is_missing(value) else str(value).strip()


def classify_product(row: Any) -> tuple[str, str]:
    c3 = _safe_text(row.get("cate_level3_name"))
    c4 = _safe_text(row.get("cate_level4_name"))
    sku_name = _safe_text(row.get("sku_name"))
    if c4 == "端盒":
        return "end_box", "c4=端盒"
    if c3 == "明盒" or "明盒" in c4:
        return "open_box", "明盒规则"
    if c3 == "小程序销售" and c4 == "抽盒机":
        return "blind_box", "小程序销售/抽盒机"
    if c3 in BLIND_LEVEL3:
        return "blind_box", f"c3={c3}"
    if "盲盒" in c4:
        return "blind_box", "c4含盲盒"
    if any(word in sku_name for word in ("盲盒", "盲袋", "盲抽")):
        return "blind_box", "sku_name盲盒fallback"
    return "other_1to1", "其他1to1"


def _business_quantity(
    sku_quantity: float, product_type: str, pkg_unit: float | None
) -> float | None:
    if product_type != "blind_box":
        return sku_quantity
    if pkg_unit is None or pkg_unit <= 0:
        return None
    return sku_quantity / pkg_unit


def _reproduction_version(
    rows: Sequence[HistoricalReproductionRow],
    *,
    label_as_of_date: date,
    product_snapshot_date: date,
    pack_snapshot_date: date,
) -> str:
    payload = {
        "spec_version": "3.2",
        "label_as_of_date": label_as_of_date.isoformat(),
        "product_snapshot_date": product_snapshot_date.isoformat(),
        "pack_snapshot_date": pack_snapshot_date.isoformat(),
        "rows": [
            {
                **asdict(row),
                "product_first_sale_date": row.product_first_sale_date.isoformat(),
                "first_actual_sale_date": row.first_actual_sale_date.isoformat(),
            }
            for row in rows
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"v3.2-reproduction-{label_as_of_date.isoformat()}-{digest[:16]}"


def build_historical_reproduction_report(
    sales_result: CohortBuildResult,
    *,
    product_frame: Any,
    product_snapshot_date: date,
    pack_frame: Any,
    pack_snapshot_date: date,
    product_first_sale_start: date,
    train_test_cutoff: date,
    expected_cohort_size: int,
    expected_train_size: int,
    expected_test_size: int,
    left_censor_buffer_days: int = 7,
    distribution_audits: Sequence[
        tuple[str, Callable[[float], bool], int, int]
    ] = V3_2_DISTRIBUTION_AUDITS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if left_censor_buffer_days < 0:
        raise ValueError("left_censor_buffer_days must be non-negative")
    product = _collapse_product_snapshot(product_frame)
    pack = _collapse_pack_snapshot(pack_frame)
    product = product.merge(pack, on="sku_code", how="left", validate="one_to_one")
    classified = product.apply(classify_product, axis=1)
    product["product_type"] = [value[0] for value in classified]
    product["product_rule"] = [value[1] for value in classified]

    window_days = sales_result.stats.window_days
    label_as_of = sales_result.stats.label_as_of_date
    left_censor_cutoff = sales_result.stats.source_partition_min + timedelta(
        days=left_censor_buffer_days
    )
    sales_by_sku = {row.sku_code: row for row in sales_result.rows}
    if len(sales_by_sku) != len(sales_result.rows):
        raise DataContractViolation("sales cohort contains duplicate sku_code")

    legacy_complete = {
        sku: row
        for sku, row in sales_by_sku.items()
        if row.first_actual_sale_date >= left_censor_cutoff
        and row.first_actual_sale_date + timedelta(days=window_days) <= label_as_of
    }
    matched_product_count = int(product["sku_code"].isin(legacy_complete).sum())
    product_date_eligible_count = 0
    missing_business_count = 0
    candidate_rows: list[dict[str, Any]] = []
    for product_row in product.itertuples(index=False):
        sales_row = legacy_complete.get(str(product_row.sku_code))
        if sales_row is None or _is_missing(product_row.first_sale_date):
            continue
        if product_row.first_sale_date < product_first_sale_start:
            continue
        product_date_eligible_count += 1
        pkg_unit = (
            None if _is_missing(product_row.pkg_unit) else float(product_row.pkg_unit)
        )
        business_qty = _business_quantity(
            sales_row.sales_26w_qty, str(product_row.product_type), pkg_unit
        )
        if business_qty is None:
            missing_business_count += 1
            continue
        candidate_rows.append(
            {
                "sku_code": str(product_row.sku_code),
                "product_first_sale_date": product_row.first_sale_date,
                "first_actual_sale_date": sales_row.first_actual_sale_date,
                "product_type": str(product_row.product_type),
                "product_rule": str(product_row.product_rule),
                "pkg_unit": pkg_unit,
                "sales_26w_sku": float(sales_row.sales_26w_qty),
                "sales_26w_business": float(business_qty),
            }
        )

    samples = [
        SkuSample(row["first_actual_sale_date"], row["sku_code"])
        for row in candidate_rows
    ]
    train, test = fixed_train_test_split(samples, cutoff=train_test_cutoff)
    train_skus = {row.sku_code for row in train}
    test_skus = {row.sku_code for row in test}
    rows = tuple(
        HistoricalReproductionRow(
            **row,
            split="Train" if row["sku_code"] in train_skus else "Test",
        )
        for row in sorted(
            candidate_rows,
            key=lambda value: (value["first_actual_sale_date"], value["sku_code"]),
        )
    )
    row_by_sku = {row.sku_code: row for row in rows}

    distribution_checks: list[dict[str, Any]] = []
    for name, predicate, expected_train, expected_test in distribution_audits:
        actual_train = sum(
            predicate(row_by_sku[sku].sales_26w_business) for sku in train_skus
        )
        actual_test = sum(
            predicate(row_by_sku[sku].sales_26w_business) for sku in test_skus
        )
        distribution_checks.append(
            {
                "rule": name,
                "train": {
                    "expected": expected_train,
                    "actual": actual_train,
                    "match": actual_train == expected_train,
                },
                "test": {
                    "expected": expected_test,
                    "actual": actual_test,
                    "match": actual_test == expected_test,
                },
            }
        )

    size_checks = {
        "cohort": {
            "expected": expected_cohort_size,
            "actual": len(rows),
            "match": len(rows) == expected_cohort_size,
        },
        "train": {
            "expected": expected_train_size,
            "actual": len(train),
            "match": len(train) == expected_train_size,
        },
        "test": {
            "expected": expected_test_size,
            "actual": len(test),
            "match": len(test) == expected_test_size,
        },
    }
    size_match = all(value["match"] for value in size_checks.values())
    distribution_match = all(
        check[side]["match"]
        for check in distribution_checks
        for side in ("train", "test")
    )
    reproduction_version = _reproduction_version(
        rows,
        label_as_of_date=label_as_of,
        product_snapshot_date=product_snapshot_date,
        pack_snapshot_date=pack_snapshot_date,
    )
    report = {
        "spec_version": "3.2",
        "purpose": (
            "independent historical reproduction from raw server sales, product, "
            "and pack snapshots; no prepared reference CSV is read"
        ),
        "status": "ok" if size_match and distribution_match else "mismatch",
        "reproduction_version": reproduction_version,
        "raw_sales_dataset_version": sales_result.dataset_version,
        "source_parameters": {
            "source_partition_min": sales_result.stats.source_partition_min.isoformat(),
            "label_as_of_date": label_as_of.isoformat(),
            "product_snapshot_date": product_snapshot_date.isoformat(),
            "pack_snapshot_date": pack_snapshot_date.isoformat(),
            "product_first_sale_start": product_first_sale_start.isoformat(),
            "left_censor_buffer_days": left_censor_buffer_days,
            "legacy_complete_window_rule": (
                f"first_actual_sale_date + {window_days} days <= label_as_of_date"
            ),
            "train_test_cutoff": train_test_cutoff.isoformat(),
        },
        "label_semantics": {
            "d0": "MIN(DATE(order_time)) WHERE latest signed sale_qty > 0",
            "raw_field": "sales_26w_sku",
            "audit_field": "sales_26w_business",
            "window": f"[D0, D0 + {window_days} days)",
            "negative_sale_qty_handling": (
                "retain signed sale_qty and net same-ds reversal events"
            ),
            "business_conversion": (
                "blind_box: sales_26w_sku/pkg_unit; all other product types: 1:1"
            ),
        },
        "filter_funnel": {
            "raw_mature_sales_skus": len(sales_result.rows),
            "legacy_complete_26w_skus": len(legacy_complete),
            "matched_product_snapshot_skus": matched_product_count,
            "product_first_sale_date_eligible_skus": product_date_eligible_count,
            "missing_business_quantity_skus": missing_business_count,
            "final_cohort_skus": len(rows),
        },
        "size_checks": size_checks,
        "distribution_audits": distribution_checks,
        "source_stats": {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in asdict(sales_result.stats).items()
        },
        "interpretation": (
            "ok means the current server sources can reproduce the frozen V3.2 "
            "counts under the historical cutoff and snapshot rules. mismatch keeps "
            "all actual rows for diagnosis and does not alter or trim the cohort."
        ),
    }
    output_rows = [
        {
            "reproduction_version": reproduction_version,
            "sku_code": row.sku_code,
            "product_first_sale_date": row.product_first_sale_date.isoformat(),
            "first_actual_sale_date": row.first_actual_sale_date.isoformat(),
            "product_type": row.product_type,
            "product_rule": row.product_rule,
            "pkg_unit": "" if row.pkg_unit is None else row.pkg_unit,
            "sales_26w_sku": row.sales_26w_sku,
            "sales_26w_business": row.sales_26w_business,
            "split": row.split,
        }
        for row in rows
    ]
    return report, output_rows


def write_historical_reproduction_artifacts(
    *, output_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "historical_reproduction_report.json"
    report_tmp = report_path.with_suffix(".json.tmp")
    report_tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_tmp.replace(report_path)

    cohort_path = output_dir / "historical_reproduction_cohort.csv"
    cohort_tmp = cohort_path.with_suffix(".csv.tmp")
    fieldnames = list(rows[0]) if rows else [
        "reproduction_version",
        "sku_code",
        "product_first_sale_date",
        "first_actual_sale_date",
        "product_type",
        "product_rule",
        "pkg_unit",
        "sales_26w_sku",
        "sales_26w_business",
        "split",
    ]
    with cohort_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    cohort_tmp.replace(cohort_path)
    return {"report": report_path, "cohort": cohort_path}
