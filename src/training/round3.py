from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from training.cohort import CohortBuildResult, CohortRow
from training.split import (
    SkuSample,
    build_time_blocks,
    chronological_ratio_split,
    expanding_window_folds,
)
from training.weights import V3_2_SALES_BAND_EDGES, long_tail_weights, sales_band


V3_2_DISTRIBUTION_AUDITS: tuple[tuple[str, Callable[[float], bool], int, int], ...] = (
    ("lt_500", lambda value: value < 500, 452, 102),
    ("lt_1000", lambda value: value < 1_000, 553, 142),
    ("ge_9000", lambda value: value >= 9_000, 117, 30),
    ("ge_15000", lambda value: value >= 15_000, 63, 19),
    ("ge_30000", lambda value: value >= 30_000, 28, 13),
    ("ge_50000", lambda value: value >= 50_000, 15, 10),
    ("ge_100000", lambda value: value >= 100_000, 5, 3),
)


def _row_by_sku(rows: Sequence[CohortRow]) -> dict[str, CohortRow]:
    result = {row.sku_code: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("cohort contains duplicate sku_code")
    return result


def build_round3_report(
    result: CohortBuildResult,
    *,
    train_ratio: float,
    n_time_blocks: int,
    sku_equal_share: float,
    band_equal_share: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows_by_sku = _row_by_sku(result.rows)
    samples = [
        SkuSample(row.first_actual_sale_date, row.sku_code) for row in result.rows
    ]
    train, test, train_test_cutoff = chronological_ratio_split(
        samples, train_ratio=train_ratio
    )
    train_skus = {row.sku_code for row in train}
    test_skus = {row.sku_code for row in test}
    train_dates = {row.first_actual_sale_date for row in train}
    test_dates = {row.first_actual_sale_date for row in test}
    if train_dates & test_dates:
        raise ValueError("the chronological split divided a first-sale date")
    blocks = build_time_blocks(train, n_time_blocks=n_time_blocks)
    folds = expanding_window_folds(blocks)
    train_labels = {
        sku: rows_by_sku[sku].sales_26w_qty for sku in sorted(train_skus)
    }
    weighted = long_tail_weights(
        train_labels,
        sku_equal_share=sku_equal_share,
        band_equal_share=band_equal_share,
    )
    weight_by_sku = {row.sku_code: row for row in weighted}
    block_by_sku = {
        row.sku_code: f"B{index}"
        for index, block in enumerate(blocks, start=1)
        for row in block
    }

    distribution_summary: list[dict[str, Any]] = []
    for name, predicate, _, _ in V3_2_DISTRIBUTION_AUDITS:
        actual_train = sum(
            predicate(rows_by_sku[sku].sales_26w_qty) for sku in train_skus
        )
        actual_test = sum(
            predicate(rows_by_sku[sku].sales_26w_qty) for sku in test_skus
        )
        distribution_summary.append(
            {
                "rule": name,
                "train_count": actual_train,
                "test_count": actual_test,
            }
        )

    block_reports = []
    for index, block in enumerate(blocks, start=1):
        dates = [row.first_actual_sale_date for row in block]
        block_reports.append(
            {
                "block": f"B{index}",
                "sku_count": len(block),
                "first_sale_min": min(dates).isoformat(),
                "first_sale_max": max(dates).isoformat(),
            }
        )
    fold_reports = [
        {
            "fold": index,
            "train_blocks": [f"B{i}" for i in range(1, index + 1)],
            "validation_block": f"B{index + 1}",
            "train_sku_count": len(fold_train),
            "validation_sku_count": len(fold_validation),
        }
        for index, (fold_train, fold_validation) in enumerate(folds, start=1)
    ]
    band_counts = {
        sales_band(lower): 0 for lower in V3_2_SALES_BAND_EDGES[:-1]
    }
    for row in weighted:
        band_counts[row.sales_band] = band_counts.get(row.sales_band, 0) + 1

    cohort_count = len(result.rows)
    train_count = len(train)
    test_count = len(test)
    cutoff_day_count = sum(
        row.first_actual_sale_date == train_test_cutoff for row in train
    )
    report = {
        "spec_version": "3.2",
        "round": 3,
        "status": "ok",
        "dataset_version": result.dataset_version,
        "scope": (
            "actual mature cohort, chronological approximate 80/20 split, "
            "B1-B5, and long-tail weight audit"
        ),
        "label_semantics": {
            "field": "sales_26w_qty",
            "purpose": "long-tail distribution, sales band, and lifecycle audit only",
            "not_universal_first_order_label": True,
            "window_days": result.stats.window_days,
            "zero_filled_skus_added": False,
            "negative_sale_qty_handling": (
                "retain the signed delivered sale_qty after latest-version selection; "
                "do not clamp it and do not subtract all-null refund fields"
            ),
        },
        "source_stats": {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in asdict(result.stats).items()
        },
        "split_summary": {
            "method": (
                "select the first-sale date boundary closest to the requested train "
                "ratio; never split SKUs sharing the same first-sale date"
            ),
            "requested_train_ratio": train_ratio,
            "selected_cutoff_date": train_test_cutoff.isoformat(),
            "cohort_count": cohort_count,
            "train_count": train_count,
            "test_count": test_count,
            "actual_train_ratio": train_count / cohort_count,
            "actual_test_ratio": test_count / cohort_count,
            "cutoff_day_train_sku_count": cutoff_day_count,
            "same_first_sale_day_split": False,
        },
        "time_blocks": block_reports,
        "expanding_window_folds": fold_reports,
        "num_folds": len(folds),
        "long_tail_weights": {
            "formula": (
                f"{sku_equal_share}/N + {band_equal_share}/(|B|*n_b(i)), "
                "then normalized"
            ),
            "sku_equal_share": sku_equal_share,
            "band_equal_share": band_equal_share,
            "train_only": True,
            "test_influences_weight_fit": False,
            "band_definition": "V3.2 section 8.1 cumulative-threshold edges",
            "nonempty_train_band_count": sum(
                count > 0 for count in band_counts.values()
            ),
            "weight_sum": sum(row.weight for row in weighted),
            "band_counts": dict(sorted(band_counts.items())),
        },
        "distribution_summary": distribution_summary,
        "warnings": (
            ([
                f"{result.stats.first_sale_on_source_min_count} SKUs first appear on "
                "the first available sales partition; their true earlier history cannot "
                "be proven from this source window."
            ]
            if result.stats.first_sale_on_source_min_count
            else [])
            + ([
                "The delivered is_return='0' source contains "
                f"{result.stats.latest_negative_order_lines} latest visible order lines "
                "with negative sale_qty. Their signed values are retained and audited; "
                "refund_qty/refund_amt are not subtracted."
            ] if result.stats.latest_negative_order_lines else [])
            + ([
                "Collapsed "
                f"{result.stats.same_ds_exact_duplicate_rows} cohort-relevant exact "
                "duplicate order-line rows within the same ds."
            ] if result.stats.same_ds_exact_duplicate_rows else [])
            + ([
                "Net-aggregated signed sale_qty within the same ds for "
                f"{result.stats.same_ds_signed_aggregation_groups} order-line groups "
                f"using {result.stats.same_ds_signed_aggregation_extra_rows} additional "
                "event rows "
                f"({result.stats.same_ds_zero_net_groups} zero-net; "
                f"{result.stats.same_ds_nonzero_net_groups} nonzero-net). Exact physical "
                "duplicates are collapsed first; groups with different SKUs or "
                "different business days are rejected."
            ] if result.stats.same_ds_signed_aggregation_groups else [])
        ),
        "reference_example_policy": (
            "The 1178/955/223 counts, 2025-11-01 boundary, and accompanying "
            "distribution counts are documentation examples only. They do not gate "
            "the rolling dataset and are not used to trim SKUs."
        ),
    }

    output_rows: list[dict[str, Any]] = []
    for row in result.rows:
        is_train = row.sku_code in train_skus
        weighted_row = weight_by_sku.get(row.sku_code)
        output_rows.append(
            {
                "dataset_version": result.dataset_version,
                "sku_code": row.sku_code,
                "first_actual_sale_date": row.first_actual_sale_date.isoformat(),
                "sales_26w_qty": row.sales_26w_qty,
                "split": "Train" if is_train else "Test",
                "time_block": block_by_sku.get(row.sku_code, ""),
                "sales_band": (
                    weighted_row.sales_band
                    if weighted_row is not None
                    else sales_band(row.sales_26w_qty)
                ),
                "long_tail_weight": (
                    weighted_row.weight if weighted_row is not None else ""
                ),
                "weight_fitted_from_train": is_train,
            }
        )
    return report, output_rows


def write_round3_artifacts(
    *, output_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "round3_report.json"
    report_tmp = report_path.with_suffix(".json.tmp")
    report_tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_tmp.replace(report_path)

    cohort_path = output_dir / "cohort.csv"
    cohort_tmp = cohort_path.with_suffix(".csv.tmp")
    fieldnames = list(rows[0]) if rows else [
        "dataset_version",
        "sku_code",
        "first_actual_sale_date",
        "sales_26w_qty",
        "split",
        "time_block",
        "sales_band",
        "long_tail_weight",
        "weight_fitted_from_train",
    ]
    with cohort_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    cohort_tmp.replace(cohort_path)
    return {"report": report_path, "cohort": cohort_path}
