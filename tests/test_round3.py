from __future__ import annotations

from datetime import date, timedelta

import pytest

from training.cohort import CohortBuildResult, CohortBuildStats, CohortRow
from training.round3 import build_round3_report


def test_round3_report_uses_dynamic_time_split_without_example_count_gate() -> None:
    start = date(2025, 1, 1)
    rows = tuple(
        CohortRow(
            sku_code=f"SKU{index:03d}",
            first_actual_sale_date=start + timedelta(days=index),
            sales_26w_qty=float(index * 1_000),
        )
        for index in range(10)
    )
    result = CohortBuildResult(
        rows=rows,
        stats=CohortBuildStats(
            source_partition_min=start,
            label_as_of_date=date(2025, 12, 31),
            window_days=182,
            partitions_scanned=10,
            physical_rows_scanned=10,
            visible_rows_scanned=10,
            order_line_versions=10,
            negative_sale_qty_versions=0,
            negative_sale_qty_sum=0.0,
            negative_sale_qty_min=None,
            distinct_latest_order_lines=10,
            latest_negative_order_lines=0,
            latest_negative_sale_qty_sum=0.0,
            latest_negative_sale_qty_min=None,
            same_ds_duplicate_rows=0,
            same_ds_exact_duplicate_rows=0,
            same_ds_signed_aggregation_extra_rows=0,
            same_ds_signed_aggregation_groups=0,
            same_ds_zero_net_groups=0,
            same_ds_nonzero_net_groups=0,
            sku_with_positive_sales=10,
            mature_sku_count=10,
            immature_sku_count=0,
            first_sale_on_source_min_count=1,
        ),
        dataset_version="v3.2-test",
    )

    report, output_rows = build_round3_report(
        result,
        train_ratio=0.8,
        n_time_blocks=4,
        sku_equal_share=0.5,
        band_equal_share=0.5,
    )

    assert report["status"] == "ok"
    assert report["split_summary"]["cohort_count"] == 10
    assert report["split_summary"]["train_count"] == 8
    assert report["split_summary"]["test_count"] == 2
    assert report["split_summary"]["requested_train_ratio"] == 0.8
    assert report["split_summary"]["actual_train_ratio"] == pytest.approx(0.8)
    assert report["split_summary"]["selected_cutoff_date"] == "2025-01-08"
    assert report["num_folds"] == 3
    assert report["split_summary"]["same_first_sale_day_split"] is False
    assert report["long_tail_weights"]["weight_sum"] == pytest.approx(1.0)
    assert "1178/955/223" in report["reference_example_policy"]
    assert len(output_rows) == 10
    assert sum(row["split"] == "Train" for row in output_rows) == 8
    assert all(
        row["long_tail_weight"] == ""
        for row in output_rows
        if row["split"] == "Test"
    )
