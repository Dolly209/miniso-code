from __future__ import annotations

from datetime import date

import pandas as pd

from training.cohort import CohortBuildResult, CohortBuildStats, CohortRow
from training.historical_reproduction import build_historical_reproduction_report


def _stats() -> CohortBuildStats:
    return CohortBuildStats(
        source_partition_min=date(2025, 1, 1),
        label_as_of_date=date(2025, 1, 29),
        window_days=7,
        partitions_scanned=29,
        physical_rows_scanned=6,
        visible_rows_scanned=6,
        order_line_versions=6,
        negative_sale_qty_versions=0,
        negative_sale_qty_sum=0.0,
        negative_sale_qty_min=None,
        distinct_latest_order_lines=6,
        latest_negative_order_lines=0,
        latest_negative_sale_qty_sum=0.0,
        latest_negative_sale_qty_min=None,
        same_ds_duplicate_rows=0,
        same_ds_exact_duplicate_rows=0,
        same_ds_signed_aggregation_extra_rows=0,
        same_ds_signed_aggregation_groups=0,
        same_ds_zero_net_groups=0,
        same_ds_nonzero_net_groups=0,
        sku_with_positive_sales=6,
        mature_sku_count=6,
        immature_sku_count=0,
        first_sale_on_source_min_count=1,
    )


def test_historical_reproduction_uses_raw_snapshots_and_business_units() -> None:
    sales = CohortBuildResult(
        rows=(
            CohortRow("EARLY", date(2025, 1, 1), 1.0),
            CohortRow("BLIND", date(2025, 1, 8), 120.0),
            CohortRow("OTHER", date(2025, 1, 9), 600.0),
            CohortRow("OLD_PRODUCT", date(2025, 1, 9), 700.0),
            CohortRow("MISSING_PACK", date(2025, 1, 9), 800.0),
            CohortRow("TEST", date(2025, 1, 15), 10_000.0),
        ),
        stats=_stats(),
        dataset_version="v3.2-test-raw",
    )
    product = pd.DataFrame(
        [
            {
                "sku_code": "EARLY",
                "sku_name": "普通商品",
                "first_sale_date": "2025-01-08",
                "cate_level3_name": "其他",
                "cate_level4_name": "其他",
            },
            {
                "sku_code": "BLIND",
                "sku_name": "测试盲盒",
                "first_sale_date": "2025-01-08",
                "cate_level3_name": "盲盒",
                "cate_level4_name": "",
            },
            # Identical multi-IP product rows must collapse to one SKU.
            {
                "sku_code": "BLIND",
                "sku_name": "测试盲盒",
                "first_sale_date": "2025-01-08",
                "cate_level3_name": "盲盒",
                "cate_level4_name": "",
            },
            {
                "sku_code": "OTHER",
                "sku_name": "普通商品",
                "first_sale_date": "2025-01-08",
                "cate_level3_name": "其他",
                "cate_level4_name": "其他",
            },
            {
                "sku_code": "OLD_PRODUCT",
                "sku_name": "普通商品",
                "first_sale_date": "2025-01-07",
                "cate_level3_name": "其他",
                "cate_level4_name": "其他",
            },
            {
                "sku_code": "MISSING_PACK",
                "sku_name": "缺箱规盲盒",
                "first_sale_date": "2025-01-08",
                "cate_level3_name": "盲盒",
                "cate_level4_name": "",
            },
            {
                "sku_code": "TEST",
                "sku_name": "普通商品",
                "first_sale_date": "2025-01-08",
                "cate_level3_name": "其他",
                "cate_level4_name": "其他",
            },
        ]
    )
    pack = pd.DataFrame(
        [
            {"sku_code": "EARLY", "pkg_unit": 1},
            {"sku_code": "BLIND", "pkg_unit": 12},
            {"sku_code": "OTHER", "pkg_unit": 1},
            {"sku_code": "OLD_PRODUCT", "pkg_unit": 1},
            {"sku_code": "MISSING_PACK", "pkg_unit": None},
            {"sku_code": "TEST", "pkg_unit": 1},
        ]
    )

    report, rows = build_historical_reproduction_report(
        sales,
        product_frame=product,
        product_snapshot_date=date(2025, 1, 29),
        pack_frame=pack,
        pack_snapshot_date=date(2025, 1, 28),
        product_first_sale_start=date(2025, 1, 8),
        train_test_cutoff=date(2025, 1, 10),
        expected_cohort_size=3,
        expected_train_size=2,
        expected_test_size=1,
        left_censor_buffer_days=7,
        distribution_audits=(
            ("lt_500", lambda value: value < 500, 1, 0),
            ("ge_500", lambda value: value >= 500, 1, 1),
        ),
    )

    assert report["status"] == "ok"
    assert report["filter_funnel"] == {
        "raw_mature_sales_skus": 6,
        "legacy_complete_26w_skus": 5,
        "matched_product_snapshot_skus": 5,
        "product_first_sale_date_eligible_skus": 4,
        "missing_business_quantity_skus": 1,
        "final_cohort_skus": 3,
    }
    by_sku = {row["sku_code"]: row for row in rows}
    assert set(by_sku) == {"BLIND", "OTHER", "TEST"}
    assert by_sku["BLIND"]["sales_26w_sku"] == 120.0
    assert by_sku["BLIND"]["sales_26w_business"] == 10.0
    assert by_sku["BLIND"]["split"] == "Train"
    assert by_sku["TEST"]["split"] == "Test"
