from __future__ import annotations

import pandas as pd

from scripts.audit_training_decision_times import _timeline, _type_summary


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sku_code": "A",
                "first_actual_sale_date": "2026-01-20",
                "split": "Train",
                "time_block": "B1",
            },
            {
                "sku_code": "B",
                "first_actual_sale_date": "2026-02-01",
                "split": "Test",
                "time_block": "",
            },
        ]
    )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ds": pd.Timestamp("2026-01-15").date(),
                "purchase_order_no": "PO1",
                "purchase_order_line_no": "10",
                "sku_code": "A",
                "demand_type": "FIRST",
                "demand_type_desc": "首单",
                "order_type": "NB",
                "order_type_desc": "新品",
                "document_status": "OK",
                "approval_status_desc": "已审批",
                "deletion_flag": "",
                "changed_date": pd.Timestamp("2026-01-10"),
                "approval_date": pd.Timestamp("2026-01-11"),
                "confirm_date": pd.Timestamp("2026-01-12"),
                "arrival_date": pd.Timestamp("2026-01-15"),
                "target_qty": 100.0,
                "received_qty": 90.0,
            },
            {
                "ds": pd.Timestamp("2026-01-18").date(),
                "purchase_order_no": "PO2",
                "purchase_order_line_no": "20",
                "sku_code": "A",
                "demand_type": "REPEAT",
                "demand_type_desc": "返单",
                "order_type": "NB",
                "order_type_desc": "新品",
                "document_status": "OK",
                "approval_status_desc": "已审批",
                "deletion_flag": "",
                "changed_date": pd.Timestamp("2026-01-16"),
                "approval_date": pd.Timestamp("2026-01-21"),
                "confirm_date": pd.NaT,
                "arrival_date": pd.Timestamp("2026-01-18"),
                "target_qty": 50.0,
                "received_qty": 50.0,
            },
        ]
    )


def test_timeline_keeps_missing_purchase_skus_and_does_not_choose_a_rule() -> None:
    result = _timeline(_cohort(), _history()).set_index("sku_code")

    assert result.loc["A", "purchase_order_count"] == 2
    assert result.loc["A", "earliest_approval_date"].startswith("2026-01-11")
    assert result.loc["A", "approval_date_days_before_first_sale"] == 9.0
    assert result.loc["A", "purchase_history_status"].endswith(
        "needs_first_order_rule"
    )
    assert result.loc["B", "purchase_line_count"] == 0
    assert result.loc["B", "purchase_history_status"] == "no_purchase_rows"


def test_type_summary_exposes_before_and_after_first_sale_dates() -> None:
    result = _type_summary(_cohort(), _history()).set_index("demand_type")

    assert result.loc["FIRST", "approval_date_before_or_on_first_sale_rows"] == 1
    assert result.loc["REPEAT", "approval_date_before_or_on_first_sale_rows"] == 0
    assert result.loc["FIRST", "sku_count"] == 1
    assert result.loc["FIRST", "target_qty_sum"] == 100.0
