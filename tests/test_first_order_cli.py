from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from scripts.run_first_order import parse_args


def test_first_order_cli_accepts_fixed_product_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_first_order.py",
            "--data-root",
            "./data",
            "--decision-cutoff-time",
            "2025-03-08T23:59:59+08:00",
            "--product-snapshot-time",
            "2026-09-30T23:59:59+08:00",
        ],
    )

    args = parse_args()

    assert args.decision_cutoff_time == datetime.fromisoformat(
        "2025-03-08T23:59:59+08:00"
    )
    assert args.product_snapshot_time == datetime.fromisoformat(
        "2026-09-30T23:59:59+08:00"
    )
