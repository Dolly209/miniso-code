from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.domestic import DataContractViolation
from scripts.build_historical_restore import _read_cohort


def _write_cohort(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "dataset_version": "test-v1",
                "sku_code": "2028156310109",
                "first_actual_sale_date": "2025-10-17",
                "split": "Train",
                "long_tail_weight": 0.5,
            },
            {
                "dataset_version": "test-v1",
                "sku_code": "2029025310107",
                "first_actual_sale_date": "2025-11-03",
                "split": "Test",
                "long_tail_weight": None,
            },
        ]
    ).to_csv(path, index=False)


def test_explicit_sku_selection_accepts_train_and_test_and_preserves_order(
    tmp_path: Path,
) -> None:
    cohort_path = tmp_path / "cohort.csv"
    _write_cohort(cohort_path)

    selected = _read_cohort(
        cohort_path,
        [" 2029025310107 ", "2028156310109"],
        limit=1,
    )

    assert selected["sku_code"].tolist() == ["2029025310107", "2028156310109"]
    assert selected["split"].tolist() == ["Test", "Train"]


def test_explicit_sku_selection_rejects_test_only_sample(tmp_path: Path) -> None:
    cohort_path = tmp_path / "cohort.csv"
    _write_cohort(cohort_path)

    with pytest.raises(DataContractViolation, match="at least one Train SKU"):
        _read_cohort(cohort_path, ["2029025310107"], limit=5)
