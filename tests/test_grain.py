from __future__ import annotations

import pytest

from data.grain import (
    DuplicateGrainError,
    aggregate_additive_measures,
    assert_unique_grain,
)


def test_multi_ip_dimension_cannot_be_joined_as_unique_sku() -> None:
    product_rows = [
        {"sku_code": "SKU001", "ip_code": "IP_A"},
        {"sku_code": "SKU001", "ip_code": "IP_B"},
    ]
    with pytest.raises(DuplicateGrainError, match="not unique"):
        assert_unique_grain(product_rows, key_fields=("sku_code",))


def test_measures_are_aggregated_before_dimension_join() -> None:
    rows = [
        {"sku_code": "SKU001", "store_code": "S1", "day": "2025-01-01", "qty": 2},
        {"sku_code": "SKU001", "store_code": "S1", "day": "2025-01-01", "qty": 3},
    ]
    aggregated = aggregate_additive_measures(
        rows,
        key_fields=("sku_code", "store_code", "day"),
        measure_fields=("qty",),
    )
    assert aggregated == [
        {"sku_code": "SKU001", "store_code": "S1", "day": "2025-01-01", "qty": 5.0}
    ]
