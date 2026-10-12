from __future__ import annotations

import pytest

from training.weights import long_tail_weights, sales_band, split_weight_across_nodes


def test_v3_2_sales_band_edges_are_half_open() -> None:
    assert sales_band(0) == "[0,500)"
    assert sales_band(499.999) == "[0,500)"
    assert sales_band(500) == "[500,1000)"
    assert sales_band(100_000) == "[100000,inf)"
    with pytest.raises(ValueError, match="non-negative"):
        sales_band(-1)


def test_long_tail_weight_is_half_sku_equal_and_half_band_equal() -> None:
    weighted = long_tail_weights({"A": 100, "B": 200, "C": 2_000})
    by_sku = {row.sku_code: row.weight for row in weighted}

    assert sum(by_sku.values()) == pytest.approx(1.0)
    assert by_sku["A"] == pytest.approx(0.5 / 3 + 0.5 / (2 * 2))
    assert by_sku["B"] == pytest.approx(by_sku["A"])
    assert by_sku["C"] == pytest.approx(0.5 / 3 + 0.5 / (2 * 1))


def test_reorder_nodes_split_each_sku_weight_without_changing_total() -> None:
    weighted = long_tail_weights({"A": 100, "B": 2_000})
    node_weights = split_weight_across_nodes(weighted, {"A": 2, "B": 3})

    assert sum(node_weights.values()) == pytest.approx(1.0)
    assert node_weights[("A", 0)] == node_weights[("A", 1)]
    assert node_weights[("B", 0)] == node_weights[("B", 2)]
