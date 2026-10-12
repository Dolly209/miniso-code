from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import inf
from typing import Iterable, Mapping


# The V3.2 section 8.1 distribution table defines these piece-unit thresholds.
# Test labels never fit or alter these boundaries.
V3_2_SALES_BAND_EDGES = (
    0.0,
    500.0,
    1_000.0,
    9_000.0,
    15_000.0,
    30_000.0,
    50_000.0,
    100_000.0,
    inf,
)


@dataclass(frozen=True)
class WeightedSku:
    sku_code: str
    label_value: float
    sales_band: str
    weight: float


def sales_band(value: float) -> str:
    if value < 0:
        raise ValueError("sales-band label must be non-negative")
    for lower, upper in zip(V3_2_SALES_BAND_EDGES[:-1], V3_2_SALES_BAND_EDGES[1:]):
        if lower <= value < upper:
            upper_text = "inf" if upper == inf else str(int(upper))
            return f"[{int(lower)},{upper_text})"
    raise AssertionError("unreachable sales band")


def long_tail_weights(
    labels: Mapping[str, float],
    *,
    sku_equal_share: float = 0.5,
    band_equal_share: float = 0.5,
) -> tuple[WeightedSku, ...]:
    if not labels:
        raise ValueError("labels must not be empty")
    if sku_equal_share < 0 or band_equal_share < 0:
        raise ValueError("weight shares must be non-negative")
    if abs(sku_equal_share + band_equal_share - 1.0) > 1e-12:
        raise ValueError("weight shares must sum to 1")
    assigned = {sku: sales_band(float(value)) for sku, value in labels.items()}
    counts = Counter(assigned.values())
    number_of_skus = len(labels)
    number_of_nonempty_bands = len(counts)
    raw: dict[str, float] = {}
    for sku, band in assigned.items():
        raw[sku] = (
            sku_equal_share / number_of_skus
            + band_equal_share / (number_of_nonempty_bands * counts[band])
        )
    normalizer = sum(raw.values())
    return tuple(
        WeightedSku(
            sku_code=sku,
            label_value=float(labels[sku]),
            sales_band=assigned[sku],
            weight=raw[sku] / normalizer,
        )
        for sku in sorted(labels)
    )


def split_weight_across_nodes(
    sku_weights: Iterable[WeightedSku], node_counts: Mapping[str, int]
) -> dict[tuple[str, int], float]:
    result: dict[tuple[str, int], float] = {}
    for row in sku_weights:
        count = node_counts.get(row.sku_code)
        if count is None or count <= 0:
            raise ValueError(f"node count must be positive for {row.sku_code}")
        for index in range(count):
            result[(row.sku_code, index)] = row.weight / count
    return result
