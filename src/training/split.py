from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class SkuSample:
    first_actual_sale_date: date
    sku_code: str


def fixed_train_test_split(
    samples: Iterable[SkuSample], *, cutoff: date
) -> tuple[tuple[SkuSample, ...], tuple[SkuSample, ...]]:
    by_sku: dict[str, SkuSample] = {}
    for sample in samples:
        existing = by_sku.get(sample.sku_code)
        if existing and existing.first_actual_sale_date != sample.first_actual_sale_date:
            raise ValueError(
                f"conflicting first_actual_sale_date for SKU {sample.sku_code}"
            )
        by_sku[sample.sku_code] = sample
    ordered = tuple(sorted(by_sku.values()))
    train = tuple(sample for sample in ordered if sample.first_actual_sale_date <= cutoff)
    test = tuple(sample for sample in ordered if sample.first_actual_sale_date > cutoff)
    return train, test


def chronological_ratio_split(
    samples: Iterable[SkuSample], *, train_ratio: float
) -> tuple[tuple[SkuSample, ...], tuple[SkuSample, ...], date]:
    """Choose the date boundary closest to the requested chronological ratio.

    Every SKU with the same first-sale date stays on the same side. Therefore the
    realized ratio is allowed to differ slightly from ``train_ratio``.
    """

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    by_sku: dict[str, SkuSample] = {}
    for sample in samples:
        existing = by_sku.get(sample.sku_code)
        if existing and existing.first_actual_sale_date != sample.first_actual_sale_date:
            raise ValueError(
                f"conflicting first_actual_sale_date for SKU {sample.sku_code}"
            )
        by_sku[sample.sku_code] = sample
    ordered = tuple(sorted(by_sku.values()))
    if len(ordered) < 2:
        raise ValueError("at least two SKUs are required for a train/test split")

    groups: list[tuple[date, int]] = []
    for sample in ordered:
        if groups and groups[-1][0] == sample.first_actual_sale_date:
            groups[-1] = (groups[-1][0], groups[-1][1] + 1)
        else:
            groups.append((sample.first_actual_sale_date, 1))
    if len(groups) < 2:
        raise ValueError(
            "at least two distinct first-sale dates are required without splitting a day"
        )

    target_train_count = len(ordered) * train_ratio
    boundaries: list[tuple[float, bool, date]] = []
    cumulative = 0
    for cutoff, count in groups[:-1]:
        cumulative += count
        boundaries.append(
            (
                abs(cumulative - target_train_count),
                cumulative > target_train_count,
                cutoff,
            )
        )
    _, _, selected_cutoff = min(boundaries)
    train, test = fixed_train_test_split(ordered, cutoff=selected_cutoff)
    if not train or not test:
        raise ValueError("chronological ratio split produced an empty partition")
    return train, test, selected_cutoff


def build_time_blocks(
    samples: Sequence[SkuSample], *, n_time_blocks: int
) -> tuple[tuple[SkuSample, ...], ...]:
    if n_time_blocks < 2:
        raise ValueError("n_time_blocks must be >= 2")
    groups: dict[date, list[SkuSample]] = {}
    for sample in sorted(samples):
        groups.setdefault(sample.first_actual_sale_date, []).append(sample)
    dated_groups = sorted(groups.items())
    if len(dated_groups) < n_time_blocks:
        raise ValueError("not enough distinct first-sale dates for n_time_blocks")

    blocks: list[list[SkuSample]] = [[] for _ in range(n_time_blocks)]
    total = sum(len(group) for _, group in dated_groups)
    current_block = 0
    assigned = 0
    for group_index, (_, group) in enumerate(dated_groups):
        groups_remaining = len(dated_groups) - group_index
        later_blocks = n_time_blocks - current_block - 1
        if (
            current_block < n_time_blocks - 1
            and blocks[current_block]
            and (
                assigned >= round(total * (current_block + 1) / n_time_blocks)
                or groups_remaining == later_blocks
            )
            and groups_remaining >= later_blocks
        ):
            current_block += 1
        blocks[current_block].extend(group)
        assigned += len(group)
    if any(not block for block in blocks):
        raise ValueError("could not create non-empty time blocks without splitting a date")
    return tuple(tuple(block) for block in blocks)


def expanding_window_folds(
    blocks: Sequence[Sequence[SkuSample]],
) -> tuple[tuple[tuple[SkuSample, ...], tuple[SkuSample, ...]], ...]:
    if len(blocks) < 2:
        raise ValueError("at least two time blocks are required")
    folds = []
    for validation_index in range(1, len(blocks)):
        train = tuple(
            sample
            for block in blocks[:validation_index]
            for sample in block
        )
        validation = tuple(blocks[validation_index])
        folds.append((train, validation))
    return tuple(folds)
