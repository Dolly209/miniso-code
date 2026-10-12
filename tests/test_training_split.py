from __future__ import annotations

from datetime import date, timedelta

from training.split import (
    SkuSample,
    build_time_blocks,
    chronological_ratio_split,
    expanding_window_folds,
    fixed_train_test_split,
)


def synthetic_samples() -> list[SkuSample]:
    start = date(2025, 1, 1)
    return [
        SkuSample(start + timedelta(days=index // 2), f"SKU{index:03d}")
        for index in range(20)
    ]


def test_fixed_split_keeps_same_day_together() -> None:
    cutoff = date(2025, 1, 5)
    train, test = fixed_train_test_split(synthetic_samples(), cutoff=cutoff)
    assert all(sample.first_actual_sale_date <= cutoff for sample in train)
    assert all(sample.first_actual_sale_date > cutoff for sample in test)
    assert {sample.first_actual_sale_date for sample in train}.isdisjoint(
        {sample.first_actual_sale_date for sample in test}
    )


def test_ratio_split_selects_closest_date_boundary() -> None:
    samples = synthetic_samples()
    train, test, cutoff = chronological_ratio_split(samples, train_ratio=0.8)
    assert cutoff == date(2025, 1, 8)
    assert len(train) == 16
    assert len(test) == 4
    assert all(sample.first_actual_sale_date <= cutoff for sample in train)
    assert all(sample.first_actual_sale_date > cutoff for sample in test)


def test_ratio_split_never_splits_a_large_same_day_group() -> None:
    samples = [
        SkuSample(date(2025, 1, 1), "EARLY"),
        *(
            SkuSample(date(2025, 1, 2), f"GROUP{index:02d}")
            for index in range(8)
        ),
        SkuSample(date(2025, 1, 3), "LATE"),
    ]
    train, test, cutoff = chronological_ratio_split(samples, train_ratio=0.8)
    assert cutoff == date(2025, 1, 2)
    assert len(train) == 9
    assert len(test) == 1
    assert {sample.first_actual_sale_date for sample in train}.isdisjoint(
        {sample.first_actual_sale_date for sample in test}
    )


def test_expanding_fold_count_is_not_hard_coded() -> None:
    blocks = build_time_blocks(synthetic_samples(), n_time_blocks=5)
    folds = expanding_window_folds(blocks)
    assert len(blocks) == 5
    assert len(folds) == 4
    for block in blocks:
        dates = {sample.first_actual_sale_date for sample in block}
        for other in blocks:
            if other is block:
                continue
            assert dates.isdisjoint(
                {sample.first_actual_sale_date for sample in other}
            )
