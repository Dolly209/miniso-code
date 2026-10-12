from __future__ import annotations

import pandas as pd

from data.normalization import (
    NONDISPLAY_FLAG_VOCABULARY,
    normalize_identifier,
    normalize_nullable_flag,
)


def test_flag_normalization_handles_full_width_and_business_aliases() -> None:
    values = pd.Series([" Ｙ ", "否", "有库存未陈列商品", "--", "new-value", None])

    normalized, tokens, unrecognized = normalize_nullable_flag(
        values, vocabulary=NONDISPLAY_FLAG_VOCABULARY
    )

    assert normalized.iloc[0] == 1
    assert normalized.iloc[1] == 0
    assert normalized.iloc[2] == 1
    assert normalized.iloc[3] == 0
    assert pd.isna(normalized.iloc[4])
    assert pd.isna(normalized.iloc[5])
    assert tokens.iloc[0] == "y"
    assert unrecognized.tolist() == [False, False, False, False, True, False]


def test_identifier_normalization_strips_without_changing_case() -> None:
    values = pd.Series([" 2028156310109 ", " Ｚ088 "])

    normalized = normalize_identifier(values, field_name="sku_code")

    assert normalized.tolist() == ["2028156310109", "Z088"]
