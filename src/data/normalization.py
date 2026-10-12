from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


def _canonical_token(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    token = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return token or None


@dataclass(frozen=True)
class FlagVocabulary:
    """Source-specific aliases for a nullable business flag."""

    true_values: frozenset[str]
    false_values: frozenset[str]
    missing_values: frozenset[str] = frozenset()

    @classmethod
    def from_values(
        cls,
        *,
        true_values: Iterable[object],
        false_values: Iterable[object],
        missing_values: Iterable[object] = (),
    ) -> "FlagVocabulary":
        true = frozenset(_canonical_token(value) for value in true_values)
        false = frozenset(_canonical_token(value) for value in false_values)
        missing = frozenset(_canonical_token(value) for value in missing_values)
        if None in true or None in false:
            raise ValueError("flag true/false aliases must not be empty")
        true_tokens = frozenset(value for value in true if value is not None)
        false_tokens = frozenset(value for value in false if value is not None)
        missing_tokens = frozenset(value for value in missing if value is not None)
        overlap = (true_tokens & false_tokens) | (true_tokens & missing_tokens) | (
            false_tokens & missing_tokens
        )
        if overlap:
            raise ValueError(f"flag aliases overlap: {sorted(overlap)}")
        return cls(
            true_values=true_tokens,
            false_values=false_tokens,
            missing_values=missing_tokens,
        )


COMMON_TRUE_FLAG_VALUES = (1, 1.0, "true", "t", "y", "yes", "是")
COMMON_FALSE_FLAG_VALUES = (0, 0.0, "false", "f", "n", "no", "否")
COMMON_MISSING_FLAG_VALUES = ("", "na", "n/a", "null", "none", "nan", "未知", "unknown")


STOCKOUT_FLAG_VOCABULARY = FlagVocabulary.from_values(
    true_values=(*COMMON_TRUE_FLAG_VALUES, "缺货", "缺货商品"),
    false_values=(*COMMON_FALSE_FLAG_VALUES, "有货", "非缺货", "非缺货商品"),
    # A dash is not assumed to mean in-stock for the stockout field. Quantity
    # evidence can still establish the state independently.
    missing_values=(*COMMON_MISSING_FLAG_VALUES, "-", "--"),
)


NONDISPLAY_FLAG_VOCABULARY = FlagVocabulary.from_values(
    true_values=(
        *COMMON_TRUE_FLAG_VALUES,
        "有库存未陈列商品",
        "有库存未陈列",
        "未陈列",
    ),
    false_values=(
        *COMMON_FALSE_FLAG_VALUES,
        "-",
        "--",
        "已陈列",
        "正常陈列",
        "无未陈列",
    ),
    missing_values=COMMON_MISSING_FLAG_VALUES,
)


def normalize_identifier(values: pd.Series, *, field_name: str) -> pd.Series:
    """Normalize join identifiers without changing their business case."""

    normalized = values.map(
        lambda value: (
            None
            if value is None or pd.isna(value)
            else unicodedata.normalize("NFKC", str(value)).strip()
        )
    )
    invalid = normalized.isna() | normalized.eq("")
    if invalid.any():
        raise ValueError(f"{field_name} contains null or blank identifiers")
    return normalized.astype(str)


def normalize_nullable_flag(
    values: pd.Series,
    *,
    vocabulary: FlagVocabulary,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return canonical 1/0/NA values, tokens, and non-null unmapped rows.

    Unexpected values deliberately remain NA. This prevents a new source label
    from silently becoming either business truth or business falsehood.
    """

    tokens = values.map(_canonical_token)
    normalized = pd.Series(pd.NA, index=values.index, dtype="Float64")
    normalized.loc[tokens.isin(vocabulary.true_values)] = 1.0
    normalized.loc[tokens.isin(vocabulary.false_values)] = 0.0
    known_missing = tokens.isna() | tokens.isin(vocabulary.missing_values)
    unrecognized = ~(known_missing | tokens.isin(vocabulary.true_values) | tokens.isin(vocabulary.false_values))
    return normalized, tokens, unrecognized.astype(bool)


def summarize_flag_normalization(
    *,
    raw_values: pd.Series,
    normalized: pd.Series,
    unrecognized: pd.Series,
    max_values: int = 20,
) -> dict[str, object]:
    tokens = raw_values.map(_canonical_token).fillna("__MISSING__")
    counts = tokens.value_counts(dropna=False).head(max_values)
    bad_counts = tokens.loc[unrecognized].value_counts(dropna=False).head(max_values)
    return {
        "rows": int(len(raw_values)),
        "true_rows": int((normalized == 1).sum()),
        "false_rows": int((normalized == 0).sum()),
        "missing_rows": int(normalized.isna().sum() - unrecognized.sum()),
        "unrecognized_rows": int(unrecognized.sum()),
        "raw_value_counts": {str(key): int(value) for key, value in counts.items()},
        "unrecognized_value_counts": {
            str(key): int(value) for key, value in bad_counts.items()
        },
    }
