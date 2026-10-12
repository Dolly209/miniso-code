from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence


class DuplicateGrainError(ValueError):
    """Raised before a join that would multiply measure rows."""


def assert_unique_grain(
    rows: Iterable[Mapping[str, object]], *, key_fields: Sequence[str]
) -> None:
    seen: set[tuple[object, ...]] = set()
    duplicates: set[tuple[object, ...]] = set()
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        examples = sorted((repr(key) for key in duplicates))[:5]
        raise DuplicateGrainError(
            f"dimension is not unique at grain {tuple(key_fields)}; examples={examples}"
        )


def aggregate_additive_measures(
    rows: Iterable[Mapping[str, object]],
    *,
    key_fields: Sequence[str],
    measure_fields: Sequence[str],
) -> list[dict[str, object]]:
    totals: dict[tuple[object, ...], dict[str, float]] = defaultdict(
        lambda: {field: 0.0 for field in measure_fields}
    )
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        for field in measure_fields:
            value = row.get(field)
            if value is not None:
                totals[key][field] += float(value)
    result = []
    for key in sorted(totals, key=repr):
        output = dict(zip(key_fields, key))
        output.update(totals[key])
        result.append(output)
    return result

