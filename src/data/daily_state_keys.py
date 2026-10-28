"""Safe daily snapshot grain normalization; snapshot quantities are not flows."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Sequence

import pandas as pd

from data.domestic import DataContractViolation
from data.domain import require_columns


class DailyStateConflict(DataContractViolation):
    def __init__(self, audit: dict[str, Any]):
        self.audit = audit
        super().__init__(
            f"{audit['source']} has conflicting daily snapshots after canonical normalization; "
            f"conflicting_keys={audit['conflicting_key_count']} "
            f"equivalent_extra_rows={audit['equivalent_extra_rows']} "
            f"examples={audit['conflict_examples']}"
        )


def collapse_equivalent_daily_states(
    frame: pd.DataFrame, *, source: str, state_columns: Sequence[str],
    flag_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge only identical normalized facts, never sum/max/choose conflicts.

    Callers first normalize IDs, dates, numeric quantities and source flag aliases.
    Unknown flag tokens stay distinguishable, rather than equating two unknowns
    merely because both became NA. Only duplicate-key rows need comparison work.
    """
    keys = ["sku_code", "store_code", "business_day"]
    require_columns(frame, (*keys, *state_columns), source)
    duplicate = frame.duplicated(keys, keep=False)
    audit: dict[str, Any] = dict(
        source=source, input_rows=len(frame), duplicate_key_rows=int(duplicate.sum()),
        duplicate_key_count=0, equivalent_extra_rows=0, output_rows=len(frame),
        conflicting_key_count=0, conflict_examples=[],
        policy="collapse_equal_canonical_snapshots_only; never_sum_quantities_or_pick_conflicts",
    )
    if not duplicate.any():
        return frame, audit
    repeated = frame.loc[duplicate]
    comparison = repeated[[*keys, *state_columns]].copy()
    for flag in flag_columns:
        require_columns(repeated, (flag + "_token", flag + "_unrecognized"), source)
        comparison[flag + "_unrecognized_token"] = repeated[flag + "_token"].where(
            repeated[flag + "_unrecognized"].fillna(False), None,
        )
    distinct = comparison.drop_duplicates()
    conflicting = distinct.duplicated(keys, keep=False)
    audit["duplicate_key_count"] = len(comparison[keys].drop_duplicates())
    audit["equivalent_extra_rows"] = len(comparison) - len(distinct)
    if conflicting.any():
        conflict_values = distinct.loc[conflicting]
        audit["conflicting_key_count"] = len(conflict_values[keys].drop_duplicates())
        audit["output_rows"] = None
        # Bounded examples: at most three keys, three distinct variants per key.
        samples = conflict_values[keys].drop_duplicates().head(3)
        variants = conflict_values.merge(samples, on=keys, how="inner", validate="many_to_one")
        variants = variants.groupby(keys, sort=False, dropna=False).head(3)
        audit["conflict_examples"] = [
            {key: value.isoformat() if isinstance(value, (date, datetime)) else value for key, value in row.items()}
            for row in variants.astype(object).where(variants.notna(), None).to_dict("records")
        ]
        raise DailyStateConflict(audit)
    # No source cache is modified. The returned state is unique before joins.
    output = frame.drop_duplicates(keys, keep="first").copy()
    audit["output_rows"] = len(output)
    return output, audit
