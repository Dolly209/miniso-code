from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol, Sequence


class DecisionType(str, Enum):
    FIRST_ORDER = "first_order"
    REORDER = "reorder"


class PointInTimeViolation(ValueError):
    """Raised when a record was not available at the decision cutoff."""


class PartitionSelectionError(ValueError):
    """Raised when no partition is eligible at the decision cutoff."""


@dataclass(frozen=True)
class PointInTimeContext:
    """Immutable boundary shared by all reads for one SKU decision.

    Visibility is governed by available_time, not event_time.  This interface does
    not map any physical source columns; that mapping is deferred until probe output
    confirms the real server schemas.
    """

    run_id: str
    sku_code: str
    decision_type: DecisionType
    decision_cutoff_time: datetime
    source_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be blank")
        if not self.sku_code.strip():
            raise ValueError("sku_code must not be blank")
        if self.decision_cutoff_time.tzinfo is None:
            raise ValueError("decision_cutoff_time must be timezone-aware")
        object.__setattr__(
            self, "source_versions", MappingProxyType(dict(self.source_versions))
        )

    def is_visible(self, *, available_time: datetime) -> bool:
        if available_time.tzinfo is None:
            raise ValueError("available_time must be timezone-aware")
        return available_time <= self.decision_cutoff_time

    def require_visible(
        self, *, event_time: datetime | None, available_time: datetime
    ) -> None:
        del event_time  # Event time is recorded for audit, never used to bypass availability.
        if not self.is_visible(available_time=available_time):
            raise PointInTimeViolation(
                f"record available at {available_time.isoformat()} is after cutoff "
                f"{self.decision_cutoff_time.isoformat()}"
            )

    def latest_partition_on_or_before(
        self, partitions: Iterable[date | datetime]
    ) -> date:
        cutoff_date = self.decision_cutoff_time.date()
        eligible: list[date] = []
        for value in partitions:
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    raise ValueError("datetime partition values must be timezone-aware")
                partition_date = value.astimezone(
                    self.decision_cutoff_time.tzinfo
                ).date()
            else:
                partition_date = value
            if partition_date <= cutoff_date:
                eligible.append(partition_date)
        if not eligible:
            raise PartitionSelectionError(
                "no ds partition is available on or before decision_cutoff_time"
            )
        return max(eligible)


class PointInTimeRepository(Protocol):
    """Read-only source repository contract to be implemented after ROUND 1."""

    def fetch_as_of(
        self,
        context: PointInTimeContext,
        *,
        source_name: str,
        columns: Sequence[str] | None = None,
    ) -> Iterable[Mapping[str, object]]:
        ...
