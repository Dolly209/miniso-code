from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AvailabilityStatus(str, Enum):
    MISSING = "missing"
    PROXY = "proxy"
    MANUAL_CONFIG = "manual/config"
    NEUTRAL = "neutral"
    DISABLED = "disabled"


@dataclass(frozen=True)
class UnavailableFieldDecision:
    """Explicit decision for a V3.2 concept absent from the delivered fields."""

    concept: str
    status: AvailabilityStatus
    reason: str
    physical_field: str | None = None

    def __post_init__(self) -> None:
        if not self.concept.strip() or not self.reason.strip():
            raise ValueError("concept and reason must not be blank")
        if self.status is AvailabilityStatus.PROXY and not self.physical_field:
            raise ValueError("proxy decisions must name the physical proxy field")
        if self.status is not AvailabilityStatus.PROXY and self.physical_field:
            raise ValueError("physical_field is only valid for proxy decisions")


DELIVERED_SALES_SEMANTICS = (
    "normal sales rows only (is_return='0'); use sale_qty without subtracting refund_qty again"
)
INVENTORY_SEMANTICS = (
    "transit_stock_qty is derived within total_stock_qty; do not add it again"
)
INVENTORY_CENTER_SEMANTICS = (
    "batch/scenario reconciliation snapshot; never aggregate across ds as transaction flow"
)

