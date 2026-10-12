"""Compatibility exports for the canonical cohort data adapter."""

from data.cohort import (
    CohortBuildResult,
    CohortBuildStats,
    CohortRow,
    build_first_order_cohort,
)

__all__ = [
    "CohortBuildResult",
    "CohortBuildStats",
    "CohortRow",
    "build_first_order_cohort",
]
