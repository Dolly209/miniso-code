"""Static store covariate enrichment of reusable standardized weekly artifacts."""
from __future__ import annotations

import pandas as pd

from data.domain import require_columns, normalize_sku_column


STORE_MODEL_ATTRIBUTES = ("city_level", "city_name", "secbiz_area_name", "area_class1_name", "operate_size", "meter_sum")


def enrich_weekly_store_attributes(weekly: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Replace old all-missing placeholders only from the audited store snapshot.

    No sales/inventory files are read, exposure totals are unchanged, and the
    reusable weekly checkpoint is not edited in place. The fixed snapshot is the
    same date used by the existing source adapter, not a newer October snapshot.
    """
    require_columns(weekly, ("store_code",), "weekly model inputs")
    require_columns(stores, ("store_code",), "fixed store snapshot")
    available = [name for name in STORE_MODEL_ATTRIBUTES if name in stores]
    dimension = normalize_sku_column(stores[["store_code", *available]], column="store_code")
    if dimension["store_code"].duplicated().any():
        raise ValueError("store model attributes are not unique by store_code")
    result = weekly.drop(columns=[name for name in available if name in weekly]).merge(
        dimension, on="store_code", how="left", validate="many_to_one", sort=False,
    )
    return result
