from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pyarrow as pa

from data.blindbox_audit import BlindBoxAuditDataAdapter, classify_blindbox_variants
from data.decision_adapter import SOURCES


def test_blindbox_variants_separate_direct_tokens_and_inferred_children() -> None:
    products = pd.DataFrame(
        [
            {
                "sku_code": "P", "sku_name": "YOYO指环爱神系列盲盒手办",
                "spu_code": "SPU1", "spu_name": "YOYO指环爱神系列盲盒手办",
            },
            {
                "sku_code": "E", "sku_name": "YOYO指环爱神系列盲盒手办-端盒",
                "spu_code": "SPU1", "spu_name": "YOYO指环爱神系列盲盒手办",
            },
            {
                "sku_code": "H", "sku_name": "YOYO指环爱神系列盲盒手办（隐藏款）",
                "spu_code": "SPU1", "spu_name": "YOYO指环爱神系列盲盒手办",
            },
            {
                "sku_code": "C", "sku_name": "YOYO指环爱神系列盲盒手办（誓言）",
                "spu_code": "SPU1", "spu_name": "YOYO指环爱神系列盲盒手办",
            },
            {
                "sku_code": "X", "sku_name": "普通毛绒挂件",
                "spu_code": "SPU2", "spu_name": "普通毛绒挂件",
            },
        ]
    )

    result = classify_blindbox_variants(products).variants.set_index("sku_code")

    assert set(result.index) == {"E", "H", "C"}
    assert result.loc["E", "variant_type"] == "end_box"
    assert result.loc["H", "variant_type"] == "hidden_variant"
    assert result.loc["C", "variant_type"] == "child_variant"
    assert result.loc["C", "classification_confidence"] == "inferred_strong_link"
    assert set(result["parent_sku_code"]) == {"P"}


def test_placeholder_spu_does_not_create_cross_family_relationship() -> None:
    products = pd.DataFrame(
        [
            {
                "sku_code": "P1", "sku_name": "系列甲手办盲盒",
                "spu_code": "00000000000000000000", "spu_name": "无",
            },
            {
                "sku_code": "P2", "sku_name": "系列乙手办盲盒",
                "spu_code": "00000000000000000000", "spu_name": "无",
            },
            {
                "sku_code": "E", "sku_name": "完全不同系列手办盲盒-端盒",
                "spu_code": "00000000000000000000", "spu_name": "无",
            },
        ]
    )

    result = classify_blindbox_variants(products).variants.set_index("sku_code")

    assert result.loc["E", "spu_code"] == ""
    assert result.loc["E", "spu_resolution_source"] == "missing"
    assert pd.isna(result.loc["E", "parent_sku_code"])
    assert result.loc["E", "family_key"] == "完全不同系列"


def test_observed_spu_id_precedes_name_fallback() -> None:
    products = pd.DataFrame(
        [
            {
                "sku_code": "P", "sku_name": "右右酱指环爱神系列手办盲盒",
                "spu_code": "SPU1", "spu_name": "右右酱指环爱神系列手办盲盒",
            },
            {
                "sku_code": "E", "sku_name": "YOYO指环爱神系列手办盲盒-端盒",
                "spu_code": "00000000000000000000", "spu_name": "无",
            },
        ]
    )

    result = classify_blindbox_variants(
        products, sku_spu_overrides={"E": "SPU1"}
    ).variants.set_index("sku_code")

    assert result.loc["E", "parent_sku_code"] == "P"
    assert result.loc["E", "parent_match_method"] == "shared_observed_spu_id"
    assert result.loc["E", "spu_resolution_source"] == "observed_source_spu_id"


def test_ambiguous_name_parent_is_not_forced() -> None:
    products = pd.DataFrame(
        [
            {
                "sku_code": "P1", "sku_name": "系列甲手办盲盒",
                "spu_code": "SPU1", "spu_name": "系列甲手办盲盒",
            },
            {
                "sku_code": "P2", "sku_name": "系列甲手办盲盒",
                "spu_code": "SPU2", "spu_name": "系列甲手办盲盒",
            },
            {
                "sku_code": "E", "sku_name": "系列甲手办盲盒-端盒",
                "spu_code": "0", "spu_name": "无",
            },
        ]
    )

    result = classify_blindbox_variants(products).variants.set_index("sku_code")

    assert pd.isna(result.loc["E", "parent_sku_code"])


class _SpuRepository:
    def iter_partition_batches(self, *, source_name: str, **_: object):
        if source_name == SOURCES["sales"]:
            frame = pd.DataFrame(
                [
                    {"ds": "2026-01-02", "sku_code": "A", "spu_id": "0"},
                    {"ds": "2026-01-03", "sku_code": "A", "spu_id": "SPU1"},
                    {"ds": "2026-01-03", "sku_code": "B", "spu_id": "SPU2"},
                ]
            )
        else:
            frame = pd.DataFrame(
                [
                    {
                        "ds": "2026-01-03", "stock_date": "2026-01-03",
                        "sku_code": "A", "spu_id": "SPU1",
                    },
                    {
                        "ds": "2026-01-03", "stock_date": "2026-01-03",
                        "sku_code": "B", "spu_id": "SPU3",
                    },
                ]
            )
        yield from pa.Table.from_pandas(frame, preserve_index=False).to_batches()


def test_source_spu_resolution_requires_one_consistent_mother_code() -> None:
    entities = pd.DataFrame(
        {
            "sku_code": ["A", "B"],
            "window_start": [date(2026, 1, 1), date(2026, 1, 1)],
            "window_end": [date(2026, 1, 31), date(2026, 1, 31)],
        }
    )
    adapter = BlindBoxAuditDataAdapter(_SpuRepository())

    result = adapter.resolve_spu_ids(
        entities=entities,
        cutoff_time=datetime(2026, 1, 31, tzinfo=timezone.utc),
    ).set_index("sku_code")

    assert result.loc["A", "resolved_observed_spu_id"] == "SPU1"
    assert result.loc["A", "observed_spu_resolution_status"] == "unique_observed_spu_id"
    assert result.loc["B", "resolved_observed_spu_id"] == ""
    assert result.loc["B", "observed_spu_candidate_count"] == 2
    assert result.loc["B", "observed_spu_resolution_status"] == "conflicting_observed_spu_ids"
