from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from data.domestic import DataContractViolation
from data.standard_ip import (
    build_server_ip_mapping,
    load_standard_ip_alias_config,
    load_standard_ip_mapping,
    path1_ip_membership,
    validate_resolved_standard_ip_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def _mapping_rows() -> pd.DataFrame:
    common = {
        "sku_name": "",
        "standard_ip_level2": "",
        "standard_ip_name": "",
        "text_ip_candidate": "",
        "evidence_field": "",
        "evidence_text": "",
        "copyright_name": "",
    }
    return pd.DataFrame(
        [
            {
                **common,
                "sku_code": "A",
                "standard_ip_level1": "IP-A",
                "standard_ip_code": "IP-A",
                "standard_ip_name": "IP-A",
                "mapping_method": "excel_exact",
                "mapping_confidence": "1.0",
                "path1_eligible": "1",
                "copyright_name": "IP-A-名创版权",
            },
            {
                **common,
                "sku_code": "A",
                "standard_ip_level1": "IP-B",
                "standard_ip_code": "IP-B",
                "standard_ip_name": "IP-B",
                "mapping_method": "excel_exact",
                "mapping_confidence": "1.0",
                "path1_eligible": "1",
            },
            {
                **common,
                "sku_code": "B",
                "standard_ip_level1": "IP-C",
                "standard_ip_code": "IP-C",
                "standard_ip_name": "IP-C",
                "mapping_method": "text_inferred_reference_alias",
                "mapping_confidence": "0.95",
                "path1_eligible": "1",
                "text_ip_candidate": "IP-C",
            },
            {
                **common,
                "sku_code": "C",
                "standard_ip_level1": "",
                "standard_ip_code": "",
                "mapping_method": "text_candidate_needs_review",
                "mapping_confidence": "0.5",
                "path1_eligible": "0",
                "text_ip_candidate": "POSSIBLE-C",
            },
        ]
    )


def test_load_and_path1_membership_preserve_multi_ip(tmp_path: Path) -> None:
    path = tmp_path / "mapping.csv"
    _mapping_rows().to_csv(path, index=False)
    mapping = load_standard_ip_mapping(path)
    membership = path1_ip_membership(mapping, product_skus=["A", "B", "C"])
    assert set(map(tuple, membership[["sku_code", "ip_code"]].to_numpy())) == {
        ("A", "IP-A"),
        ("A", "IP-B"),
        ("B", "IP-C"),
    }
    assert "C" not in set(membership["sku_code"])


def test_server_mapping_audit_distinguishes_sources(tmp_path: Path) -> None:
    path = tmp_path / "mapping.csv"
    _mapping_rows().to_csv(path, index=False)
    mapping = load_standard_ip_mapping(path)
    products = pd.DataFrame(
        [
            {"sku_code": "A", "sku_name": "Alpha"},
            {"sku_code": "B", "sku_name": "Beta"},
            {"sku_code": "C", "sku_name": "Candidate"},
            {"sku_code": "D", "sku_name": "Missing"},
        ]
    )
    raw = pd.DataFrame(
        [
            {"sku_code": "A", "ip_code": "ip000057"},
            {"sku_code": "B", "ip_code": ""},
            {"sku_code": "C", "ip_code": "ip000057"},
            {"sku_code": "D", "ip_code": "ip000034"},
        ]
    )
    result = build_server_ip_mapping(
        products=products, raw_ip_membership=raw, mapping=mapping
    )
    status = result.audit.set_index("sku_code")["mapping_status"].to_dict()
    assert status == {
        "A": "mapped_excel",
        "B": "mapped_text_inferred",
        "C": "candidate_needs_review",
        "D": "not_in_reference",
    }
    assert result.audit.set_index("sku_code").loc["A", "standard_ip_count"] == 2


def test_server_mapping_infers_unreferenced_sku_from_unique_copyright(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mapping.csv"
    _mapping_rows().to_csv(path, index=False)
    mapping = load_standard_ip_mapping(path)
    products = pd.DataFrame(
        [
            {
                "sku_code": "NEW",
                "sku_name": "New product",
                "spu_code": "00000000000000000000",
                "spu_name": "无",
                "copyright_name": "IP-A-名创版权",
            }
        ]
    )
    raw = pd.DataFrame([{"sku_code": "NEW", "ip_code": "ip000057"}])
    result = build_server_ip_mapping(
        products=products, raw_ip_membership=raw, mapping=mapping
    )
    eligible = result.mapping.loc[result.mapping["path1_eligible"].eq(1)]
    assert eligible[["sku_code", "standard_ip_code"]].to_records(index=False).tolist() == [
        ("NEW", "IP-A")
    ]
    assert eligible.iloc[0]["mapping_method"] == "server_inferred_taxonomy_text"
    assert result.audit.iloc[0]["mapping_status"] == "mapped_text_inferred"


def test_text_alias_does_not_scan_broad_copyright_labels(tmp_path: Path) -> None:
    path = tmp_path / "mapping.csv"
    rows = _mapping_rows()
    broad_category = rows.iloc[0].copy()
    broad_category["sku_code"] = "CATEGORY"
    broad_category["sku_name"] = "ABAO花间协奏曲系列手办盲盒"
    broad_category["standard_ip_level1"] = "工作室艺术家"
    broad_category["standard_ip_code"] = "工作室艺术家"
    broad_category["standard_ip_name"] = "工作室艺术家"
    broad_category["copyright_name"] = "工作室艺术家-ABAO-名创版权"
    rows = pd.concat([rows, broad_category.to_frame().T], ignore_index=True)
    rows.to_csv(path, index=False)
    mapping = load_standard_ip_mapping(path)
    products = pd.DataFrame(
        [
            {
                "sku_code": "NEW",
                "sku_name": "IP-A新品端盒",
                "spu_code": "0",
                "spu_name": "无",
                "copyright_name": "工作室艺术家- IP-A-名创版权",
            }
        ]
    )
    raw = pd.DataFrame([{"sku_code": "NEW", "ip_code": "ip000057"}])

    result = build_server_ip_mapping(
        products=products, raw_ip_membership=raw, mapping=mapping
    )

    assert set(result.mapping["standard_ip_code"]) == {"IP-A"}


def test_checked_in_reference_mapping_is_valid() -> None:
    mapping = load_standard_ip_mapping(
        ROOT / "reference" / "sku_standard_ip_mapping.csv"
    )
    assert mapping["sku_code"].nunique() == 2114
    counts = mapping["mapping_method"].value_counts().to_dict()
    assert counts == {
        "excel_exact": 2097,
        "text_candidate_needs_review": 7,
        "text_inferred_reference_alias": 5,
        "unmapped": 5,
    }


def test_statistics_taxonomy_maps_mickey_and_sanrio_children() -> None:
    mapping = load_standard_ip_mapping(
        ROOT / "reference" / "sku_standard_ip_mapping.csv"
    )
    alias_config = load_standard_ip_alias_config(
        ROOT / "config" / "standard_ip_aliases.json"
    )
    products = pd.DataFrame(
        [
            {
                "sku_code": "MICKEY",
                "sku_name": "迪士尼米奇欢乐派对系列手办盲盒-端盒",
                "spu_code": "0",
                "spu_name": "无",
                "copyright_name": "迪士尼-名创版权",
            },
            {
                "sku_code": "MELODY",
                "sku_name": "Sanrio characters花园系列手办-美乐蒂",
                "spu_code": "0",
                "spu_name": "无",
                "copyright_name": "三丽鸥-名创版权",
            },
        ]
    )
    raw = pd.DataFrame(
        {"sku_code": ["MICKEY", "MELODY"], "ip_code": ["ip000057", "ip000057"]}
    )

    result = build_server_ip_mapping(
        products=products,
        raw_ip_membership=raw,
        mapping=mapping,
        alias_config=alias_config,
    )
    codes = result.mapping.groupby("sku_code")["standard_ip_code"].apply(set).to_dict()
    assert codes == {
        "MICKEY": {"迪士尼::米奇和朋友们"},
        "MELODY": {"三丽鸥::美乐蒂"},
    }


def test_statistics_taxonomy_collapses_synonyms_and_rejects_placeholder_spu() -> None:
    mapping = load_standard_ip_mapping(
        ROOT / "reference" / "sku_standard_ip_mapping.csv"
    )
    alias_config = load_standard_ip_alias_config(
        ROOT / "config" / "standard_ip_aliases.json"
    )
    products = pd.DataFrame(
        [
            {
                "sku_code": "WAKUKU",
                "sku_name": "WAKUKU哇库库-胖达限定吊卡",
                "spu_code": "0",
                "spu_name": "无",
                "copyright_name": "",
            },
            {
                "sku_code": "STRAWBERRY",
                "sku_name": "草莓熊系列心花怒放盲盒挂件-端盒",
                "spu_code": "1",
                "spu_name": "无",
                "copyright_name": "",
            },
            {
                "sku_code": "TOM_JERRY",
                "sku_name": "Tom and Jerry 经典MOMENT",
                "spu_code": "0",
                "spu_name": "无",
                "copyright_name": "TOM and JERRY-12509-供应商版权",
            },
        ]
    )
    raw = pd.DataFrame(
        {
            "sku_code": ["WAKUKU", "STRAWBERRY", "TOM_JERRY"],
            "ip_code": ["", "", ""],
        }
    )

    result = build_server_ip_mapping(
        products=products,
        raw_ip_membership=raw,
        mapping=mapping,
        alias_config=alias_config,
    )
    codes = result.mapping.groupby("sku_code")["standard_ip_code"].apply(set).to_dict()
    assert codes == {
        "WAKUKU": {"WAKUKU"},
        "STRAWBERRY": {"草莓熊"},
        "TOM_JERRY": {"猫和老鼠"},
    }


def test_unique_copyright_resolves_conflicting_spu_text() -> None:
    mapping = load_standard_ip_mapping(
        ROOT / "reference" / "sku_standard_ip_mapping.csv"
    )
    alias_config = load_standard_ip_alias_config(
        ROOT / "config" / "standard_ip_aliases.json"
    )
    products = pd.DataFrame(
        [
            {
                "sku_code": "SNOOPY",
                "sku_name": "史努比旅途拍档系列手办盲盒-舞动时刻",
                "spu_code": "25643731014293000020",
                "spu_name": "迪士尼小熊维尼宅家时刻系列手办盲盒",
                "copyright_name": "史努比-名创版权",
            }
        ]
    )
    raw = pd.DataFrame({"sku_code": ["SNOOPY"], "ip_code": [""]})

    result = build_server_ip_mapping(
        products=products,
        raw_ip_membership=raw,
        mapping=mapping,
        alias_config=alias_config,
    )

    assert set(result.mapping["standard_ip_code"]) == {"史努比"}


def test_duplicate_statistics_child_and_parent_defaults_use_parent_context() -> None:
    mapping = load_standard_ip_mapping(
        ROOT / "reference" / "sku_standard_ip_mapping.csv"
    )
    alias_config = load_standard_ip_alias_config(
        ROOT / "config" / "standard_ip_aliases.json"
    )
    products = pd.DataFrame(
        [
            {
                "sku_code": "DISNEY_TOY_STORY",
                "sku_name": "玩具总动员系列音乐盒-三眼仔",
                "spu_code": "0",
                "spu_name": "无",
                "copyright_name": "迪士尼-名创版权",
            },
            {
                "sku_code": "PIXAR_TOY_STORY",
                "sku_name": "迪士尼皮克斯玩具总动员玩具人生系列手办盲盒-端盒",
                "spu_code": "0",
                "spu_name": "无",
                "copyright_name": "迪士尼-名创版权",
            },
            {
                "sku_code": "PIXAR_GENERIC",
                "sku_name": "迪士尼皮克斯系列光栅卡片盲盒-端盒",
                "spu_code": "0",
                "spu_name": "无",
                "copyright_name": "迪士尼-名创版权",
            },
        ]
    )
    raw = pd.DataFrame(
        {"sku_code": products["sku_code"], "ip_code": ["", "", ""]}
    )

    result = build_server_ip_mapping(
        products=products,
        raw_ip_membership=raw,
        mapping=mapping,
        alias_config=alias_config,
    )

    codes = result.mapping.groupby("sku_code")["standard_ip_code"].apply(set).to_dict()
    assert codes == {
        "DISNEY_TOY_STORY": {"迪士尼::玩具总动员"},
        "PIXAR_TOY_STORY": {"皮克斯::玩具总动员"},
        "PIXAR_GENERIC": {"皮克斯::皮克斯综合"},
    }


def _resolved_snapshot_fixture() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    version = "v3.2-2026-09-30-test"
    mapping = _mapping_rows().copy()
    mapping["path1_eligible"] = pd.to_numeric(mapping["path1_eligible"])
    mapping["dataset_version"] = version
    mapping["mapping_status"] = mapping["sku_code"].map(
        {"A": "mapped_excel", "B": "mapped_text_inferred", "C": "unmapped_reference"}
    )
    cohort = pd.DataFrame(
        {
            "sku_code": ["A", "B", "C"],
            "dataset_version": [version, version, version],
        }
    )
    report: dict[str, object] = {
        "spec_version": "3.2",
        "stage": "standard IP mapping",
        "status": "implemented_output",
        "decision_cutoff_time": "2026-09-30T23:59:59+08:00",
        "source_product_partition": "2026-09-30",
        "cohort_skus": 3,
        "mapped_cohort_skus": 2,
    }
    return mapping, cohort, report


def test_resolved_snapshot_matches_cohort_and_cutoff() -> None:
    mapping, cohort, report = _resolved_snapshot_fixture()

    version = validate_resolved_standard_ip_snapshot(
        mapping,
        cohort=cohort,
        report=report,
        decision_cutoff_time=datetime.fromisoformat("2026-09-30T23:59:59+08:00"),
    )

    assert version == "v3.2-2026-09-30-test"


def test_resolved_snapshot_rejects_stale_version_cutoff_and_sku_universe() -> None:
    mapping, cohort, report = _resolved_snapshot_fixture()
    cutoff = datetime.fromisoformat("2026-09-30T23:59:59+08:00")

    stale_mapping = mapping.copy()
    stale_mapping["dataset_version"] = "v3.2-2026-09-29-stale"
    with pytest.raises(DataContractViolation, match="dataset_version"):
        validate_resolved_standard_ip_snapshot(
            stale_mapping, cohort=cohort, report=report, decision_cutoff_time=cutoff
        )

    stale_report = dict(report)
    stale_report["decision_cutoff_time"] = "2026-09-29T23:59:59+08:00"
    with pytest.raises(DataContractViolation, match="cutoff"):
        validate_resolved_standard_ip_snapshot(
            mapping, cohort=cohort, report=stale_report, decision_cutoff_time=cutoff
        )

    incomplete_mapping = mapping.loc[mapping["sku_code"].ne("C")]
    with pytest.raises(DataContractViolation, match="SKU universe"):
        validate_resolved_standard_ip_snapshot(
            incomplete_mapping, cohort=cohort, report=report, decision_cutoff_time=cutoff
        )
