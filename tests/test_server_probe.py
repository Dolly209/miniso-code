from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from config.settings import AppConfig, TableSpec
from data.server_probe import profile_table, run_server_probe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "data"
DB_ENV_NAMES = (
    "MINISO_DB_HOST",
    "MINISO_DB_PORT",
    "MINISO_DB_USER",
    "MINISO_DB_PASSWORD",
    "MINISO_DB_NAME",
)


def digest_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_probe_is_read_only_and_writes_all_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in DB_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    before = digest_tree(FIXTURE_ROOT)
    output = tmp_path / "probe"
    config = AppConfig.load(PROJECT_ROOT)
    logger = logging.getLogger("synthetic-probe-test")
    paths = run_server_probe(
        data_root=FIXTURE_ROOT,
        output_dir=output,
        config=config,
        logger=logger,
    )
    assert set(paths) == {
        "environment",
        "tables",
        "schema_report",
        "partition_report",
        "data_quality_summary",
        "mysql_report",
        "probe_summary",
    }
    assert all(path.is_file() for path in paths.values())
    assert digest_tree(FIXTURE_ROOT) == before

    tables = json.loads(paths["tables"].read_text(encoding="utf-8"))["tables"]
    assert len(tables) == 10
    goods = next(
        item for item in tables if item["table_name"] == "scm.dim_goods_info_ip_forecast"
    )
    assert goods["detected_format"] == "csv"
    assert goods["partition_count"] == 2

    schema = json.loads(paths["schema_report"].read_text(encoding="utf-8"))["tables"]
    goods_schema = next(
        item for item in schema if item["table_name"] == "scm.dim_goods_info_ip_forecast"
    )
    assert "retail_price" in goods_schema["missing_required_fields"]
    missing_row = next(
        item for item in goods_schema["columns"] if item["column_name"] == "retail_price"
    )
    assert missing_row["required_by_v3_2"] is True
    assert missing_row["status"] == "missing"

    quality = json.loads(
        paths["data_quality_summary"].read_text(encoding="utf-8")
    )["tables"]
    sales = next(
        item
        for item in quality
        if item["table_name"] == "scm.dwd_trd_order_all_flow_di_ip_forecast"
    )
    assert sales["key_duplicate_check"]["duplicate_row_count"] == 2
    goods_quality = next(
        item
        for item in quality
        if item["table_name"] == "scm.dim_goods_info_ip_forecast"
    )
    assert goods_quality["potential_multi_ip_duplication"]["groups_with_multiple_ip"] == 1

    environment = json.loads(paths["environment"].read_text(encoding="utf-8"))
    assert environment["spec_version"] == "3.2"
    assert "cpu" in environment
    assert "memory" in environment


def test_probe_refuses_to_write_inside_data_root(tmp_path: Path) -> None:
    config = AppConfig.load(PROJECT_ROOT)
    with pytest.raises(ValueError, match="must not be inside"):
        run_server_probe(
            data_root=FIXTURE_ROOT,
            output_dir=FIXTURE_ROOT / "artifacts",
            config=config,
            logger=logging.getLogger("synthetic-probe-safety-test"),
        )


def test_quality_probe_skips_empty_part_files_within_attempt_bound(tmp_path: Path) -> None:
    table_path = tmp_path / "source"
    latest = table_path / "ds=2025-01-02" / "part-001.csv"
    older = table_path / "ds=2025-01-01" / "part-001.csv"
    latest.parent.mkdir(parents=True)
    older.parent.mkdir(parents=True)
    latest.write_text("sku_code,value\n", encoding="utf-8")
    older.write_text("sku_code,value\nSKU1,1\n", encoding="utf-8")
    spec = TableSpec(
        table_name="source",
        partition_column="ds",
        required_fields=("ds", "sku_code", "value"),
        sku_column="sku_code",
        key_columns=("ds", "sku_code"),
        date_columns=("ds",),
    )
    schema_columns = {
        name: {"column_name": name, "sample_null_rate": None}
        for name in spec.required_fields
    }

    report = profile_table(
        spec,
        table_path,
        [latest, older],
        schema_columns,
        sample_files_per_table=1,
        max_sample_files_to_attempt=2,
        sample_rows_per_file=10,
        max_sample_rows=10,
        logger=logging.getLogger("empty-file-probe-test"),
    )

    assert report["status"] == "ok"
    assert report["sample_rows"] == 1
    assert report["sampling"]["files_attempted"] == 2
    assert report["sampling"]["empty_files"] == ["ds=2025-01-02/part-001.csv"]
