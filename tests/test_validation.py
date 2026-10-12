from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import TableSpec
from data.parquet_repository import ParquetDataRepository
from data.validation import build_validation_report


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_round2_validation_uses_probe_evidence_and_bounded_read(tmp_path: Path) -> None:
    source_name = "scm.synthetic"
    spec = TableSpec(
        table_name=source_name,
        partition_column="ds",
        required_fields=("ds", "sku_code"),
        read_mode="latest_snapshot",
        business_date_column="ds",
        grain_columns=("ds", "sku_code"),
        sku_column="sku_code",
    )
    partition = tmp_path / "data" / source_name / "ds=2025-01-01"
    partition.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"sku_code": "SKU1"}]), partition / "part-00000"
    )
    probe = tmp_path / "probe"
    probe.mkdir()
    base = {"spec_version": "3.2"}
    _write_json(
        probe / "tables.json",
        {**base, "tables": [{"table_name": source_name, "exists": True, "detected_format": "parquet"}]},
    )
    _write_json(
        probe / "schema_report.json",
        {**base, "tables": [{"table_name": source_name, "missing_required_fields": [], "errors": []}]},
    )
    _write_json(
        probe / "partition_report.json",
        {**base, "tables": [{"table_name": source_name, "max_ds": "2025-01-01"}]},
    )
    _write_json(
        probe / "data_quality_summary.json",
        {
            **base,
            "tables": [
                {
                    "table_name": source_name,
                    "status": "ok",
                    "critical_field_missing_rate": {"sku_code": 0.0},
                }
            ],
        },
    )
    repository = ParquetDataRepository(
        data_root=tmp_path / "data", table_specs=[spec]
    )

    report = build_validation_report(
        repository=repository,
        table_specs=[spec],
        probe_dir=probe,
        cutoff_time=datetime.fromisoformat("2025-01-02T12:00:00+08:00"),
        sample_rows=10,
    )

    assert report["status"] == "ok"
    assert report["source_reads"][0]["selected_partition"] == "2025-01-01"
    assert report["source_reads"][0]["sample_rows"] == 1
    assert report["source_reads"][0]["bounded_sample"] is True
