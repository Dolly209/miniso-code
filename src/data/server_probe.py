from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from config.settings import AppConfig, TableSpec, load_table_specs
from common.logging_utils import log_exception


PACKAGE_NAMES = (
    "pandas",
    "numpy",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "sqlalchemy",
    "pymysql",
    "statsmodels",
)
PARTITION_RE = re.compile(r"^([^=/]+)=(.*)$")
VERSION_PARTITION_NAMES = {
    "version",
    "data_version",
    "snapshot_version",
    "batch",
    "batch_id",
    "hour",
    "hh",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date, Path)):
        return str(value)
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_output_dir(data_root: Path, output_dir: Path) -> None:
    source = data_root.resolve()
    target = output_dir.resolve()
    try:
        target.relative_to(source)
    except ValueError:
        return
    raise ValueError(
        f"output-dir must not be inside the read-only data root: {target}"
    )


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _memory_info() -> dict[str, int | None]:
    total: int | None = None
    available: int | None = None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
        available = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return {"total_bytes": total, "available_bytes": available}


def collect_environment(data_root: Path) -> dict[str, Any]:
    versions = _package_versions()
    disk_path = data_root if data_root.exists() else data_root.parent
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    disk = shutil.disk_usage(disk_path)
    return {
        "generated_at_utc": _utc_now(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {
            "logical_count": os.cpu_count(),
            "processor": platform.processor() or None,
        },
        "memory": _memory_info(),
        "packages": versions,
        "missing_packages": sorted(name for name, value in versions.items() if value is None),
        "free_disk_space": {
            "checked_path": str(disk_path),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }


def _candidate_table_paths(data_root: Path, table_name: str) -> list[Path]:
    candidates = [data_root / table_name, data_root.joinpath(*table_name.split("."))]
    result: list[Path] = []
    for candidate in candidates:
        if candidate not in result:
            result.append(candidate)
    return result


def resolve_table_path(data_root: Path, table_name: str) -> tuple[Path, list[str]]:
    candidates = _candidate_table_paths(data_root, table_name)
    existing = [path for path in candidates if path.is_dir()]
    if existing:
        warnings = []
        if len(existing) > 1:
            warnings.append(
                "multiple candidate table directories exist: "
                + ", ".join(str(path) for path in existing)
            )
        return existing[0], warnings
    return candidates[0], []


def _detect_file_format(path: Path) -> str:
    suffix = path.suffix.lower()
    suffix_map = {
        ".parquet": "parquet",
        ".pq": "parquet",
        ".orc": "orc",
        ".csv": "csv",
        ".tsv": "tsv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".avro": "avro",
    }
    if suffix in suffix_map:
        return suffix_map[suffix]
    try:
        with path.open("rb") as stream:
            prefix = stream.read(4)
            if prefix == b"PAR1":
                return "parquet"
            if prefix[:3] == b"ORC":
                return "orc"
    except OSError:
        return "unreadable"
    return "unknown"


def _partition_items(path: Path, table_path: Path) -> dict[str, str]:
    try:
        relative = path.relative_to(table_path)
    except ValueError:
        return {}
    result: dict[str, str] = {}
    for part in relative.parts[:-1]:
        match = PARTITION_RE.match(part)
        if match:
            result[match.group(1)] = match.group(2)
    return result


def scan_table_files(
    table_path: Path, *, sample_file_limit: int
) -> tuple[dict[str, Any], list[Path], dict[str, set[str]], dict[str, set[str]]]:
    files: list[Path] = []
    file_count = 0
    total_bytes = 0
    formats: Counter[str] = Counter()
    partition_values: dict[str, set[str]] = defaultdict(set)
    partition_locations: dict[str, set[str]] = defaultdict(set)
    partition_tuples: set[tuple[tuple[str, str], ...]] = set()
    errors: list[str] = []
    if not table_path.is_dir():
        return (
            {
                "exists": False,
                "detected_format": None,
                "file_count": 0,
                "partition_columns": [],
                "partition_count": 0,
                "sample_files": [],
                "approx_size_bytes": 0,
                "format_counts": {},
                "scan_errors": [],
            },
            files,
            partition_values,
            partition_locations,
        )

    def record_walk_error(exc: OSError) -> None:
        errors.append(f"{exc.filename}: {type(exc).__name__}: {exc}")

    for root, directory_names, file_names in os.walk(
        table_path, followlinks=False, onerror=record_walk_error
    ):
        directory_names[:] = sorted(name for name in directory_names if not name.startswith("."))
        for file_name in sorted(file_names):
            if (
                file_name.startswith(".")
                or file_name in {"_SUCCESS", "_metadata", "_common_metadata"}
                or file_name.endswith((".crc", ".tmp"))
            ):
                continue
            path = Path(root) / file_name
            try:
                stat = path.stat()
            except OSError as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
            if not path.is_file():
                continue
            file_count += 1
            total_bytes += stat.st_size
            detected = _detect_file_format(path)
            formats[detected] += 1
            parts = _partition_items(path, table_path)
            if parts:
                partition_tuples.add(tuple(sorted(parts.items())))
                for key, value in parts.items():
                    partition_values[key].add(value)
                    partition_locations[f"{key}={value}"].add(str(path.parent))
            files.append(path)

    ordered = sorted(
        files,
        key=lambda item: (
            _partition_items(item, table_path).get("ds", ""),
            str(item),
        ),
        reverse=True,
    )
    nonzero_formats = [name for name, count in formats.items() if count]
    detected_format = None
    if len(nonzero_formats) == 1:
        detected_format = nonzero_formats[0]
    elif nonzero_formats:
        detected_format = "mixed"
    return (
        {
            "exists": True,
            "detected_format": detected_format,
            "file_count": file_count,
            "partition_columns": sorted(partition_values),
            "partition_count": len(partition_tuples),
            "sample_files": [
                str(path.relative_to(table_path))
                for path in ordered[:sample_file_limit]
            ],
            "approx_size_bytes": total_bytes,
            "format_counts": dict(sorted(formats.items())),
            "scan_errors": errors[:100],
        },
        ordered,
        partition_values,
        partition_locations,
    )


def _inspect_arrow_file(path: Path, detected_format: str) -> list[dict[str, Any]]:
    if detected_format == "parquet":
        import pyarrow.parquet as pq

        schema = pq.ParquetFile(path).schema_arrow
    elif detected_format == "orc":
        import pyarrow.orc as orc

        schema = orc.ORCFile(path).schema
    else:
        raise ValueError(f"not an Arrow metadata format: {detected_format}")
    return [
        {
            "column_name": field.name,
            "physical_type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def _inspect_text_file(path: Path, detected_format: str) -> list[dict[str, Any]]:
    import pandas as pd

    if detected_format in {"csv", "tsv"}:
        frame = pd.read_csv(path, sep="\t" if detected_format == "tsv" else ",", nrows=200)
    elif detected_format in {"json", "jsonl"}:
        frame = pd.read_json(path, lines=detected_format == "jsonl", nrows=200)
    else:
        raise ValueError(f"unsupported text format: {detected_format}")
    return [
        {
            "column_name": str(column),
            "physical_type": str(frame[column].dtype),
            "nullable": None,
        }
        for column in frame.columns
    ]


def inspect_schema(
    spec: TableSpec,
    table_path: Path,
    files: Sequence[Path],
    *,
    sample_files_per_table: int,
    logger: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    union: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    inspected_files: list[str] = []
    for path in files[:sample_files_per_table]:
        detected = _detect_file_format(path)
        try:
            if detected in {"parquet", "orc"}:
                columns = _inspect_arrow_file(path, detected)
            elif detected in {"csv", "tsv", "json", "jsonl"}:
                columns = _inspect_text_file(path, detected)
            else:
                errors.append({"file": str(path), "error": f"unsupported format: {detected}"})
                continue
            inspected_files.append(str(path.relative_to(table_path)))
            for column in columns:
                name = column["column_name"]
                existing = union.get(name)
                if existing and existing["physical_type"] != column["physical_type"]:
                    existing["physical_type"] = (
                        f"mixed<{existing['physical_type']},{column['physical_type']}>"
                    )
                    existing["nullable"] = None
                else:
                    union.setdefault(name, column)
        except Exception as exc:  # probe must continue across corrupt/unreadable files
            log_exception(logger, module="server_probe.schema", table=spec.table_name, exc=exc)
            errors.append(
                {"file": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )

    physical_columns = set(union)
    for path in files[:sample_files_per_table]:
        for partition_name in _partition_items(path, table_path):
            union.setdefault(
                partition_name,
                {
                    "column_name": partition_name,
                    "physical_type": "partition<string>",
                    "nullable": False,
                },
            )
    required = set(spec.required_fields)
    missing = sorted(required - set(union))
    rows: list[dict[str, Any]] = []
    for name in sorted(set(union) | required):
        base = dict(
            union.get(
                name,
                {"column_name": name, "physical_type": None, "nullable": None},
            )
        )
        base.update(
            {
                "partition_column": name == spec.partition_column,
                "sample_null_rate": None,
                "required_by_v3_2": name in required,
                "status": "missing" if name in missing else "present",
                "physical_in_file": name in physical_columns,
            }
        )
        rows.append(base)
    return (
        {
            "table_name": spec.table_name,
            "path": str(table_path),
            "status": "missing_required_fields" if missing else ("ok" if union else "unreadable"),
            "inspected_files": inspected_files,
            "columns": rows,
            "missing_required_fields": missing,
            "errors": errors,
        },
        {row["column_name"]: row for row in rows},
    )


def _read_sample(
    path: Path,
    detected_format: str,
    requested_columns: Sequence[str],
    limit: int,
) -> Any:
    import pandas as pd

    if detected_format == "parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        selected = [name for name in requested_columns if name in available]
        if not selected:
            return pd.DataFrame()
        batches = parquet.iter_batches(batch_size=limit, columns=selected)
        try:
            return next(batches).to_pandas().head(limit)
        except StopIteration:
            return pd.DataFrame(columns=selected)
    if detected_format == "orc":
        import pyarrow.orc as orc

        orc_file = orc.ORCFile(path)
        available = set(orc_file.schema.names)
        selected = [name for name in requested_columns if name in available]
        if not selected or orc_file.nstripes == 0:
            return pd.DataFrame(columns=selected)
        arrow_table = orc_file.read_stripe(0, columns=selected)
        return arrow_table.slice(0, limit).to_pandas()
    if detected_format in {"csv", "tsv"}:
        header = pd.read_csv(
            path, sep="\t" if detected_format == "tsv" else ",", nrows=0
        )
        selected = [name for name in requested_columns if name in header.columns]
        return pd.read_csv(
            path,
            sep="\t" if detected_format == "tsv" else ",",
            usecols=selected or None,
            nrows=limit,
        )
    if detected_format in {"json", "jsonl"}:
        frame = pd.read_json(path, lines=detected_format == "jsonl", nrows=limit)
        selected = [name for name in requested_columns if name in frame.columns]
        return frame[selected]
    raise ValueError(f"unsupported sample format: {detected_format}")


def profile_table(
    spec: TableSpec,
    table_path: Path,
    files: Sequence[Path],
    schema_columns: dict[str, dict[str, Any]],
    *,
    sample_files_per_table: int,
    max_sample_files_to_attempt: int,
    sample_rows_per_file: int,
    max_sample_rows: int,
    logger: Any,
) -> dict[str, Any]:
    requested = list(
        dict.fromkeys(
            list(spec.required_fields)
            + list(spec.key_columns)
            + list(spec.date_columns)
            + [name for name in (spec.sku_column, spec.store_column, spec.ip_column) if name]
        )
    )
    frames: list[Any] = []
    errors: list[dict[str, str]] = []
    attempted_files = 0
    empty_files: list[str] = []
    try:
        import pandas as pd
    except ImportError as exc:
        return {
            "table_name": spec.table_name,
            "status": "skipped_missing_dependency",
            "reason": str(exc),
            "sample_rows": 0,
        }

    for path in files[:max_sample_files_to_attempt]:
        if sum(len(frame) for frame in frames) >= max_sample_rows:
            break
        if len(frames) >= sample_files_per_table:
            break
        attempted_files += 1
        try:
            frame = _read_sample(
                path, _detect_file_format(path), requested, sample_rows_per_file
            )
            parts = _partition_items(path, table_path)
            for name, value in parts.items():
                if name not in frame.columns:
                    frame[name] = value
            if not frame.empty:
                frames.append(frame)
            else:
                empty_files.append(str(path.relative_to(table_path)))
        except Exception as exc:
            log_exception(logger, module="server_probe.quality", table=spec.table_name, exc=exc)
            errors.append(
                {"file": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
    if not frames:
        return {
            "table_name": spec.table_name,
            "status": "no_readable_sample",
            "sample_rows": 0,
            "sampling": {
                "files_attempted": attempted_files,
                "empty_files": empty_files[:20],
                "max_files_to_attempt": max_sample_files_to_attempt,
                "full_table_scan": False,
            },
            "errors": errors,
        }
    sample = pd.concat(frames, ignore_index=True).head(max_sample_rows)
    present_columns = set(sample.columns)
    sample_rows = len(sample)
    null_rates = {
        name: float(sample[name].isna().mean())
        for name in spec.required_fields
        if name in present_columns
    }
    for name, rate in null_rates.items():
        if name in schema_columns:
            schema_columns[name]["sample_null_rate"] = rate

    def distinct(column: str | None) -> dict[str, Any] | None:
        if not column or column not in present_columns:
            return None
        return {
            "column": column,
            "count": int(sample[column].nunique(dropna=True)),
            "method": "exact_within_bounded_sample",
        }

    key_columns = [column for column in spec.key_columns if column in present_columns]
    duplicate = None
    if key_columns:
        duplicated = sample.duplicated(subset=key_columns, keep=False)
        duplicate = {
            "columns": key_columns,
            "duplicate_row_count": int(duplicated.sum()),
            "duplicate_group_count": int(
                sample.loc[duplicated].groupby(key_columns, dropna=False).ngroups
            )
            if duplicated.any()
            else 0,
            "method": "bounded_sample",
        }
    ranges: dict[str, dict[str, Any]] = {}
    for column in spec.date_columns:
        if column not in present_columns:
            continue
        parsed = pd.to_datetime(sample[column], errors="coerce")
        valid = parsed.dropna()
        ranges[column] = {
            "min": valid.min().isoformat() if not valid.empty else None,
            "max": valid.max().isoformat() if not valid.empty else None,
            "parse_success_rate": float(parsed.notna().mean()),
        }

    multi_ip = None
    if spec.ip_column and spec.ip_column in present_columns and spec.sku_column in present_columns:
        group_columns = [spec.sku_column]
        if spec.partition_column in present_columns:
            group_columns.insert(0, spec.partition_column)
        counts = sample.groupby(group_columns, dropna=False)[spec.ip_column].nunique(dropna=True)
        multi = counts[counts > 1]
        multi_ip = {
            "group_columns": group_columns,
            "groups_with_multiple_ip": int(len(multi)),
            "max_ip_per_group": int(multi.max()) if not multi.empty else 0,
            "interpretation": "potential multi-IP rows; not automatically treated as duplicates",
            "method": "bounded_sample",
        }
    return {
        "table_name": spec.table_name,
        "status": "ok",
        "sample_rows": sample_rows,
        "sampling": {
            "files_read": len(frames),
            "files_attempted": attempted_files,
            "empty_files": empty_files[:20],
            "max_files_to_attempt": max_sample_files_to_attempt,
            "max_rows": max_sample_rows,
            "full_table_scan": False,
        },
        "sku_distinct": distinct(spec.sku_column),
        "store_distinct": distinct(spec.store_column),
        "date_ranges": ranges,
        "key_duplicate_check": duplicate,
        "critical_field_missing_rate": null_rates,
        "potential_multi_ip_duplication": multi_ip,
        "errors": errors,
    }


def _parse_ds(value: str) -> date | None:
    for pattern in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            pass
    return None


def build_partition_report(
    spec: TableSpec,
    partition_values: dict[str, set[str]],
    partition_locations: dict[str, set[str]],
    *,
    recent_limit: int,
) -> dict[str, Any]:
    values = sorted(partition_values.get(spec.partition_column, set()))
    parsed = [(value, _parse_ds(value)) for value in values]
    valid = [(value, parsed_date) for value, parsed_date in parsed if parsed_date]
    formats = sorted(
        {
            "YYYY-MM-DD" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
            else "YYYYMMDD" if re.fullmatch(r"\d{8}", value)
            else "YYYY/MM/DD" if re.fullmatch(r"\d{4}/\d{2}/\d{2}", value)
            else "other"
            for value in values
        }
    )
    duplicate_locations = {
        key: sorted(locations)
        for key, locations in partition_locations.items()
        if key.startswith(f"{spec.partition_column}=") and len(locations) > 1
    }
    version_columns = sorted(set(partition_values) & VERSION_PARTITION_NAMES)
    missing_dates: list[str] = []
    continuity_status = "unknown"
    if valid:
        valid_dates = sorted({parsed_date for _, parsed_date in valid})
        expected_days = (valid_dates[-1] - valid_dates[0]).days + 1
        continuity_status = "continuous" if expected_days == len(valid_dates) else "has_gaps"
        if expected_days <= 20_000:
            observed = set(valid_dates)
            cursor = valid_dates[0]
            while cursor <= valid_dates[-1] and len(missing_dates) < 100:
                if cursor not in observed:
                    missing_dates.append(cursor.isoformat())
                cursor = date.fromordinal(cursor.toordinal() + 1)
    multi_version_status = "not_detected"
    if duplicate_locations or version_columns:
        multi_version_status = "potential_multiple_versions"
    return {
        "table_name": spec.table_name,
        "partition_column": spec.partition_column,
        "min_ds": min((item[1] for item in valid), default=None),
        "max_ds": max((item[1] for item in valid), default=None),
        "partition_count": len(values),
        "recent_partitions": sorted(values, reverse=True)[:recent_limit],
        "detected_ds_formats": formats,
        "has_ds_equals_yyyy_mm_dd": "YYYY-MM-DD" in formats,
        "unparseable_partition_values": [value for value, parsed_date in parsed if not parsed_date][
            :100
        ],
        "continuity": {
            "status": continuity_status,
            "missing_date_count_at_least": len(missing_dates),
            "first_missing_dates": missing_dates,
        },
        "same_business_date_multiple_versions": {
            "status": multi_version_status,
            "version_like_partition_columns": version_columns,
            "duplicate_ds_partition_paths": duplicate_locations,
        },
    }


def probe_mysql(logger: Any | None = None) -> dict[str, Any]:
    required = (
        "MINISO_DB_HOST",
        "MINISO_DB_PORT",
        "MINISO_DB_USER",
        "MINISO_DB_PASSWORD",
        "MINISO_DB_NAME",
    )
    missing = [name for name in required if not os.environ.get(name)]
    requested_database = os.environ.get("MINISO_DB_NAME")
    if missing:
        return {
            "status": "skipped_missing_environment",
            "missing_environment_variables": missing,
            "requested_database": requested_database,
        }
    if requested_database != "miniso":
        return {
            "status": "rejected_invalid_database",
            "requested_database": requested_database,
            "error": "MINISO_DB_NAME must be miniso",
        }
    if os.environ["MINISO_DB_USER"].strip().lower() == "root":
        return {
            "status": "rejected_forbidden_user",
            "requested_database": requested_database,
            "error": "root is forbidden for the algorithm application",
        }
    try:
        import pymysql

        connection = pymysql.connect(
            host=os.environ["MINISO_DB_HOST"],
            port=int(os.environ["MINISO_DB_PORT"]),
            user=os.environ["MINISO_DB_USER"],
            password=os.environ["MINISO_DB_PASSWORD"],
            database=requested_database,
            charset="utf8mb4",
            autocommit=True,
            connect_timeout=10,
            read_timeout=30,
            cursorclass=pymysql.cursors.DictCursor,
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT VERSION() AS version")
                version = cursor.fetchone()["version"]
                cursor.execute("SELECT DATABASE() AS database_name")
                selected_database = cursor.fetchone()["database_name"]
                cursor.execute("SELECT 1 AS ok")
                select_one = cursor.fetchone()["ok"]
                cursor.execute(
                    "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA "
                    "WHERE SCHEMA_NAME = %s",
                    (requested_database,),
                )
                database_exists = cursor.fetchone() is not None
        finally:
            connection.close()
        return {
            "status": "ok",
            "connection": True,
            "server_version": version,
            "selected_database": selected_database,
            "select_1": select_one,
            "requested_database": requested_database,
            "database_exists": database_exists,
            "read_only_probe": True,
        }
    except Exception as exc:
        if logger is not None:
            log_exception(logger, module="server_probe.mysql", exc=exc)
        return {
            "status": "error",
            "connection": False,
            "requested_database": requested_database,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _summary_lines(
    table_reports: list[dict[str, Any]],
    schema_reports: list[dict[str, Any]],
    partition_reports: list[dict[str, Any]],
    quality_reports: list[dict[str, Any]],
    mysql_report: dict[str, Any],
    environment: dict[str, Any],
) -> list[str]:
    lines = ["MINISO V3.2 server probe summary", f"generated_at_utc={_utc_now()}", ""]
    for table, schema, partition, quality in zip(
        table_reports, schema_reports, partition_reports, quality_reports
    ):
        name = table["table_name"]
        if not table["exists"]:
            lines.append(f"[ERROR] {name}: table directory not found at {table['path']}")
            continue
        lines.append(
            f"[OK] {name}: format={table['detected_format']} files={table['file_count']} "
            f"partitions={table['partition_count']}"
        )
        if schema["missing_required_fields"]:
            lines.append(
                f"[WARNING] {name}: missing required fields: "
                + ", ".join(schema["missing_required_fields"])
            )
        if schema["errors"]:
            lines.append(f"[WARNING] {name}: schema inspection errors={len(schema['errors'])}")
        lines.append(
            f"[OK] {name}: ds min={partition['min_ds']} max={partition['max_ds']} "
            f"continuity={partition['continuity']['status']}"
        )
        if quality["status"] != "ok":
            lines.append(
                f"[WARNING] {name}: data quality sample status={quality['status']}"
            )
    missing_packages = environment["missing_packages"]
    if missing_packages:
        lines.append("[WARNING] missing packages: " + ", ".join(missing_packages))
    if mysql_report["status"] == "ok":
        lines.append(
            f"[OK] mysql reachable; version={mysql_report['server_version']} "
            f"database_exists={mysql_report['database_exists']}"
        )
    else:
        detail = mysql_report.get("error") or ", ".join(
            mysql_report.get("missing_environment_variables", [])
        )
        lines.append(f"[WARNING] mysql status={mysql_report['status']}: {detail}")
    return lines


def run_server_probe(
    *,
    data_root: Path,
    output_dir: Path,
    config: AppConfig,
    logger: Any,
) -> dict[str, Path]:
    _safe_output_dir(data_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = load_table_specs(config.probe.table_specs_path)
    environment = collect_environment(data_root)
    table_reports: list[dict[str, Any]] = []
    schema_reports: list[dict[str, Any]] = []
    partition_reports: list[dict[str, Any]] = []
    quality_reports: list[dict[str, Any]] = []

    for spec in specs:
        logger.info("probing table=%s", spec.table_name)
        table_path, path_warnings = resolve_table_path(data_root, spec.table_name)
        scan, files, partition_values, partition_locations = scan_table_files(
            table_path, sample_file_limit=config.probe.sample_file_limit
        )
        table_report = {
            "table_name": spec.table_name,
            "path": str(table_path),
            **scan,
            "path_warnings": path_warnings,
        }
        schema_report, schema_columns = inspect_schema(
            spec,
            table_path,
            files,
            sample_files_per_table=config.probe.sample_files_per_table,
            logger=logger,
        )
        partition_report = build_partition_report(
            spec,
            partition_values,
            partition_locations,
            recent_limit=config.probe.recent_partition_limit,
        )
        quality_report = profile_table(
            spec,
            table_path,
            files,
            schema_columns,
            sample_files_per_table=config.probe.sample_files_per_table,
            max_sample_files_to_attempt=config.probe.max_sample_files_to_attempt,
            sample_rows_per_file=config.probe.sample_rows_per_file,
            max_sample_rows=config.probe.max_sample_rows_per_table,
            logger=logger,
        )
        table_reports.append(table_report)
        schema_reports.append(schema_report)
        partition_reports.append(partition_report)
        quality_reports.append(quality_report)

    mysql_report = probe_mysql(logger)
    generated = _utc_now()
    environment["spec_version"] = config.spec_version
    environment["probe_scope"] = "read_only_bounded_sample"
    outputs = {
        "environment": output_dir / "environment.json",
        "tables": output_dir / "tables.json",
        "schema_report": output_dir / "schema_report.json",
        "partition_report": output_dir / "partition_report.json",
        "data_quality_summary": output_dir / "data_quality_summary.json",
        "mysql_report": output_dir / "mysql_report.json",
        "probe_summary": output_dir / "probe_summary.txt",
    }
    _write_json(outputs["environment"], environment)
    _write_json(
        outputs["tables"],
        {
            "generated_at_utc": generated,
            "spec_version": config.spec_version,
            "tables": table_reports,
        },
    )
    _write_json(
        outputs["schema_report"],
        {
            "generated_at_utc": generated,
            "spec_version": config.spec_version,
            "required_field_source": str(config.probe.table_specs_path),
            "tables": schema_reports,
        },
    )
    _write_json(
        outputs["partition_report"],
        {
            "generated_at_utc": generated,
            "spec_version": config.spec_version,
            "tables": partition_reports,
        },
    )
    _write_json(
        outputs["data_quality_summary"],
        {
            "generated_at_utc": generated,
            "spec_version": config.spec_version,
            "scope": "bounded samples only; no full data scan",
            "tables": quality_reports,
        },
    )
    _write_json(outputs["mysql_report"], mysql_report)
    outputs["probe_summary"].write_text(
        "\n".join(
            _summary_lines(
                table_reports,
                schema_reports,
                partition_reports,
                quality_reports,
                mysql_report,
                environment,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("probe complete output_dir=%s", output_dir)
    return outputs
