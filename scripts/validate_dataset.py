#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig, load_table_specs  # noqa: E402
from data.decision_adapter import build_repository  # noqa: E402
from data.validation import build_validation_report, write_validation_report  # noqa: E402


def _parse_cutoff(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cutoff must be ISO-8601, for example 2026-10-09T12:00:00+08:00"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("cutoff must include a UTC offset")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the ROUND 2 Parquet data layer with bounded, read-only scans."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, default=Path("artifacts/server_probe"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/data_validation")
    )
    parser.add_argument(
        "--cutoff-time", "--decision-cutoff-time", dest="cutoff_time",
        type=_parse_cutoff, required=True,
    )
    parser.add_argument("--limit-skus", type=int, default=None)
    parser.add_argument("--sku-code", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=100)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT, args.config)
    logger, log_path = setup_run_logger("validate_dataset", config.log_dir, config.timezone)
    logger.info(
        "starting ROUND 2 validation data_root=%s cutoff=%s output_dir=%s",
        args.data_root,
        args.cutoff_time.isoformat(),
        args.output_dir,
    )
    if args.dry_run:
        print("status: dry_run")
        print("stage: bounded data contract validation")
        print(f"cutoff: {args.cutoff_time.isoformat()}")
        print("sku filters: accepted for CLI consistency; schema validation remains table-level")
        print(f"log: {log_path}")
        return 0
    try:
        if not _inside(args.output_dir, PROJECT_ROOT):
            raise ValueError(f"output-dir must be inside project root {PROJECT_ROOT}")
        if _inside(args.output_dir, args.data_root):
            raise ValueError("output-dir must not be inside the read-only data root")
        if not _inside(args.probe_dir, PROJECT_ROOT):
            raise ValueError(f"probe-dir must be inside project root {PROJECT_ROOT}")
        specs = load_table_specs(config.probe.table_specs_path)
        repository = build_repository(data_root=args.data_root, config=config)
        report = build_validation_report(
            repository=repository,
            table_specs=specs,
            probe_dir=args.probe_dir,
            cutoff_time=args.cutoff_time,
            sample_rows=args.sample_rows,
        )
        report_path = args.output_dir / "validation_report.json"
        write_validation_report(report_path, report)
    except Exception as exc:
        log_exception(logger, module="validate_dataset", cutoff=args.cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print(f"log: {log_path}")
        return 1
    print(f"status: {report['status']}")
    print(f"report: {report_path}")
    print(f"log: {log_path}")
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
