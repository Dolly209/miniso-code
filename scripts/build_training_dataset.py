#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig, load_table_specs  # noqa: E402
from data.cohort import build_first_order_cohort  # noqa: E402
from data.decision_adapter import build_repository  # noqa: E402
from data.validation import validate_probe_evidence  # noqa: E402
from training.round3 import build_round3_report, write_round3_artifacts  # noqa: E402


def _parse_cutoff(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cutoff must be ISO-8601, for example 2026-09-30T23:59:59+08:00"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("cutoff must include a UTC offset")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and audit the V3.2 ROUND 3 SKU cohort from the delivered "
            "normal-sales Parquet source."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--probe-dir", type=Path, default=Path("artifacts/server_probe_round2")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/round3_dataset")
    )
    parser.add_argument("--cutoff-time", type=_parse_cutoff, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Arrow record-batch size; defaults to config training value.",
    )
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
    logger, log_path = setup_run_logger(
        "build_training_dataset", config.log_dir, config.timezone
    )
    cache_path: Path | None = None
    logger.info(
        "starting ROUND 3 cohort build data_root=%s cutoff=%s output_dir=%s",
        args.data_root,
        args.cutoff_time.isoformat(),
        args.output_dir,
    )
    try:
        if not _inside(args.output_dir, PROJECT_ROOT):
            raise ValueError(f"output-dir must be inside project root {PROJECT_ROOT}")
        if _inside(args.output_dir, args.data_root):
            raise ValueError("output-dir must not be inside the read-only data root")
        if not _inside(args.probe_dir, PROJECT_ROOT):
            raise ValueError(f"probe-dir must be inside project root {PROJECT_ROOT}")
        batch_size = args.batch_size or config.training.cohort_scan_batch_size
        if batch_size <= 0:
            raise ValueError("batch-size must be positive")

        specs = load_table_specs(config.probe.table_specs_path)
        evidence = validate_probe_evidence(
            probe_dir=args.probe_dir, table_specs=specs
        )
        if evidence["status"] != "ok":
            raise ValueError(
                "ROUND 2 probe evidence failed: " + "; ".join(evidence["errors"])
            )
        for warning in evidence["warnings"]:
            logger.warning("probe_evidence=%s", warning)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        file_descriptor, raw_cache_path = tempfile.mkstemp(
            prefix=".round3_sales_", suffix=".sqlite", dir=args.output_dir
        )
        os.close(file_descriptor)
        cache_path = Path(raw_cache_path)
        repository = build_repository(data_root=args.data_root, config=config)

        def report_progress(index: int, total: int, partition: object) -> None:
            logger.info(
                "sales_partition_progress starting=%s total=%s ds=%s",
                index,
                total,
                partition,
            )

        cohort = build_first_order_cohort(
            repository=repository,
            cutoff_time=args.cutoff_time,
            cache_path=cache_path,
            window_weeks=config.training.label_distribution_window_weeks,
            batch_size=batch_size,
            progress=report_progress,
        )
        report, rows = build_round3_report(
            cohort,
            train_ratio=config.training.train_ratio,
            n_time_blocks=config.training.n_time_blocks,
            sku_equal_share=config.training.long_tail_sku_equal_weight,
            band_equal_share=config.training.long_tail_sales_band_equal_weight,
        )
        report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["decision_cutoff_time"] = args.cutoff_time.isoformat()
        report["probe_evidence"] = {
            "status": evidence["status"],
            "warnings": evidence["warnings"],
        }
        paths = write_round3_artifacts(
            output_dir=args.output_dir, report=report, rows=rows
        )
        logger.info(
            "ROUND 3 complete status=%s dataset_version=%s cohort=%s train=%s test=%s",
            report["status"],
            report["dataset_version"],
            report["split_summary"]["cohort_count"],
            report["split_summary"]["train_count"],
            report["split_summary"]["test_count"],
        )
    except Exception as exc:
        log_exception(
            logger,
            module="build_training_dataset",
            cutoff=args.cutoff_time.isoformat(),
            exc=exc,
        )
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print(f"log: {log_path}")
        return 1
    finally:
        if cache_path is not None:
            cache_path.unlink(missing_ok=True)

    print(f"status: {report['status']}")
    print(
        "counts: "
        f"cohort={report['split_summary']['cohort_count']} "
        f"train={report['split_summary']['train_count']} "
        f"test={report['split_summary']['test_count']}"
    )
    print(
        "split: "
        f"requested_train_ratio={report['split_summary']['requested_train_ratio']:.4f} "
        f"actual_train_ratio={report['split_summary']['actual_train_ratio']:.4f} "
        f"cutoff={report['split_summary']['selected_cutoff_date']}"
    )
    print(f"dataset_version: {report['dataset_version']}")
    print(f"report: {paths['report']}")
    print(f"cohort: {paths['cohort']}")
    print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
