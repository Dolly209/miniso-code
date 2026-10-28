#!/usr/bin/env python3
"""Freeze approximately 20 Train + 5 Test targets without changing source split."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts._pipeline_common import parse_cutoff, positive_int  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from training.checkpoint_replay import normalize_replay_schedule  # noqa: E402
from training.pilot import freeze_development_pilot  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-train-skus", type=positive_int, default=20)
    parser.add_argument("--limit-test-skus", type=positive_int, default=5)
    parser.add_argument("--decision-cutoff-time", type=parse_cutoff, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _inside_project(path: Path) -> Path:
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("pilot inputs and outputs must be inside the project") from exc
    return resolved


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("freeze_prelaunch_pilot", config.log_dir, config.timezone)
    try:
        store = StageArtifactStore()
        schedule_path = _inside_project(args.schedule_file)
        output_dir = _inside_project(args.output_dir)
        schedule = normalize_replay_schedule(store.read_frame(schedule_path), available_through=args.decision_cutoff_time)
        result = freeze_development_pilot(schedule, train_skus=args.limit_train_skus, test_skus=args.limit_test_skus,
                                          n_time_blocks=config.training.n_time_blocks)
        report = dict(result.report, source_schedule=str(args.schedule_file),
                      decision_cutoff_time=args.decision_cutoff_time.isoformat(),
                      source_schedule_sha256=hashlib.sha256(schedule_path.read_bytes()).hexdigest())
        outputs = {"pilot_schedule.csv": result.schedule, "pilot_skus.csv": result.skus,
                   "train_time_blocks.csv": result.time_blocks}
        report_path = output_dir / "pilot_report.json"
        if args.dry_run:
            print("status: dry_run")
            print(f"pilot_version: {report['pilot_version']} train={report['train_skus']} test={report['test_skus']}")
            print(f"source_label_model_health: {report['label_source_model_health']}")
            print(f"output_dir: {output_dir}")
            return 0
        if report_path.is_file():
            previous = store.read_json(report_path)
            if previous.get("content_sha256") != report["content_sha256"]:
                raise ValueError("pilot directory is already frozen with different inputs/selection; use a new output directory")
            for name in outputs:
                path = output_dir / name
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != previous.get("output_sha256", {}).get(name):
                    raise ValueError("frozen pilot artifact is missing/modified; preserve it and use a new output directory")
            print(f"status: frozen_pilot_reused\npilot_version: {report['pilot_version']}\nreport: {report_path}")
            return 0
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("pilot output directory is not empty and has no complete freeze manifest; use a new directory")
        for name, frame in outputs.items():
            store.write_frame(output_dir / name, frame)
        report["output_sha256"] = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in outputs}
        store.write_json(report_path, report)
        logger.info("pilot fixed version=%s train=%s test=%s", report["pilot_version"], report["train_skus"], report["test_skus"])
        print(f"status: {report['status']}\nschedule: {output_dir / 'pilot_schedule.csv'}\nreport: {report_path}\nlog: {log_path}")
        return 0
    except Exception as exc:
        log_exception(logger, module="freeze_prelaunch_pilot", exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
