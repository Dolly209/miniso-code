from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_cutoff(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("decision cutoff must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("decision cutoff must include a UTC offset")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def add_execution_arguments(
    parser: argparse.ArgumentParser,
    *,
    cutoff_required: bool = True,
    default_output: str,
) -> None:
    parser.add_argument("--limit-skus", type=positive_int, default=None)
    parser.add_argument("--sku-code", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--decision-cutoff-time",
        type=parse_cutoff,
        required=cutoff_required,
    )
    parser.add_argument("--output-dir", type=Path, default=Path(default_output))
    parser.add_argument("--algorithm-config", type=Path, default=Path("config/algorithm_v3_2.json"))


def dry_run_payload(stage: str, args: argparse.Namespace, outputs: list[str]) -> dict[str, Any]:
    return {
        "status": "dry_run",
        "stage": stage,
        "executes_other_stages": False,
        "decision_cutoff_time": (
            args.decision_cutoff_time.isoformat()
            if getattr(args, "decision_cutoff_time", None) is not None else None
        ),
        "sku_codes": list(getattr(args, "sku_code", [])),
        "limit_skus": getattr(args, "limit_skus", None),
        "planned_outputs": outputs,
    }


def print_dry_run(stage: str, args: argparse.Namespace, outputs: list[str]) -> None:
    print(json.dumps(dry_run_payload(stage, args, outputs), ensure_ascii=False, indent=2))
