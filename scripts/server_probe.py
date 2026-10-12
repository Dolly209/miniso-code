#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.server_probe import run_server_probe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only probe of MINISO server data and MySQL connectivity."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-skus", type=int, default=None)
    parser.add_argument("--sku-code", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT, args.config)
    logger, log_path = setup_run_logger("server_probe", config.log_dir, config.timezone)
    logger.info(
        "starting read-only probe data_root=%s output_dir=%s log=%s",
        args.data_root,
        args.output_dir,
        log_path,
    )
    if args.dry_run:
        print("status: dry_run")
        print("stage: read-only server probe")
        print("sku filters: accepted for CLI consistency; table/schema probe remains table-level")
        print(f"log: {log_path}")
        return 0
    try:
        try:
            args.output_dir.resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"output-dir must be inside project root {PROJECT_ROOT}"
            ) from exc
        outputs = run_server_probe(
            data_root=args.data_root,
            output_dir=args.output_dir,
            config=config,
            logger=logger,
        )
    except Exception as exc:
        log_exception(logger, module="server_probe", exc=exc)
        return 1
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
