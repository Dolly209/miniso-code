from __future__ import annotations

import logging
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def setup_run_logger(
    module: str, log_dir: Path, timezone: str = "Asia/Shanghai"
) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(ZoneInfo(timezone)).strftime("%Y%m%d_%H%M%S_%f")
    path = log_dir / f"run_{timestamp}.log"
    logger = logging.getLogger(f"miniso_v3_2.{module}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s module=%(name)s %(message)s"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger, path


def log_exception(
    logger: logging.Logger,
    *,
    module: str,
    exc: BaseException,
    table: str | None = None,
    sku: str | None = None,
    cutoff: str | None = None,
) -> None:
    logger.error(
        "module=%s table=%s sku=%s cutoff=%s exception_type=%s "
        "exception_message=%r traceback=%r",
        module,
        table,
        sku,
        cutoff,
        type(exc).__name__,
        str(exc),
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )
