"""Consistent console and file logging for pipeline commands."""

from __future__ import annotations

import logging
from pathlib import Path
import sys


FORMAT = "%(asctime)sZ | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(
    *,
    name: str = "precipitation",
    log_file: Path | str | None = None,
    verbose: bool = False,
) -> logging.Logger:
    """Configure an idempotent logger with UTC timestamps."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(FORMAT)
    formatter.converter = __import__("time").gmtime

    if not any(getattr(handler, "_pipeline_console", False) for handler in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.DEBUG if verbose else logging.INFO)
        console.setFormatter(formatter)
        console._pipeline_console = True
        logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve()
        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == resolved
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger
