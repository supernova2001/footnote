"""Structured logging: JSON in production, readable lines in a terminal. Never logs secrets."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

import structlog


def configure_logging(*, stream: TextIO = sys.stdout, json: bool | None = None, level: int = logging.INFO) -> None:
    if json is None:
        json = not stream.isatty()
    logging.basicConfig(stream=stream, level=level, format="%(message)s")
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        cache_logger_on_first_use=True,
    )
