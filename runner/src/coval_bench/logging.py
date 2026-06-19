# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Structlog JSON configuration helper.

Call ``configure_logging()`` once at process startup (e.g. in ``__main__.py``
or the FastAPI lifespan) before emitting any log records.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
) -> None:
    """Configure structlog for JSON output to stdout.

    Args:
        level: The minimum log level to emit.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Route third-party stdlib loggers (uvicorn, google-cloud, ...) to stdout at
    # the same level. These emit plain text, not JSON — only loggers obtained via
    # structlog.get_logger() go through the JSON pipeline above.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
