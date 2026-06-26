# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Structlog JSON configuration helper.

Call ``configure_logging()`` once at process startup (e.g. in ``__main__.py``
or the FastAPI lifespan) before emitting any log records.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog
from structlog.typing import EventDict, WrappedLogger

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def add_gcp_severity(_logger: WrappedLogger, _method_name: str, event_dict: EventDict) -> EventDict:
    """Copy ``level`` into a top-level ``severity`` Cloud Logging can filter on.

    Must run after ``add_log_level``; GCP expects DEBUG/INFO/WARNING/ERROR.
    """
    level = event_dict.get("level")
    if level is not None:
        event_dict["severity"] = level.upper()
    return event_dict


def configure_logging(
    level: LogLevel = "INFO",
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
            add_gcp_severity,
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

    # Route third-party stdlib loggers (google-cloud, httpx, ...) to stdout at
    # the same level. These emit plain text, not JSON — only loggers obtained via
    # structlog.get_logger() go through the JSON pipeline above. uvicorn's own
    # loggers are handled separately by uvicorn_log_config().
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(log_level, logging.WARNING))


def uvicorn_log_config(level: LogLevel = "INFO") -> dict[str, Any]:
    """Return a uvicorn ``log_config`` that renders uvicorn's logs as JSON.

    Pass to ``uvicorn.run(log_config=...)`` so uvicorn's startup and error logs
    match the structlog JSON the app emits on stdout, instead of uvicorn's default
    plain colored text. ``uvicorn.access`` is given no handler: the
    ``RequestLoggingMiddleware`` emits the canonical per-request line, so uvicorn's
    duplicate access log is suppressed (also pass ``access_log=False``).

    Args:
        level: The minimum log level for uvicorn's loggers.
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
                "foreign_pre_chain": [
                    structlog.processors.add_log_level,
                    add_gcp_severity,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.format_exc_info,
                ],
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": level, "propagate": False},
            "uvicorn.access": {"handlers": [], "level": level, "propagate": False},
        },
    }
