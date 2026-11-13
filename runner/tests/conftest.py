# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the coval-bench test suite."""

from __future__ import annotations

from typing import Any

import pytest
import structlog

from coval_bench.config import Settings

_structlog_configure = structlog.configure


def _configure_uncached(*args: Any, **kwargs: Any) -> None:
    kwargs["cache_logger_on_first_use"] = False
    _structlog_configure(*args, **kwargs)


def pytest_configure(config: pytest.Config) -> None:
    structlog.configure = _configure_uncached

    # Before collection, so module-level Settings never read a developer's .env either.
    Settings.model_config["env_file"] = None

    # The default renderer pretty-prints exc_info tracebacks through Rich, which
    # costs a quarter second or more per logged exception.
    processors = structlog.get_config()["processors"]
    structlog.configure(
        processors=[
            *processors[:-1],
            structlog.dev.ConsoleRenderer(
                colors=False, exception_formatter=structlog.dev.plain_traceback
            ),
        ]
    )
