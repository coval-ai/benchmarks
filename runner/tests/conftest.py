# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the coval-bench test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import structlog

import coval_bench.scenarios as scenarios
from coval_bench.config import Settings
from tests.tool_fixtures import SYNTHETIC_SUITE, TOOL_DEFINITIONS_JSON

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


@pytest.fixture
def synthetic_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Supply an independent suite through the package's contract-file reader."""
    contract = tmp_path / SYNTHETIC_SUITE
    contract.mkdir()
    (contract / "tool-definitions.json").write_text(TOOL_DEFINITIONS_JSON)
    read_bytes = scenarios._read_bytes

    def read(*parts: str) -> bytes:
        if parts[0] == SYNTHETIC_SUITE:
            return contract.joinpath(*parts[1:]).read_bytes()
        return read_bytes(*parts)

    monkeypatch.setattr(scenarios, "_read_bytes", read)
    return contract
