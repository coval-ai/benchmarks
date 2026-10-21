# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the coval-bench test suite."""

from __future__ import annotations

import pytest
import structlog

from coval_bench.config import Settings


def pytest_configure(config: pytest.Config) -> None:
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


@pytest.fixture(autouse=False)
def override_settings(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Settings:
    """Return a Settings instance with safe test defaults.

    Patches ``get_settings`` so that any code calling it during a test
    receives the test settings rather than reading real env vars.
    """
    test_settings = Settings(
        _env_file=None,
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        log_level="DEBUG",
    )
    monkeypatch.setattr("coval_bench.config.get_settings", lambda: test_settings)
    return test_settings
