# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for coval_bench.logging.

These lock the two things that bit us before: the JSON output format, and the
``event="RUN_FAILED"`` field that the ``benchmark_run_failure`` Cloud Logging
metric filters on. Output is captured from stdout and parsed as JSON rather than
via ``structlog.testing.capture_logs`` so the real renderer chain is exercised.
"""

from __future__ import annotations

import json
import logging
import logging.config
from collections.abc import Iterator
from typing import Any

import pytest
import structlog

from coval_bench.logging import configure_logging, uvicorn_log_config


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Restore structlog and stdlib logging defaults after each test — both are global."""
    yield
    structlog.reset_defaults()
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.setLevel(logging.NOTSET)
        if name:
            lg.propagate = True


def test_configure_logging_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO")
    structlog.get_logger("test").info("hello", run_id=7)

    record: dict[str, Any] = json.loads(capsys.readouterr().out.strip())
    assert record["event"] == "hello"
    assert record["level"] == "info"
    assert record["run_id"] == 7
    assert "timestamp" in record


def test_run_failed_event_matches_alert_filter(capsys: pytest.CaptureFixture[str]) -> None:
    # benchmark_run_failure filters on jsonPayload.event="RUN_FAILED" (see
    # benchmark-infra modules/alerting). The key name and value are a prod-alerting
    # contract — a rename here silently breaks the alert, so pin it.
    configure_logging(level="INFO")
    structlog.get_logger("coval_bench.runner").error("RUN_FAILED", error="boom")

    record: dict[str, Any] = json.loads(capsys.readouterr().out.strip())
    assert record["event"] == "RUN_FAILED"
    assert record["level"] == "error"


def test_level_filtering_drops_below_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="WARNING")
    log = structlog.get_logger("test")
    log.info("dropped")
    log.warning("kept")

    out = capsys.readouterr().out
    assert "dropped" not in out
    assert json.loads(out.strip())["event"] == "kept"


def test_uvicorn_config_silences_access_log() -> None:
    logging.config.dictConfig(uvicorn_log_config("INFO"))

    access = logging.getLogger("uvicorn.access")
    assert access.handlers == []
    assert access.propagate is False


def test_uvicorn_config_renders_error_log_as_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logging.config.dictConfig(uvicorn_log_config("INFO"))

    logging.getLogger("uvicorn.error").info("Application startup complete.")
    logging.getLogger("uvicorn.access").info('127.0.0.1 - "GET / HTTP/1.1" 200')

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 1  # access log produced nothing
    record: dict[str, Any] = json.loads(lines[0])
    assert record["event"] == "Application startup complete."
    assert record["level"] == "info"
