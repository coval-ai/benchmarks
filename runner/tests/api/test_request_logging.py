# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the request-logging middleware's level policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from coval_bench.api import request_logging
from coval_bench.api.request_logging import RequestLoggingMiddleware


class _RecordingLogger:
    """Records (level, event, kwargs) for every log call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str) -> Callable[..., None]:
        def _log(event: str, **kwargs: Any) -> None:
            self.calls.append((level, event, kwargs))

        return _log

    def http_requests(self) -> list[tuple[str, int]]:
        """Return (level, status) for each http_request line logged."""
        return [
            (level, kwargs["status"])
            for level, event, kwargs in self.calls
            if event == "http_request"
        ]


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _RecordingLogger:
    rec = _RecordingLogger()
    monkeypatch.setattr(request_logging, "logger", rec)
    return rec


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/ok")
    async def ok() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/fail")
    async def fail() -> JSONResponse:
        return JSONResponse({"detail": "boom"}, status_code=500)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.add_middleware(RequestLoggingMiddleware)
    return app


async def _get(path: str) -> None:
    transport = ASGITransport(app=_make_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get(path)


async def test_success_logs_debug(recorder: _RecordingLogger) -> None:
    """2xx responses log at debug."""
    await _get("/ok")
    assert recorder.http_requests() == [("debug", 200)]


async def test_client_error_logs_info(recorder: _RecordingLogger) -> None:
    """4xx responses are expected client traffic and log at info, not warning."""
    await _get("/missing")
    assert recorder.http_requests() == [("info", 404)]


async def test_server_error_logs_error(recorder: _RecordingLogger) -> None:
    """5xx responses log at error."""
    await _get("/fail")
    assert recorder.http_requests() == [("error", 500)]


async def test_quiet_path_success_not_logged(recorder: _RecordingLogger) -> None:
    """Successful health probes are not logged at all."""
    await _get("/healthz")
    assert recorder.http_requests() == []
