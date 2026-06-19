# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-request logging for the API.

A pure-ASGI middleware that binds a ``request_id`` to structlog's contextvars
for the duration of each request — so every log line emitted while handling it
carries the id — and emits one canonical ``http_request`` line per request with
the method, path, status, and duration. It is pure ASGI rather than
``BaseHTTPMiddleware`` on purpose: the latter runs the endpoint in a separate
context, so contextvars bound in it would not reach the route handler.

Cloud Run's ``X-Cloud-Trace-Context`` trace id is used as the id when present so
logs line up with Cloud Trace; otherwise a random id is generated. The id is
echoed back in the ``X-Request-ID`` response header.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_log = structlog.get_logger("coval_bench.api.access")

_TRACE_HEADER = b"x-cloud-trace-context"

_QUIET_PATHS = frozenset({"/healthz", "/readyz"})


class RequestLoggingMiddleware:
    """Bind a per-request ``request_id`` and log one line per request."""

    def __init__(self, app: ASGIApp, *, quiet_paths: frozenset[str] = _QUIET_PATHS) -> None:
        self.app = app
        self.quiet_paths = quiet_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id(scope)
        method: str = scope["method"]
        path: str = scope["path"]
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append(
                    (b"x-request-id", request_id.encode("latin-1"))
                )
            await send(message)

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            _log.error(
                "http_request",
                method=method,
                path=path,
                status=500,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
                exc_info=True,
            )
            raise
        else:
            if not (path in self.quiet_paths and status < 400):
                emit = _log.error if status >= 500 else _log.warning if status >= 400 else _log.info
                emit(
                    "http_request",
                    method=method,
                    path=path,
                    status=status,
                    duration_ms=round((time.perf_counter() - start) * 1000, 1),
                )
        finally:
            structlog.contextvars.clear_contextvars()


def _request_id(scope: Scope) -> str:
    """Return the Cloud Trace id from the request headers, or a random id."""
    headers: Iterable[tuple[bytes, bytes]] = scope["headers"]
    for key, value in headers:
        if key == _TRACE_HEADER:
            trace = value.decode("latin-1").split("/", 1)[0]
            if trace:
                return trace
            break
    return uuid.uuid4().hex
