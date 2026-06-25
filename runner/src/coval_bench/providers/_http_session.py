# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared httpx clients with timed-transport diagnostics.

HTTP providers grab a module-level ``httpx.AsyncClient`` keyed by provider.
The client uses a long ``keepalive_expiry`` so the pool survives between
dataset items; once the first request opens a connection, every subsequent
request on that pool reuses it (no DNS, no TCP, no TLS).

``TimedTransport`` stamps each request with ``__t_submit`` / ``__t_headers``
in ``request.extensions``. ``submit_to_headers_ms`` reads those stamps so
providers can record the interval on every result row: a small, stable value
confirms the pool stayed warm; a spike means the connection reconnected and
TTFA reabsorbed connect time for that row.

Lifecycle is managed by the orchestrator: ``close_all()`` runs in the
``finally`` of ``run_benchmarks``.
"""

from __future__ import annotations

import time
from typing import Final

import httpx
import structlog

logger = structlog.get_logger(__name__)

_KEEPALIVE_S: Final[float] = 300.0
_TIMEOUT_S: Final[float] = 30.0
_MAX_KEEPALIVE_CONNECTIONS: Final[int] = 8
_MAX_CONNECTIONS: Final[int] = 8

_CLIENTS: dict[str, httpx.AsyncClient] = {}


class TimedTransport(httpx.AsyncHTTPTransport):
    """httpx transport that timestamps each request for warmth diagnostics.

    The interval ``__t_headers - __t_submit`` contains connect time (if the
    pool was cold) plus server-side processing. After the orchestrator's
    one-time warmup primes the pool, this interval should be small and
    stable; large or growing values indicate the pool is reconnecting
    mid-run.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        opened_connection = False

        async def _trace(name: str, info: dict[str, object]) -> None:
            nonlocal opened_connection
            if "connect_tcp" in name:
                opened_connection = True

        request.extensions["trace"] = _trace
        t_submit = time.monotonic()
        response = await super().handle_async_request(request)
        t_headers = time.monotonic()
        request.extensions["__t_submit"] = t_submit
        request.extensions["__t_headers"] = t_headers
        request.extensions["__connection_reused"] = not opened_connection
        logger.debug(
            "http_request_timing",
            host=request.url.host,
            method=request.method,
            submit_to_headers_ms=round((t_headers - t_submit) * 1000, 1),
            connection_reused=not opened_connection,
        )
        return response


def submit_to_headers_ms(request: httpx.Request) -> float | None:
    t_submit = request.extensions.get("__t_submit")
    t_headers = request.extensions.get("__t_headers")
    if t_submit is None or t_headers is None:
        return None
    return round((float(t_headers) - float(t_submit)) * 1000, 1)


def connection_reused(request: httpx.Request) -> bool | None:
    reused = request.extensions.get("__connection_reused")
    if reused is None:
        return None
    return bool(reused)


def get_shared_client(provider_key: str, base_url: str) -> httpx.AsyncClient:
    """Return the module-level ``httpx.AsyncClient`` for *provider_key*.

    Lazy-initialised on first call. Subsequent calls return the same instance
    so connection-pool state is preserved across providers' lifetimes.

    Raises ``ValueError`` if called again with the same *provider_key* but a
    different *base_url* — the cached client is pinned to the first host.
    """
    existing = _CLIENTS.get(provider_key)
    if existing is not None:
        if str(existing.base_url).rstrip("/") != base_url.rstrip("/"):
            raise ValueError(
                f"shared client {provider_key!r} already bound to "
                f"{existing.base_url!r}, cannot rebind to {base_url!r}"
            )
        return existing

    _CLIENTS[provider_key] = httpx.AsyncClient(
        base_url=base_url,
        transport=TimedTransport(
            retries=0,
            http2=True,
            limits=httpx.Limits(
                max_connections=_MAX_CONNECTIONS,
                max_keepalive_connections=_MAX_KEEPALIVE_CONNECTIONS,
                keepalive_expiry=_KEEPALIVE_S,
            ),
        ),
        timeout=_TIMEOUT_S,
    )
    return _CLIENTS[provider_key]


async def close_all() -> None:
    """Close every shared client. Idempotent; safe to call in a finally block."""
    for key, client in list(_CLIENTS.items()):
        try:
            await client.aclose()
        except Exception as exc:  # noqa: BLE001 — best-effort teardown
            logger.warning("http_client_close_failed", provider=key, exc_info=exc)
    _CLIENTS.clear()
