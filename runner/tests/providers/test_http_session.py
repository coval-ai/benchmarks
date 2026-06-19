# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared httpx client registry."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast

import pytest

from coval_bench.providers import _http_session


@pytest.fixture(autouse=True)
def _reset_clients() -> Generator[None, None, None]:
    _http_session._CLIENTS.clear()
    yield
    _http_session._CLIENTS.clear()


# ---------------------------------------------------------------------------
# get_shared_client — memoization
# ---------------------------------------------------------------------------


def test_get_shared_client_memoizes() -> None:
    a = _http_session.get_shared_client("foo", "https://example.com")
    b = _http_session.get_shared_client("foo", "https://example.com")
    assert a is b


def test_get_shared_client_distinct_per_key() -> None:
    a = _http_session.get_shared_client("foo", "https://example.com")
    b = _http_session.get_shared_client("bar", "https://example.org")
    assert a is not b
    assert {"foo", "bar"} <= set(_http_session._CLIENTS)


def test_get_shared_client_rejects_rebind_to_new_base_url() -> None:
    _http_session.get_shared_client("foo", "https://example.com")
    with pytest.raises(ValueError, match="already bound"):
        _http_session.get_shared_client("foo", "https://other.example.com")


def test_get_shared_client_uses_timed_transport() -> None:
    c = _http_session.get_shared_client("foo", "https://example.com")
    # The default transport wraps the chosen one; httpx exposes it directly.
    assert isinstance(c._transport, _http_session.TimedTransport)


def test_get_shared_client_applies_http2_and_limits() -> None:
    """http2/limits must live on the transport's pool, not the ignored client args."""
    transport = cast(Any, _http_session.get_shared_client("foo", "https://example.com")._transport)
    pool = transport._pool
    assert pool._http2 is True
    assert pool._keepalive_expiry == _http_session._KEEPALIVE_S
    assert pool._max_keepalive_connections == _http_session._MAX_KEEPALIVE_CONNECTIONS
    assert pool._max_connections == _http_session._MAX_CONNECTIONS


# ---------------------------------------------------------------------------
# submit_to_headers_ms — reads TimedTransport stamps
# ---------------------------------------------------------------------------


def test_submit_to_headers_ms_none_without_stamps() -> None:
    import httpx

    request = httpx.Request("GET", "https://example.com")
    assert _http_session.submit_to_headers_ms(request) is None


def test_submit_to_headers_ms_computes_interval() -> None:
    import httpx

    request = httpx.Request("GET", "https://example.com")
    request.extensions["__t_submit"] = 100.0
    request.extensions["__t_headers"] = 100.5
    assert _http_session.submit_to_headers_ms(request) == 500.0


# ---------------------------------------------------------------------------
# TimedTransport — stamps timing and connection-reuse on each request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timed_transport_flags_new_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request that opens a TCP connection is stamped not-reused."""
    import httpx

    async def fake_super(self: object, request: httpx.Request) -> httpx.Response:
        trace = request.extensions.get("trace")
        if trace is not None:
            await trace("connection.connect_tcp.started", {})
        return httpx.Response(200, request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_super)
    request = httpx.Request("GET", "https://example.com")
    await _http_session.TimedTransport().handle_async_request(request)

    assert _http_session.submit_to_headers_ms(request) is not None
    assert _http_session.connection_reused(request) is False


@pytest.mark.asyncio
async def test_timed_transport_flags_reused_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request with no connect event reused a pooled connection."""
    import httpx

    async def fake_super(self: object, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_super)
    request = httpx.Request("GET", "https://example.com")
    await _http_session.TimedTransport().handle_async_request(request)

    assert _http_session.connection_reused(request) is True


@pytest.mark.asyncio
async def test_timed_transport_trace_is_async_through_real_httpcore() -> None:
    """A real request exercises httpcore's async trace path.

    Regression guard: a sync ``trace`` callback raises ``TypeError`` here
    (httpcore rejects non-coroutine callbacks on the async path); an async one
    lets the request proceed to a genuine connection error.
    """
    import httpx

    transport = _http_session.TimedTransport()
    request = httpx.Request("GET", "http://127.0.0.1:1/")
    try:
        with pytest.raises(httpx.ConnectError):
            await transport.handle_async_request(request)
    finally:
        await transport.aclose()


def test_connection_reused_none_without_stamp() -> None:
    import httpx

    request = httpx.Request("GET", "https://example.com")
    assert _http_session.connection_reused(request) is None


# ---------------------------------------------------------------------------
# close_all — clears registry and aclose()s each client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_all_clears_registry() -> None:
    _http_session.get_shared_client("foo", "https://example.com")
    _http_session.get_shared_client("bar", "https://example.org")
    assert len(_http_session._CLIENTS) == 2

    await _http_session.close_all()
    assert _http_session._CLIENTS == {}


@pytest.mark.asyncio
async def test_close_all_is_idempotent() -> None:
    await _http_session.close_all()  # empty registry
    await _http_session.close_all()  # second call, still empty
    assert _http_session._CLIENTS == {}


@pytest.mark.asyncio
async def test_close_all_swallows_per_client_errors() -> None:
    class _BrokenClient:
        async def aclose(self) -> None:
            raise RuntimeError("synthetic close failure")

    _http_session._CLIENTS["broken"] = cast(Any, _BrokenClient())
    await _http_session.close_all()  # must not raise
    assert _http_session._CLIENTS == {}
