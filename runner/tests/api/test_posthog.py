# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""PostHog wiring for the API: lifespan client creation, the real get_posthog
dependency, and each router's event name and payload.

Uses ``dependency_overrides`` plus ``unittest.mock`` so no extra test
dependencies are needed and no real PostHog network calls are made. The
lifespan tests stub the DB pool, so they do not spin up Postgres.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import MagicMock, create_autospec

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from posthog import Posthog

from coval_bench.api.deps import get_posthog

AppFactory = Callable[[dict[str, str] | None], Awaitable[FastAPI]]


async def test_lifespan_builds_client_and_route_captures(
    app_factory: AppFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Token set: lifespan builds the client, stores it on app.state, the real
    get_posthog returns it, the route captures, and shutdown flushes."""
    fake = create_autospec(Posthog, instance=True)
    monkeypatch.setattr("coval_bench.api.app.Posthog", lambda *args, **kwargs: fake)
    app = await app_factory({"POSTHOG_PROJECT_TOKEN": "phc_test", "POSTHOG_DISABLED": "false"})

    async with LifespanManager(app):
        assert app.state.posthog is fake
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/providers")
        assert response.status_code == 200
        fake.capture.assert_called()
    fake.shutdown.assert_called()


async def test_disabled_builds_no_client(
    app_factory: AppFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Analytics disabled: no client is constructed and the route still succeeds."""
    sentinel = MagicMock()
    monkeypatch.setattr("coval_bench.api.app.Posthog", sentinel)
    app = await app_factory(None)

    async with LifespanManager(app):
        assert app.state.posthog is None
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/providers")
        assert response.status_code == 200
    sentinel.assert_not_called()


@pytest.mark.parametrize(
    ("path", "params", "event", "required_keys"),
    [
        ("/v1/providers", {}, "providers listed", {"stt_provider_count", "tts_provider_count"}),
        ("/v1/runs", {}, "runs listed", {"limit", "run_count"}),
        (
            "/v1/results",
            {"benchmark": "STT"},
            "results queried",
            {"benchmark", "result_count", "limit"},
        ),
        (
            "/v1/leaderboard",
            {"metric": "WER", "benchmark": "STT"},
            "leaderboard queried",
            {"metric", "window", "entry_count"},
        ),
    ],
)
async def test_router_emits_event_with_payload(
    app: FastAPI,
    client: AsyncClient,
    path: str,
    params: dict[str, str],
    event: str,
    required_keys: set[str],
) -> None:
    """Each router emits its named event with the expected property keys and the
    $process_person_profile guard set to False."""
    fake = create_autospec(Posthog, instance=True)
    app.dependency_overrides[get_posthog] = lambda: fake
    try:
        response = await client.get(path, params=params)
    finally:
        app.dependency_overrides.pop(get_posthog, None)

    assert response.status_code == 200
    fake.capture.assert_called_once()
    assert fake.capture.call_args.args[0] == event
    assert fake.capture.call_args.kwargs["distinct_id"] == "coval-bench-api"
    properties = fake.capture.call_args.kwargs["properties"]
    assert required_keys <= set(properties.keys())
    assert properties["$process_person_profile"] is False


async def test_capture_failure_does_not_break_endpoint(app: FastAPI, client: AsyncClient) -> None:
    """A raising PostHog client is swallowed; the route still returns 200."""
    fake = create_autospec(Posthog, instance=True)
    fake.capture.side_effect = RuntimeError("posthog down")
    app.dependency_overrides[get_posthog] = lambda: fake
    try:
        response = await client.get("/v1/providers")
    finally:
        app.dependency_overrides.pop(get_posthog, None)

    assert response.status_code == 200
    fake.capture.assert_called_once()
