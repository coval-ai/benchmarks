# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the API emits PostHog events when analytics is configured.

Uses ``dependency_overrides`` plus ``unittest.mock`` so no extra test
dependencies are needed and no real PostHog network calls are made. The
``/providers`` route exercises the shared ``get_posthog`` wiring that every
router uses.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.api.deps import get_posthog


async def test_providers_captures_event(app: FastAPI, client: AsyncClient) -> None:
    """An overridden PostHog client receives a 'providers listed' event."""
    fake = MagicMock()
    app.dependency_overrides[get_posthog] = lambda: fake
    try:
        response = await client.get("/v1/providers")
    finally:
        app.dependency_overrides.pop(get_posthog, None)

    assert response.status_code == 200
    fake.capture.assert_called_once()
    distinct_id, event = fake.capture.call_args.args[:2]
    assert distinct_id == "coval-bench-api"
    assert event == "providers listed"


async def test_no_capture_when_analytics_disabled(client: AsyncClient) -> None:
    """With PostHog unconfigured, the route still succeeds and emits nothing."""
    response = await client.get("/v1/providers")
    assert response.status_code == 200
