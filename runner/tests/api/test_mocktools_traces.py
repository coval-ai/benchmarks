# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The /mock appliance queues Coval's copy of a call, only when it knows which simulation."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from pydantic import SecretStr

import coval_bench.api.routers.mocktools as router_module
from coval_bench.mocktools.traces import SERVICE_NAME as TRACE_SERVICE_NAME
from tests.api.test_mocktools import AUTH, PHONE, SIMULATION_ID
from tests.api.test_mocktools import mock_app as _mock_app

# pytest registers a fixture under the module attribute that holds it, so the
# shared appliance fixture is rebound here under its own name.
mock_app = _mock_app

COVAL_KEY = "test-coval-key"  # noqa: S105 — a fixture value, not a credential


@pytest.fixture
def queued(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what the router queues for Coval instead of posting anything."""
    seen: list[dict[str, Any]] = []

    def fake_schedule(**kwargs: Any) -> bool:  # noqa: ANN401
        seen.append(
            {
                "simulation_id": kwargs["simulation_id"],
                "api_key": kwargs["api_key"].get_secret_value() if kwargs["api_key"] else None,
                "api_base": kwargs["api_base"],
                "service": dict(kwargs["provider"].resource.attributes)["service.name"],
                "platform": kwargs["platform"],
                "tools": [call.tool for call in kwargs["calls"]],
                "modes": [outcome.mode for outcome in kwargs["outcomes"]],
                "ordered": kwargs["started_ns"] <= kwargs["ended_ns"],
            }
        )
        return kwargs["api_key"] is not None

    monkeypatch.setattr(router_module, "schedule_export", fake_schedule)
    return seen


def _with_key(app: FastAPI) -> None:
    app.state.settings = app.state.settings.model_copy(
        update={"coval_api_key": SecretStr(COVAL_KEY)}
    )


async def test_a_correlated_call_is_queued_for_coval(
    client: AsyncClient, mock_app: FastAPI, queued: list[dict[str, Any]]
) -> None:
    _with_key(mock_app)
    response = await client.post(
        "/mock/generic/lookup_patient",
        json={"phone": PHONE},
        headers={**AUTH, "X-Coval-Simulation-Id": SIMULATION_ID},
    )
    assert response.status_code == 200
    assert queued == [
        {
            "simulation_id": SIMULATION_ID,
            "api_key": COVAL_KEY,
            "api_base": mock_app.state.settings.coval_api_base,
            "service": TRACE_SERVICE_NAME,
            "platform": "generic",
            "tools": ["lookup_patient"],
            "modes": ["exact"],
            "ordered": True,
        }
    ]


async def test_a_call_without_a_simulation_id_queues_nothing(
    client: AsyncClient, mock_app: FastAPI, queued: list[dict[str, Any]]
) -> None:
    _with_key(mock_app)
    response = await client.post(
        "/mock/generic/lookup_patient", json={"phone": PHONE}, headers=AUTH
    )
    assert response.status_code == 200
    assert queued == []


async def test_the_key_decision_is_left_to_the_scheduler(
    client: AsyncClient, mock_app: FastAPI, queued: list[dict[str, Any]]
) -> None:
    """The router hands over whatever key it has; the scheduler owns the once-only warning."""
    mock_app.state.settings = mock_app.state.settings.model_copy(update={"coval_api_key": None})
    response = await client.post(
        "/mock/generic/lookup_patient",
        json={"phone": PHONE},
        headers={**AUTH, "X-Coval-Simulation-Id": SIMULATION_ID},
    )
    assert response.status_code == 200
    assert [entry["api_key"] for entry in queued] == [None]


async def test_a_vapi_batch_is_queued_once_with_every_call(
    client: AsyncClient, mock_app: FastAPI, queued: list[dict[str, Any]]
) -> None:
    _with_key(mock_app)
    body = {
        "message": {
            "toolCallList": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "lookup_patient", "arguments": f'{{"phone": "{PHONE}"}}'},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "check_availability",
                        "arguments": '{"date": "2030-05-06", "appointment_type": "cleaning"}',
                    },
                },
            ]
        }
    }
    response = await client.post(
        "/mock/vapi", json=body, headers={**AUTH, "X-Coval-Simulation-Id": SIMULATION_ID}
    )
    assert response.status_code == 200
    assert len(queued) == 1
    assert queued[0]["platform"] == "vapi"
    assert queued[0]["tools"] == ["lookup_patient", "check_availability"]
