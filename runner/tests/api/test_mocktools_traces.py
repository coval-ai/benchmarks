# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The /mock appliance hands Coval a span per call, only when it knows which simulation."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from pydantic import SecretStr

import coval_bench.api.routers.mocktools as router_module
from tests.api.test_mocktools import AUTH, PHONE, SIMULATION_ID
from tests.api.test_mocktools import mock_app as _mock_app

# pytest registers a fixture under the module attribute that holds it, so the
# shared appliance fixture is rebound here under its own name.
mock_app = _mock_app

COVAL_KEY = "test-coval-key"  # noqa: S105 — a fixture value, not a credential


@pytest.fixture
def exported(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what the router hands to the exporter instead of posting to Coval."""
    seen: list[dict[str, Any]] = []

    def fake_export(
        exporter: Any,  # noqa: ANN401
        resource: Any,  # noqa: ANN401
        platform: str,
        calls: Any,  # noqa: ANN401
        outcomes: Any,  # noqa: ANN401
        started_ns: int,
        ended_ns: int,
    ) -> int:
        seen.append(
            {
                "simulation_id": exporter._headers["X-Simulation-Id"],  # noqa: SLF001
                "api_key": exporter._headers["X-API-Key"],  # noqa: SLF001
                "service": dict(resource.attributes)["service.name"],
                "platform": platform,
                "tools": [call.tool for call in calls],
                "modes": [o.resolution.mode if o.resolution else "rejected" for o in outcomes],
                "ordered": started_ns <= ended_ns,
            }
        )
        return len(calls)

    monkeypatch.setattr(router_module, "export_tool_spans", fake_export)
    return seen


def _with_key(app: FastAPI) -> None:
    app.state.settings = app.state.settings.model_copy(
        update={"coval_api_key": SecretStr(COVAL_KEY)}
    )


async def test_a_correlated_call_is_handed_to_coval(
    client: AsyncClient, mock_app: FastAPI, exported: list[dict[str, Any]]
) -> None:
    _with_key(mock_app)
    response = await client.post(
        "/mock/generic/lookup_patient",
        json={"phone": PHONE},
        headers={**AUTH, "X-Coval-Simulation-Id": SIMULATION_ID},
    )
    assert response.status_code == 200
    assert exported == [
        {
            "simulation_id": SIMULATION_ID,
            "api_key": COVAL_KEY,
            "service": router_module.TRACE_SERVICE_NAME,
            "platform": "generic",
            "tools": ["lookup_patient"],
            "modes": ["exact"],
            "ordered": True,
        }
    ]


async def test_a_call_without_a_simulation_id_posts_nothing(
    client: AsyncClient, mock_app: FastAPI, exported: list[dict[str, Any]]
) -> None:
    _with_key(mock_app)
    response = await client.post(
        "/mock/generic/lookup_patient", json={"phone": PHONE}, headers=AUTH
    )
    assert response.status_code == 200
    assert exported == []


async def test_a_call_without_a_coval_key_posts_nothing(
    client: AsyncClient, mock_app: FastAPI, exported: list[dict[str, Any]]
) -> None:
    mock_app.state.settings = mock_app.state.settings.model_copy(update={"coval_api_key": None})
    response = await client.post(
        "/mock/generic/lookup_patient",
        json={"phone": PHONE},
        headers={**AUTH, "X-Coval-Simulation-Id": SIMULATION_ID},
    )
    assert response.status_code == 200
    assert exported == []


async def test_a_vapi_batch_is_one_export_with_every_call(
    client: AsyncClient, mock_app: FastAPI, exported: list[dict[str, Any]]
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
    assert len(exported) == 1
    assert exported[0]["platform"] == "vapi"
    assert exported[0]["tools"] == ["lookup_patient", "check_availability"]
