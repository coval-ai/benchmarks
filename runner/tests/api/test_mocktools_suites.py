# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``?suite=`` picks a loaded suite; a checked suite's calls are queued for scoring."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from pydantic import SecretStr

import coval_bench.api.routers.mocktools as router_module
from coval_bench.mocktools import dispatch
from coval_bench.mocktools.dispatch import build_corral_dispatcher
from tests.api.test_mocktools import AUTH, PHONE, SIMULATION_ID
from tests.api.test_mocktools import mock_app as _mock_app

mock_app = _mock_app


@pytest.fixture
def roulette(mock_app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(
        dispatch, "read_private_fixture", lambda _suite: b'{"lookup": {"value": {"a": "KD7B"}}}'
    )
    mock_app.state.mock_suites = {"id_roulette": build_corral_dispatcher("id_roulette")}
    mock_app.state.settings = mock_app.state.settings.model_copy(
        update={"coval_api_key": SecretStr("test-coval-key")}
    )
    return mock_app


@pytest.fixture
def scored(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def fake_scoring(**kwargs: Any) -> bool:  # noqa: ANN401
        seen.append(
            {
                "simulation_id": kwargs["simulation_id"],
                "platform": kwargs["platform"],
                "checks": set(kwargs["checks"]),
                "tools": [call.tool for call in kwargs["calls"]],
            }
        )
        return True

    monkeypatch.setattr(router_module, "schedule_export", lambda **_: True)
    monkeypatch.setattr(router_module, "schedule_scoring", fake_scoring)
    return seen


async def test_the_named_suite_answers(
    client: AsyncClient, roulette: FastAPI, scored: list[dict[str, Any]]
) -> None:
    response = await client.post(
        "/mock/generic/lookup?suite=id_roulette",
        json={"value": "KD7BO0RWNLMX0O4BYD"},
        headers={**AUTH, "X-Coval-Simulation-Id": SIMULATION_ID},
    )
    assert response.status_code == 200
    assert response.json() == {"received": True}
    assert scored == [
        {
            "simulation_id": SIMULATION_ID,
            "platform": "generic",
            "checks": {"lookup"},
            "tools": ["lookup"],
        }
    ]


async def test_without_a_suite_the_default_answers_and_nothing_is_scored(
    client: AsyncClient, roulette: FastAPI, scored: list[dict[str, Any]]
) -> None:
    response = await client.post(
        "/mock/generic/lookup_patient",
        json={"phone": PHONE},
        headers={**AUTH, "X-Coval-Simulation-Id": SIMULATION_ID},
    )
    assert response.status_code == 200
    assert scored == []


async def test_an_unloaded_suite_is_404(client: AsyncClient, roulette: FastAPI) -> None:
    response = await client.post(
        "/mock/generic/lookup?suite=nope", json={"value": "x"}, headers=AUTH
    )
    assert response.status_code == 404


async def test_without_a_simulation_id_nothing_is_scored(
    client: AsyncClient, roulette: FastAPI, scored: list[dict[str, Any]]
) -> None:
    response = await client.post(
        "/mock/generic/lookup?suite=id_roulette", json={"value": "x"}, headers=AUTH
    )
    assert response.status_code == 200
    assert scored == []


async def test_a_vapi_batch_of_lookups_is_answered_and_queued_together(
    client: AsyncClient, roulette: FastAPI, scored: list[dict[str, Any]]
) -> None:
    body = {
        "message": {
            "type": "tool-calls",
            "toolCallList": [
                {"id": "tc_1", "name": "lookup", "arguments": {"value": "KD7B"}},
                {"id": "tc_2", "name": "lookup", "arguments": {"value": "KD7 B"}},
            ],
        }
    }
    response = await client.post(
        "/mock/vapi?suite=id_roulette",
        json=body,
        headers={**AUTH, "X-Coval-Simulation-Id": SIMULATION_ID},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["toolCallId"] for r in results] == ["tc_1", "tc_2"]
    assert [json.loads(r["result"]) for r in results] == [{"received": True}] * 2
    assert scored == [
        {
            "simulation_id": SIMULATION_ID,
            "platform": "vapi",
            "checks": {"lookup"},
            "tools": ["lookup", "lookup"],
        }
    ]
