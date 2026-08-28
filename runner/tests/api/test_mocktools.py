# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The /mock appliance: auth, the latency budget, and the tool-call log."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import psycopg
import psycopg.rows
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient

from coval_bench.mocktools.dispatch import Dispatcher, load_tool_specs
from coval_bench.mocktools.fixtures import MockFixtures, Seed, ToolFixture
from tests.api.conftest import MOCK_TOOLS_KEY, _make_db_url

PHONE = "2065550180"
AUTH = {"X-Mock-Tools-Key": MOCK_TOOLS_KEY}
SIMULATION_ID = "sim_abc123"

# A response big enough that gzip would fire if /mock were not excluded.
BULKY = {"patient_name": "Marcus Lee", "notes": "x" * 4096}


def _test_fixtures() -> MockFixtures:
    """Seeds covering every tool the committed contract declares.

    Built here rather than read from ``_private/``, which is never committed and
    so does not exist in CI.
    """

    def plain(name: str) -> ToolFixture:
        return ToolFixture(fallback=Seed(id=f"{name}_fallback", response={"ok": True}))

    return MockFixtures(
        tools={
            "lookup_patient": ToolFixture(
                seeds=(
                    Seed(id="marcus_lee", match={"phone": PHONE}, response=BULKY),
                    Seed(
                        id="broken",
                        match={"phone": "0000000000"},
                        response={"error": "upstream_unavailable"},
                        http_status=500,
                    ),
                ),
                fallback=Seed(id="no_match", response={"found": False}),
            ),
            "check_availability": plain("check_availability"),
            "book_appointment": plain("book_appointment"),
            "cancel_appointment": plain("cancel_appointment"),
            "transfer_to_front_desk": plain("transfer_to_front_desk"),
        }
    )


@pytest_asyncio.fixture
async def mock_app(app: FastAPI) -> FastAPI:
    """The app with the test seeds bound, and no artificial latency."""
    app.state.mock_dispatcher = Dispatcher(load_tool_specs("dental"), _test_fixtures())
    app.state.settings = app.state.settings.model_copy(update={"mock_tools_latency_ms": 0.0})
    return app


async def _rows(postgresql: Any) -> list[dict[str, Any]]:
    conn = await psycopg.AsyncConnection.connect(
        _make_db_url(postgresql), autocommit=True, row_factory=psycopg.rows.dict_row
    )
    try:
        cur = await conn.execute(
            "SELECT * FROM benchmarks_v2.mock_tool_calls ORDER BY created_at, id"
        )
        return list(await cur.fetchall())
    finally:
        await conn.close()


# --- auth ------------------------------------------------------------------


async def test_a_call_without_the_secret_is_rejected(
    client: AsyncClient, mock_app: FastAPI
) -> None:
    response = await client.post("/mock/lookup_patient", json={"phone": PHONE})
    assert response.status_code == 401


async def test_a_call_with_the_wrong_secret_is_rejected(
    client: AsyncClient, mock_app: FastAPI
) -> None:
    response = await client.post(
        "/mock/lookup_patient", json={"phone": PHONE}, headers={"X-Mock-Tools-Key": "nope"}
    )
    assert response.status_code == 401


async def test_an_unconfigured_secret_closes_the_appliance(
    client: AsyncClient, mock_app: FastAPI
) -> None:
    """Fails closed: an open mock would let anyone write into the tool-call log."""
    mock_app.state.settings = mock_app.state.settings.model_copy(update={"mock_tools_secret": None})
    response = await client.post("/mock/lookup_patient", json={"phone": PHONE}, headers=AUTH)
    assert response.status_code == 503


# --- answers ---------------------------------------------------------------


async def test_a_seeded_call_returns_its_seed(client: AsyncClient, mock_app: FastAPI) -> None:
    response = await client.post("/mock/lookup_patient", json={"phone": PHONE}, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["patient_name"] == "Marcus Lee"


async def test_an_unseeded_call_returns_the_fallback(
    client: AsyncClient, mock_app: FastAPI
) -> None:
    response = await client.post("/mock/lookup_patient", json={"phone": "5105550141"}, headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {"found": False}


async def test_a_seeded_failure_reaches_the_caller(client: AsyncClient, mock_app: FastAPI) -> None:
    response = await client.post("/mock/lookup_patient", json={"phone": "0000000000"}, headers=AUTH)
    assert response.status_code == 500
    assert response.json()["error"] == "upstream_unavailable"


async def test_an_unknown_tool_answers_404(client: AsyncClient, mock_app: FastAPI) -> None:
    response = await client.post("/mock/read_chart", json={"phone": PHONE}, headers=AUTH)
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_tool"


async def test_a_missing_required_argument_answers_422(
    client: AsyncClient, mock_app: FastAPI
) -> None:
    response = await client.post("/mock/lookup_patient", json={}, headers=AUTH)
    assert response.status_code == 422
    assert response.json()["missing"] == ["phone"]


# --- the log ---------------------------------------------------------------


async def test_a_call_lands_a_row_carrying_the_simulation_id(
    client: AsyncClient, mock_app: FastAPI, postgresql: Any
) -> None:
    await client.post(
        "/mock/lookup_patient",
        json={"phone": PHONE},
        headers={**AUTH, "X-Coval-Simulation-Id": SIMULATION_ID, "X-Coval-Caller-Number": PHONE},
    )
    rows = await _rows(postgresql)
    assert len(rows) == 1
    assert rows[0]["simulation_id"] == SIMULATION_ID
    assert rows[0]["caller_number"] == PHONE
    assert rows[0]["tool"] == "lookup_patient"
    assert rows[0]["args"] == {"phone": PHONE}
    assert rows[0]["matched_seed"] == "marcus_lee"
    assert rows[0]["latency_ms"] >= 0


async def test_a_fallback_row_names_no_seed(
    client: AsyncClient, mock_app: FastAPI, postgresql: Any
) -> None:
    await client.post("/mock/lookup_patient", json={"phone": "5105550141"}, headers=AUTH)
    rows = await _rows(postgresql)
    assert rows[0]["matched_seed"] is None


async def test_a_rejected_call_is_still_recorded(
    client: AsyncClient, mock_app: FastAPI, postgresql: Any
) -> None:
    """An agent inventing a tool is evidence, not a reason to lose the row."""
    await client.post("/mock/read_chart", json={"phone": PHONE}, headers=AUTH)
    rows = await _rows(postgresql)
    assert len(rows) == 1
    assert rows[0]["tool"] == "read_chart"
    assert rows[0]["matched_seed"] is None


async def test_calls_are_recoverable_in_the_order_they_fired(
    client: AsyncClient, mock_app: FastAPI, postgresql: Any
) -> None:
    """`now()` is constant within a transaction, so `id` is what orders a burst."""
    for _ in range(5):
        await client.post("/mock/check_availability", json={"date": "2030-11-12"}, headers=AUTH)
    rows = await _rows(postgresql)
    assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)
    assert len(rows) == 5


# --- the appliance's own guarantees ----------------------------------------


async def test_every_answer_is_held_to_the_latency_budget(
    client: AsyncClient, app: FastAPI
) -> None:
    app.state.mock_dispatcher = Dispatcher(load_tool_specs("dental"), _test_fixtures())
    app.state.settings = app.state.settings.model_copy(update={"mock_tools_latency_ms": 120.0})
    started = time.perf_counter()
    response = await client.post("/mock/lookup_patient", json={"phone": PHONE}, headers=AUTH)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert response.status_code == 200
    assert elapsed_ms >= 120.0


async def test_the_budget_is_configurable(client: AsyncClient, mock_app: FastAPI) -> None:
    """Zeroed by the fixture: the budget is a setting, not a constant in the route."""
    started = time.perf_counter()
    await client.post("/mock/lookup_patient", json={"phone": PHONE}, headers=AUTH)
    assert (time.perf_counter() - started) * 1000 < 120.0


async def test_the_appliance_is_never_rate_limited(client: AsyncClient, mock_app: FastAPI) -> None:
    """A burst of tool calls is a scenario working; a 429 would be graded as the agent failing."""
    responses = await asyncio.gather(
        *(
            client.post("/mock/lookup_patient", json={"phone": PHONE}, headers=AUTH)
            for _ in range(80)
        )
    )
    assert {r.status_code for r in responses} == {200}


async def test_answers_are_not_compressed(client: AsyncClient, mock_app: FastAPI) -> None:
    """Compression would add a variable cost to a deliberately fixed-latency response."""
    response = await client.post(
        "/mock/lookup_patient",
        json={"phone": PHONE},
        headers={**AUTH, "Accept-Encoding": "gzip"},
    )
    assert len(response.json()["notes"]) == 4096
    assert "content-encoding" not in response.headers


@pytest.mark.parametrize("tool", ["check_availability", "book_appointment", "cancel_appointment"])
async def test_every_contract_tool_is_answerable(
    client: AsyncClient, mock_app: FastAPI, tool: str
) -> None:
    """Routes come from the contract, so no tool needs a hand-written one."""
    response = await client.post(f"/mock/{tool}", json={}, headers=AUTH)
    assert response.status_code in (200, 422)
