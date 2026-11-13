# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The admin health endpoint over HTTP."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from httpx import AsyncClient

from tests.api.conftest import COVAL_ORG, EA_ORG, _make_db_url, bearer

ADMIN = bearer(sub="user_admin", org_id=COVAL_ORG, email="admin@coval.dev")
NOW = datetime.now(UTC)


def _probe(conn: Any, model_id: int, hours_ago: float, **failure: Any) -> None:
    at = NOW - timedelta(hours=hours_ago)
    conn.execute(
        "INSERT INTO benchmarks_v2.probe_results"
        " (model_id, scheduled_at, sent_at, status, ttfa_ms, error, failure_origin, status_code)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (
            model_id,
            at,
            at,
            "failed" if failure else "succeeded",
            None if failure else 180.0,
            failure.get("error"),
            failure.get("origin"),
            failure.get("status_code"),
        ),
    )


def _two_models(postgresql: Any) -> tuple[int, int]:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        ids = [
            row[0]
            for row in conn.execute("SELECT id FROM benchmarks_v2.models ORDER BY id LIMIT 2")
        ]
        up, down = ids
        _probe(conn, up, 1)
        _probe(conn, up, 30, error="old outage", origin="provider")
        _probe(conn, down, 2)
        _probe(conn, down, 1, error="request failed", origin="runner", status_code=401)
        _probe(conn, down, 0.5, error="timed out after 30s", origin="provider")
    return up, down


async def test_health_requires_a_coval_token(client: AsyncClient) -> None:
    assert (await client.get("/v1/admin/provider-health")).status_code == 401
    other_org = bearer(sub="u", org_id=EA_ORG)
    assert (await client.get("/v1/admin/provider-health", headers=other_org)).status_code == 403


async def test_health_lists_each_probed_model_with_its_latest_failure(
    client: AsyncClient, postgresql: Any
) -> None:
    up, down = _two_models(postgresql)

    response = await client.get("/v1/admin/provider-health", headers=ADMIN)

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    models = response.json()["models"]
    assert [m["model_id"] for m in models] == [down, up]
    failing, fine = models
    assert (failing["healthy"], failing["probe_count"], failing["failed_count"]) == (False, 3, 2)
    assert failing["last_failure"]["kind"] == "timeout"
    assert failing["last_failure"]["origin"] == "provider"
    assert (fine["healthy"], fine["probe_count"], fine["failed_count"]) == (True, 1, 0)
    assert fine["last_failure"] is None


async def test_a_401_is_reported_as_an_auth_failure(client: AsyncClient, postgresql: Any) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        model_id = conn.execute("SELECT id FROM benchmarks_v2.models LIMIT 1").fetchone()[0]  # type: ignore[index]
        _probe(conn, model_id, 1, error="request failed", origin="runner", status_code=401)

    failure = (await client.get("/v1/admin/provider-health", headers=ADMIN)).json()["models"][0][
        "last_failure"
    ]

    assert (failure["kind"], failure["status_code"], failure["origin"]) == ("auth", 401, "runner")
