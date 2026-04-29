# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GET /v1/runs."""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient

from tests.api.conftest import _insert_run


async def test_empty_db_returns_empty_list(client: AsyncClient, postgresql: Any) -> None:
    """Empty DB returns an empty list with null next_before."""
    response = await client.get("/v1/runs")
    assert response.status_code == 200
    data = response.json()
    assert data["runs"] == []
    assert data["next_before"] is None


async def test_three_runs_returned_newest_first(client: AsyncClient, postgresql: Any) -> None:
    """Three seeded runs are returned newest-first (descending id)."""
    id1 = await _insert_run(postgresql)
    id2 = await _insert_run(postgresql)
    id3 = await _insert_run(postgresql)

    response = await client.get("/v1/runs")
    assert response.status_code == 200
    run_ids = [r["id"] for r in response.json()["runs"]]
    assert run_ids == [id3, id2, id1]


async def test_cursor_pagination(client: AsyncClient, postgresql: Any) -> None:
    """Cursor pagination works: ?limit=2&before=<id> returns the next page."""
    id1 = await _insert_run(postgresql)
    id2 = await _insert_run(postgresql)
    id3 = await _insert_run(postgresql)

    # First page: limit=2, should return id3, id2
    r1 = await client.get("/v1/runs", params={"limit": 2})
    assert r1.status_code == 200
    page1 = r1.json()
    assert [r["id"] for r in page1["runs"]] == [id3, id2]
    assert page1["next_before"] == id2

    # Second page using cursor
    r2 = await client.get("/v1/runs", params={"limit": 2, "before": page1["next_before"]})
    assert r2.status_code == 200
    page2 = r2.json()
    assert [r["id"] for r in page2["runs"]] == [id1]
    assert page2["next_before"] is None  # fewer rows than limit


async def test_limit_zero_returns_422(client: AsyncClient) -> None:
    """limit=0 violates the ge=1 constraint and returns 422."""
    response = await client.get("/v1/runs", params={"limit": 0})
    assert response.status_code == 422


async def test_next_before_set_when_full_page(client: AsyncClient, postgresql: Any) -> None:
    """next_before is the smallest id in the page when page is full."""
    for _ in range(5):
        await _insert_run(postgresql)
    response = await client.get("/v1/runs", params={"limit": 3})
    assert response.status_code == 200
    data = response.json()
    assert len(data["runs"]) == 3
    assert data["next_before"] == min(r["id"] for r in data["runs"])
