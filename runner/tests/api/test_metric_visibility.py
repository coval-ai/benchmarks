# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-metric visibility over HTTP: public, partner and coval callers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from httpx import AsyncClient

from tests.api.conftest import (
    COVAL_ORG,
    EA_ORG,
    _insert_run,
    _insert_value,
    _make_db_url,
    _publish,
    bearer,
)

_STT = {"benchmark": "STT", "window": "24h"}


def _set_level(postgresql: Any, code: str, level: str) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        conn.execute(
            "UPDATE benchmarks_v2.metrics SET visibility = %s WHERE code = %s", (level, code)
        )


async def _seed(postgresql: Any, level: str) -> None:
    run_id = await _insert_run(postgresql, scheduled_at=datetime.now(UTC))
    await _insert_value(postgresql, run_id, 3.5)
    await _insert_value(postgresql, run_id, 120.0, metric_type="TTFT")
    await _publish(postgresql)
    _set_level(postgresql, "TTFT", level)


async def _where_ttft_shows(client: AsyncClient, headers: dict[str, str]) -> dict[str, bool]:
    async def body(path: str, params: dict[str, str]) -> Any:
        response = await client.get(path, params=params, headers=headers)
        assert response.status_code == 200, (path, response.text)
        return response.json()

    def has_ttft(rows: list[dict[str, Any]]) -> bool:
        assert any(row["metric_type"] == "WER" for row in rows)
        return any(row["metric_type"] == "TTFT" for row in rows)

    aggregates = await body("/v1/results/aggregates", _STT)
    by_dataset = await body("/v1/results/aggregates/by-dataset", _STT)
    return {
        "aggregates": has_ttft(aggregates["model_stats"]),
        "series": has_ttft(aggregates["series"]),
        "by_dataset": has_ttft(
            [stat for block in by_dataset["blocks"] for stat in block["model_stats"]]
        ),
        "timeline": bool(
            (await body("/v1/results/timeline", {**_STT, "metric_type": "TTFT"}))["points"]
        ),
        "results": has_ttft((await body("/v2/results", {"benchmark": "STT"}))["results"]),
        "leaderboard": bool((await body("/v1/leaderboard", {**_STT, "metric": "TTFT"}))["entries"]),
    }


@pytest.mark.parametrize(
    ("level", "caller", "shown"),
    [
        ("coval", {}, False),
        ("coval", {"org_id": EA_ORG}, False),
        ("coval", {"org_id": COVAL_ORG}, True),
        ("partner", {}, False),
        ("partner", {"org_id": EA_ORG}, True),
        ("partner", {"org_id": COVAL_ORG}, True),
        ("public", {}, True),
    ],
)
async def test_metric_level_decides_who_gets_it_on_every_score_route(
    client: AsyncClient, postgresql: Any, level: str, caller: dict[str, str], shown: bool
) -> None:
    await _seed(postgresql, level)

    seen = await _where_ttft_shows(client, bearer(**caller) if caller else {})

    assert seen == dict.fromkeys(seen, shown)


async def test_a_partner_timeline_is_never_served_to_the_public_from_cache(
    client: AsyncClient, postgresql: Any
) -> None:
    await _seed(postgresql, "partner")
    params = {**_STT, "metric_type": "TTFT"}

    partner = await client.get("/v1/results/timeline", params=params, headers=bearer(org_id=EA_ORG))
    public = await client.get("/v1/results/timeline", params=params)

    assert partner.json()["points"]
    assert public.json()["points"] == []


async def test_a_signed_in_caller_outside_any_mapped_org_is_public(
    client: AsyncClient, postgresql: Any
) -> None:
    await _seed(postgresql, "partner")

    seen = await _where_ttft_shows(client, bearer(org_id="org_test_stranger"))

    assert not any(seen.values())


async def test_unreadable_levels_fail_score_routes_but_not_the_roster(
    client: AsyncClient, postgresql: Any
) -> None:
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        conn.execute("ALTER TABLE benchmarks_v2.metrics DROP COLUMN visibility")

    assert (await client.get("/v1/results/aggregates", params=_STT)).status_code == 503
    assert (await client.get("/v1/providers")).status_code == 200
