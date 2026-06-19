# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Voice Arena read endpoints (GET /v1/arena/*)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
from httpx import AsyncClient

from tests.api.conftest import _make_db_url


async def _apply_arena_schema(dsn: str) -> None:
    """Create the arena tables read by the endpoints (mirrors migration 20260615_0007)."""
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        await aconn.execute("CREATE SCHEMA IF NOT EXISTS arena")
        await aconn.execute("""
            CREATE TABLE IF NOT EXISTS arena.battles (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                provider_a  TEXT NOT NULL,
                model_a     TEXT NOT NULL,
                provider_b  TEXT NOT NULL,
                model_b     TEXT NOT NULL,
                domain      TEXT,
                prompt_text TEXT NOT NULL,
                audio_a_url TEXT NOT NULL,
                audio_b_url TEXT NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await aconn.execute("""
            CREATE TABLE IF NOT EXISTS arena.leaderboard_snapshots (
                id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                computed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
                metric_name         TEXT NOT NULL,
                methodology_version TEXT NOT NULL,
                domain              TEXT NOT NULL DEFAULT 'all',
                provider            TEXT NOT NULL,
                model               TEXT NOT NULL,
                rating_elo          NUMERIC NOT NULL,
                rating_bt           NUMERIC NOT NULL,
                ci_low              NUMERIC,
                ci_high             NUMERIC,
                ci_half_width       NUMERIC,
                votes_total         INTEGER NOT NULL,
                wins                NUMERIC NOT NULL,
                losses              NUMERIC NOT NULL,
                ties                NUMERIC NOT NULL,
                status              TEXT NOT NULL
            )
        """)
    finally:
        await aconn.close()


async def _insert_battle(postgresql: Any, **kwargs: Any) -> str:
    """Insert a battle row and return its id as a string."""
    dsn = _make_db_url(postgresql)
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        defaults: dict[str, Any] = {
            "provider_a": "elevenlabs",
            "model_a": "eleven_multilingual_v2",
            "provider_b": "cartesia",
            "model_b": "sonic-3",
            "domain": "support",
            "prompt_text": "Tell me about your refund policy.",
            "audio_a_url": "https://example.test/a.wav",
            "audio_b_url": "https://example.test/b.wav",
        }
        defaults.update(kwargs)
        row = await aconn.execute(
            """
            INSERT INTO arena.battles
                (provider_a, model_a, provider_b, model_b, domain,
                 prompt_text, audio_a_url, audio_b_url)
            VALUES
                (%(provider_a)s, %(model_a)s, %(provider_b)s, %(model_b)s, %(domain)s,
                 %(prompt_text)s, %(audio_a_url)s, %(audio_b_url)s)
            RETURNING id
            """,
            defaults,
        )
        result = await row.fetchone()
        assert result is not None
        return str(result[0])
    finally:
        await aconn.close()


async def _insert_snapshot(postgresql: Any, **kwargs: Any) -> None:
    """Insert a leaderboard_snapshots row."""
    dsn = _make_db_url(postgresql)
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        defaults: dict[str, Any] = {
            "computed_at": datetime(2026, 6, 18, 12, 0, tzinfo=UTC),
            "metric_name": "naturalness",
            "methodology_version": "davidson-v1",
            "domain": "all",
            "provider": "elevenlabs",
            "model": "eleven_multilingual_v2",
            "rating_elo": 1500.0,
            "rating_bt": 0.0,
            "ci_low": 1450.0,
            "ci_high": 1550.0,
            "ci_half_width": 50.0,
            "votes_total": 10,
            "wins": 6,
            "losses": 4,
            "ties": 0,
            "status": "usable",
        }
        defaults.update(kwargs)
        await aconn.execute(
            """
            INSERT INTO arena.leaderboard_snapshots
                (computed_at, metric_name, methodology_version, domain, provider, model,
                 rating_elo, rating_bt, ci_low, ci_high, ci_half_width,
                 votes_total, wins, losses, ties, status)
            VALUES
                (%(computed_at)s, %(metric_name)s, %(methodology_version)s, %(domain)s,
                 %(provider)s, %(model)s, %(rating_elo)s, %(rating_bt)s, %(ci_low)s,
                 %(ci_high)s, %(ci_half_width)s, %(votes_total)s, %(wins)s, %(losses)s,
                 %(ties)s, %(status)s)
            """,
            defaults,
        )
    finally:
        await aconn.close()


async def test_get_battle_404_when_empty(client: AsyncClient, postgresql: Any) -> None:
    """No battles seeded -> 404."""
    await _apply_arena_schema(_make_db_url(postgresql))
    response = await client.get("/v1/arena/battle")
    assert response.status_code == 404


async def test_get_battle_is_blind(client: AsyncClient, postgresql: Any) -> None:
    """A served battle exposes only the blind fields, never provider/model identities."""
    await _apply_arena_schema(_make_db_url(postgresql))
    await _insert_battle(postgresql)

    response = await client.get("/v1/arena/battle")
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"id", "prompt_text", "domain", "audio_a_url", "audio_b_url"}
    assert "provider_a" not in data
    assert "model_a" not in data
    assert data["domain"] == "support"


async def test_get_battle_by_id(client: AsyncClient, postgresql: Any) -> None:
    """GET /arena/battle/{id} returns the matching battle."""
    await _apply_arena_schema(_make_db_url(postgresql))
    battle_id = await _insert_battle(postgresql)

    response = await client.get(f"/v1/arena/battle/{battle_id}")
    assert response.status_code == 200
    assert response.json()["id"] == battle_id


async def test_get_battle_by_id_unknown_returns_404(client: AsyncClient, postgresql: Any) -> None:
    """A well-formed but unknown UUID returns 404."""
    await _apply_arena_schema(_make_db_url(postgresql))
    response = await client.get("/v1/arena/battle/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


async def test_get_battle_by_id_malformed_returns_422(client: AsyncClient, postgresql: Any) -> None:
    """A non-UUID id is rejected by validation with 422."""
    await _apply_arena_schema(_make_db_url(postgresql))
    response = await client.get("/v1/arena/battle/not-a-uuid")
    assert response.status_code == 422


async def test_leaderboard_empty_when_no_snapshots(client: AsyncClient, postgresql: Any) -> None:
    """No snapshots -> 200 with empty entries and null computed_at."""
    await _apply_arena_schema(_make_db_url(postgresql))
    response = await client.get("/v1/arena/leaderboard")
    assert response.status_code == 200
    data = response.json()
    assert data["metric"] == "naturalness"
    assert data["domain"] == "all"
    assert data["entries"] == []
    assert data["computed_at"] is None


async def test_leaderboard_returns_latest_board_sorted(
    client: AsyncClient, postgresql: Any
) -> None:
    """Only the most recent computed_at board is returned, sorted by rating_elo desc."""
    await _apply_arena_schema(_make_db_url(postgresql))
    stale = datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    latest = datetime(2026, 6, 18, 12, 0, tzinfo=UTC)

    # An older board that must be excluded.
    await _insert_snapshot(postgresql, computed_at=stale, provider="old", model="m", rating_elo=999)
    # The latest board: two models, inserted out of rank order.
    await _insert_snapshot(
        postgresql, computed_at=latest, provider="cartesia", model="sonic-3", rating_elo=1480
    )
    await _insert_snapshot(
        postgresql, computed_at=latest, provider="elevenlabs", model="v2", rating_elo=1520
    )

    response = await client.get("/v1/arena/leaderboard")
    assert response.status_code == 200
    data = response.json()
    entries = data["entries"]
    assert len(entries) == 2
    assert [e["model"] for e in entries] == ["v2", "sonic-3"]
    assert data["methodology_version"] == "davidson-v1"


async def test_leaderboard_domain_filter(client: AsyncClient, postgresql: Any) -> None:
    """domain filter returns only that domain's board, excluding the global 'all' board."""
    await _apply_arena_schema(_make_db_url(postgresql))
    computed = datetime(2026, 6, 18, 12, 0, tzinfo=UTC)
    await _insert_snapshot(
        postgresql, computed_at=computed, domain="all", provider="elevenlabs", model="v2"
    )
    await _insert_snapshot(
        postgresql, computed_at=computed, domain="support", provider="cartesia", model="sonic-3"
    )

    support = await client.get("/v1/arena/leaderboard", params={"domain": "support"})
    assert support.status_code == 200
    assert support.json()["domain"] == "support"
    assert [e["model"] for e in support.json()["entries"]] == ["sonic-3"]

    default = await client.get("/v1/arena/leaderboard")
    assert [e["model"] for e in default.json()["entries"]] == ["v2"]


async def test_leaderboard_latest_is_scoped_per_metric(
    client: AsyncClient, postgresql: Any
) -> None:
    """The latest board is per metric: a metric computed earlier still returns its own rows."""
    await _apply_arena_schema(_make_db_url(postgresql))
    await _insert_snapshot(
        postgresql,
        computed_at=datetime(2026, 6, 18, 12, 0, tzinfo=UTC),
        metric_name="naturalness",
        provider="elevenlabs",
        model="v2",
    )
    await _insert_snapshot(
        postgresql,
        computed_at=datetime(2026, 6, 17, 12, 0, tzinfo=UTC),
        metric_name="clarity",
        provider="cartesia",
        model="sonic-3",
    )

    response = await client.get("/v1/arena/leaderboard", params={"metric": "clarity"})
    assert response.status_code == 200
    data = response.json()
    assert data["metric"] == "clarity"
    assert [e["model"] for e in data["entries"]] == ["sonic-3"]
