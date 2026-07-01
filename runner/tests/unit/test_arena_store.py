# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.db.arena_store.

Uses ``pytest-postgresql`` (embedded ``pg_ctl``, no Docker) to spin up a real
Postgres. No remote DB is ever contacted.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import psycopg
import psycopg.errors
import psycopg.rows
import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql

from coval_bench.db.arena_store import ArenaStore
from coval_bench.db.models import (
    Battle,
    LeaderboardSnapshot,
    SnapshotStatus,
    VoteOutcome,
    VoterType,
)

arena_pg = postgresql("pg_proc")  # shared server from conftest, own per-test DB

_INI_PATH = Path(__file__).parents[2] / "alembic.ini"


def _alembic_cfg(dsn: str) -> AlembicConfig:
    cfg = AlembicConfig(str(_INI_PATH))
    cfg.set_main_option(
        "sqlalchemy.url",
        dsn.replace("postgresql://", "postgresql+psycopg://"),
    )
    return cfg


def _async_dsn(conn: psycopg.Connection[Any]) -> str:
    info = conn.info
    host = info.host or "localhost"
    port = info.port or 5432
    dbname = info.dbname or "test"
    user = info.user or ""
    password = info.password or ""
    if password:
        return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{dbname}"
    return f"postgresql://{quote_plus(user)}@{host}:{port}/{dbname}"


def _apply_migrations(conn: psycopg.Connection[Any]) -> None:
    alembic_command.upgrade(_alembic_cfg(_async_dsn(conn)), "head")


async def _make_pool(
    conn: psycopg.Connection[Any],
) -> AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]:
    pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]] = AsyncConnectionPool(
        conninfo=_async_dsn(conn),
        min_size=1,
        max_size=4,
        open=False,
        kwargs={
            "autocommit": False,
            "row_factory": psycopg.rows.dict_row,
        },
    )
    await pool.open()
    return pool


def _make_battle(*, provider_b: str = "elevenlabs", model_b: str = "flash") -> Battle:
    return Battle(
        provider_a="cartesia",
        model_a="sonic-3.5",
        provider_b=provider_b,
        model_b=model_b,
        domain="general",
        prompt_text="hello there",
        audio_a_url="https://example.test/a.wav",
        audio_b_url="https://example.test/b.wav",
    )


def _snapshot(
    provider: str,
    model: str,
    rating_elo: float,
    *,
    metric_name: str = "naturalness",
    domain: str = "all",
) -> LeaderboardSnapshot:
    return LeaderboardSnapshot(
        metric_name=metric_name,
        methodology_version="davidson-bt-1",
        domain=domain,
        provider=provider,
        model=model,
        rating_elo=rating_elo,
        rating_bt=0.0,
        ci_half_width=25.0,
        votes_total=100,
        wins=50.0,
        losses=45.0,
        ties=5.0,
        status=SnapshotStatus.USABLE,
    )


def test_get_latest_ratings_returns_only_latest_board(
    arena_pg: psycopg.Connection[Any],
) -> None:
    _apply_migrations(arena_pg)

    async def _run() -> None:
        pool = await _make_pool(arena_pg)
        try:
            store = ArenaStore(pool)
            # Older board, then a newer one (distinct computed_at via now()).
            await store.insert_snapshot_board([_snapshot("cartesia", "sonic-3.5", 1000.0)])
            await asyncio.sleep(0.05)
            await store.insert_snapshot_board(
                [
                    _snapshot("cartesia", "sonic-3.5", 1200.0),
                    _snapshot("openai", "gpt-4o-mini-tts", 1100.0),
                ]
            )
            # A different metric must not leak into the naturalness board.
            await store.insert_snapshot_board(
                [_snapshot("cartesia", "sonic-3.5", 999.0, metric_name="other")]
            )

            ratings = await store.get_latest_ratings(metric_name="naturalness", domain="all")
            assert set(ratings) == {("cartesia", "sonic-3.5"), ("openai", "gpt-4o-mini-tts")}
            assert ratings[("cartesia", "sonic-3.5")].rating_elo == 1200.0
            assert ratings[("cartesia", "sonic-3.5")].ci_half_width == 25.0
        finally:
            await pool.close()

    asyncio.run(_run())


def test_get_latest_ratings_empty_before_any_snapshot(
    arena_pg: psycopg.Connection[Any],
) -> None:
    _apply_migrations(arena_pg)

    async def _run() -> None:
        pool = await _make_pool(arena_pg)
        try:
            store = ArenaStore(pool)
            ratings = await store.get_latest_ratings(metric_name="naturalness", domain="all")
            assert ratings == {}
        finally:
            await pool.close()

    asyncio.run(_run())


def test_insert_and_read_battle(arena_pg: psycopg.Connection[Any]) -> None:
    _apply_migrations(arena_pg)

    async def _run() -> None:
        pool = await _make_pool(arena_pg)
        try:
            store = ArenaStore(pool)
            created = await store.insert_battle(_make_battle())
            assert created.id is not None
            assert created.created_at is not None
            fetched = await store.get_battle(created.id)
            assert fetched == created
        finally:
            await pool.close()

    asyncio.run(_run())


def test_count_battles_today(arena_pg: psycopg.Connection[Any]) -> None:
    _apply_migrations(arena_pg)

    async def _run() -> None:
        pool = await _make_pool(arena_pg)
        try:
            store = ArenaStore(pool)
            assert await store.count_battles_today() == 0
            await store.insert_battle(_make_battle())
            await store.insert_battle(_make_battle())
            assert await store.count_battles_today() == 2
        finally:
            await pool.close()

    asyncio.run(_run())


def test_upsert_vote_dedup_updates_in_place(arena_pg: psycopg.Connection[Any]) -> None:
    _apply_migrations(arena_pg)

    async def _run() -> None:
        pool = await _make_pool(arena_pg)
        try:
            store = ArenaStore(pool)
            battle = await store.insert_battle(_make_battle())
            assert battle.id is not None

            first = await store.upsert_vote(
                battle_id=battle.id,
                outcome=VoteOutcome.A_WIN,
                voter_type=VoterType.LABELER,
                voter_id="labeler-1",
            )
            second = await store.upsert_vote(
                battle_id=battle.id,
                outcome=VoteOutcome.B_WIN,
                voter_type=VoterType.LABELER,
                voter_id="labeler-1",
            )

            votes = await store.list_votes(battle_id=battle.id)
            assert len(votes) == 1
            assert votes[0].outcome == VoteOutcome.B_WIN
            assert second.id == first.id
            assert second.updated_at is not None
            assert second.created_at is not None
            assert second.updated_at >= second.created_at

        finally:
            await pool.close()

    asyncio.run(_run())


def test_self_battle_rejected(arena_pg: psycopg.Connection[Any]) -> None:
    _apply_migrations(arena_pg)

    async def _run() -> None:
        pool = await _make_pool(arena_pg)
        try:
            store = ArenaStore(pool)
            with pytest.raises(psycopg.errors.CheckViolation):
                await store.insert_battle(
                    Battle(
                        provider_a="cartesia",
                        model_a="sonic-3.5",
                        provider_b="cartesia",
                        model_b="sonic-3.5",
                        prompt_text="x",
                        audio_a_url="https://example.test/a.wav",
                        audio_b_url="https://example.test/b.wav",
                    )
                )
        finally:
            await pool.close()

    asyncio.run(_run())


def test_list_battles_filters_by_domain(arena_pg: psycopg.Connection[Any]) -> None:
    _apply_migrations(arena_pg)

    async def _run() -> None:
        pool = await _make_pool(arena_pg)
        try:
            store = ArenaStore(pool)
            general = _make_battle()
            support = _make_battle(provider_b="openai", model_b="gpt-4o-mini-tts")
            support = support.model_copy(update={"domain": "support"})
            await store.insert_battle(general)
            await store.insert_battle(support)

            support_only = await store.list_battles(domain="support")
            assert len(support_only) == 1
            assert support_only[0].domain == "support"
            assert len(await store.list_battles()) == 2
        finally:
            await pool.close()

    asyncio.run(_run())


def test_list_votes_scopes_to_battle(arena_pg: psycopg.Connection[Any]) -> None:
    _apply_migrations(arena_pg)

    async def _run() -> None:
        pool = await _make_pool(arena_pg)
        try:
            store = ArenaStore(pool)
            battle_one = await store.insert_battle(_make_battle())
            battle_two = await store.insert_battle(
                _make_battle(provider_b="openai", model_b="gpt-4o-mini-tts")
            )
            assert battle_one.id is not None
            assert battle_two.id is not None

            await store.upsert_vote(
                battle_id=battle_one.id,
                outcome=VoteOutcome.A_WIN,
                voter_type=VoterType.LABELER,
                voter_id="labeler-1",
            )
            await store.upsert_vote(
                battle_id=battle_two.id,
                outcome=VoteOutcome.TIE,
                voter_type=VoterType.LABELER,
                voter_id="labeler-1",
            )

            assert len(await store.list_votes(battle_id=battle_one.id)) == 1
            assert len(await store.list_votes()) == 2
            assert len(await store.list_votes(limit=1)) == 1
        finally:
            await pool.close()

    asyncio.run(_run())
