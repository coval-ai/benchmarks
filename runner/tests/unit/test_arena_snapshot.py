# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.arena.snapshot.

Uses ``pytest-postgresql`` (embedded ``pg_ctl``, no Docker) to spin up a real
Postgres. No remote DB is ever contacted. A separate session server (random
port) keeps these independent of the other db tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import psycopg
import psycopg.rows
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql, postgresql_proc

from coval_bench.arena.snapshot import run_snapshot
from coval_bench.db.arena_store import ArenaStore
from coval_bench.db.models import Battle, VoteOutcome, VoterType

snap_pg_proc = postgresql_proc(port=None)
snap_pg = postgresql("snap_pg_proc")

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


async def _reset(pool: AsyncConnectionPool[Any]) -> None:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("DELETE FROM arena.leaderboard_snapshots")
        await cur.execute("DELETE FROM arena.votes")
        await cur.execute("DELETE FROM arena.battles")
        await conn.commit()


async def _snapshot_row_count(pool: AsyncConnectionPool[Any]) -> int:
    async with (
        pool.connection() as conn,
        conn.cursor(row_factory=psycopg.rows.tuple_row) as cur,
    ):
        await cur.execute("SELECT count(*) FROM arena.leaderboard_snapshots")
        row = await cur.fetchone()
    assert row is not None
    return int(row[0])


def _battle(provider_b: str, model_b: str) -> Battle:
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


_VOTES = [VoteOutcome.A_WIN] * 5 + [VoteOutcome.B_WIN, VoteOutcome.TIE]


async def _seed_battle(store: ArenaStore, *, domain: str, provider_b: str, model_b: str) -> None:
    battle = await store.insert_battle(
        Battle(
            provider_a="cartesia",
            model_a="sonic-3.5",
            provider_b=provider_b,
            model_b=model_b,
            domain=domain,
            prompt_text="hello there",
            audio_a_url="https://example.test/a.wav",
            audio_b_url="https://example.test/b.wav",
        )
    )
    assert battle.id is not None
    for idx, outcome in enumerate(_VOTES):
        await store.upsert_vote(
            battle_id=battle.id,
            outcome=outcome,
            voter_type=VoterType.LABELER,
            voter_id=f"labeler-{idx + 1}",
        )


async def _seed_one_battle(store: ArenaStore) -> None:
    await _seed_battle(store, domain="general", provider_b="openai", model_b="gpt-4o-mini-tts")


def test_snapshot_persists_one_board(snap_pg: psycopg.Connection[Any]) -> None:
    _apply_migrations(snap_pg)

    async def _run() -> None:
        pool = await _make_pool(snap_pg)
        try:
            await _reset(pool)
            store = ArenaStore(pool)
            await _seed_one_battle(store)

            result = await run_snapshot(store, bootstrap_rounds=50, seed=0)
            assert result is not None
            assert len(result.models) == 2

            async with (
                pool.connection() as conn,
                conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
            ):
                await cur.execute(
                    "SELECT provider, model, metric_name, methodology_version, domain,"
                    " rating_elo, rating_bt, status, votes_total, computed_at"
                    " FROM arena.leaderboard_snapshots"
                )
                rows = await cur.fetchall()

            assert len(rows) == 2
            assert {r["metric_name"] for r in rows} == {"naturalness"}
            assert {r["domain"] for r in rows} == {"all"}
            assert {r["methodology_version"] for r in rows} == {"davidson-bt-001"}
            assert len({r["computed_at"] for r in rows}) == 1
            assert {(r["provider"], r["model"]) for r in rows} == {
                ("cartesia", "sonic-3.5"),
                ("openai", "gpt-4o-mini-tts"),
            }
            assert sum(r["votes_total"] for r in rows) == 2 * len(_VOTES)
        finally:
            await pool.close()

    asyncio.run(_run())


def test_snapshot_with_no_votes_writes_nothing(snap_pg: psycopg.Connection[Any]) -> None:
    _apply_migrations(snap_pg)

    async def _run() -> None:
        pool = await _make_pool(snap_pg)
        try:
            await _reset(pool)
            store = ArenaStore(pool)
            result = await run_snapshot(store, bootstrap_rounds=50, seed=0)
            assert result is not None
            assert result.models == []
            assert await _snapshot_row_count(pool) == 0
        finally:
            await pool.close()

    asyncio.run(_run())


def test_snapshot_skips_when_lock_is_held(snap_pg: psycopg.Connection[Any]) -> None:
    _apply_migrations(snap_pg)

    async def _run() -> None:
        pool = await _make_pool(snap_pg)
        try:
            await _reset(pool)
            store = ArenaStore(pool)
            await _seed_one_battle(store)

            async with store.snapshot_lock() as held:
                assert held
                skipped = await run_snapshot(store, bootstrap_rounds=50, seed=0)

            assert skipped is None
            assert await _snapshot_row_count(pool) == 0
        finally:
            await pool.close()

    asyncio.run(_run())


def test_snapshot_force_runs_despite_held_lock(snap_pg: psycopg.Connection[Any]) -> None:
    _apply_migrations(snap_pg)

    async def _run() -> None:
        pool = await _make_pool(snap_pg)
        try:
            await _reset(pool)
            store = ArenaStore(pool)
            await _seed_one_battle(store)

            async with store.snapshot_lock() as held:
                assert held
                forced = await run_snapshot(store, bootstrap_rounds=50, seed=0, force=True)

            assert forced is not None
            assert len(forced.models) == 2
            assert await _snapshot_row_count(pool) == 2
        finally:
            await pool.close()

    asyncio.run(_run())


def test_snapshot_scoped_to_domain_excludes_other_domains(snap_pg: psycopg.Connection[Any]) -> None:
    _apply_migrations(snap_pg)

    async def _run() -> None:
        pool = await _make_pool(snap_pg)
        try:
            await _reset(pool)
            store = ArenaStore(pool)
            await _seed_battle(
                store, domain="general", provider_b="openai", model_b="gpt-4o-mini-tts"
            )
            await _seed_battle(store, domain="support", provider_b="deepgram", model_b="aura-2")

            result = await run_snapshot(store, domain="support", bootstrap_rounds=50, seed=0)
            assert result is not None
            assert {entry.model_id for entry in result.models} == {
                "cartesia/sonic-3.5",
                "deepgram/aura-2",
            }

            async with (
                pool.connection() as conn,
                conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
            ):
                await cur.execute("SELECT domain, provider, model FROM arena.leaderboard_snapshots")
                rows = await cur.fetchall()

            assert {r["domain"] for r in rows} == {"support"}
            assert {(r["provider"], r["model"]) for r in rows} == {
                ("cartesia", "sonic-3.5"),
                ("deepgram", "aura-2"),
            }
        finally:
            await pool.close()

    asyncio.run(_run())
