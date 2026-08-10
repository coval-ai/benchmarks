# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.arena.snapshot.

Uses ``pytest-postgresql`` (embedded ``pg_ctl``, no Docker) to spin up a real
Postgres. No remote DB is ever contacted.
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
from pytest_postgresql.factories import postgresql

from coval_bench.arena.snapshot import run_snapshot
from coval_bench.db.arena_store import ArenaStore
from coval_bench.db.models import Battle, Gender, VoteOutcome, VoterType

snap_pg = postgresql("pg_proc")  # shared server from conftest, own per-test DB

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
        voice_a="voice-a",
        voice_b="voice-b",
        gender=Gender.FEMALE,
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
            voice_a="voice-a",
            voice_b="voice-b",
            gender=Gender.FEMALE,
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
            assert {r["methodology_version"] for r in rows} == {"davidson-bt-002"}
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


def test_battle_voices_and_gender_survive_a_round_trip(snap_pg: psycopg.Connection[Any]) -> None:
    """Written by ``insert_battle``, read back by both battle getters.

    The INSERT and the two SELECTs carry independent column lists, so a column
    added to one and forgotten in another reads back as ``None`` rather than
    failing — this is the test that catches it.
    """
    _apply_migrations(snap_pg)

    async def _run() -> None:
        pool = await _make_pool(snap_pg)
        try:
            await _reset(pool)
            store = ArenaStore(pool)
            inserted = await store.insert_battle(_battle("openai", "gpt-4o-mini-tts"))

            assert inserted.voice_a == "voice-a"
            assert inserted.voice_b == "voice-b"
            assert inserted.gender is Gender.FEMALE

            assert inserted.id is not None
            fetched = await store.get_battle(inserted.id)
            assert fetched is not None
            assert (fetched.voice_a, fetched.voice_b) == ("voice-a", "voice-b")
            assert fetched.gender is Gender.FEMALE

            listed = await store.list_battles(limit=None)
            assert [b.gender for b in listed] == [Gender.FEMALE]
            assert [b.voice_a for b in listed] == ["voice-a"]
        finally:
            await pool.close()

    asyncio.run(_run())


def test_count_battles_by_gender_ignores_ungendered_rows(
    snap_pg: psycopg.Connection[Any],
) -> None:
    _apply_migrations(snap_pg)

    async def _run() -> None:
        pool = await _make_pool(snap_pg)
        try:
            await _reset(pool)
            store = ArenaStore(pool)
            assert await store.count_battles_by_gender() == {}

            await store.insert_battle(_battle("openai", "gpt-4o-mini-tts"))
            male = _battle("deepgram", "aura-2").model_copy(update={"gender": Gender.MALE})
            await store.insert_battle(male)
            legacy = _battle("rime", "mistv3").model_copy(update={"gender": None})
            await store.insert_battle(legacy)

            assert await store.count_battles_by_gender() == {Gender.FEMALE: 1, Gender.MALE: 1}
        finally:
            await pool.close()

    asyncio.run(_run())


def test_snapshot_excludes_pre_gender_battles(snap_pg: psycopg.Connection[Any]) -> None:
    """Cross-gender votes from the retired methodology never reach a new board.

    Their voices have since been replaced, so they are not evidence about the
    models as they sound now. The rows stay; only the board excludes them.
    """
    _apply_migrations(snap_pg)

    async def _run() -> None:
        pool = await _make_pool(snap_pg)
        try:
            await _reset(pool)
            store = ArenaStore(pool)

            legacy = await store.insert_battle(
                _battle("openai", "gpt-4o-mini-tts").model_copy(update={"gender": None})
            )
            assert legacy.id is not None
            for idx, outcome in enumerate(_VOTES):
                await store.upsert_vote(
                    battle_id=legacy.id,
                    outcome=outcome,
                    voter_type=VoterType.LABELER,
                    voter_id=f"labeler-{idx + 1}",
                )

            result = await run_snapshot(store, bootstrap_rounds=50, seed=0)
            assert result is not None
            assert result.models == []
            assert await _snapshot_row_count(pool) == 0

            # The battle and its votes are still on disk, just not in the board.
            assert len(await store.list_battles(limit=None)) == 1
            assert len(await store.list_votes()) == len(_VOTES)
        finally:
            await pool.close()

    asyncio.run(_run())


def _battle_columns(conn: psycopg.Connection[Any]) -> set[str]:
    conn.rollback()  # see the catalog as alembic left it, not this session's snapshot
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_schema = 'arena' AND table_name = 'battles'"
        )
        return {str(row[0]) for row in cur.fetchall()}


def test_battle_voice_migration_reverses(snap_pg: psycopg.Connection[Any]) -> None:
    """0016 adds three columns and takes exactly those three back off.

    A downgrade that leaves debris behind makes the migration unsafe to roll
    back, which is the only thing standing between a bad deploy and a restore.
    """
    _apply_migrations(snap_pg)
    cfg = _alembic_cfg(_async_dsn(snap_pg))
    added = {"voice_a", "voice_b", "gender"}

    after_upgrade = _battle_columns(snap_pg)
    assert added <= after_upgrade

    alembic_command.downgrade(cfg, "-1")
    after_downgrade = _battle_columns(snap_pg)
    assert added.isdisjoint(after_downgrade)
    assert after_downgrade == after_upgrade - added

    alembic_command.upgrade(cfg, "head")
    assert _battle_columns(snap_pg) == after_upgrade


def test_gender_check_constraint_rejects_other_values(snap_pg: psycopg.Connection[Any]) -> None:
    """Postgres refuses a bad gender even when the write bypasses Pydantic."""
    _apply_migrations(snap_pg)
    snap_pg.rollback()
    with snap_pg.cursor() as cur:
        try:
            cur.execute(
                "INSERT INTO arena.battles"
                " (provider_a, model_a, provider_b, model_b, prompt_text,"
                "  audio_a_url, audio_b_url, gender)"
                " VALUES ('a', 'm', 'b', 'n', 'p', 'x', 'y', 'nonbinary')"
            )
        except psycopg.errors.CheckViolation:
            pass
        else:  # pragma: no cover - the constraint is missing
            raise AssertionError("arena.battles accepted a gender outside the CHECK")
    snap_pg.rollback()
