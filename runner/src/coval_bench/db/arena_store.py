# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``ArenaStore`` — typed read/write helpers for the ``arena`` schema.

Mirrors ``db/writer.py``: async, uses the shared psycopg pool, all SQL is
parameterised (``%s``) — no string interpolation with caller data. Raw votes
are the source of truth; ratings are refit from them elsewhere.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

from coval_bench.db.models import Battle, LeaderboardSnapshot, Vote, VoteOutcome, VoterType

ArenaPool = AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]

_SNAPSHOT_LOCK_ACQUIRE = "SELECT pg_try_advisory_lock(hashtextextended('arena_snapshot', 0))"
_SNAPSHOT_LOCK_RELEASE = "SELECT pg_advisory_unlock(hashtextextended('arena_snapshot', 0))"


class ArenaStore:
    """Per-pool persistence helper for arena battles and votes."""

    def __init__(
        self,
        pool: ArenaPool,
    ) -> None:
        self._pool = pool

    async def insert_battle(self, battle: Battle) -> Battle:
        """Insert one ``arena.battles`` row; return it with ``id``/``created_at``."""
        sql = """
            INSERT INTO arena.battles
                (provider_a, model_a, provider_b, model_b, domain,
                 prompt_text, audio_a_url, audio_b_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, provider_a, model_a, provider_b, model_b, domain,
                      prompt_text, audio_a_url, audio_b_url, created_at
        """
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    sql,
                    (
                        battle.provider_a,
                        battle.model_a,
                        battle.provider_b,
                        battle.model_b,
                        battle.domain,
                        battle.prompt_text,
                        battle.audio_a_url,
                        battle.audio_b_url,
                    ),
                )
                row = await cur.fetchone()
                if row is None:  # pragma: no cover — unreachable after INSERT RETURNING
                    raise RuntimeError("INSERT INTO arena.battles returned no row")
            await conn.commit()
        return Battle.model_validate(dict(row))

    async def upsert_vote(
        self,
        *,
        battle_id: UUID,
        outcome: VoteOutcome,
        voter_type: VoterType,
        voter_id: str,
    ) -> Vote:
        """Record a vote, or update the existing one for this identity.

        The schema enforces one vote per ``(battle_id, voter_type, voter_id)``;
        ``ON CONFLICT DO UPDATE`` turns a re-vote into an update of that row
        (the ``BEFORE UPDATE`` trigger refreshes ``updated_at``).
        """
        sql = """
            INSERT INTO arena.votes (battle_id, outcome, voter_type, voter_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (battle_id, voter_type, voter_id)
            DO UPDATE SET outcome = EXCLUDED.outcome
            RETURNING id, battle_id, outcome, voter_type, voter_id,
                      created_at, updated_at
        """
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(sql, (battle_id, outcome, voter_type, voter_id))
                row = await cur.fetchone()
                if row is None:  # pragma: no cover — unreachable after UPSERT RETURNING
                    raise RuntimeError("UPSERT arena.votes returned no row")
            await conn.commit()
        return Vote.model_validate(dict(row))

    async def get_battle(self, battle_id: UUID) -> Battle | None:
        """Return the battle with this id, or ``None`` if it does not exist."""
        sql = """
            SELECT id, provider_a, model_a, provider_b, model_b, domain,
                   prompt_text, audio_a_url, audio_b_url, created_at
            FROM arena.battles
            WHERE id = %s
        """
        async with (
            self._pool.connection() as conn,
            conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
        ):
            await cur.execute(sql, (battle_id,))
            row = await cur.fetchone()
        return Battle.model_validate(dict(row)) if row is not None else None

    async def list_battles(
        self,
        *,
        domain: str | None = None,
        limit: int | None = 100,
    ) -> list[Battle]:
        """Return battles ordered by creation time, optionally filtered by domain.

        ``limit`` defaults to 100; pass ``None`` for no cap, which the rating
        refit needs so it can map every vote back to its battle.
        """
        clauses = [
            "SELECT id, provider_a, model_a, provider_b, model_b, domain,"
            " prompt_text, audio_a_url, audio_b_url, created_at FROM arena.battles",
        ]
        params: list[object] = []
        if domain is not None:
            clauses.append("WHERE domain = %s")
            params.append(domain)
        clauses.append("ORDER BY created_at")
        if limit is not None:
            clauses.append("LIMIT %s")
            params.append(limit)
        sql = "\n".join(clauses)
        async with (
            self._pool.connection() as conn,
            conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
        ):
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [Battle.model_validate(dict(row)) for row in rows]

    async def insert_snapshot_board(self, rows: list[LeaderboardSnapshot]) -> int:
        """Persist one leaderboard board atomically; return the row count.

        Every row is inserted in a single transaction with ``computed_at`` left
        to the DB default, so the whole board shares one timestamp and a reader
        taking ``max(computed_at)`` never sees a partial board. An empty board
        (no votes yet) writes nothing.
        """
        if not rows:
            return 0
        sql = """
            INSERT INTO arena.leaderboard_snapshots
                (metric_name, methodology_version, domain, provider, model,
                 rating_elo, rating_bt, ci_low, ci_high, ci_half_width,
                 votes_total, wins, losses, ties, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        async with (
            self._pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            for r in rows:
                await cur.execute(
                    sql,
                    (
                        r.metric_name,
                        r.methodology_version,
                        r.domain,
                        r.provider,
                        r.model,
                        r.rating_elo,
                        r.rating_bt,
                        r.ci_low,
                        r.ci_high,
                        r.ci_half_width,
                        r.votes_total,
                        r.wins,
                        r.losses,
                        r.ties,
                        r.status,
                    ),
                )
        return len(rows)

    @asynccontextmanager
    async def snapshot_lock(self) -> AsyncIterator[bool]:
        """Hold the global snapshot advisory lock; yield whether it was acquired.

        Non-blocking: ``pg_try_advisory_lock`` returns ``False`` at once if another
        run already holds the lock, so the caller can skip instead of queueing.
        The lock is session-scoped — it lives on this one connection and survives
        commits — so the connection is held open for the whole job and released on
        exit. Postgres drops it if the connection dies, so a crashed run never
        wedges the lock.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
                await cur.execute(_SNAPSHOT_LOCK_ACQUIRE)
                row = await cur.fetchone()
            await conn.commit()
            acquired = bool(row[0]) if row is not None else False
            try:
                yield acquired
            finally:
                if acquired:
                    async with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
                        await cur.execute(_SNAPSHOT_LOCK_RELEASE)
                    await conn.commit()

    async def list_votes(
        self,
        *,
        battle_id: UUID | None = None,
        limit: int | None = None,
    ) -> list[Vote]:
        """Return votes ordered by creation time.

        Scope to one battle with ``battle_id``. ``limit`` defaults to ``None``
        (no cap) because the rating refit must see every vote; pass an int to
        bound the result for a paginated caller.
        """
        clauses = [
            "SELECT id, battle_id, outcome, voter_type, voter_id,"
            " created_at, updated_at FROM arena.votes",
        ]
        params: list[object] = []
        if battle_id is not None:
            clauses.append("WHERE battle_id = %s")
            params.append(battle_id)
        clauses.append("ORDER BY created_at")
        if limit is not None:
            clauses.append("LIMIT %s")
            params.append(limit)
        sql = "\n".join(clauses)
        async with (
            self._pool.connection() as conn,
            conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
        ):
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [Vote.model_validate(dict(row)) for row in rows]
