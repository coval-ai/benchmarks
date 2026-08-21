# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Model lifecycle state — the database half of the model registry.

The registry (``coval_bench.registries.models``) says which models exist and
how to run them; ``benchmarks_v2.model_state`` says whether each one is
benchmarked (``running``) and whether the public site shows it (``shown``).
The registry's old five-value ``status`` collapsed onto these two booleans:
Active (running+shown), Hidden (running only, the old EARLY_ACCESS embargo),
Paused (shown only), Stopped (neither, the old RETIRED/PENDING).

Every consumer goes through :func:`fetch_model_states`, which joins the
registry against the table: a registry entry with no row gets
:data:`DEFAULT_STATE` (Hidden — a newly merged model runs under embargo until
someone flips it in the admin page), and a row whose model has left the
registry is dropped. Neither direction of drift can expose a hidden model.

State is written only by :func:`set_model_state`, which records every change
in ``model_state_history``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import psycopg.rows
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import MODEL_REGISTRY

# The registry's natural key: (benchmark, provider, model).
ModelKey = tuple[Benchmark, str, str]


class ModelState(BaseModel, frozen=True, extra="forbid"):
    """Whether one model is benchmarked and whether the site shows it."""

    running: bool
    shown: bool
    updated_by: str | None = None  # None: the implicit default, no row written yet
    updated_at: datetime | None = None


class ModelStateChange(BaseModel, frozen=True, extra="forbid"):
    """One recorded state change; old values are None when no row existed before."""

    old_running: bool | None
    old_shown: bool | None
    new_running: bool
    new_shown: bool
    changed_by: str
    changed_at: datetime


# A registry entry with no row: benchmarked under embargo (the old EARLY_ACCESS).
DEFAULT_STATE = ModelState(running=True, shown=False)


def registry_keys() -> list[ModelKey]:
    """The natural key of every current registry entry, in registry order."""
    return [(m.benchmark, m.provider, m.model) for m in MODEL_REGISTRY]


def assume_all_active() -> dict[ModelKey, ModelState]:
    """Every registry model treated as running and shown.

    For offline tools (scale tuning, dev fixtures) that need a roster without a
    database. Never a substitute for :func:`fetch_model_states` in serving code.
    """
    return dict.fromkeys(registry_keys(), ModelState(running=True, shown=True))


async def fetch_model_states(pool: AsyncConnectionPool[Any]) -> dict[ModelKey, ModelState]:
    """The state of every current registry entry, in registry order."""
    async with pool.connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(
            """
            SELECT benchmark, provider, model, running, shown, updated_by, updated_at
            FROM benchmarks_v2.model_state
            """
        )
        rows = await cur.fetchall()
    stored = {
        (Benchmark(r["benchmark"]), r["provider"], r["model"]): ModelState(
            running=r["running"],
            shown=r["shown"],
            updated_by=r["updated_by"],
            updated_at=r["updated_at"],
        )
        for r in rows
    }
    return {key: stored.get(key, DEFAULT_STATE) for key in registry_keys()}


async def set_model_state(
    pool: AsyncConnectionPool[Any],
    *,
    benchmark: Benchmark,
    provider: str,
    model: str,
    running: bool,
    shown: bool,
    changed_by: str,
) -> ModelState:
    """Upsert one model's state and append the change to the history.

    Both writes share a transaction; the old row is locked so concurrent
    changes serialize and each history entry records the true prior values.
    """
    params = {
        "benchmark": benchmark.value,
        "provider": provider,
        "model": model,
        "running": running,
        "shown": shown,
        "changed_by": changed_by,
    }
    async with (
        pool.connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
    ):
        await cur.execute(
            """
            SELECT running, shown FROM benchmarks_v2.model_state
            WHERE benchmark = %(benchmark)s AND provider = %(provider)s AND model = %(model)s
            FOR UPDATE
            """,
            params,
        )
        old = await cur.fetchone()
        await cur.execute(
            """
            INSERT INTO benchmarks_v2.model_state
                (benchmark, provider, model, running, shown, updated_by, updated_at)
            VALUES
                (%(benchmark)s, %(provider)s, %(model)s, %(running)s, %(shown)s,
                 %(changed_by)s, now())
            ON CONFLICT (benchmark, provider, model) DO UPDATE
                SET running = EXCLUDED.running, shown = EXCLUDED.shown,
                    updated_by = EXCLUDED.updated_by, updated_at = EXCLUDED.updated_at
            RETURNING running, shown, updated_by, updated_at
            """,
            params,
        )
        row = await cur.fetchone()
        assert row is not None  # noqa: S101 — INSERT ... RETURNING always yields a row
        await cur.execute(
            """
            INSERT INTO benchmarks_v2.model_state_history
                (benchmark, provider, model, old_running, old_shown,
                 new_running, new_shown, changed_by, changed_at)
            VALUES
                (%(benchmark)s, %(provider)s, %(model)s, %(old_running)s, %(old_shown)s,
                 %(running)s, %(shown)s, %(changed_by)s, %(changed_at)s)
            """,
            {
                **params,
                "old_running": old["running"] if old else None,
                "old_shown": old["shown"] if old else None,
                "changed_at": row["updated_at"],
            },
        )
    return ModelState(
        running=row["running"],
        shown=row["shown"],
        updated_by=row["updated_by"],
        updated_at=row["updated_at"],
    )


async def fetch_recent_history(
    pool: AsyncConnectionPool[Any], *, per_model: int = 10
) -> dict[ModelKey, list[ModelStateChange]]:
    """The most recent changes per model, newest first. Orphans are dropped."""
    async with pool.connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(
            """
            SELECT benchmark, provider, model, old_running, old_shown,
                   new_running, new_shown, changed_by, changed_at
            FROM (
                SELECT h.*, row_number() OVER (
                    PARTITION BY benchmark, provider, model
                    ORDER BY changed_at DESC, id DESC
                ) AS recency
                FROM benchmarks_v2.model_state_history h
            ) ranked
            WHERE recency <= %(per_model)s
            ORDER BY benchmark, provider, model, changed_at DESC, id DESC
            """,
            {"per_model": per_model},
        )
        rows = await cur.fetchall()
    known = set(registry_keys())
    history: dict[ModelKey, list[ModelStateChange]] = {}
    for r in rows:
        key = (Benchmark(r["benchmark"]), r["provider"], r["model"])
        if key not in known:
            continue
        history.setdefault(key, []).append(
            ModelStateChange(
                old_running=r["old_running"],
                old_shown=r["old_shown"],
                new_running=r["new_running"],
                new_shown=r["new_shown"],
                changed_by=r["changed_by"],
                changed_at=r["changed_at"],
            )
        )
    return history
