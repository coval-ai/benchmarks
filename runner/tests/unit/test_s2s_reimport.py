# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Replacing stored S2S runs: the swap on real Postgres, and the per-run decision."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import psycopg.rows
import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql

from coval_bench.db.models import (
    Benchmark,
    MetricEvaluation,
    MetricExecutor,
    Observation,
    ObservationSourceKind,
    ObservationStatus,
    ProcessingStatus,
    RunStatus,
)
from coval_bench.db.writer import RunWriter
from coval_bench.registries import Metric
from coval_bench.s2s.reimport import StoredRun, SwapRefused, decide, stored_runs, swap_runs

pg_conn = postgresql("pg_proc")
_INI_PATH = Path(__file__).parents[2] / "alembic.ini"
_NOW = datetime(2026, 11, 8, tzinfo=UTC)
_SHA = "a" * 64
_METRICS = (Metric.CALL_LENGTH, Metric.CHARACTER_ERROR_RATE)


def _dsn(conn: psycopg.Connection[Any]) -> str:
    info = conn.info
    auth = f"{info.user}:{info.password}@" if info.password else f"{info.user}@"
    return (
        f"postgresql://{auth}{info.host or 'localhost'}:{info.port or 5432}/{info.dbname or 'test'}"
    )


async def _pool(conn: psycopg.Connection[Any]) -> AsyncConnectionPool[Any]:
    config = AlembicConfig(str(_INI_PATH))
    config.set_main_option(
        "sqlalchemy.url", _dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    config.attributes["allow_metric_code_cleanup"] = True
    alembic_command.upgrade(config, "head")
    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        conninfo=_dsn(conn), open=False, kwargs={"row_factory": psycopg.rows.dict_row}
    )
    await pool.open()
    return pool


async def _stored_run(
    writer: RunWriter, coval_run_id: str, calls: int, *, provider: str = "vapi"
) -> int:
    run = await writer.start_run(
        dataset_id="s2s-id-roulette-quiet-room-v1", dataset_sha256=_SHA, scheduled_at=_NOW
    )
    assert run.id is not None
    for i in range(calls):
        await _call(writer, run.id, f"{coval_run_id}/s{i}", provider=provider)
    await writer.finish_run(run.id, status=RunStatus.SUCCEEDED)
    return run.id


async def _call(writer: RunWriter, run_id: int, sample_id: str, *, provider: str = "vapi") -> None:
    observation = await writer.insert_observation(
        Observation(
            run_id=run_id,
            dataset_id="s2s-id-roulette-quiet-room-v1",
            dataset_sha256=_SHA,
            sample_id=sample_id,
            provider=provider,
            model="pinned",
            benchmark=Benchmark.S2S,
            source_kind=ObservationSourceKind.CONVERSATION_AUDIO,
            status=ObservationStatus.SUCCEEDED,
        )
    )
    assert observation.id is not None
    for metric in _METRICS:
        queued = await writer.insert_metric_evaluation(
            MetricEvaluation(
                observation_id=observation.id,
                metric_type=str(metric),
                metric_version="v1",
                executor=MetricExecutor.INLINE,
                status=ProcessingStatus.QUEUED,
            )
        )
        assert queued.id is not None
        await writer.start_metric_evaluation_exact(queued.id, started_at=_NOW)
        await writer.fail_metric_evaluation_exact(
            queued.id, started_at=_NOW, finished_at=_NOW, error="no value"
        )


async def _count(pool: AsyncConnectionPool[Any], sql: str) -> int:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(sql)
        row = await cur.fetchone()
        assert row is not None
        return int(row["count"])


@pytest.mark.asyncio
async def test_swap_deletes_exactly_the_replaced_runs(pg_conn: psycopg.Connection[Any]) -> None:
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        old = await _stored_run(writer, "R1", calls=2)
        new = await _stored_run(writer, "R1", calls=3)
        other = await _stored_run(writer, "R2", calls=1)
        stored = await stored_runs(pool, "vapi", "R1")
        assert {run_id: run.calls for run_id, run in stored.items()} == {old: 2, new: 3}

        await swap_runs(
            pool, provider="vapi", coval_run_id="R1", keep=new, replace={old: stored[old]}
        )

        assert await stored_runs(pool, "vapi", "R1") == {new: stored[new]}
        assert set(await stored_runs(pool, "vapi", "R2")) == {other}
        assert await _count(pool, "SELECT count(*) FROM benchmarks_v2.runs") == 2
        assert await _count(pool, "SELECT count(*) FROM benchmarks_v2.metric_evaluations") == 8
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_swap_changes_nothing_when_the_old_rows_moved(
    pg_conn: psycopg.Connection[Any],
) -> None:
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        old = await _stored_run(writer, "R1", calls=2)
        new = await _stored_run(writer, "R1", calls=3)
        planned = await stored_runs(pool, "vapi", "R1")
        await _call(writer, old, "R1/late")

        with pytest.raises(SwapRefused, match="changed since they were counted"):
            await swap_runs(
                pool, provider="vapi", coval_run_id="R1", keep=new, replace={old: planned[old]}
            )

        assert set(await stored_runs(pool, "vapi", "R1")) == {old, new}
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_swap_refuses_to_replace_the_kept_run(pg_conn: psycopg.Connection[Any]) -> None:
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        run = await _stored_run(writer, "R1", calls=1)
        stored = await stored_runs(pool, "vapi", "R1")
        with pytest.raises(SwapRefused):
            await swap_runs(pool, provider="vapi", coval_run_id="R1", keep=run, replace=stored)
        with pytest.raises(SwapRefused):
            await swap_runs(pool, provider="vapi", coval_run_id="R1", keep=run, replace={})
        assert set(await stored_runs(pool, "vapi", "R1")) == {run}
    finally:
        await pool.close()


_NAMED = frozenset({"CallLength", "CharacterErrorRate"})


def _run(run_id: int, calls: int, metrics: frozenset[str] = _NAMED) -> StoredRun:
    return StoredRun(run_id=run_id, calls=calls, counts={(m, "succeeded"): calls for m in metrics})


@pytest.mark.parametrize(
    ("stored", "calls_now", "action"),
    [
        ({}, 5, "refuse"),
        ({1: _run(1, 3)}, None, "refuse"),
        ({1: _run(1, 3, frozenset({"CallLength"}))}, 5, "refuse"),
        ({1: _run(1, 3)}, 5, "replace"),
        ({1: _run(1, 5)}, 5, "unchanged"),
        ({1: _run(1, 3), 2: _run(2, 5)}, 5, "finish"),
        ({1: _run(1, 3), 2: _run(2, 4)}, 5, "replace"),
    ],
)
def test_decide(stored: dict[int, StoredRun], calls_now: int | None, action: str) -> None:
    assert decide(stored, _NAMED, calls_now)[0] == action
