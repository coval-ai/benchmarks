# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Replacing stored S2S runs: the swap on real Postgres, and the per-run decision."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
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
from coval_bench.s2s.conditions import condition_for
from coval_bench.s2s.coval_runs import CovalRun
from coval_bench.s2s.reimport import (
    Candidate,
    StoredRun,
    SwapRefused,
    coval_rows,
    decide,
    discard_run,
    stored_runs,
    swap_runs,
)
from coval_bench.s2s.specs import AgentSpec

pg_conn = postgresql("pg_proc")
_INI_PATH = Path(__file__).parents[2] / "alembic.ini"
_NOW = datetime(2026, 11, 8, tzinfo=UTC)
_SHA = "a" * 64
_METRICS = (Metric.CALL_LENGTH, Metric.CHARACTER_ERROR_RATE)
_NAMED = frozenset(str(m) for m in _METRICS)


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
    writer: RunWriter,
    coval_run_id: str,
    calls: int,
    *,
    provider: str = "vapi",
    metrics: tuple[Metric, ...] = _METRICS,
) -> int:
    run = await writer.start_run(
        dataset_id="s2s-id-roulette-quiet-room-v1", dataset_sha256=_SHA, scheduled_at=_NOW
    )
    assert run.id is not None
    for i in range(calls):
        await _call(writer, run.id, f"{coval_run_id}/s{i}", provider=provider, metrics=metrics)
    await writer.finish_run(run.id, status=RunStatus.SUCCEEDED)
    return run.id


async def _call(
    writer: RunWriter,
    run_id: int,
    sample_id: str,
    *,
    provider: str = "vapi",
    metrics: tuple[Metric, ...] = _METRICS,
) -> None:
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
    for metric in metrics:
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
            pool,
            provider="vapi",
            coval_run_id="R1",
            keep=new,
            replace={old: stored[old]},
            named=_NAMED,
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
                pool,
                provider="vapi",
                coval_run_id="R1",
                keep=new,
                replace={old: planned[old]},
                named=_NAMED,
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
            await swap_runs(
                pool, provider="vapi", coval_run_id="R1", keep=run, replace=stored, named=_NAMED
            )
        with pytest.raises(SwapRefused):
            await swap_runs(
                pool, provider="vapi", coval_run_id="R1", keep=run, replace={}, named=_NAMED
            )
        assert set(await stored_runs(pool, "vapi", "R1")) == {run}
    finally:
        await pool.close()


def _run(run_id: int, rows: dict[str, int]) -> StoredRun:
    return StoredRun(
        run_id=run_id,
        calls=max(rows.values()),
        counts={(metric, "succeeded"): n for metric, n in rows.items()},
    )


_FULL = {"CallLength": 3, "CharacterErrorRate": 3}
_CUT = {"CallLength": 3, "CharacterErrorRate": 2}


@pytest.mark.parametrize(
    ("stored", "rows_now", "action"),
    [
        ({}, _FULL, "refuse"),
        ({1: _run(1, _CUT)}, None, "refuse"),
        ({1: _run(1, _CUT)}, {"CallLength": 3}, "refuse"),
        ({1: _run(1, {"CallLength": 3})}, _FULL, "refuse"),
        ({1: _run(1, _CUT)}, _FULL, "replace"),
        ({1: _run(1, _FULL)}, _FULL, "unchanged"),
        ({1: _run(1, _CUT), 2: _run(2, _FULL)}, _FULL, "finish"),
        ({1: _run(1, _CUT), 2: _run(2, _CUT)}, _FULL, "replace"),
    ],
)
def test_decide(stored: dict[int, StoredRun], rows_now: dict[str, int] | None, action: str) -> None:
    assert decide(stored, _NAMED, rows_now)[0] == action


@pytest.mark.asyncio
async def test_swap_refuses_a_new_run_missing_a_named_metric(
    pg_conn: psycopg.Connection[Any],
) -> None:
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        old = await _stored_run(writer, "R1", calls=2)
        new = await _stored_run(writer, "R1", calls=3, metrics=(Metric.CALL_LENGTH,))
        stored = await stored_runs(pool, "vapi", "R1")
        with pytest.raises(SwapRefused, match="not every named metric"):
            await swap_runs(
                pool,
                provider="vapi",
                coval_run_id="R1",
                keep=new,
                replace={old: stored[old]},
                named=_NAMED,
            )
        assert set(await stored_runs(pool, "vapi", "R1")) == {old, new}
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_discard_removes_only_the_new_run(pg_conn: psycopg.Connection[Any]) -> None:
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        old = await _stored_run(writer, "R1", calls=2)
        new = await _stored_run(writer, "R1", calls=3)
        mixed = await _stored_run(writer, "R1", calls=1)
        await _call(writer, mixed, "R2/s0")

        await discard_run(pool, provider="vapi", coval_run_id="R1", run_id=new)
        with pytest.raises(SwapRefused, match="other runs' rows"):
            await discard_run(pool, provider="vapi", coval_run_id="R1", run_id=mixed)

        assert set(await stored_runs(pool, "vapi", "R1")) == {old, mixed}
    finally:
        await pool.close()


def _candidate(dataset_id: str) -> Candidate:
    return Candidate(
        spec=AgentSpec(
            provider="vapi",
            model="pinned",
            agent_id="A",
            metric_ids={
                Metric.V2V: "lat",
                Metric.CALL_LENGTH: "len",
                Metric.CHARACTER_ERROR_RATE: "cer",
            },
            uses_shared_metrics=False,
        ),
        coval_run=CovalRun(run_id="R1", create_time=_NOW),
        dataset_id=dataset_id,
        dataset_sha256=_SHA,
        condition=condition_for(dataset_id),
    )


def _client(metrics: dict[str, Any], output_ids: list[str]) -> httpx.AsyncClient:
    body = {"run": {"results": {"metrics": metrics, "output_ids": output_ids}}}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    return httpx.AsyncClient(transport=transport, base_url="https://coval.test")


def _values(*ids: str, value: float = 1.0) -> dict[str, Any]:
    return {"values": [{"simulation_output_id": i, "value": value} for i in ids]}


@pytest.mark.asyncio
async def test_coval_rows_follows_ingest_including_failed_latency_rows() -> None:
    candidate = _candidate("s2s-id-roulette-quiet-room-v1")
    metrics = {
        "lat": _values("s1", "s2"),
        "len": _values("s1", "s2", "s3", value=60.0),
        "cer": _values("s1", "s2", "s3", value=0.1),
    }
    named = frozenset({"V2V", "CallLength", "CharacterErrorRate"})
    async with _client(metrics, ["s1", "s2", "s3"]) as client:
        rows = await coval_rows(client, candidate, named)
    assert rows is not None
    assert rows["V2V"] == 3
    if candidate.condition.required is Metric.V2V:
        assert rows == {"V2V": 3, "CallLength": 2, "CharacterErrorRate": 2}
    else:
        assert rows == {"V2V": 3, "CallLength": 3, "CharacterErrorRate": 3}
    async with _client({}, ["s1"]) as client:
        assert await coval_rows(client, candidate, named) is None


@pytest.mark.asyncio
async def test_swap_refuses_an_old_run_holding_another_coval_run(
    pg_conn: psycopg.Connection[Any],
) -> None:
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        old = await _stored_run(writer, "R1", calls=2)
        await _call(writer, old, "R2/s0")
        new = await _stored_run(writer, "R1", calls=3)
        stored = await stored_runs(pool, "vapi", "R1")
        with pytest.raises(SwapRefused, match="also hold other runs' rows"):
            await swap_runs(
                pool,
                provider="vapi",
                coval_run_id="R1",
                keep=new,
                replace={old: stored[old]},
                named=_NAMED,
            )
        assert set(await stored_runs(pool, "vapi", "R2")) == {old}
    finally:
        await pool.close()
