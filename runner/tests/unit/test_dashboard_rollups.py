"""Rollup rebuilds: grouping, the slot queue, draining, retries, and retention."""

import asyncio
from datetime import UTC, datetime, timedelta
from importlib import import_module
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import psycopg
import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.db import dashboard_rollups
from coval_bench.db.dashboard_rollups import (
    DrainResult,
    drain_rollup_queue,
    floor_rollup,
)
from coval_bench.db.models import MetricValue, RunStatus
from coval_bench.db.writer import RunWriter
from coval_bench.registries import Benchmark
from coval_bench.s2s.conditions import (
    PLATFORM_SLUGS,
    ROOMS,
    SCENARIO_SLUGS,
    TIERS,
    Condition,
    pooled_dataset_id,
    room_dataset_id,
    scenario_dataset_id,
)
from tests.unit import test_db_storage as storage
from tests.unit.conftest import apply_migrations

pg_conn = postgresql("pg_proc")


def test_floor_follows_the_grain() -> None:
    at = datetime(2026, 9, 14, 13, 59, 41, tzinfo=UTC)
    assert floor_rollup(at, "1h") == datetime(2026, 9, 14, 13, tzinfo=UTC)
    assert floor_rollup(at, "4h") == datetime(2026, 9, 14, 12, tzinfo=UTC)
    assert floor_rollup(at, "run") == at


async def _seed_two_runs(pool: Any) -> int:
    writer = RunWriter(pool)
    run_id, first = await storage._observation(writer, sample="a")
    second_run_id, second = await storage._observation(writer, sample="b")
    for observation, value in ((first, 10.0), (second, 30.0)):
        evaluation = await storage._evaluation(writer, observation)
        evaluation_id = storage._required(evaluation.id)
        scale = value / 10.0
        values = [
            item.model_copy(update={"value": item.value * scale})
            for item in storage._wer_values(evaluation_id)
        ]
        await writer.complete_metric_evaluation(
            evaluation_id, values=values, finished_at=storage._NOW
        )
    await writer.finish_run(run_id, status=RunStatus.SUCCEEDED)
    await writer.finish_run(second_run_id, status=RunStatus.PARTIAL)
    return run_id


def _counts(evaluation_id: Any, counts: tuple[float, float, float, float]) -> list[MetricValue]:
    keys = ("substitution_count", "deletion_count", "insertion_count", "reference_words")
    return [
        MetricValue(metric_evaluation_id=evaluation_id, value_key=key, unit="count", value=v)
        for key, v in zip(keys, counts, strict=True)
    ]


async def _rows(pool: Any, sql: str, *params: Any) -> list[dict[str, Any]]:
    async with pool.connection() as conn:
        return list(await (await conn.execute(sql, params)).fetchall())


@pytest.mark.asyncio
async def test_finished_run_queues_its_slot_even_when_the_inline_rebuild_fails(
    pg_conn: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        run_id = await _seed_two_runs(pool)
        with monkeypatch.context() as patch:
            patch.setattr(dashboard_rollups, "FILL_ROLLUP_SQL", "SELECT nope")
            with pytest.raises(psycopg.Error):
                await RunWriter(pool).rebuild_run_rollup(run_id)
        queued = await _rows(pool, "SELECT slot_at FROM benchmarks_v2.dashboard_rollup_queue")
        assert [q["slot_at"] for q in queued] == [storage._NOW]

        await RunWriter(pool).rebuild_run_rollup(run_id)

        rows = await _rows(
            pool,
            """SELECT dataset_id, value_key, value_sum, sample_count, p50, p95, max_value,
                      wer_error_words, latest_run_at
               FROM benchmarks_v2.dashboard_rollups WHERE grain = 'run'
               ORDER BY dataset_id, value_key""",
        )
        assert [(r["dataset_id"], r["value_key"]) for r in rows] == [
            ("__all__", "primary"),
            ("observation-dataset", "primary"),
        ]
        primary = rows[0]
        assert (primary["value_sum"], primary["sample_count"]) == (40.0, 2)
        assert (primary["p50"], primary["p95"], primary["max_value"]) == (20.0, 29.0, 30.0)
        assert primary["wer_error_words"] == 40.0
        assert primary["latest_run_at"] == storage._NOW
        buckets = await _rows(
            pool, "SELECT count(*) AS n FROM benchmarks_v2.dashboard_rollups WHERE grain <> 'run'"
        )
        assert buckets == [{"n": 0}]
    finally:
        await pool.close()


def test_every_s2s_scenario_condition_pools_into_its_scenario_version(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    cases: list[tuple[str, str, str | None]] = []
    for slug in SCENARIO_SLUGS:
        for condition in (Condition.CLEAN, *TIERS):
            dataset = scenario_dataset_id(slug, condition)
            assert dataset is not None
            cases.append(("S2S", dataset, pooled_dataset_id(slug)))
            cases.append(("S2S", dataset.replace("-v1", "-v2"), pooled_dataset_id(slug, 2)))
    for slug in PLATFORM_SLUGS:
        for room in ROOMS:
            cases.append(("S2S", room_dataset_id(slug, room), pooled_dataset_id(slug)))
    cases += [
        ("S2S", pooled_dataset_id("workflow-bank"), None),
        ("S2S", "s2s-dental-v1", None),
        ("S2S", "s2s-dental-noisy-v1", None),
        ("S2S", "s2s-multiturn-v1", None),
        ("S2S", "s2s-v1", None),
        ("STT", "s2s-workflow-bank-v1", None),
        ("TTS", "tts-v2", None),
    ]
    for benchmark, dataset, expected in cases:
        row = pg_conn.execute(
            "SELECT benchmarks_v2.pooled_dataset_id(%s, %s)", (benchmark, dataset)
        ).fetchone()
        assert row is not None
        assert row[0] == expected, (benchmark, dataset)


@pytest.mark.asyncio
async def test_workflow_callers_also_roll_up_as_one_pooled_dataset(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        run_ids = []
        for sample, dataset, value in (
            ("a", "s2s-workflow-bank-v1", 10.0),
            ("b", "s2s-workflow-bank-hard-v1", 30.0),
            ("c", "s2s-bank-v1", 90.0),
        ):
            run_id, observation = await storage._observation(
                writer, sample=sample, observation_dataset=dataset, benchmark=Benchmark.S2S
            )
            evaluation = await storage._evaluation(writer, observation)
            evaluation_id = storage._required(evaluation.id)
            values = [
                item.model_copy(update={"value": item.value * value / 10.0})
                for item in storage._wer_values(evaluation_id)
            ]
            await writer.complete_metric_evaluation(
                evaluation_id, values=values, finished_at=storage._NOW
            )
            await writer.finish_run(run_id, status=RunStatus.SUCCEEDED)
            run_ids.append(run_id)
        await writer.rebuild_run_rollup(run_ids[0])

        rows = await _rows(
            pool,
            """SELECT dataset_id, sample_count, p50 FROM benchmarks_v2.dashboard_rollups
               WHERE grain = 'run' AND value_key = 'primary' ORDER BY dataset_id""",
        )
        assert [(r["dataset_id"], r["sample_count"]) for r in rows] == [
            ("__all__", 3),
            ("s2s-bank-all-conditions-v1", 1),
            ("s2s-bank-v1", 1),
            ("s2s-workflow-bank-all-conditions-v1", 2),
            ("s2s-workflow-bank-hard-v1", 1),
            ("s2s-workflow-bank-v1", 1),
        ]
        pooled = rows[3]
        assert pooled["p50"] == pytest.approx((rows[4]["p50"] + rows[5]["p50"]) / 2)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_the_pooling_migration_queues_recent_s2s_slots_only(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        slots = {}
        for sample, benchmark, age in (
            ("a", Benchmark.S2S, timedelta(days=2)),
            ("b", Benchmark.STT, timedelta(days=3)),
            ("c", Benchmark.S2S, timedelta(days=40)),
        ):
            run_id, _ = await storage._observation(writer, sample=sample, benchmark=benchmark)
            await writer.finish_run(run_id, status=RunStatus.SUCCEEDED)
            slot = floor_rollup(datetime.now(tz=UTC) - age, "1h")
            async with pool.connection() as conn:
                await conn.execute(
                    "UPDATE benchmarks_v2.runs SET scheduled_at = %s WHERE id = %s", (slot, run_id)
                )
                await conn.execute("DELETE FROM benchmarks_v2.dashboard_rollup_queue")
            slots[sample] = slot
        async with pool.connection() as conn:
            await conn.execute(_pooling_migration()._QUEUE_S2S_SLOTS)
        queued = await _rows(pool, "SELECT slot_at FROM benchmarks_v2.dashboard_rollup_queue")
        assert [row["slot_at"] for row in queued] == [slots["a"]]
    finally:
        await pool.close()


def test_the_pooling_downgrade_keeps_the_function_the_rollups_call(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    migration = _pooling_migration()
    with patch.object(migration, "op", SimpleNamespace(execute=pg_conn.execute)):
        migration.downgrade()
        migration.upgrade()
    row = pg_conn.execute(
        "SELECT benchmarks_v2.pooled_dataset_id('S2S', 's2s-workflow-bank-hard-v1')"
    ).fetchone()
    assert row is not None
    assert row[0] == "s2s-workflow-bank-all-conditions-v1"


def _pooling_migration() -> Any:
    return import_module("coval_bench.db.migrations.versions.20261107_0049_pool_s2s_conditions")


@pytest.mark.asyncio
async def test_fill_sums_wer_word_counts_and_leaves_uncounted_clips_out(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        run_id, first = await storage._observation(writer, sample="a")
        _, second = await storage._observation(writer, sample="b")
        _, third = await storage._observation(writer, sample="c")
        counts = {"a": (1, 1, 1, 20), "b": (0, 0, 2, 30), "c": None}
        for observation in (first, second, third):
            evaluation = await storage._evaluation(writer, observation)
            evaluation_id = storage._required(evaluation.id)
            clip = counts[observation.sample_id]
            percents = [v for v in storage._wer_values(evaluation_id) if v.unit == "percent"]
            values = percents + (_counts(evaluation_id, clip) if clip else [])
            await writer.complete_metric_evaluation(
                evaluation_id, values=values, finished_at=storage._NOW
            )
        await writer.finish_run(run_id, status=RunStatus.SUCCEEDED)
        for observation in (second, third):
            await writer.finish_run(
                storage._required(observation.run_id), status=RunStatus.SUCCEEDED
            )
        await writer.rebuild_run_rollup(run_id)
        [row] = await _rows(
            pool,
            """SELECT sample_count, wer_error_words, wer_reference_words
               FROM benchmarks_v2.dashboard_rollups
               WHERE grain = 'run' AND dataset_id = '__all__'""",
        )
        assert (row["sample_count"], row["wer_error_words"], row["wer_reference_words"]) == (
            2,
            5.0,
            50.0,
        )
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_drain_rebuilds_slot_and_both_buckets_then_dequeues(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        run_id = await _seed_two_runs(pool)
        await RunWriter(pool).rebuild_run_rollup(run_id)
        as_of = storage._NOW + timedelta(hours=5)

        assert await drain_rollup_queue(pool, as_of=as_of) == DrainResult(1, 0)
        grains = await _rows(
            pool,
            """SELECT grain, bucket_at, sample_count FROM benchmarks_v2.dashboard_rollups
               WHERE dataset_id = '__all__' ORDER BY grain""",
        )
        assert [(g["grain"], g["bucket_at"], g["sample_count"]) for g in grains] == [
            ("1h", storage._NOW, 2),
            ("4h", storage._NOW, 2),
            ("run", storage._NOW, 2),
        ]
        assert await _rows(pool, "SELECT 1 FROM benchmarks_v2.dashboard_rollup_queue") == []
        assert await drain_rollup_queue(pool, as_of=as_of) == DrainResult(0, 0)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_drain_stops_at_the_deadline_and_keeps_failed_slots_queued(
    pg_conn: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        slots = [storage._NOW + timedelta(hours=offset) for offset in (0, 1)]
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO benchmarks_v2.dashboard_rollup_queue (slot_at) VALUES (%s), (%s)",
                slots,
            )
        as_of = storage._NOW + timedelta(days=1)
        expired = await drain_rollup_queue(
            pool, as_of=as_of, deadline=asyncio.get_running_loop().time() - 1
        )
        assert expired == DrainResult(0, 2)

        monkeypatch.setattr(dashboard_rollups, "FILL_ROLLUP_SQL", "SELECT nope")
        with pytest.raises(psycopg.Error):
            await drain_rollup_queue(pool, as_of=as_of)
        queued = await _rows(pool, "SELECT slot_at FROM benchmarks_v2.dashboard_rollup_queue")
        assert [q["slot_at"] for q in queued] == slots
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_drain_batches_slots_and_prunes_past_retention(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        run_id = await _seed_two_runs(pool)
        await RunWriter(pool).rebuild_run_rollup(run_id)
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO benchmarks_v2.dashboard_rollup_queue (slot_at) VALUES (%s), (%s)",
                (storage._NOW + timedelta(minutes=30), storage._NOW - timedelta(days=31)),
            )
        assert await drain_rollup_queue(pool, as_of=storage._NOW) == DrainResult(3, 0)
        hour_rows = await _rows(
            pool,
            "SELECT count(*) AS n FROM benchmarks_v2.dashboard_rollups"
            " WHERE grain = '1h' AND bucket_at = %s AND dataset_id = '__all__'",
            storage._NOW,
        )
        assert hour_rows == [{"n": 1}]
        assert await drain_rollup_queue(pool, as_of=storage._NOW + timedelta(days=31)) == (
            DrainResult(0, 0)
        )
        assert await _rows(pool, "SELECT 1 FROM benchmarks_v2.dashboard_rollups") == []
    finally:
        await pool.close()
