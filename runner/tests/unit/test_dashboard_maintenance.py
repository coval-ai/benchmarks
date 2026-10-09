"""The hourly maintenance job: fill closed buckets, then publish summaries."""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.db import dashboard_aggregates
from coval_bench.db.dashboard_aggregates import MaintenanceResult, repair_dashboard_aggregates
from coval_bench.db.dashboard_buckets import FillResult
from coval_bench.db.dashboard_summaries import RefreshResult
from coval_bench.db.models import RunStatus
from coval_bench.db.writer import RunWriter
from tests.unit import test_normalized_db_writer as storage
from tests.unit.conftest import apply_migrations

pg_conn = postgresql("pg_proc")


@pytest.mark.asyncio
async def test_reconciliation_fills_then_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = MagicMock()
    fill = AsyncMock(return_value=FillResult(3, 1))
    refresh = AsyncMock(return_value=RefreshResult("published", 7))
    monkeypatch.setattr(dashboard_aggregates, "fill_closed_buckets", fill)
    monkeypatch.setattr(dashboard_aggregates, "refresh_summary_snapshots", refresh)
    at = datetime(2026, 9, 16, tzinfo=UTC)

    result = await dashboard_aggregates.reconcile_dashboard_aggregates(pool, as_of=at)

    assert (result.filled, result.remaining, result.summary) == (
        3,
        1,
        RefreshResult("published", 7),
    )
    assert fill.await_args is not None and fill.await_args.kwargs["as_of"] == at
    refresh.assert_awaited_once_with(pool, as_of=at)


@pytest.mark.asyncio
async def test_fill_failure_still_publishes_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = MagicMock()
    monkeypatch.setattr(
        dashboard_aggregates, "fill_closed_buckets", AsyncMock(side_effect=RuntimeError("boom"))
    )
    refresh = AsyncMock(return_value=RefreshResult("published", 1))
    monkeypatch.setattr(dashboard_aggregates, "refresh_summary_snapshots", refresh)

    with pytest.raises(RuntimeError, match="missing buckets are retained"):
        await dashboard_aggregates.reconcile_dashboard_aggregates(
            pool, as_of=datetime(2026, 9, 16, tzinfo=UTC)
        )
    refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconciliation_requires_timezone() -> None:
    with pytest.raises(ValueError, match="timezone"):
        await dashboard_aggregates.reconcile_dashboard_aggregates(
            MagicMock(), as_of=datetime(2026, 9, 16)
        )


def test_aggregation_fingerprint_is_hash_seed_independent() -> None:
    code = (
        "from coval_bench.db.dashboard_contracts import aggregation_fingerprint; "
        "print(aggregation_fingerprint())"
    )
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.append(
            subprocess.check_output(  # noqa: S603
                [sys.executable, "-c", code], env=env, text=True
            ).strip()
        )
    assert outputs[0] == outputs[1]


@pytest.mark.asyncio
async def test_repair_timestamp_move_rebuilds_old_and_new_buckets(
    pg_conn: psycopg.Connection[Any],
) -> None:
    apply_migrations(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        run_id, observation = await storage._observation(writer)
        evaluation = await storage._evaluation(writer, observation)
        evaluation_id = storage._required(evaluation.id)
        await writer.complete_metric_evaluation(
            evaluation_id, values=storage._wer_values(evaluation_id), finished_at=storage._NOW
        )
        await writer.finish_run(run_id, status=RunStatus.SUCCEEDED)
        await writer.refresh_metric_values_bucket(run_id)
        old = storage._NOW
        new = old + timedelta(hours=2)
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE benchmarks_v2.runs SET scheduled_at=%s WHERE id=%s", (new, run_id)
            )
            await conn.commit()
        result = await repair_dashboard_aggregates(
            pool, buckets=[old, new], as_of=new + timedelta(hours=5)
        )
        assert isinstance(result, MaintenanceResult) and result.filled == 3
        async with pool.connection() as conn:
            source = await (
                await conn.execute(
                    """SELECT bucket_at, dataset_id, count(*) AS n
                FROM benchmarks_v2.metric_values_by_bucket
                WHERE bucket_at IN (%s, %s)
                GROUP BY bucket_at, dataset_id ORDER BY bucket_at, dataset_id""",
                    (old, new),
                )
            ).fetchall()
            buckets = await (
                await conn.execute(
                    """SELECT interval_seconds, bucket_at, dataset_id, count(*) AS n
                FROM benchmarks_v2.dashboard_bucket_aggregates
                GROUP BY interval_seconds, bucket_at, dataset_id
                ORDER BY interval_seconds, bucket_at, dataset_id""",
                )
            ).fetchall()
        assert source == [
            {"bucket_at": new, "dataset_id": "__all__", "n": 4},
            {"bucket_at": new, "dataset_id": "observation-dataset", "n": 4},
        ]
        assert buckets == [
            {"interval_seconds": 3600, "bucket_at": new, "dataset_id": "__all__", "n": 4},
            {
                "interval_seconds": 3600,
                "bucket_at": new,
                "dataset_id": "observation-dataset",
                "n": 4,
            },
            {"interval_seconds": 14400, "bucket_at": old, "dataset_id": "__all__", "n": 4},
            {
                "interval_seconds": 14400,
                "bucket_at": old,
                "dataset_id": "observation-dataset",
                "n": 4,
            },
        ]
    finally:
        await pool.close()
