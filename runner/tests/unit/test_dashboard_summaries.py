"""Focused tests for summary rule validation and refresh orchestration."""
# ruff: noqa: E501

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from pytest_postgresql.factories import postgresql
from structlog.testing import capture_logs

from coval_bench.db.dashboard_summaries import (
    SUMMARY_VIEWS,
    RefreshResult,
    validate_summary_rules,
)
from coval_bench.registries.metrics import METRIC_VALUE_CONTRACTS, Metric

from .conftest import apply_migrations, open_pool

summary_pg = postgresql("pg_proc")


def test_summary_views_cover_every_saved_window() -> None:
    assert SUMMARY_VIEWS == {
        "24h": "benchmarks_v2.normalized_results_24h",
        "7d": "benchmarks_v2.normalized_results_7d",
        "30d": "benchmarks_v2.normalized_results_30d",
    }


def test_registered_contracts_are_supported() -> None:
    validate_summary_rules()


def test_changed_wer_ratio_contract_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = METRIC_VALUE_CONTRACTS[(Metric.WER, "v1")]
    monkeypatch.setitem(
        METRIC_VALUE_CONTRACTS,
        (Metric.WER, "v1"),
        original.model_copy(update={"ratio_scale": 1}),
    )
    with pytest.raises(ValueError, match="ratio contract"):
        validate_summary_rules()


def test_naive_as_of_is_rejected() -> None:
    # The refresh boundary is persisted and must have an unambiguous timezone.
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock

    from coval_bench.db.dashboard_summaries import refresh_summary_snapshots

    with pytest.raises(ValueError, match="timezone-aware"):
        asyncio.run(refresh_summary_snapshots(AsyncMock(), as_of=datetime(2026, 9, 14, 12)))


def test_empty_snapshot_publishes_state_and_all_views(
    summary_pg: psycopg.Connection[Any],
) -> None:
    apply_migrations(summary_pg)

    async def scenario() -> None:
        pool = await open_pool(summary_pg)
        try:
            from coval_bench.db.dashboard_summaries import refresh_summary_snapshots

            with capture_logs() as logs:
                result = await refresh_summary_snapshots(pool)
            assert result.status == "published"
            assert result.generation == 1
            completed = [
                event
                for event in logs
                if event["event"] == "dashboard_summary_view_refresh_completed"
            ]
            assert [event["window"] for event in completed] == ["24h", "7d", "30d"]
            assert all(event["elapsed_seconds"] >= 0 for event in completed)
            async with pool.connection() as conn:
                state = await conn.execute(
                    "SELECT generation,as_of,published_at "
                    "FROM benchmarks_v2.dashboard_summary_state"
                )
                row = await state.fetchone()
                assert row is not None and row["as_of"] is not None
                for view in SUMMARY_VIEWS.values():
                    count = await conn.execute(f"SELECT count(*) AS n FROM {view}")  # noqa: S608
                    assert (await count.fetchone())["n"] == 0
        finally:
            await pool.close()

    asyncio.run(scenario())


def test_refresh_materializes_wer_percentiles_and_pooled_values(
    summary_pg: psycopg.Connection[Any],
) -> None:
    """Exercise the real projection, frozen view SQL, and publication state."""
    apply_migrations(summary_pg)
    as_of = datetime(2026, 9, 14, 12, tzinfo=UTC)
    with summary_pg.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.runs
            (started_at,finished_at,runner_sha,dataset_id,dataset_sha256,status,scheduled_at)
            VALUES (%s,%s,'test','d',%s,'succeeded',%s) RETURNING id""",
            (as_of, as_of, "a" * 64, as_of),
        )
        run_row = cur.fetchone()
        assert run_row is not None
        run_id = run_row[0]
        for i, (value, ins, dele, sub, ref) in enumerate(
            ((10, 1, 1, 1, 10), (20, 1, 0, 0, 10), (30, 0, 1, 0, 10))
        ):
            oid = uuid4()
            eid = uuid4()
            captured = as_of - timedelta(hours=1)
            cur.execute(
                """INSERT INTO benchmarks_v2.benchmark_observations
                (id,run_id,dataset_id,dataset_sha256,sample_id,provider,model,benchmark,source_kind,captured_at,status)
                VALUES (%s,%s,'d',%s,%s,'p','m','STT','dataset_audio',%s,'succeeded')""",
                (oid, run_id, "b" * 64, str(i), captured),
            )
            cur.execute(
                """INSERT INTO benchmarks_v2.metric_evaluations
                (id,observation_id,metric_type,metric_version,executor,status,started_at,created_at,updated_at)
                VALUES (%s,%s,'WER','v1','test','queued',NULL,%s,%s)""",
                (eid, oid, captured, captured),
            )
            cur.execute(
                "UPDATE benchmarks_v2.metric_evaluations SET status='running',started_at=%s WHERE id=%s",
                (captured, eid),
            )
            values = [
                ("primary", "percent", value, "primary"),
                ("insertions", "percent", ins, "component"),
                ("deletions", "percent", dele, "component"),
                ("substitutions", "percent", sub, "component"),
                ("insertion_count", "count", ins, "component"),
                ("deletion_count", "count", dele, "component"),
                ("substitution_count", "count", sub, "component"),
                ("reference_words", "count", ref, "component"),
            ]
            cur.executemany(
                "INSERT INTO benchmarks_v2.metric_values (metric_evaluation_id,value_key,unit,value,value_role) VALUES (%s,%s,%s,%s,%s)",
                [(eid, *v) for v in values],
            )
            cur.execute(
                "UPDATE benchmarks_v2.metric_evaluations SET status='succeeded',finished_at=%s,updated_at=%s WHERE id=%s",
                (captured, captured, eid),
            )
    summary_pg.commit()

    async def scenario() -> None:
        pool = await open_pool(summary_pg)
        try:
            from coval_bench.db.dashboard_summaries import refresh_summary_snapshots

            result = await refresh_summary_snapshots(pool, as_of=as_of)
            assert result.status == "published"
            async with pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT * FROM benchmarks_v2.normalized_results_24h WHERE dataset_id='d'"
                )
                row = await cur.fetchone()
                assert row is not None
                assert row["sample_count"] == 3 and row["primary_sample_count"] == 3
                assert row["p25"] == pytest.approx(15) and row["p50"] == pytest.approx(20)
                assert row["p75"] == pytest.approx(25) and row["p90"] == pytest.approx(28)
                assert row["p95"] == pytest.approx(29) and row["p99"] == pytest.approx(29.8)
                assert row["avg_value"] == pytest.approx(500 / 30)
                assert row["pooled_insertions_pct"] == pytest.approx(200 / 30)
        finally:
            await pool.close()

    asyncio.run(scenario())


def test_unknown_summary_metric_rolls_back_state_and_views(
    summary_pg: psycopg.Connection[Any],
) -> None:
    """An eligible unknown source code cannot replace the prior snapshot."""
    apply_migrations(summary_pg)
    as_of = datetime(2026, 9, 14, 12, tzinfo=UTC)

    async def scenario() -> None:
        pool = await open_pool(summary_pg)
        try:
            from coval_bench.db.dashboard_summaries import refresh_summary_snapshots

            await refresh_summary_snapshots(pool, as_of=as_of)
            async with pool.connection() as conn:
                before = await (
                    await conn.execute(
                        "SELECT generation, as_of FROM benchmarks_v2.dashboard_summary_state"
                    )
                ).fetchone()
                before_view = await (
                    await conn.execute(
                        "SELECT count(*) AS n FROM benchmarks_v2.normalized_results_24h"
                    )
                ).fetchone()
            run_id = _insert_summary_run(summary_pg, as_of)
            observation_id = uuid4()
            evaluation_id = uuid4()
            with summary_pg.cursor() as cur:
                cur.execute(
                    "INSERT INTO benchmarks_v2.metrics (code,display_name) "
                    "VALUES ('UnknownSummaryMetric','Unknown summary metric')"
                )
                cur.execute(
                    """INSERT INTO benchmarks_v2.benchmark_observations
                    (id,run_id,dataset_id,dataset_sha256,sample_id,provider,model,benchmark,
                     source_kind,captured_at,status)
                    VALUES (%s,%s,'d',%s,'unknown','p','m','STT','dataset_audio',%s,'succeeded')""",
                    (observation_id, run_id, "c" * 64, as_of - timedelta(hours=1)),
                )
                cur.execute(
                    """INSERT INTO benchmarks_v2.metric_evaluations
                    (id,observation_id,metric_type,metric_version,executor,status,
                     started_at,created_at,updated_at)
                    VALUES (%s,%s,'UnknownSummaryMetric','v1','test','queued',NULL,%s,%s)""",
                    (evaluation_id, observation_id, as_of, as_of),
                )
                cur.execute(
                    "UPDATE benchmarks_v2.metric_evaluations SET status='running', started_at=%s WHERE id=%s",
                    (as_of, evaluation_id),
                )
                cur.execute(
                    """INSERT INTO benchmarks_v2.metric_values
                    (metric_evaluation_id,value_key,unit,value,value_role)
                    VALUES (%s,'primary','seconds',10,'primary')""",
                    (evaluation_id,),
                )
                cur.execute(
                    "UPDATE benchmarks_v2.metric_evaluations SET status='succeeded', started_at=%s, finished_at=%s WHERE id=%s",
                    (as_of, as_of, evaluation_id),
                )
            summary_pg.commit()
            with pytest.raises(ValueError, match="UnknownSummaryMetric"):
                await refresh_summary_snapshots(pool, as_of=as_of)
            async with pool.connection() as conn:
                after = await (
                    await conn.execute(
                        "SELECT generation, as_of FROM benchmarks_v2.dashboard_summary_state"
                    )
                ).fetchone()
                after_view = await (
                    await conn.execute(
                        "SELECT count(*) AS n FROM benchmarks_v2.normalized_results_24h"
                    )
                ).fetchone()
            assert after == before
            assert after_view == before_view
        finally:
            await pool.close()

    def _insert_summary_run(conn: psycopg.Connection[Any], timestamp: datetime) -> int:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO benchmarks_v2.runs
                (started_at,finished_at,runner_sha,dataset_id,dataset_sha256,status,scheduled_at)
                VALUES (%s,%s,'summary-test','d',%s,'succeeded',%s) RETURNING id""",
                (timestamp, timestamp, "d" * 64, timestamp),
            )
            row = cur.fetchone()
            assert row is not None
            return int(row[0])

    asyncio.run(scenario())


def test_refresh_result_is_immutable_and_typed() -> None:
    result = RefreshResult("published", 4)
    assert result.status == "published"
    assert result.generation == 4
    with pytest.raises(AttributeError):
        result.generation = 5  # type: ignore[misc]
