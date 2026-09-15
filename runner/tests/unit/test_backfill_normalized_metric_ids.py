# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Real-Postgres coverage for the normalized metric identity backfill."""

# Embedded SQL and DSN assertions are intentionally kept readable.
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from click.testing import CliRunner
from pytest_postgresql.factories import postgresql

from coval_bench.migrations import backfill_normalized_metric_ids as metric_ids
from coval_bench.migrations.backfill_normalized_metric_ids import backfill

backfill_pg = postgresql("pg_proc")
_INI_PATH = Path(__file__).parents[2] / "alembic.ini"
_SHA = "a" * 64


def _dsn(conn: psycopg.Connection[Any]) -> str:
    info = conn.info
    auth = f"{info.user}:{info.password}@" if info.password else f"{info.user}@"
    return (
        f"postgresql://{auth}{info.host or 'localhost'}:{info.port or 5432}/{info.dbname or 'test'}"
    )


def _upgrade(conn: psycopg.Connection[Any], revision: str) -> None:
    config = AlembicConfig(str(_INI_PATH))
    config.set_main_option(
        "sqlalchemy.url", _dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    alembic_command.upgrade(config, revision)


def _seed_0035(conn: psycopg.Connection[Any]) -> None:
    _upgrade(conn, "20260914_0035")
    run = conn.execute(
        """INSERT INTO benchmarks_v2.runs
           (started_at, finished_at, runner_sha, dataset_id, dataset_sha256, status)
           VALUES (now(), now(), 'test', 'metric-id-test', %s, 'succeeded') RETURNING id""",
        (_SHA,),
    ).fetchone()[0]
    observation = conn.execute(
        """INSERT INTO benchmarks_v2.benchmark_observations
           (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark,
            source_kind, status)
           VALUES (%s, 'metric-id-test', %s, 'sample-1', 'provider', 'model', 'STT',
                   'dataset_audio', 'succeeded') RETURNING id""",
        (run, _SHA),
    ).fetchone()[0]
    evaluation = conn.execute(
        """INSERT INTO benchmarks_v2.metric_evaluations
           (observation_id, metric_type, metric_version, executor, status)
           VALUES (%s, 'WER', 'v1', 'test', 'queued') RETURNING id""",
        (observation,),
    ).fetchone()[0]
    conn.execute(
        """INSERT INTO benchmarks_v2.dashboard_metric_values
           (evaluation_id, observation_id, metric_type, metric_version, evaluation_variant,
            has_primary_role, value)
           VALUES (%s, %s, 'WER', 'v1', 'default', true, 1.0)""",
        (evaluation, observation),
    )
    conn.execute(
        """INSERT INTO benchmarks_v2.metric_values_by_bucket
           (provider, model, benchmark, dataset_id, metric_type, metric_version,
            evaluation_variant, value_key, unit, bucket_at, min_value, p25, p50, p75,
            max_value, value_sum, sample_count)
           VALUES ('provider', 'model', 'STT', 'metric-id-test', 'WER', 'v1', 'default',
                   'primary', '%', now(), 1, 1, 1, 1, 1, 1, 1)"""
    )
    conn.commit()
    _upgrade(conn, "head")


@pytest.fixture
def seeded(backfill_pg: Any) -> Any:
    _seed_0035(backfill_pg)
    backfill_pg.autocommit = True
    return backfill_pg


def _counts(conn: psycopg.Connection[Any]) -> tuple[int, int, int]:
    row = conn.execute(
        """SELECT (SELECT COUNT(*) FROM benchmarks_v2.metric_evaluations WHERE metric_id IS NULL),
                  (SELECT COUNT(*) FROM benchmarks_v2.dashboard_metric_values WHERE metric_id IS NULL),
                  (SELECT COUNT(*) FROM benchmarks_v2.metric_values_by_bucket WHERE metric_id IS NULL)"""
    ).fetchone()
    return tuple(int(value) for value in row)


def test_dry_run_does_not_mutate_any_normalized_table(seeded: Any) -> None:
    before = _counts(seeded)
    report = backfill(seeded, apply=False)
    assert report.mode == "preflight"
    assert report.status == "incomplete"
    assert _counts(seeded) == before == (1, 1, 1)


def test_apply_is_bounded_committed_and_resumable(seeded: Any) -> None:
    first = backfill(seeded, apply=True, batch_size=1, max_batches=1)
    assert first.updated["metric_evaluations"] == 1
    assert _counts(seeded) == (0, 1, 1)
    # A separate session proves the first bounded batch was committed rather
    # than merely visible through the caller's transaction.
    with psycopg.connect(_dsn(seeded), autocommit=True) as observer:
        assert observer.execute(
            "SELECT COUNT(*) FROM benchmarks_v2.metric_evaluations WHERE metric_id IS NULL"
        ).fetchone() == (0,)
    second = backfill(seeded, apply=True, batch_size=1)
    assert second.status == "completed"
    assert second.updated["dashboard_metric_values"] == 1
    assert second.updated["metric_values_by_bucket"] == 1
    assert _counts(seeded) == (0, 0, 0)


def test_interrupted_batch_keeps_prior_commit(seeded: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    original = metric_ids._update_batch_safe
    calls = 0

    def interrupt_once(conn: Any, table: str, batch_size: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise psycopg.errors.QueryCanceled("simulated interruption")
        return original(conn, table, batch_size)

    monkeypatch.setattr(metric_ids, "_update_batch_safe", interrupt_once)
    report = backfill(seeded, apply=True, batch_size=1, max_runtime_seconds=0.1)
    assert report.updated["metric_evaluations"] == 1
    with psycopg.connect(_dsn(seeded), autocommit=True) as observer:
        assert observer.execute(
            "SELECT COUNT(*) FROM benchmarks_v2.metric_evaluations WHERE metric_id IS NULL"
        ).fetchone() == (0,)


def test_advisory_owner_lock_is_reported_and_released(seeded: Any) -> None:
    other = psycopg.connect(_dsn(seeded), autocommit=True)
    try:
        other.execute(
            "SELECT pg_advisory_lock(hashtextextended(%s, 0))", ("normalized_metric_ids_backfill",)
        )
        assert backfill(seeded).status == "busy"
        other.execute(
            "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
            ("normalized_metric_ids_backfill",),
        )
        assert backfill(seeded).lock_acquired is True
    finally:
        other.close()


def test_locked_rows_are_skipped_then_resume(seeded: Any) -> None:
    other = psycopg.connect(_dsn(seeded), autocommit=False)
    try:
        row_id = other.execute("SELECT id FROM benchmarks_v2.metric_evaluations").fetchone()[0]
        other.execute(
            "SELECT id FROM benchmarks_v2.metric_evaluations WHERE id = %s FOR UPDATE", (row_id,)
        )
        report = backfill(seeded, apply=True, max_runtime_seconds=2)
        assert report.pending["metric_evaluations"] == 1
        other.rollback()
        assert backfill(seeded, apply=True).pending["metric_evaluations"] == 0
    finally:
        other.close()


def test_unknown_code_fails_closed_without_known_writes(backfill_pg: Any) -> None:
    _upgrade(backfill_pg, "20260914_0035")
    run = backfill_pg.execute(
        """INSERT INTO benchmarks_v2.runs
           (started_at, finished_at, runner_sha, dataset_id, dataset_sha256, status)
           VALUES (now(), now(), 'test', 'unknown', %s, 'succeeded') RETURNING id""",
        (_SHA,),
    ).fetchone()[0]
    obs = backfill_pg.execute(
        """INSERT INTO benchmarks_v2.benchmark_observations
           (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark, source_kind, status)
           VALUES (%s, 'unknown', %s, 's', 'p', 'm', 'STT', 'dataset_audio', 'succeeded') RETURNING id""",
        (run, _SHA),
    ).fetchone()[0]
    backfill_pg.execute(
        """INSERT INTO benchmarks_v2.metric_evaluations
           (observation_id, metric_type, metric_version, executor, status)
           VALUES (%s, 'UNKNOWN', 'v1', 'test', 'queued')""",
        (obs,),
    )
    backfill_pg.commit()
    _upgrade(backfill_pg, "head")
    backfill_pg.autocommit = True
    report = backfill(backfill_pg, apply=True)
    assert report.status == "needs_reconciliation"
    assert report.updated == {table: 0 for table in report.updated}


def test_projection_accepts_effective_identity_of_pending_parent(seeded: Any) -> None:
    seeded.execute(
        """UPDATE benchmarks_v2.dashboard_metric_values
           SET metric_id = (SELECT id FROM benchmarks_v2.metrics WHERE code = 'WER')"""
    )
    report = backfill(seeded)
    assert report.status == "incomplete"
    assert report.verification_complete
    assert not any(report.mismatched_ids.values())
    assert report.pending["metric_evaluations"] == 1
    assert report.pending["dashboard_metric_values"] == 0


def test_reconciliation_timeout_returns_report_and_releases_lock(
    seeded: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded.execute("SET statement_timeout = '2s'")
    original_counts = metric_ids._counts
    calls = 0

    def slow_final_counts(conn: Any, table: str) -> tuple[int, int, int]:
        nonlocal calls
        calls += 1
        if calls == 4:
            conn.execute("SELECT pg_sleep(1)")
        return original_counts(conn, table)

    monkeypatch.setattr(metric_ids, "_counts", slow_final_counts)
    report = backfill(seeded, max_runtime_seconds=0.2)
    assert report.status == "time_limit"
    assert not report.verification_complete
    assert report.errors
    assert seeded.execute("SHOW statement_timeout").fetchone() == ("2s",)
    with psycopg.connect(_dsn(seeded), autocommit=True) as other:
        assert other.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
            ("normalized_metric_ids_backfill",),
        ).fetchone() == (True,)


def test_rejects_caller_transaction_before_writing(seeded: Any) -> None:
    with (
        psycopg.connect(_dsn(seeded)) as transactional,
        pytest.raises(ValueError, match="idle autocommit"),
    ):
        backfill(transactional, apply=True)


def test_pending_indexes_require_explicit_apply() -> None:
    result = CliRunner().invoke(
        metric_ids.backfill_normalized_metric_ids_cli, ["--create-pending-indexes"]
    )
    assert result.exit_code == 2
    assert "requires --apply" in result.output
