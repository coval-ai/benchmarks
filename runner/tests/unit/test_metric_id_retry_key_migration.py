# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""PostgreSQL regression coverage for metric identity retry keys."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic, sleep
from typing import Any, cast

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from pytest_postgresql.factories import postgresql
from sqlalchemy import event
from sqlalchemy.engine import Engine

pg_conn = postgresql("pg_proc")


def _dsn(conn: psycopg.Connection[Any]) -> str:
    info = conn.info
    auth = f"{info.user}:{info.password}@" if info.password else f"{info.user}@"
    return (
        f"postgresql://{auth}{info.host or 'localhost'}:{info.port or 5432}/{info.dbname or 'test'}"
    )


def _migrate(conn: Any, revision: str, *, down: bool = False) -> None:
    conn.commit()
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    config.set_main_option(
        "sqlalchemy.url", _dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    (command.downgrade if down else command.upgrade)(config, revision)


@pytest.fixture
def seeded(pg_conn: Any) -> Any:
    _migrate(pg_conn, "20260921_0041")
    pg_conn.autocommit = True
    with pg_conn.transaction():
        run_id = pg_conn.execute(
            """INSERT INTO benchmarks_v2.runs
               (dataset_id, dataset_sha256, runner_sha, status, finished_at, scheduled_at)
               VALUES ('dataset', %s, 'test', 'succeeded', now(), date_trunc('hour', now()))
               RETURNING id""",
            ("a" * 64,),
        ).fetchone()[0]
        observation_id = pg_conn.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
               (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark,
                source_kind, status)
               VALUES (%s, 'dataset', %s, 'sample', 'provider', 'model', 'STT',
                       'dataset_audio', 'succeeded') RETURNING id""",
            (run_id, "b" * 64),
        ).fetchone()[0]
        evaluation_id = pg_conn.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations
               (observation_id, metric_type, metric_version, evaluation_variant, executor, status)
               VALUES (%s, 'WER', 'v1', 'default', 'inline', 'queued') RETURNING id""",
            (observation_id,),
        ).fetchone()[0]
        pg_conn.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET status='running', started_at=now() WHERE id=%s""",
            (evaluation_id,),
        )
        pg_conn.execute(
            """INSERT INTO benchmarks_v2.metric_values
               (metric_evaluation_id, value_key, unit, value, value_role)
               VALUES (%s, 'primary', 'percent', 10, 'primary')""",
            (evaluation_id,),
        )
        pg_conn.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET status='succeeded', finished_at=now() WHERE id=%s""",
            (evaluation_id,),
        )
        metric_id = pg_conn.execute(
            "SELECT metric_id FROM benchmarks_v2.metric_evaluations WHERE id=%s", (evaluation_id,)
        ).fetchone()[0]
        bucket = pg_conn.execute(
            "SELECT scheduled_at FROM benchmarks_v2.runs WHERE id=%s", (run_id,)
        ).fetchone()[0]
        pg_conn.execute(
            """INSERT INTO benchmarks_v2.metric_values_by_bucket
               (provider, model, benchmark, dataset_id, metric_id, metric_type, metric_version,
                evaluation_variant, value_key, unit, bucket_at, min_value, p25, p50, p75,
                max_value, value_sum, sample_count)
               VALUES ('provider', 'model', 'STT', 'dataset', %s, 'WER', 'v1', 'default',
                       'primary', 'percent', %s, 10, 10, 10, 10, 10, 10, 1)""",
            (metric_id, bucket),
        )
        pg_conn.execute(
            """UPDATE benchmarks_v2.dashboard_summary_state
               SET generation=7, as_of=now(), published_at=now(), definition_revision=2,
                   definition_fingerprint='published' WHERE id=true"""
        )
        pg_conn.execute(
            """INSERT INTO benchmarks_v2.dashboard_hourly_aggregates
               (provider, model, benchmark, dataset_id, metric_id, metric_type, metric_version,
                evaluation_variant, hour_at, primary_sum, sample_count, coverage_complete,
                source_count, definition_revision)
               VALUES ('provider', 'model', 'STT', 'dataset', %s, 'WER', 'v1', 'default',
                       %s, 10, 1, true, 1, 2)""",
            (metric_id, bucket),
        )
    return pg_conn


def _payloads(conn: Any) -> dict[str, list[Any]]:
    return {
        table: conn.execute(
            f"SELECT to_jsonb(t) FROM benchmarks_v2.{table} t ORDER BY 1"
        ).fetchall()
        for table in (
            "metric_evaluations",
            "dashboard_metric_values",
            "metric_values_by_bucket",
            "dashboard_hourly_aggregates",
            "dashboard_summary_state",
        )
    }


def _index_names(conn: Any) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            """SELECT indexname FROM pg_indexes WHERE schemaname='benchmarks_v2'
               AND indexname IN ('metric_evaluations_metric_identity_key',
                 'metric_values_by_bucket_metric_identity_key',
                 'dashboard_hourly_aggregates_metric_identity_key')"""
        ).fetchall()
    }


def _index_definition(conn: Any, name: str) -> tuple[Any, ...]:
    return cast(
        tuple[Any, ...],
        conn.execute(
            """SELECT i.indrelid::regclass::text, i.indisunique, i.indisvalid, i.indisready,
                  i.indpred IS NULL, i.indexprs IS NULL,
                  array_agg(a.attname ORDER BY x.ordinality)
           FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
           LEFT JOIN LATERAL unnest(i.indkey) WITH ORDINALITY x(attnum, ordinality) ON true
           LEFT JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=x.attnum
           WHERE c.oid=%s::regclass
           GROUP BY i.indrelid,i.indisunique,i.indisvalid,i.indisready,i.indpred,i.indexprs""",
            (f"benchmarks_v2.{name}",),
        ).fetchone(),
    )


def test_upgrade_keys_all_dimensions_and_variations(seeded: Any) -> None:
    _migrate(seeded, "20261002_0042")
    assert len(_index_names(seeded)) == 3
    sync_definition = seeded.execute(
        "SELECT pg_get_functiondef('benchmarks_v2.sync_metric_evaluation_identity()'::regprocedure)"
    ).fetchone()[0]
    assert "pg_advisory_xact_lock" in sync_definition
    assert _index_definition(seeded, "metric_evaluations_metric_identity_key") == (
        "benchmarks_v2.metric_evaluations",
        True,
        True,
        True,
        True,
        True,
        ["observation_id", "metric_id", "metric_version", "evaluation_variant"],
    )
    assert _index_definition(seeded, "metric_values_by_bucket_metric_identity_key") == (
        "benchmarks_v2.metric_values_by_bucket",
        True,
        True,
        True,
        True,
        True,
        [
            "provider",
            "model",
            "benchmark",
            "dataset_id",
            "metric_id",
            "metric_version",
            "evaluation_variant",
            "value_key",
            "bucket_at",
        ],
    )
    assert _index_definition(seeded, "dashboard_hourly_aggregates_metric_identity_key") == (
        "benchmarks_v2.dashboard_hourly_aggregates",
        True,
        True,
        True,
        True,
        True,
        [
            "provider",
            "model",
            "benchmark",
            "dataset_id",
            "metric_id",
            "metric_version",
            "evaluation_variant",
            "hour_at",
        ],
    )
    eval_row = seeded.execute(
        "SELECT observation_id, metric_id FROM benchmarks_v2.metric_evaluations LIMIT 1"
    ).fetchone()
    code_only_id = seeded.execute(
        """INSERT INTO benchmarks_v2.metric_evaluations
           (observation_id, metric_type, metric_version, evaluation_variant, executor, status)
           VALUES (%s, 'WER', 'v1', 'code-only', 'inline', 'queued') RETURNING metric_id""",
        (eval_row[0],),
    ).fetchone()[0]
    assert code_only_id == eval_row[1]
    with pytest.raises(psycopg.errors.UniqueViolation):
        seeded.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations
               (observation_id, metric_id, metric_type, metric_version, evaluation_variant, executor, status)
               VALUES (%s, %s, 'WER', 'v1', 'default', 'inline', 'queued')""",
            eval_row,
        )
    seeded.execute(
        """INSERT INTO benchmarks_v2.metric_evaluations
           (observation_id, metric_id, metric_type, metric_version, evaluation_variant, executor, status)
           VALUES (%s, %s, 'WER', 'v2', 'alternate', 'inline', 'queued')""",
        eval_row,
    )
    source = seeded.execute(
        """SELECT provider, model, benchmark, dataset_id, metric_id, bucket_at
           FROM benchmarks_v2.metric_values_by_bucket LIMIT 1"""
    ).fetchone()
    with pytest.raises(psycopg.errors.UniqueViolation):
        seeded.execute(
            """INSERT INTO benchmarks_v2.metric_values_by_bucket
               (provider, model, benchmark, dataset_id, metric_id, metric_type, metric_version,
                evaluation_variant, value_key, unit, bucket_at, min_value, p25, p50, p75,
                max_value, value_sum, sample_count)
               VALUES (%s,%s,%s,%s,%s,'WER','v1','default','primary','percent',%s,1,1,1,1,1,1,1)""",
            source,
        )


def test_downgrade_reupgrade_preserves_payloads_and_old_keys(seeded: Any) -> None:
    _migrate(seeded, "20261002_0042")
    before = _payloads(seeded)
    assert seeded.execute(
        """SELECT pg_get_triggerdef(oid) NOT LIKE '%WHEN (%'
           FROM pg_trigger WHERE tgname='metric_evaluations_validate_update'"""
    ).fetchone() == (True,)
    _migrate(seeded, "20260921_0041", down=True)
    assert _index_names(seeded) == set()
    assert _payloads(seeded) == before
    assert seeded.execute(
        """SELECT pg_get_triggerdef(oid) LIKE '%WHEN (%'
           FROM pg_trigger WHERE tgname='metric_evaluations_validate_update'"""
    ).fetchone() == (True,)
    sync_definition = seeded.execute(
        "SELECT pg_get_functiondef('benchmarks_v2.sync_metric_evaluation_identity()'::regprocedure)"
    ).fetchone()[0]
    assert "pg_advisory_xact_lock" not in sync_definition
    _migrate(seeded, "20261002_0042")
    assert _index_names(seeded) == {
        "metric_evaluations_metric_identity_key",
        "metric_values_by_bucket_metric_identity_key",
        "dashboard_hourly_aggregates_metric_identity_key",
    }
    assert _payloads(seeded) == before


def test_stale_named_indexes_are_recovered(seeded: Any) -> None:
    seeded.execute(
        """CREATE UNIQUE INDEX metric_evaluations_metric_identity_key
           ON benchmarks_v2.metric_evaluations (observation_id) WHERE status='queued'"""
    )
    _migrate(seeded, "20261002_0042")
    row = seeded.execute(
        """SELECT indisunique, indisvalid, indisready, indpred IS NULL, indexprs IS NULL
           FROM pg_index WHERE indexrelid='benchmarks_v2.metric_evaluations_metric_identity_key'::regclass"""
    ).fetchone()
    assert row == (True, True, True, True, True)


def test_partial_index_migration_retries_and_preserves_first_oid(seeded: Any) -> None:
    seen = False

    def interrupt_after_first_index(
        conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool
    ) -> None:
        nonlocal seen
        if (
            not seen
            and "CREATE UNIQUE INDEX CONCURRENTLY metric_evaluations_metric_identity_key"
            in statement
        ):
            seen = True
            raise RuntimeError("simulated migration interruption")

    event.listen(Engine, "after_cursor_execute", interrupt_after_first_index)
    try:
        with pytest.raises(RuntimeError, match="simulated migration interruption"):
            _migrate(seeded, "20261002_0042")
    finally:
        event.remove(Engine, "after_cursor_execute", interrupt_after_first_index)
    first_oid = seeded.execute(
        "SELECT 'benchmarks_v2.metric_evaluations_metric_identity_key'::regclass::oid"
    ).fetchone()[0]
    assert seeded.execute("SELECT version_num FROM alembic_version").fetchone() == (
        "20260921_0041",
    )
    _migrate(seeded, "20261002_0042")
    assert seeded.execute(
        "SELECT 'benchmarks_v2.metric_evaluations_metric_identity_key'::regclass::oid"
    ).fetchone() == (first_oid,)
    assert len(_index_names(seeded)) == 3


def test_identity_preflight_rejects_null_or_mismatch(seeded: Any) -> None:
    seeded.execute("SET session_replication_role='replica'")
    seeded.execute(
        "UPDATE benchmarks_v2.metric_evaluations SET metric_id=999 WHERE evaluation_variant='default'"
    )
    seeded.execute("SET session_replication_role='origin'")
    with pytest.raises(RuntimeError, match="mismatched metric identities"):
        _migrate(seeded, "20261002_0042")


def test_identity_preflight_rejects_nullable_column_before_indexes(seeded: Any) -> None:
    seeded.execute(
        "ALTER TABLE benchmarks_v2.metric_values_by_bucket ALTER COLUMN metric_id DROP NOT NULL"
    )
    with pytest.raises(RuntimeError, match="must be NOT NULL"):
        _migrate(seeded, "20261002_0042")
    assert _index_names(seeded) == set()


def test_canceled_concurrent_build_is_recovered(seeded: Any) -> None:
    name = "metric_values_by_bucket_metric_identity_key"
    before = _payloads(seeded)
    with (
        psycopg.connect(_dsn(seeded)) as blocker,
        psycopg.connect(_dsn(seeded), autocommit=True) as builder,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        blocker.execute("UPDATE benchmarks_v2.metric_values_by_bucket SET value_sum=value_sum")
        future = executor.submit(
            builder.execute,
            f"""CREATE UNIQUE INDEX CONCURRENTLY {name}
                ON benchmarks_v2.metric_values_by_bucket
                (provider,model,benchmark,dataset_id,metric_id,metric_version,
                 evaluation_variant,value_key,bucket_at)""",
        )
        try:
            deadline = monotonic() + 5
            while monotonic() < deadline:
                state = seeded.execute(
                    """SELECT i.indisvalid FROM pg_index i
                       JOIN pg_class c ON c.oid=i.indexrelid
                       JOIN pg_namespace n ON n.oid=c.relnamespace
                       WHERE n.nspname='benchmarks_v2' AND c.relname=%s""",
                    (name,),
                ).fetchone()
                if state == (False,):
                    break
                sleep(0.01)
            else:
                pytest.fail("concurrent index build did not reach its unready state")
            builder.cancel()
            with pytest.raises(psycopg.errors.QueryCanceled):
                future.result(timeout=5)
        finally:
            if not future.done():
                builder.cancel()
            blocker.rollback()
    assert _index_definition(seeded, name)[2] is False
    _migrate(seeded, "20261002_0042")
    assert _index_definition(seeded, name)[1:6] == (True, True, True, True, True)
    assert _payloads(seeded) == before
