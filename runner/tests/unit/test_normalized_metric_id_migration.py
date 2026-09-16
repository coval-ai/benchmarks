# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Standalone PostgreSQL checks for the normalized metric identity migration."""

# ruff: noqa: E501, S608

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from pytest_postgresql.factories import postgresql

pg_conn = postgresql("pg_proc")


def _dsn(conn: psycopg.Connection[Any]) -> str:
    info = conn.info
    auth = f"{info.user}:{info.password}@" if info.password else f"{info.user}@"
    return (
        f"postgresql://{auth}{info.host or 'localhost'}:{info.port or 5432}/{info.dbname or 'test'}"
    )


def _migrate(conn: psycopg.Connection[Any], revision: str, *, down: bool = False) -> None:
    conn.commit()
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    config.set_main_option(
        "sqlalchemy.url", _dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    (command.downgrade if down else command.upgrade)(config, revision)


def _payloads(conn: Any, *, metric_column: bool = True) -> dict[str, list[Any]]:
    projection = "to_jsonb(t) - 'metric_id'" if metric_column else "to_jsonb(t)"
    return {
        table: conn.execute(
            f"SELECT {projection} FROM benchmarks_v2.{table} t ORDER BY 1"
        ).fetchall()
        for table in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket")
    }


@pytest.fixture
def legacy(pg_conn: Any) -> Any:
    _migrate(pg_conn, "20260914_0035")
    pg_conn.autocommit = True
    with pg_conn.transaction():
        run_id = pg_conn.execute(
            """INSERT INTO benchmarks_v2.runs
               (dataset_id, dataset_sha256, runner_sha, status, finished_at, scheduled_at)
               VALUES ('d', %s, 'test', 'succeeded', now(), date_trunc('hour', now())) RETURNING id""",
            ("a" * 64,),
        ).fetchone()[0]
        observation_id = pg_conn.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
               (run_id, dataset_id, dataset_sha256, sample_id, provider, model,
                benchmark, source_kind, status)
               VALUES (%s, 'd', %s, 'sample', 'p', 'm', 'STT', 'dataset_audio', 'succeeded')
               RETURNING id""",
            (run_id, "a" * 64),
        ).fetchone()[0]
        for status in ("queued", "running", "succeeded", "failed"):
            variant = f"legacy-{status}"
            evaluation_id = pg_conn.execute(
                """INSERT INTO benchmarks_v2.metric_evaluations
                   (observation_id, metric_type, metric_version, evaluation_variant, executor, status)
                   VALUES (%s, 'WER', 'v1', %s, 'inline', 'queued') RETURNING id""",
                (observation_id, variant),
            ).fetchone()[0]
            if status == "queued":
                continue
            pg_conn.execute(
                "UPDATE benchmarks_v2.metric_evaluations SET status='running', started_at=now() WHERE id=%s",
                (evaluation_id,),
            )
            if status == "running":
                continue
            if status == "succeeded":
                pg_conn.execute(
                    """INSERT INTO benchmarks_v2.metric_values
                       (metric_evaluation_id, value_key, unit, value, value_role)
                       VALUES (%s, 'primary', 'percent', 9, 'primary')""",
                    (evaluation_id,),
                )
            pg_conn.execute(
                """UPDATE benchmarks_v2.metric_evaluations
                   SET status=%s, finished_at=now(), error=%s WHERE id=%s""",
                (status, "expected failure" if status == "failed" else None, evaluation_id),
            )
        pg_conn.execute(
            """INSERT INTO benchmarks_v2.metric_values_by_bucket
               (provider, model, benchmark, dataset_id, metric_type, metric_version,
                evaluation_variant, value_key, unit, bucket_at, min_value, p25, p50,
                p75, max_value, value_sum, sample_count)
               VALUES ('p', 'm', 'STT', 'd', 'WER', 'v1', 'default', 'primary',
                       'percent', now(), 1, 2, 3, 4, 5, 5, 1)"""
        )
    _migrate(pg_conn, "20260915_0036")
    return pg_conn


def test_additive_nullable_columns_and_foreign_keys(legacy: Any) -> None:
    for table in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"):
        assert legacy.execute(
            """SELECT is_nullable FROM information_schema.columns
               WHERE table_schema='benchmarks_v2' AND table_name=%s AND column_name='metric_id'""",
            (table,),
        ).fetchone() == ("YES",)
        assert legacy.execute(
            """SELECT count(*) FROM pg_constraint
               WHERE conrelid=%s::regclass AND conname LIKE '%%metric_id_fkey'""",
            (f"benchmarks_v2.{table}",),
        ).fetchone() == (1,)


@pytest.mark.parametrize("status", ["queued", "running", "succeeded", "failed"])
def test_hydration_rejects_combined_payload_changes(legacy: Any, status: str) -> None:
    with pytest.raises(psycopg.errors.RaiseException):
        legacy.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET metric_id=benchmarks_v2.metric_id_for_code(metric_type),
                   updated_at=updated_at + interval '1 second'
               WHERE evaluation_variant=%s""",
            (f"legacy-{status}",),
        )


def test_hydration_preserves_all_lifecycle_payloads(legacy: Any) -> None:
    before = _payloads(legacy)
    legacy.execute(
        """UPDATE benchmarks_v2.metric_evaluations
           SET metric_id=benchmarks_v2.metric_id_for_code(metric_type)
           WHERE metric_id IS NULL"""
    )
    legacy.execute(
        """UPDATE benchmarks_v2.dashboard_metric_values
           SET metric_id=benchmarks_v2.metric_id_for_code(metric_type)
           WHERE metric_id IS NULL"""
    )
    legacy.execute(
        """UPDATE benchmarks_v2.metric_values_by_bucket
           SET metric_id=benchmarks_v2.metric_id_for_code(metric_type)
           WHERE metric_id IS NULL"""
    )
    assert _payloads(legacy) == before
    assert legacy.execute(
        "SELECT count(*) FROM benchmarks_v2.metric_evaluations WHERE metric_id IS NOT NULL"
    ).fetchone() == (4,)


def test_code_only_current_writes_resolve_and_mismatch_is_rejected(legacy: Any) -> None:
    observation_id = legacy.execute(
        "SELECT observation_id FROM benchmarks_v2.metric_evaluations LIMIT 1"
    ).fetchone()[0]
    row = legacy.execute(
        """INSERT INTO benchmarks_v2.metric_evaluations
           (observation_id, metric_type, metric_version, evaluation_variant, executor, status)
           VALUES (%s, 'TTFA', 'v1', 'new', 'inline', 'queued') RETURNING metric_id""",
        (observation_id,),
    ).fetchone()
    assert row == (legacy.execute("SELECT benchmarks_v2.metric_id_for_code('TTFA')").fetchone()[0],)
    with pytest.raises(psycopg.errors.CheckViolation, match="same definition"):
        legacy.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET metric_id=benchmarks_v2.metric_id_for_code('TTFA')
               WHERE evaluation_variant='legacy-queued'"""
        )


def test_projection_parent_guard_and_lifecycle_guard(legacy: Any) -> None:
    with pytest.raises(psycopg.errors.CheckViolation, match="disagrees with parent"):
        legacy.execute(
            """UPDATE benchmarks_v2.dashboard_metric_values
               SET metric_type='TTFA', metric_id=benchmarks_v2.metric_id_for_code('TTFA')"""
        )
    with pytest.raises(psycopg.errors.RaiseException, match="terminal"):
        legacy.execute(
            "DELETE FROM benchmarks_v2.metric_evaluations WHERE evaluation_variant='legacy-succeeded'"
        )
    with pytest.raises(psycopg.errors.CheckViolation, match="same definition"):
        legacy.execute(
            """INSERT INTO benchmarks_v2.metric_values_by_bucket
               (provider, model, benchmark, dataset_id, metric_id, metric_type, metric_version,
                evaluation_variant, value_key, unit, bucket_at, min_value, p25, p50, p75,
                max_value, value_sum, sample_count)
               VALUES ('x', 'x', 'STT', 'x', -1, 'WER', 'v1', 'bad', 'primary', 'percent',
                       now(), 1, 2, 3, 4, 5, 5, 1)"""
        )


def test_downgrade_restores_legacy_trigger_and_rows(legacy: Any) -> None:
    before = _payloads(legacy)
    original = legacy.execute(
        "SELECT pg_get_functiondef('benchmarks_v2.validate_metric_transition()'::regprocedure)"
    ).fetchone()
    _migrate(legacy, "20260914_0035", down=True)
    assert _payloads(legacy, metric_column=False) == before
    assert (
        legacy.execute(
            "SELECT pg_get_functiondef('benchmarks_v2.validate_metric_transition()'::regprocedure)"
        ).fetchone()
        == original
    )
    with pytest.raises(psycopg.errors.RaiseException, match="terminal"):
        legacy.execute(
            "UPDATE benchmarks_v2.metric_evaluations SET updated_at=now() "
            "WHERE evaluation_variant='legacy-succeeded'"
        )
