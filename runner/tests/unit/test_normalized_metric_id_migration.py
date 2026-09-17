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
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

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


def test_not_null_upgrade_fails_without_hydration_and_is_resumable(legacy: Any) -> None:
    with pytest.raises(IntegrityError, match="violated"):
        _migrate(legacy, "20260917_0039")

    assert legacy.execute("SELECT version_num FROM alembic_version").fetchone() == (
        "20260917_0038",
    )
    assert all(
        legacy.execute(
            """SELECT NOT attnotnull FROM pg_attribute
               WHERE attrelid=%s::regclass AND attname='metric_id'""",
            (f"benchmarks_v2.{table}",),
        ).fetchone()
        == (True,)
        for table in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket")
    )

    for table in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"):
        legacy.execute(
            f"UPDATE benchmarks_v2.{table} SET metric_id=benchmarks_v2.metric_id_for_code(metric_type) "
            "WHERE metric_id IS NULL"
        )
    _migrate(legacy, "20260917_0039")


def test_not_null_upgrade_validates_constraints_cleans_checks_and_preserves_rows(
    legacy: Any,
) -> None:
    before = _payloads(legacy)
    for table in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"):
        legacy.execute(
            f"UPDATE benchmarks_v2.{table} SET metric_id=benchmarks_v2.metric_id_for_code(metric_type) "
            "WHERE metric_id IS NULL"
        )
    _migrate(legacy, "20260917_0039")

    assert _payloads(legacy) == before
    for table in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"):
        assert legacy.execute(
            """SELECT attnotnull FROM pg_attribute
               WHERE attrelid=%s::regclass AND attname='metric_id'""",
            (f"benchmarks_v2.{table}",),
        ).fetchone() == (True,)
        assert legacy.execute(
            """SELECT convalidated FROM pg_constraint
               WHERE conrelid=%s::regclass AND conname=%s""",
            (f"benchmarks_v2.{table}", f"{table}_metric_id_fkey"),
        ).fetchone() == (True,)
        assert legacy.execute(
            """SELECT count(*) FROM pg_constraint
               WHERE conrelid=%s::regclass AND conname=%s""",
            (f"benchmarks_v2.{table}", f"{table}_metric_id_not_null"),
        ).fetchone() == (0,)

    # Existing identity and lifecycle triggers remain active after enforcement.
    observation_id = legacy.execute(
        "SELECT observation_id FROM benchmarks_v2.metric_evaluations LIMIT 1"
    ).fetchone()[0]
    assert (
        legacy.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations
           (observation_id, metric_type, metric_version, evaluation_variant, executor, status)
           VALUES (%s, 'TTFA', 'v1', 'enforced', 'inline', 'queued') RETURNING metric_id""",
            (observation_id,),
        ).fetchone()[0]
        == legacy.execute("SELECT benchmarks_v2.metric_id_for_code('TTFA')").fetchone()[0]
    )


def test_not_null_downgrade_and_reupgrade_preserve_payloads(legacy: Any) -> None:
    for table in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"):
        legacy.execute(
            f"UPDATE benchmarks_v2.{table} SET metric_id=benchmarks_v2.metric_id_for_code(metric_type) "
            "WHERE metric_id IS NULL"
        )
    _migrate(legacy, "20260917_0039")
    before = _payloads(legacy)
    _migrate(legacy, "20260917_0038", down=True)
    assert _payloads(legacy) == before
    _migrate(legacy, "20260917_0039")
    assert _payloads(legacy) == before


def test_not_null_rejects_physical_null_when_identity_trigger_is_bypassed(legacy: Any) -> None:
    for table in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"):
        legacy.execute(
            f"UPDATE benchmarks_v2.{table} SET metric_id=benchmarks_v2.metric_id_for_code(metric_type) "
            "WHERE metric_id IS NULL"
        )
    _migrate(legacy, "20260917_0039")

    with (
        pytest.raises(
            psycopg.errors.NotNullViolation, match='null value in column \\"metric_id\\"'
        ),
        legacy.transaction(),
    ):
        legacy.execute("SET LOCAL session_replication_role='replica'")
        legacy.execute(
            "UPDATE benchmarks_v2.metric_evaluations SET metric_id=NULL "
            "WHERE evaluation_variant='legacy-queued'"
        )
    assert legacy.execute(
        "SELECT metric_id IS NOT NULL FROM benchmarks_v2.metric_evaluations "
        "WHERE evaluation_variant='legacy-queued'"
    ).fetchone() == (True,)


@pytest.mark.parametrize(
    "table", ["metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"]
)
def test_not_null_upgrade_rejects_historical_orphan_fk_and_retries_after_repair(
    legacy: Any, table: str
) -> None:
    for target in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"):
        legacy.execute(
            f"UPDATE benchmarks_v2.{target} SET metric_id=benchmarks_v2.metric_id_for_code(metric_type) "
            "WHERE metric_id IS NULL"
        )
    with legacy.transaction():
        legacy.execute("SET LOCAL session_replication_role='replica'")
        legacy.execute(
            f"UPDATE benchmarks_v2.{table} SET metric_id=999 WHERE metric_id IS NOT NULL"
        )

    with pytest.raises(IntegrityError, match="violates foreign key constraint"):
        _migrate(legacy, "20260917_0039")
    assert legacy.execute("SELECT version_num FROM alembic_version").fetchone() == (
        "20260917_0038",
    )

    with legacy.transaction():
        legacy.execute("SET LOCAL session_replication_role='replica'")
        legacy.execute(
            f"UPDATE benchmarks_v2.{table} SET metric_id=benchmarks_v2.metric_id_for_code(metric_type) "
            "WHERE metric_id=999"
        )
    _migrate(legacy, "20260917_0039")
    assert legacy.execute(
        f"SELECT count(*) FROM benchmarks_v2.{table} WHERE metric_id IS NULL OR metric_id=999"
    ).fetchone() == (0,)


@pytest.mark.parametrize(
    "table", ["metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"]
)
def test_not_null_preflight_failure_isolated_to_each_table(legacy: Any, table: str) -> None:
    for target in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"):
        legacy.execute(
            f"UPDATE benchmarks_v2.{target} SET metric_id=benchmarks_v2.metric_id_for_code(metric_type) "
            "WHERE metric_id IS NULL"
        )
    with legacy.transaction():
        legacy.execute("SET LOCAL session_replication_role='replica'")
        legacy.execute(
            f"UPDATE benchmarks_v2.{table} SET metric_id=NULL WHERE metric_id IS NOT NULL"
        )
    with pytest.raises(IntegrityError, match="violated"):
        _migrate(legacy, "20260917_0039")
    for target in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"):
        assert legacy.execute(
            "SELECT attnotnull FROM pg_attribute "
            "WHERE attrelid=%s::regclass AND attname='metric_id'",
            (f"benchmarks_v2.{target}",),
        ).fetchone() == (False,)


def test_not_null_retry_finishes_after_first_set_not_null_commits(legacy: Any) -> None:
    for table in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"):
        legacy.execute(
            f"UPDATE benchmarks_v2.{table} SET metric_id=benchmarks_v2.metric_id_for_code(metric_type) "
            "WHERE metric_id IS NULL"
        )
    seen = 0

    def interrupt_after_first_set_not_null(
        conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool
    ) -> None:
        nonlocal seen
        if "ALTER COLUMN metric_id SET NOT NULL" in statement:
            seen += 1
            if seen == 1:
                raise RuntimeError("simulated interruption after committed ALTER")

    event.listen(Engine, "after_cursor_execute", interrupt_after_first_set_not_null)
    try:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            _migrate(legacy, "20260917_0039")
    finally:
        event.remove(Engine, "after_cursor_execute", interrupt_after_first_set_not_null)
    assert legacy.execute(
        "SELECT attnotnull FROM pg_attribute WHERE attrelid='benchmarks_v2.metric_evaluations'::regclass "
        "AND attname='metric_id'"
    ).fetchone() == (True,)
    _migrate(legacy, "20260917_0039")


def test_not_null_validation_releases_table_lock_before_next_step(legacy: Any) -> None:
    for table in ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket"):
        legacy.execute(
            f"UPDATE benchmarks_v2.{table} SET metric_id=benchmarks_v2.metric_id_for_code(metric_type) "
            "WHERE metric_id IS NULL"
        )
    lock_acquired: list[bool] = []

    def acquire_row_lock_after_validation(
        conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool
    ) -> None:
        if "VALIDATE CONSTRAINT metric_evaluations_metric_id_not_null" in statement:
            with psycopg.connect(_dsn(legacy)) as other:
                other.execute("SET lock_timeout='1s'")
                try:
                    other.execute(
                        "LOCK TABLE benchmarks_v2.metric_evaluations IN ROW EXCLUSIVE MODE NOWAIT"
                    )
                    lock_acquired.append(True)
                except psycopg.errors.LockNotAvailable:
                    lock_acquired.append(False)

    event.listen(Engine, "after_cursor_execute", acquire_row_lock_after_validation)
    try:
        _migrate(legacy, "20260917_0039")
    finally:
        event.remove(Engine, "after_cursor_execute", acquire_row_lock_after_validation)
    assert lock_acquired == [True]
