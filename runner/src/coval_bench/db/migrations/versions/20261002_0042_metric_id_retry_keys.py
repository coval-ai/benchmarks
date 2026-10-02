# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""Add retry keys using persisted metric identities."""

from __future__ import annotations

from alembic import op
from sqlalchemy.engine import Connection

revision = "20261002_0042"
down_revision = "20260921_0041"
branch_labels = None
depends_on = None

_INDEXES = {
    "metric_evaluations_metric_identity_key": (
        "metric_evaluations",
        "observation_id, metric_id, metric_version, evaluation_variant",
    ),
    "metric_values_by_bucket_metric_identity_key": (
        "metric_values_by_bucket",
        "provider, model, benchmark, dataset_id, metric_id, metric_version, evaluation_variant, value_key, bucket_at",
    ),
    "dashboard_hourly_aggregates_metric_identity_key": (
        "dashboard_hourly_aggregates",
        "provider, model, benchmark, dataset_id, metric_id, metric_version, evaluation_variant, hour_at",
    ),
}


def _is_not_null(bind: Connection, table: str) -> bool:
    return bool(
        bind.exec_driver_sql(
            """SELECT attnotnull FROM pg_attribute
               WHERE attrelid = %s::regclass AND attname = 'metric_id' AND NOT attisdropped""",
            (f"benchmarks_v2.{table}",),
        ).scalar()
    )


def _check_identity(bind: Connection) -> None:
    for table in (
        "metric_evaluations",
        "dashboard_metric_values",
        "metric_values_by_bucket",
        "dashboard_hourly_aggregates",
    ):
        if not _is_not_null(bind, table):
            raise RuntimeError(f"{table}.metric_id must be NOT NULL before retry indexes")
    row = bind.exec_driver_sql(
        """SELECT count(*) FROM benchmarks_v2.metric_evaluations e
           LEFT JOIN benchmarks_v2.metrics m ON m.id = e.metric_id
           WHERE m.id IS NULL OR m.code IS DISTINCT FROM e.metric_type"""
    ).scalar()
    if row:
        raise RuntimeError(f"metric_evaluations contains {row} mismatched metric identities")
    row = bind.exec_driver_sql(
        """SELECT count(*) FROM benchmarks_v2.metric_values_by_bucket b
           LEFT JOIN benchmarks_v2.metrics m ON m.id = b.metric_id
           WHERE m.id IS NULL OR m.code IS DISTINCT FROM b.metric_type"""
    ).scalar()
    if row:
        raise RuntimeError(f"metric_values_by_bucket contains {row} mismatched metric identities")
    row = bind.exec_driver_sql(
        """SELECT count(*) FROM benchmarks_v2.dashboard_hourly_aggregates h
           LEFT JOIN benchmarks_v2.metrics m ON m.id = h.metric_id
           WHERE m.id IS NULL OR m.code IS DISTINCT FROM h.metric_type"""
    ).scalar()
    if row:
        raise RuntimeError(
            f"dashboard_hourly_aggregates contains {row} mismatched metric identities"
        )
    row = bind.exec_driver_sql(
        """SELECT count(*) FROM benchmarks_v2.dashboard_metric_values p
           JOIN benchmarks_v2.metric_evaluations e ON e.id = p.evaluation_id
           WHERE p.metric_id IS DISTINCT FROM e.metric_id"""
    ).scalar()
    if row:
        raise RuntimeError(f"dashboard_metric_values contains {row} mismatched parent identities")
    row = bind.exec_driver_sql(
        """SELECT count(*) FROM benchmarks_v2.dashboard_metric_values p
           LEFT JOIN benchmarks_v2.metrics m ON m.id = p.metric_id
           WHERE m.id IS NULL OR m.code IS DISTINCT FROM p.metric_type"""
    ).scalar()
    if row:
        raise RuntimeError(f"dashboard_metric_values contains {row} mismatched metric identities")


def _index_state(bind: Connection, name: str, table: str, columns: str) -> tuple[bool, bool]:
    row = bind.exec_driver_sql(
        """SELECT i.indisunique, i.indisvalid AND i.indisready,
                  pg_get_expr(i.indpred, i.indrelid), pg_get_expr(i.indexprs, i.indrelid),
                  array_agg(a.attname ORDER BY x.ordinality)
           FROM pg_class c
           JOIN pg_namespace n ON n.oid = c.relnamespace
           JOIN pg_index i ON i.indexrelid = c.oid
           LEFT JOIN LATERAL unnest(i.indkey) WITH ORDINALITY x(attnum, ordinality) ON true
           LEFT JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = x.attnum
           WHERE n.nspname = 'benchmarks_v2' AND c.relname = %s
             AND i.indrelid = %s::regclass
           GROUP BY i.indisunique, i.indisvalid, i.indisready, i.indpred, i.indexprs, i.indrelid""",
        (name, f"benchmarks_v2.{table}"),
    ).fetchone()
    if row is None:
        return False, False
    expected = [column.strip() for column in columns.split(",")]
    exact = row[2] is None and row[3] is None and row[4] == expected
    return bool(row[0] and exact), bool(row[1] and exact)


def _ensure_index(bind: Connection, name: str, table: str, columns: str) -> None:
    exact, ready = _index_state(bind, name, table, columns)
    if exact and ready:
        return
    if exact or ready is False:
        bind.exec_driver_sql(f"DROP INDEX CONCURRENTLY IF EXISTS benchmarks_v2.{name}")
    bind.exec_driver_sql(
        f"CREATE UNIQUE INDEX CONCURRENTLY {name} ON benchmarks_v2.{table} ({columns})"
    )


def _install_retry_serialization(bind: Connection) -> None:
    bind.exec_driver_sql(
        """CREATE OR REPLACE FUNCTION benchmarks_v2.sync_metric_evaluation_identity()
        RETURNS trigger LANGUAGE plpgsql AS $fn$
        BEGIN
          IF NEW.metric_id IS NULL THEN
            NEW.metric_id := benchmarks_v2.metric_id_for_code(NEW.metric_type);
          ELSIF NEW.metric_type IS NULL THEN
            NEW.metric_type := benchmarks_v2.metric_code_for_id(NEW.metric_id);
          ELSIF NEW.metric_id <> benchmarks_v2.metric_id_for_code(NEW.metric_type) THEN
            RAISE EXCEPTION 'metric code and id must refer to the same definition'
              USING ERRCODE='23514';
          END IF;
          IF TG_OP = 'INSERT' THEN
            PERFORM pg_advisory_xact_lock(hashtextextended(
              'metric_evaluations:' ||
              ROW(NEW.observation_id, NEW.metric_id, NEW.metric_version,
                  NEW.evaluation_variant)::text, 0));
          END IF;
          RETURN NEW;
        END $fn$"""
    )


def _restore_identity_sync(bind: Connection) -> None:
    bind.exec_driver_sql(
        """CREATE OR REPLACE FUNCTION benchmarks_v2.sync_metric_evaluation_identity()
        RETURNS trigger LANGUAGE plpgsql AS $fn$
        BEGIN
          IF NEW.metric_id IS NULL THEN
            NEW.metric_id := benchmarks_v2.metric_id_for_code(NEW.metric_type);
          ELSIF NEW.metric_type IS NULL THEN
            NEW.metric_type := benchmarks_v2.metric_code_for_id(NEW.metric_id);
          ELSIF NEW.metric_id <> benchmarks_v2.metric_id_for_code(NEW.metric_type) THEN
            RAISE EXCEPTION 'metric code and id must refer to the same definition'
              USING ERRCODE='23514';
          END IF;
          RETURN NEW;
        END $fn$"""
    )


def upgrade() -> None:
    bind = op.get_bind()
    _check_identity(bind)
    _install_retry_serialization(bind)
    with op.get_context().autocommit_block():
        for name, (table, columns) in _INDEXES.items():
            _ensure_index(bind, name, table, columns)
    # The lifecycle guard and parent projection function are changed only after
    # every retry key is ready, with no window in which updates are unguarded.
    bind.exec_driver_sql(
        "DROP TRIGGER IF EXISTS metric_evaluations_validate_update ON benchmarks_v2.metric_evaluations"
    )
    bind.exec_driver_sql(
        """CREATE TRIGGER metric_evaluations_validate_update
           BEFORE UPDATE ON benchmarks_v2.metric_evaluations FOR EACH ROW
           EXECUTE FUNCTION benchmarks_v2.validate_metric_transition()"""
    )
    bind.exec_driver_sql(
        """CREATE OR REPLACE FUNCTION benchmarks_v2.guard_dashboard_metric_parent() RETURNS trigger
           LANGUAGE plpgsql AS $fn$
           DECLARE parent_metric_id BIGINT;
           BEGIN
             SELECT metric_id INTO parent_metric_id
               FROM benchmarks_v2.metric_evaluations WHERE id = NEW.evaluation_id;
             IF FOUND AND parent_metric_id IS DISTINCT FROM NEW.metric_id THEN
               RAISE EXCEPTION 'dashboard projection metric identity disagrees with parent' USING ERRCODE='23514';
             END IF;
             RETURN NEW;
           END $fn$"""
    )


def downgrade() -> None:
    bind = op.get_bind()
    with op.get_context().autocommit_block():
        for name in _INDEXES:
            bind.exec_driver_sql(f"DROP INDEX CONCURRENTLY IF EXISTS benchmarks_v2.{name}")
    _restore_identity_sync(bind)
    bind.exec_driver_sql(
        "DROP TRIGGER IF EXISTS metric_evaluations_validate_update ON benchmarks_v2.metric_evaluations"
    )
    bind.exec_driver_sql(
        """CREATE TRIGGER metric_evaluations_validate_update
           BEFORE UPDATE ON benchmarks_v2.metric_evaluations FOR EACH ROW
           WHEN (NOT (OLD.metric_id IS NULL AND NEW.metric_id IS NOT NULL
             AND NEW.metric_id = benchmarks_v2.metric_id_for_code(OLD.metric_type)
             AND (to_jsonb(OLD) - 'metric_id') = (to_jsonb(NEW) - 'metric_id')))
           EXECUTE FUNCTION benchmarks_v2.validate_metric_transition()"""
    )
    bind.exec_driver_sql(
        """CREATE OR REPLACE FUNCTION benchmarks_v2.guard_dashboard_metric_parent() RETURNS trigger
           LANGUAGE plpgsql AS $fn$
           DECLARE parent_metric_id BIGINT;
           BEGIN
             SELECT COALESCE(metric_id, benchmarks_v2.metric_id_for_code(metric_type))
               INTO parent_metric_id FROM benchmarks_v2.metric_evaluations WHERE id = NEW.evaluation_id;
             IF FOUND AND parent_metric_id IS DISTINCT FROM NEW.metric_id THEN
               RAISE EXCEPTION 'dashboard projection metric identity disagrees with parent' USING ERRCODE='23514';
             END IF;
             RETURN NEW;
           END $fn$"""
    )
