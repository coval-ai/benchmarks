# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""Resumable, identity-only backfill for normalized metric storage.

The command is intentionally separate from Alembic.  It is safe to stop and
rerun: every phase selects only rows whose identity is still null, and each
batch is committed independently. Final reconciliation reports rows skipped
because of contention; an exhausted runtime explicitly defers verification.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import click
import psycopg

from coval_bench.config import get_settings

_ADVISORY_LOCK = "normalized_metric_ids_backfill"
_TABLES = ("metric_evaluations", "dashboard_metric_values", "metric_values_by_bucket")
_PENDING_INDEXES = {
    "metric_evaluations": "metric_evaluations_metric_type_pending_idx",
    "dashboard_metric_values": "dashboard_metric_values_metric_type_pending_idx",
    "metric_values_by_bucket": "metric_values_by_bucket_metric_type_pending_idx",
}


@dataclass
class BackfillReport:
    """Aggregate-only outcome suitable for CLI JSON and operator logs."""

    mode: str
    status: str = "completed"
    batches: int = 0
    updated: dict[str, int] = field(default_factory=lambda: {table: 0 for table in _TABLES})
    pending: dict[str, int] = field(default_factory=lambda: {table: 0 for table in _TABLES})
    unknown_codes: dict[str, int] = field(default_factory=lambda: {table: 0 for table in _TABLES})
    mismatched_ids: dict[str, int] = field(default_factory=lambda: {table: 0 for table in _TABLES})
    skipped_batches: int = 0
    lock_acquired: bool = False
    verification_complete: bool = False
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": "normalized_metric_ids_backfill",
            "mode": self.mode,
            "status": self.status,
            "batches": self.batches,
            "updated": self.updated,
            "pending": self.pending,
            "unknown_codes": self.unknown_codes,
            "mismatched_ids": self.mismatched_ids,
            "skipped_batches": self.skipped_batches,
            "lock_acquired": self.lock_acquired,
            "verification_complete": self.verification_complete,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "errors": self.errors,
        }


def _registry(conn: psycopg.Connection[Any], *, apply: bool) -> dict[str, int]:
    """Return the catalog mapping, registering known definitions only on apply."""
    if apply:
        # The synchronous helper is added alongside the writer changes.  Keep
        # this import late so a read-only preflight remains usable during a
        # rolling deployment.
        from coval_bench.db.metric_definitions import register_metric_definitions_sync

        return register_metric_definitions_sync(conn)
    rows = conn.execute("SELECT code, id FROM benchmarks_v2.metrics ORDER BY code").fetchall()
    return {
        str(row["code"] if isinstance(row, dict) else row[0]): int(
            row["id"] if isinstance(row, dict) else row[1]
        )
        for row in rows
    }


def ensure_pending_indexes(conn: psycopg.Connection[Any]) -> None:
    """Create/recover temporary concurrent indexes for large backfills."""
    for table, index in _PENDING_INDEXES.items():
        invalid = conn.execute(
            """SELECT c.relname FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid
               WHERE c.relname = %s AND NOT i.indisvalid""",
            (index,),
        ).fetchone()
        if invalid:
            conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS benchmarks_v2.{index}")
        conn.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index} "
            f"ON benchmarks_v2.{table} (metric_type) WHERE metric_id IS NULL"
        )


def _counts(conn: psycopg.Connection[Any], table: str) -> tuple[int, int, int]:
    """Count pending, unknown-code, and code/ID mismatch rows."""
    parent_join = (
        "LEFT JOIN benchmarks_v2.metric_evaluations parent ON parent.id = t.evaluation_id "
        "LEFT JOIN benchmarks_v2.metrics parent_metric ON parent_metric.code = parent.metric_type"
        if table == "dashboard_metric_values"
        else ""
    )
    mismatch = (
        "m.id IS NULL OR (t.metric_id IS NOT NULL AND t.metric_id <> m.id) "
        "OR parent.id IS NULL OR parent_metric.id IS NULL "
        "OR m.id IS DISTINCT FROM COALESCE(parent.metric_id, parent_metric.id)"
        if table == "dashboard_metric_values"
        else "t.metric_id IS NOT NULL AND (m.id IS NULL OR t.metric_id <> m.id)"
    )
    row = conn.execute(
        f"""
        SELECT
          COUNT(*) FILTER (WHERE t.metric_id IS NULL AND m.id IS NOT NULL) AS pending,
          COUNT(*) FILTER (WHERE m.id IS NULL) AS unknown,
          COUNT(*) FILTER (WHERE {mismatch}) AS mismatch
        FROM benchmarks_v2.{table} t
        LEFT JOIN benchmarks_v2.metrics m ON m.code = t.metric_type
        {parent_join}
        """  # table names are constants above
    ).fetchone()
    if row is None:
        raise RuntimeError("metric identity coverage query returned no row")
    if isinstance(row, dict):
        return int(row["pending"]), int(row["unknown"]), int(row["mismatch"])
    return int(row[0]), int(row[1]), int(row[2])


def _update_batch_safe(conn: psycopg.Connection[Any], table: str, batch_size: int) -> int:
    """Use a table-specific key expression without interpolating user input."""
    if table == "metric_evaluations":
        identity = "t.id = p.id"
        columns = "t.id"
        parent_filter = ""
    elif table == "dashboard_metric_values":
        identity = "t.evaluation_id = p.evaluation_id"
        columns = "t.evaluation_id, e.metric_id AS parent_metric_id"
        parent_filter = "JOIN benchmarks_v2.metric_evaluations e ON e.id = t.evaluation_id AND e.metric_id IS NOT NULL"
    else:
        identity = "t.provider = p.provider AND t.model = p.model AND t.benchmark = p.benchmark AND t.dataset_id = p.dataset_id AND t.metric_type = p.metric_type AND t.metric_version = p.metric_version AND t.evaluation_variant = p.evaluation_variant AND t.value_key = p.value_key AND t.bucket_at = p.bucket_at"
        columns = "t.provider, t.model, t.benchmark, t.dataset_id, t.metric_type, t.metric_version, t.evaluation_variant, t.value_key, t.bucket_at"
        parent_filter = ""
    return int(
        conn.execute(
            f"""
            WITH pending AS (
              SELECT {columns}
              FROM benchmarks_v2.{table} t
              JOIN benchmarks_v2.metrics m ON m.code = t.metric_type
              {parent_filter}
              WHERE t.metric_id IS NULL
              ORDER BY t.metric_type
              LIMIT %(limit)s
              FOR UPDATE OF t SKIP LOCKED
            )
            UPDATE benchmarks_v2.{table} t
               SET metric_id = {"p.parent_metric_id" if table == "dashboard_metric_values" else "m.id"}
              FROM benchmarks_v2.metrics m, pending p
             WHERE m.code = t.metric_type AND t.metric_id IS NULL AND {identity}
               {"AND p.parent_metric_id = m.id" if table == "dashboard_metric_values" else ""}
            """,
            {"limit": batch_size},
        ).rowcount
    )


def backfill(
    conn: psycopg.Connection[Any],
    *,
    apply: bool = False,
    batch_size: int = 1000,
    max_batches: int | None = None,
    max_runtime_seconds: float = 600.0,
    lock_timeout_ms: int = 1000,
    clock: Callable[[], float] = time.monotonic,
) -> BackfillReport:
    """Run a bounded preflight or resumable backfill on an existing connection."""
    if batch_size <= 0 or (max_batches is not None and max_batches <= 0):
        raise ValueError("batch_size and max_batches must be positive")
    if not conn.autocommit or conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        raise ValueError("backfill requires an idle autocommit connection for committed batches")
    if max_runtime_seconds <= 0 or lock_timeout_ms <= 0:
        raise ValueError("runtime and lock timeout must be positive")
    started = clock()
    report = BackfillReport(mode="apply" if apply else "preflight")
    original_timeout = conn.execute("SHOW statement_timeout").fetchone()
    if original_timeout is None:
        raise RuntimeError("statement timeout query returned no row")
    original_timeout_value = (
        original_timeout["statement_timeout"]
        if isinstance(original_timeout, dict)
        else original_timeout[0]
    )

    def set_timeout() -> None:
        remaining = max(1, int((max_runtime_seconds - (clock() - started)) * 1000))
        conn.execute(
            "SELECT set_config('statement_timeout', %s, false)",
            (f"{remaining}ms",),
        )

    try:
        set_timeout()
        lock_row = conn.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired", (_ADVISORY_LOCK,)
        ).fetchone()
        if lock_row is None:
            raise RuntimeError("backfill owner lock query returned no row")
        report.lock_acquired = bool(
            lock_row["acquired"] if isinstance(lock_row, dict) else lock_row[0]
        )
        if not report.lock_acquired:
            report.status = "busy"
            return report
        if apply:
            with conn.transaction():
                mapping = _registry(conn, apply=True)
        else:
            mapping = _registry(conn, apply=False)
        if not mapping:
            report.errors.append("metric catalog is empty")
            report.status = "needs_reconciliation"
        # Unknown and mismatched rows are fail-closed diagnostics.  They are
        # never candidates for an update.
        for table in _TABLES:
            set_timeout()
            pending, unknown, mismatch = _counts(conn, table)
            report.pending[table] = pending
            report.unknown_codes[table] = unknown
            report.mismatched_ids[table] = mismatch

        if any(report.unknown_codes.values()) or any(report.mismatched_ids.values()):
            report.status = "needs_reconciliation"
        if apply and report.status == "completed" and not report.errors:
            for table in _TABLES:
                while True:
                    if clock() - started >= max_runtime_seconds:
                        report.status = "time_limit"
                        break
                    if max_batches is not None and report.batches >= max_batches:
                        report.status = "batch_limit"
                        break
                    try:
                        with conn.transaction():
                            remaining = max(
                                1, int((max_runtime_seconds - (clock() - started)) * 1000)
                            )
                            conn.execute(
                                "SELECT set_config('statement_timeout', %s, true)",
                                (f"{remaining}ms",),
                            )
                            conn.execute(
                                "SELECT set_config('lock_timeout', %s, true)",
                                (f"{lock_timeout_ms}ms",),
                            )
                            changed = _update_batch_safe(conn, table, batch_size)
                        report.batches += 1
                        report.updated[table] += changed
                        if changed == 0:
                            break
                    except (psycopg.errors.LockNotAvailable, psycopg.errors.QueryCanceled) as error:
                        conn.rollback()
                        if isinstance(error, psycopg.errors.QueryCanceled):
                            report.status = "time_limit"
                        else:
                            report.skipped_batches += 1
                        if clock() - started >= max_runtime_seconds:
                            report.status = "time_limit"
                            break
                if report.status in {"time_limit", "batch_limit"}:
                    break
        if clock() - started >= max_runtime_seconds:
            report.status = "time_limit"
            report.errors.append(
                "final verification deferred because the runtime limit was reached"
            )
            return report
        for table in _TABLES:
            set_timeout()
            pending, unknown, mismatch = _counts(conn, table)
            report.pending[table] = pending
            report.unknown_codes[table] = unknown
            report.mismatched_ids[table] = mismatch
        report.verification_complete = True
        if any(report.unknown_codes.values()) or any(report.mismatched_ids.values()):
            report.status = "needs_reconciliation"
        elif any(report.pending.values()) and report.status == "completed":
            report.status = "incomplete"
    except psycopg.errors.QueryCanceled:
        report.status = "time_limit"
        report.errors.append(
            "verification did not finish within the runtime limit; rerun preflight"
        )
    finally:
        conn.rollback()
        conn.execute("SELECT set_config('statement_timeout', %s, false)", (original_timeout_value,))
        if report.lock_acquired:
            try:
                conn.rollback()
                conn.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (_ADVISORY_LOCK,)
                )
            except psycopg.Error:
                # A canceled statement may leave the transaction aborted; the
                # connection context will close and release the session lock.
                report.errors.append("could not release advisory lock cleanly")
        report.elapsed_seconds = clock() - started
    return report


@click.command()
@click.option("--apply", is_flag=True, help="Commit metric ID updates; default is read-only.")
@click.option("--batch-size", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option("--max-batches", type=click.IntRange(min=1), default=None)
@click.option(
    "--max-runtime-seconds", type=click.FloatRange(min=0.1), default=600.0, show_default=True
)
@click.option("--lock-timeout-ms", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option(
    "--create-pending-indexes",
    is_flag=True,
    help="Create/recover temporary concurrent indexes before the run.",
)
def backfill_normalized_metric_ids_cli(
    apply: bool,
    batch_size: int,
    max_batches: int | None,
    max_runtime_seconds: float,
    lock_timeout_ms: int,
    create_pending_indexes: bool,
) -> None:
    """Backfill normalized metric IDs (read-only unless --apply is supplied)."""
    if create_pending_indexes and not apply:
        raise click.UsageError("--create-pending-indexes requires --apply")
    url = str(get_settings().database_url or os.environ.get("DATABASE_URL", ""))
    if not url:
        raise click.ClickException("DATABASE_URL is required")
    try:
        # Every apply batch owns a real transaction.  Autocommit keeps the
        # advisory lock and preflight reads from wrapping later batches in an
        # outer transaction/savepoint.
        with psycopg.connect(url, autocommit=True) as conn:
            if create_pending_indexes:
                ensure_pending_indexes(conn)
            report = backfill(
                conn,
                apply=apply,
                batch_size=batch_size,
                max_batches=max_batches,
                max_runtime_seconds=max_runtime_seconds,
                lock_timeout_ms=lock_timeout_ms,
            )
    except psycopg.Error as error:
        raise click.ClickException(f"backfill failed: {error}") from error
    click.echo(json.dumps(report.as_dict(), sort_keys=True))
    if report.status in {"busy", "needs_reconciliation"}:
        raise click.exceptions.Exit(2)
