# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Write the transactionally published dashboard summary snapshots."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

from coval_bench.db.dashboard_contracts import DEFINITION_REVISION, aggregation_fingerprint
from coval_bench.db.metric_definitions import register_metric_definitions
from coval_bench.registries.metrics import METRIC_VALUE_CONTRACTS

SUMMARY_DEFINITION_REVISION = DEFINITION_REVISION
SUMMARY_VIEWS = {
    "24h": "benchmarks_v2.normalized_results_24h",
    "7d": "benchmarks_v2.normalized_results_7d",
    "30d": "benchmarks_v2.normalized_results_30d",
}
_WINDOWS = {"24h": "24 hours", "7d": "7 days", "30d": "30 days"}


@dataclass(frozen=True)
class RefreshResult:
    status: Literal["published", "skipped_lock", "skipped_siblings"]
    generation: int | None = None


def validate_summary_rules() -> None:
    """Ensure registered contracts can be represented by the frozen projection."""
    for (metric, version), contract in METRIC_VALUE_CONTRACTS.items():
        if version != "v1":
            continue
        if (str(metric) == "WER" or contract.aggregation_method == "ratio") and (
            str(metric) != "WER"
            or contract.aggregation_method != "ratio"
            or contract.ratio_scale != 100
            or contract.ratio_fallback != "mean"
            or set(contract.numerator_keys)
            != {"substitution_count", "deletion_count", "insertion_count"}
            or contract.denominator_key != "reference_words"
            or next((d.unit for d in contract.values if d.key == "primary"), None) != "percent"
            or any(
                definition.unit
                != (
                    "percent"
                    if definition.key in {"insertions", "deletions", "substitutions"}
                    else "count"
                )
                for definition in contract.values
                if definition.key != "primary"
            )
        ):
            raise ValueError(f"unsupported summary ratio contract for {metric}/{version}")


async def refresh_summary_snapshots(
    pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]],
    *,
    as_of: dt.datetime | None = None,
    run_id: int | None = None,
) -> RefreshResult:
    """Refresh all three views and publish one generation atomically."""
    validate_summary_rules()
    captured = as_of or dt.datetime.now(dt.UTC)
    if captured.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    captured = captured.astimezone(dt.UTC)
    fingerprint = aggregation_fingerprint()
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        if run_id is not None:
            sibling = await conn.execute(
                """SELECT EXISTS (
                    SELECT 1 FROM benchmarks_v2.runs sibling
                    JOIN benchmarks_v2.runs own ON own.id = %(run_id)s
                    WHERE sibling.scheduled_at = own.scheduled_at
                      AND sibling.id <> own.id AND sibling.status = 'running'
                ) AS siblings_running""",
                {"run_id": run_id},
            )
            sibling_row = await sibling.fetchone()
            if sibling_row is not None and sibling_row["siblings_running"]:
                return RefreshResult("skipped_siblings")
        lock = await conn.execute(
            "SELECT pg_try_advisory_xact_lock(hashtextextended('dashboard_summary', 0)) AS locked"
        )
        lock_row = await lock.fetchone()
        if lock_row is None or not lock_row["locked"]:
            return RefreshResult("skipped_lock")
        await conn.execute("SET LOCAL statement_timeout = '550s'")
        metric_ids = await register_metric_definitions(conn)
        source_result = await conn.execute(
            """SELECT DISTINCT e.metric_type
               FROM benchmarks_v2.dashboard_metric_values e
               JOIN benchmarks_v2.benchmark_observations o ON o.id = e.observation_id
               JOIN benchmarks_v2.runs r ON r.id = o.run_id
               WHERE o.status = 'succeeded' AND r.status IN ('succeeded', 'partial')
                 AND e.metric_version = 'v1' AND e.evaluation_variant = 'default'
                 AND o.captured_at >= %(since)s - interval '30 days'
                 AND o.captured_at < %(until)s""",
            {"since": captured, "until": captured},
        )
        source_codes = {str(row["metric_type"]) for row in (await source_result.fetchall())}
        supported = {metric.value for metric, version in METRIC_VALUE_CONTRACTS if version == "v1"}
        invalid = sorted((source_codes - set(metric_ids)) | (source_codes - supported))
        if invalid:
            raise ValueError("invalid metrics in summary source: " + ", ".join(invalid))
        # The defining SQL reads this transaction-local snapshot boundary.  A
        # failed refresh rolls this provisional update back with the views.
        await conn.execute(
            "UPDATE benchmarks_v2.dashboard_summary_state SET as_of=%(as_of)s WHERE id=true",
            {"as_of": captured},
        )
        state = await conn.execute(
            "SELECT generation FROM benchmarks_v2.dashboard_summary_state "
            "WHERE id = true FOR UPDATE"
        )
        row = await state.fetchone()
        if row is None:
            raise RuntimeError("dashboard summary state is missing")
        generation = int(row["generation"]) + 1
        for _key, view in SUMMARY_VIEWS.items():
            populated = await conn.execute(
                "SELECT relispopulated FROM pg_class WHERE oid = %(view)s::regclass",
                {"view": view},
            )
            populated_row = await populated.fetchone()
            if populated_row is None:
                raise RuntimeError(f"dashboard summary view is missing: {view}")
            is_populated = populated_row["relispopulated"]
            mode = "CONCURRENTLY " if is_populated else ""
            await conn.execute(f"REFRESH MATERIALIZED VIEW {mode}{view}")  # noqa: S608
        await conn.execute(
            """UPDATE benchmarks_v2.dashboard_summary_state
                   SET generation=%(generation)s, as_of=%(as_of)s, published_at=clock_timestamp(),
                       definition_revision=%(revision)s, definition_fingerprint=%(fingerprint)s,
                       metadata='{"schema_version":1}'::jsonb WHERE id=true""",
            {
                "generation": generation,
                "as_of": captured,
                "revision": SUMMARY_DEFINITION_REVISION,
                "fingerprint": fingerprint,
            },
        )
        return RefreshResult("published", generation)
