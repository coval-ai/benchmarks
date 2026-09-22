# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, read-only parity audit for the arena provider-health reader.

This module is intentionally synchronous so an operator can pass an already-authenticated
``psycopg.Connection`` from a one-shot diagnostic wrapper. It never writes, logs provider
errors, or makes a rollout decision; it returns aggregate-safe evidence for review.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

import psycopg
import psycopg.rows

from .provider_health import BENCHING_REASONS, KeyFailure, classify_failure

_DEFAULT_ROW_LIMIT = 100_000

_LEGACY_ROWS_SQL = """
    WITH latest_run AS (
        SELECT DISTINCT ON (r.provider) r.provider, r.run_id
        FROM benchmarks_v2.results r
        JOIN benchmarks_v2.runs u ON u.id = r.run_id
        WHERE r.benchmark = 'TTS' AND r.metric_type = 'TTFA'
          AND u.started_at > %(as_of)s - interval '1 day'
        ORDER BY r.provider, u.started_at DESC, r.run_id DESC
    )
    SELECT l.provider, l.run_id, r.status, r.error
    FROM latest_run l
    JOIN benchmarks_v2.results r ON r.run_id = l.run_id AND r.provider = l.provider
    WHERE r.benchmark = 'TTS' AND r.metric_type = 'TTFA'
    LIMIT %(row_limit)s
"""

_NORMALIZED_ROWS_SQL = """
    WITH latest_run AS (
        SELECT DISTINCT ON (o.provider) o.provider, o.run_id
        FROM benchmarks_v2.benchmark_observations o
        JOIN benchmarks_v2.metric_evaluations e ON e.observation_id = o.id
        JOIN benchmarks_v2.runs u ON u.id = o.run_id
        WHERE o.benchmark = 'TTS'
          AND e.metric_type = 'TTFA' AND e.metric_version = 'v1'
          AND e.evaluation_variant = 'default'
          AND e.status IN ('succeeded', 'failed')
          AND u.started_at > %(as_of)s - interval '1 day'
        ORDER BY o.provider, u.started_at DESC, o.run_id DESC
    )
    SELECT l.provider, l.run_id, e.status, e.error
    FROM latest_run l
    JOIN benchmarks_v2.benchmark_observations o
      ON o.run_id = l.run_id AND o.provider = l.provider
    JOIN benchmarks_v2.metric_evaluations e ON e.observation_id = o.id
    WHERE o.benchmark = 'TTS'
      AND e.metric_type = 'TTFA' AND e.metric_version = 'v1'
      AND e.evaluation_variant = 'default'
      AND e.status IN ('succeeded', 'failed')
    LIMIT %(row_limit)s
"""


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    providers: dict[str, dict[str, Any]] = {}
    for row in rows:
        provider = str(row["provider"])
        summary = providers.setdefault(
            provider,
            {
                "run_id": int(row["run_id"]),
                "succeeded": 0,
                "failed": 0,
                "classified": {reason.value: 0 for reason in KeyFailure} | {"other": 0},
            },
        )
        if row["status"] in ("success", "succeeded"):
            summary["succeeded"] += 1
            continue
        summary["failed"] += 1
        reason = classify_failure(None, row["error"])
        summary["classified"][reason.value if reason is not None else "other"] += 1
    for summary in providers.values():
        summary["benched"] = (
            summary["failed"] > 0
            and summary["succeeded"] == 0
            and any(summary["classified"][reason.value] for reason in BENCHING_REASONS)
        )
    return providers


def audit_provider_health_parity(
    conn: psycopg.Connection[Any],
    *,
    as_of: datetime | None = None,
    sample_limit: int = 25,
    row_limit: int = _DEFAULT_ROW_LIMIT,
) -> dict[str, Any]:
    """Compare legacy and normalized arena decisions in one repeatable-read snapshot.

    ``conn`` must be a caller-supplied connection. The result contains selected run IDs,
    terminal TTFA populations, classifications, excluded provider sets, exact mismatch
    counts, and at most ``sample_limit`` redacted samples. Error bodies are never returned.
    """
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")
    if row_limit <= 0:
        raise ValueError("row_limit must be positive")
    snapshot_time = as_of or datetime.now(UTC)
    with conn.transaction():
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        conn.execute("SET LOCAL statement_timeout = '5s'")
        conn.execute("SET LOCAL lock_timeout = '1s'")
        conn.execute("SET LOCAL idle_in_transaction_session_timeout = '10s'")
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            params = {"as_of": snapshot_time, "row_limit": row_limit + 1}
            cur.execute(_LEGACY_ROWS_SQL, params)
            legacy_rows = list(cur.fetchall())
            cur.execute(_NORMALIZED_ROWS_SQL, params)
            normalized_rows = list(cur.fetchall())

    legacy_complete = len(legacy_rows) <= row_limit
    normalized_complete = len(normalized_rows) <= row_limit
    legacy_rows = legacy_rows[:row_limit]
    normalized_rows = normalized_rows[:row_limit]

    legacy = _summarize(legacy_rows)
    normalized = _summarize(normalized_rows)
    providers = sorted(set(legacy) | set(normalized))
    mismatches: list[dict[str, Any]] = []
    mismatch_counts: Counter[str] = Counter()
    for provider in providers:
        left, right = legacy.get(provider), normalized.get(provider)
        if left is None or right is None:
            kind = "legacy_only" if right is None else "normalized_only"
        elif left["run_id"] != right["run_id"]:
            kind = "selected_run_id"
        elif (
            left["succeeded"],
            left["failed"],
            left["classified"],
            left["benched"],
        ) != (
            right["succeeded"],
            right["failed"],
            right["classified"],
            right["benched"],
        ):
            kind = "population_or_decision"
        else:
            continue
        mismatch_counts[kind] += 1
        if len(mismatches) < sample_limit:
            mismatches.append({"provider": provider, "kind": kind})

    legacy_set = {provider for provider, summary in legacy.items() if summary["benched"]}
    normalized_set = {provider for provider, summary in normalized.items() if summary["benched"]}
    return {
        "as_of": snapshot_time.isoformat(),
        "window": "one_day",
        "row_limit": row_limit,
        "complete": legacy_complete and normalized_complete,
        "legacy_rows_truncated": not legacy_complete,
        "normalized_rows_truncated": not normalized_complete,
        "legacy": legacy,
        "normalized": normalized,
        "excluded_sets": {
            "legacy_only": sorted(legacy_set - normalized_set),
            "normalized_only": sorted(normalized_set - legacy_set),
        },
        "mismatch_count": sum(mismatch_counts.values()),
        "mismatch_counts": dict(mismatch_counts),
        "mismatches": mismatches,
        "mismatches_truncated": (
            not legacy_complete
            or not normalized_complete
            or len(mismatches) < sum(mismatch_counts.values())
        ),
    }
