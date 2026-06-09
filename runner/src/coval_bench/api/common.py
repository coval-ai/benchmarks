# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared API-layer literals and SQL fragments.

Single home for definitions that multiple routers (and schemas) must agree
on — adding a window or benchmark here updates every endpoint at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BenchmarkLiteral = Literal["STT", "TTS"]
WindowLiteral = Literal["24h", "7d", "30d"]


@dataclass(frozen=True)
class WindowSpec:
    """Per-window query parameters.

    interval: Postgres interval string. Fixed here — looked up by Python,
        never user-interpolated into SQL.
    series_bucket_seconds: Width of the aggregates series buckets; None keeps
        the scheduler-period buckets (exact scheduled_at).
    """

    interval: str
    series_bucket_seconds: int | None


WINDOWS: dict[WindowLiteral, WindowSpec] = {
    "24h": WindowSpec(interval="24 hours", series_bucket_seconds=None),
    "7d": WindowSpec(interval="7 days", series_bucket_seconds=2 * 3600),
    "30d": WindowSpec(interval="30 days", series_bucket_seconds=12 * 3600),
}

# Note: the leaderboard router keeps its own 7d/30d-only dict because its
# 24h window is served by the results_24h materialized view, not a live query.
WINDOW_INTERVALS: dict[str, str] = {w: spec.interval for w, spec in WINDOWS.items()}


def floor_epoch_sql(ts_expr: str, param: str) -> str:
    """SQL expression flooring ``ts_expr`` to ``%(param)s``-second buckets."""
    return f"to_timestamp(floor(extract(epoch FROM {ts_expr}) / %({param})s) * %({param})s)"


# Bucket expression for chart timestamps: the run's cron trigger time,
# falling back to created_at floored to the scheduler period for legacy rows.
# Shared by /results and /results/aggregates so both bucket identically.
# Expects ``r`` (results) and ``rn`` (runs) table aliases and a
# ``schedule_period`` query parameter.
SCHEDULED_AT_BUCKET_SQL = (
    "COALESCE(rn.scheduled_at, " + floor_epoch_sql("r.created_at", "schedule_period") + ")"
)
