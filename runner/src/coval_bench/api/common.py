# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared API-layer literals and SQL fragments.

Single home for definitions that multiple routers (and schemas) must agree
on — adding a window or benchmark here updates every endpoint at once.
"""

from __future__ import annotations

from typing import Literal

BenchmarkLiteral = Literal["STT", "TTS"]
WindowLiteral = Literal["24h", "7d", "30d"]

# Fixed interval strings — looked up by Python, never user-interpolated into
# SQL. Note: the leaderboard router keeps its own 7d/30d-only dict because its
# 24h window is served by the results_24h materialized view, not a live query.
WINDOW_INTERVALS: dict[str, str] = {
    "24h": "24 hours",
    "7d": "7 days",
    "30d": "30 days",
}

# Bucket expression for chart timestamps: the run's cron trigger time,
# falling back to created_at floored to the scheduler period for legacy rows.
# Shared by /results and /results/aggregates so both bucket identically.
# Expects ``r`` (results) and ``rn`` (runs) table aliases and a
# ``schedule_period`` query parameter.
SCHEDULED_AT_BUCKET_SQL = (
    "COALESCE(rn.scheduled_at,"
    " to_timestamp(floor(extract(epoch FROM r.created_at) / %(schedule_period)s)"
    " * %(schedule_period)s))"
)
