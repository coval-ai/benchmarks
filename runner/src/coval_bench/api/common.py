# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared API-layer literals and SQL fragments.

Single home for definitions that multiple routers (and schemas) must agree
on — adding a window or benchmark here updates every endpoint at once.
"""

from __future__ import annotations

from typing import Literal

BenchmarkLiteral = Literal["STT", "TTS", "S2S"]
WindowLiteral = Literal["24h", "7d", "30d"]

# Fixed interval strings — looked up by Python, never user-interpolated into
# SQL. Used by live queries (the aggregates series block).
WINDOW_INTERVALS: dict[str, str] = {
    "24h": "24 hours",
    "7d": "7 days",
    "30d": "30 days",
}

# Per-window stats materialized views (schema-qualified). Looked up by Python
# from the validated WindowLiteral, never user-interpolated into SQL.
WINDOW_VIEWS: dict[str, str] = {
    "24h": "benchmarks_v2.results_24h",
    "7d": "benchmarks_v2.results_7d",
    "30d": "benchmarks_v2.results_30d",
}

# Fewest scored samples a headline stat may rest on, per modality. Below this a
# stat is still returned but flagged insufficient, so the frontend shows "n/a"
# rather than presenting one lucky sample as a provider's real number.
#
# Deliberately low. The floors sit just under the smallest legitimate cohort we
# serve: a per-run breakdown metric lands 10 samples at a time, and S2S runs a
# single ~30-item batch a day. A stricter floor would blank those. It is set to
# catch collapse (a provider that scored 1 of 480 attempts), not thin-but-real
# data, so a model failing most of its items still reports — see the
# success/attempt ratio work for that case.
#
# Read-side only: every row stays in the database exactly as written.
MIN_SCORED_SAMPLES: dict[str, int] = {
    "STT": 5,
    "TTS": 5,
    "S2S": 5,
}


def has_enough_samples(benchmark: str, sample_count: int) -> bool:
    """Whether *sample_count* clears the modality's floor for a headline stat.

    An unknown benchmark passes: a new modality should surface its numbers
    rather than silently read as "n/a" until someone adds a floor for it.
    """
    return sample_count >= MIN_SCORED_SAMPLES.get(benchmark, 0)


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
