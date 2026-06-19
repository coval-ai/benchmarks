# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic v2 response schemas for the Coval Benchmarks API.

Note: ``transcript`` is intentionally not exposed — keeps payloads small and
avoids leaking dataset content. A ``?include_transcript=true`` flag can be
added later if needed.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from coval_bench.api.common import BenchmarkLiteral, WindowLiteral


class RunOut(BaseModel):
    """API response schema for a benchmark run row."""

    id: int
    started_at: datetime
    finished_at: datetime | None
    status: Literal["RUNNING", "SUCCEEDED", "PARTIAL", "FAILED"]
    runner_sha: str
    dataset_id: str
    dataset_sha256: str
    error: str | None

    model_config = ConfigDict(from_attributes=True)


class ResultOut(BaseModel):
    """API response schema for a single benchmark result row.

    ``status`` is sourced from the *parent run*, not from the result row's own
    ``status`` column (which is always ``'success'`` because we filter on it).
    The parent-run status is denormalized here at the API boundary via SQL JOIN
    so the frontend does not need a second round-trip.
    """

    id: int
    run_id: int
    provider: str
    model: str
    voice: str | None
    benchmark: Literal["STT", "TTS"]
    metric_type: str
    metric_value: float | None
    metric_units: str | None
    audio_filename: str | None
    created_at: datetime
    scheduled_at: datetime
    status: Literal["RUNNING", "SUCCEEDED", "PARTIAL", "FAILED"]

    model_config = ConfigDict(from_attributes=True)


class LeaderboardEntry(BaseModel):
    """A single entry in the leaderboard response."""

    provider: str
    model: str
    avg: float
    p50: float
    p95: float
    n: int


class ModelInfo(BaseModel):
    """A single model entry under a provider, with admin-disabled flag."""

    model: str
    disabled: bool = False


class ProviderInfo(BaseModel):
    """Information about a single provider's models."""

    provider: str
    models: list[ModelInfo]
    modes: list[str] | None = None  # only TTS uses this


class ProvidersResponse(BaseModel):
    """Response schema for GET /v1/providers."""

    stt: list[ProviderInfo]
    tts: list[ProviderInfo]


class ResultsResponse(BaseModel):
    """Response schema for GET /v1/results."""

    results: list[ResultOut]


class ModelStatEntry(BaseModel):
    """Per-(provider, model, metric_type) aggregate stats.

    Lets us compute the stats server-side and just send the summaries.
    """

    provider: str
    model: str
    metric_type: str
    avg_value: float
    stddev_value: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    min_value: float
    max_value: float
    sample_count: int


class SeriesPoint(BaseModel):
    """Per-(provider, model, metric_type) distribution for one scheduled_at bucket.

    Latency timelines render p50; WER renders value_sum / sample_count.
    """

    provider: str
    model: str
    metric_type: str
    scheduled_at: datetime
    min_value: float
    p25: float
    p50: float
    p75: float
    max_value: float
    value_sum: float
    sample_count: int


class AggregatesResponse(BaseModel):
    """Response schema for GET /v1/results/aggregates.

    Wraps and returns all our ModelStatEntry and SeriesPoint data for a time
    window.
    """

    benchmark: BenchmarkLiteral
    window: WindowLiteral
    model_stats: list[ModelStatEntry]
    series: list[SeriesPoint]


class RunsResponse(BaseModel):
    """Response schema for GET /v1/runs."""

    runs: list[RunOut]
    next_before: int | None = None


class LeaderboardResponse(BaseModel):
    """Response schema for GET /v1/leaderboard."""

    metric: Literal["WER", "TTFA", "TTFT", "TTFS"]
    window: Literal["24h", "7d", "30d"]
    entries: list[LeaderboardEntry]


class BattleOut(BaseModel):
    """A battle to vote on. Blind by design: no provider/model identities."""

    id: uuid.UUID
    prompt_text: str
    domain: str | None
    audio_a_url: str
    audio_b_url: str


class LeaderboardEntryOut(BaseModel):
    """One model's row within an arena leaderboard board."""

    provider: str
    model: str
    rating_elo: float
    rating_bt: float
    ci_low: float | None
    ci_high: float | None
    ci_half_width: float | None
    votes_total: int
    wins: float
    losses: float
    ties: float
    status: str


class ArenaLeaderboardResponse(BaseModel):
    """The latest board for a metric/domain, or empty if none computed yet."""

    metric: str
    domain: str
    computed_at: datetime | None
    methodology_version: str | None
    entries: list[LeaderboardEntryOut]
