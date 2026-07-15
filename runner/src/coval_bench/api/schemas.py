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

from pydantic import BaseModel, ConfigDict, Field

from coval_bench.api.common import BenchmarkLiteral, WindowLiteral
from coval_bench.registries import TagCategory


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
    benchmark: Literal["STT", "TTS", "S2S"]
    dataset_id: str
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


class ModelTagOut(BaseModel):
    """A faceted filter tag: its category, raw value, and display label."""

    category: TagCategory
    value: str
    label: str


class TagCategoryOut(BaseModel):
    """A facet category's display metadata. Sent in display order."""

    category: TagCategory
    label: str
    # Values are provider/creator ids the client formats, not the per-tag label.
    provider_valued: bool = False


class ModelInfo(BaseModel):
    """A single model entry under a provider, with admin-disabled flag."""

    model: str
    disabled: bool = False
    tags: list[ModelTagOut] = []


class ProviderInfo(BaseModel):
    """Information about a single provider's models."""

    provider: str
    models: list[ModelInfo]
    modes: list[str] | None = None  # only TTS uses this


class ProvidersResponse(BaseModel):
    """Response schema for GET /v1/providers."""

    stt: list[ProviderInfo]
    tts: list[ProviderInfo]
    s2s: list[ProviderInfo]
    # Facet vocabulary in display order, shared across STT, TTS, and S2S.
    tag_categories: list[TagCategoryOut]


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
    window. ``dataset`` echoes the dataset the blocks were computed over
    ('__all__' = pooled across datasets); ``datasets`` lists the dataset ids
    with data in the window.
    """

    benchmark: BenchmarkLiteral
    window: WindowLiteral
    dataset: str
    datasets: list[str]
    model_stats: list[ModelStatEntry]
    series: list[SeriesPoint]


class RunsResponse(BaseModel):
    """Response schema for GET /v1/runs."""

    runs: list[RunOut]
    next_before: int | None = None


class LeaderboardResponse(BaseModel):
    """Response schema for GET /v1/leaderboard."""

    metric: Literal["WER", "TTFA", "TTFT", "TTFS", "V2V"]
    window: Literal["24h", "7d", "30d"]
    entries: list[LeaderboardEntry]


class BattleOut(BaseModel):
    """A battle to vote on. Blind by design: no provider/model identities."""

    id: uuid.UUID
    prompt_text: str
    domain: str | None
    audio_a_url: str
    audio_b_url: str


ArenaDomain = Literal["customer-service", "healthcare", "sales", "receptionist-booking", "other"]
"""Domains a battle can be tagged with. Each doubles as a leaderboard key, so the set is
closed and excludes ``all`` — that key is reserved for the aggregate board."""


class BattleCreate(BaseModel):
    """Request to generate a new battle from a user prompt."""

    prompt: str = Field(..., max_length=500)
    domain: ArenaDomain | None = None


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


class VoteIn(BaseModel):
    """Request body for POST /arena/vote (voter_type is set server-side, never sent)."""

    battle_id: uuid.UUID
    outcome: Literal["A_WIN", "B_WIN", "TIE"]
    voter_id: str


class VoteOut(BaseModel):
    """A recorded arena vote (the row as persisted)."""

    id: uuid.UUID
    battle_id: uuid.UUID
    outcome: str
    voter_type: str
    voter_id: str
    created_at: datetime
    updated_at: datetime


class RevealModelOut(BaseModel):
    """One side of a battle, de-anonymized after a vote."""

    provider: str
    model: str
    label: str


class RevealOut(BaseModel):
    """Post-vote reveal of both sides' identities."""

    a: RevealModelOut
    b: RevealModelOut
