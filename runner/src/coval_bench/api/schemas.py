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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from coval_bench.api.common import BenchmarkLiteral, WindowLiteral
from coval_bench.arena.domains import ArenaDomain
from coval_bench.registries import Benchmark, Licensing, Source, TagCategory, Voice


class RunOut(BaseModel):
    """API response schema for a benchmark run row."""

    id: int
    started_at: datetime
    finished_at: datetime | None
    status: Literal["RUNNING", "SUCCEEDED", "PARTIAL", "FAILED"]
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
    # True when ``n`` is under the modality's floor: the values are real but rest
    # on too few samples to rank on, so clients render "n/a". Sent alongside the
    # numbers rather than in place of them, so a caller that wants the raw
    # measurement still has it.
    insufficient_samples: bool = False


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
    early_access: bool = False
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
    # True when sample_count is under the modality's floor: the values are real
    # but rest on too few samples to present, so clients render "n/a". Sent
    # alongside the numbers rather than in place of them, so a caller that wants
    # the raw measurement still has it.
    insufficient_samples: bool = False
    # WER only, percentage points summing to avg_value; null pre-0014 rows —
    # clients fall back to the total alone.
    wer_insertions_pct: float | None = None
    wer_deletions_pct: float | None = None
    wer_substitutions_pct: float | None = None


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


class DatasetAggregates(BaseModel):
    """Per-model stats for one dataset — one block of the by-dataset response."""

    dataset: str
    model_stats: list[ModelStatEntry]


class AggregatesByDatasetResponse(BaseModel):
    """Response schema for GET /v1/results/aggregates/by-dataset.

    One block per dataset with data in the window, sorted by dataset id.
    Series are deliberately absent: per-dataset timelines would multiply the
    payload by the dataset count and nothing consumes them batched — the
    single-dataset endpoint serves that long tail.
    """

    benchmark: BenchmarkLiteral
    window: WindowLiteral
    blocks: list[DatasetAggregates]


class RunsResponse(BaseModel):
    """Response schema for GET /v1/runs."""

    runs: list[RunOut]
    next_before: int | None = None


class LeaderboardResponse(BaseModel):
    """Response schema for GET /v1/leaderboard."""

    metric: Literal["WER", "TTFA", "TTFT", "TTFS", "V2V"]
    window: Literal["24h", "7d", "30d"]
    entries: list[LeaderboardEntry]


class S2SSampleTurnOut(BaseModel):
    """One spoken turn of a sampled conversation."""

    index: int
    role: str
    content: str
    start_offset: float | None = None
    end_offset: float | None = None


class S2SSampleRecordingOut(BaseModel):
    """One model's recording. ``audio_path`` is an API route, not a storage URL."""

    provider: str
    model: str
    audio_path: str
    coval_run_id: str
    sim_id: str
    agent_id: str | None = None
    turns: list[S2SSampleTurnOut] = Field(default_factory=list)


class S2SSampleOut(BaseModel):
    """One sample, filtered to the recordings this caller may see."""

    schema_version: int | None = None
    sample_id: str
    test_case_id: str
    test_set_id: str | None = None
    persona_name: str | None = None
    transcript: str | None = None
    recordings: list[S2SSampleRecordingOut]


class S2SSampleAudioOut(BaseModel):
    """A freshly signed URL for one recording, with the moment it stops working.

    Handed over as a body rather than a redirect: a browser cannot carry its
    early-access header through a cross-origin redirect to storage, so the caller
    fetches this with its proof and then points an audio element at ``url``.
    ``expires_at`` lets the caller re-ask before playing rather than after failing.
    """

    url: str
    expires_at: datetime


class BattleOut(BaseModel):
    """A battle to vote on. Blind by design: no provider/model identities."""

    id: uuid.UUID
    prompt_text: str
    domain: str | None
    audio_a_url: str
    audio_b_url: str


class BattleCreate(BaseModel):
    """Request to generate a new battle from a user prompt."""

    prompt: str = Field(..., max_length=500)
    domain: ArenaDomain | None = None


class ExamplePromptOut(BaseModel):
    """A random seed-bank prompt for the arena UI's example picker."""

    prompt: str
    domain: ArenaDomain


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


class TagOut(BaseModel):
    """A MODE or FEATURES vocabulary entry."""

    value: str = Field(min_length=1)
    category: Literal["mode", "features"]
    label: str = Field(min_length=1)


class TagsResponse(BaseModel):
    """Response schema for GET /v1/tags."""

    tags: list[TagOut]


class AdminChangeOut(BaseModel):
    """One history row, embedded under its model. ``old`` is NULL on create."""

    id: int
    old: dict[str, Any] | None = None
    new: dict[str, Any]
    changed_by_user_id: str
    changed_by_org_id: str | None = None
    changed_by_email: str | None = None
    changed_at: datetime


class AdminModelOut(BaseModel):
    """A registered model with full state, provenance, and recent history."""

    id: int
    modality: Benchmark
    provider: str
    model: str
    voice: str | None = None
    voices: list[Voice] = []
    creator: str | None = None
    source: Source
    licensing: Licensing
    on_prem: bool
    region: Literal["us", "eu", "asia"] | None = None
    arena_enabled: bool
    collected: bool
    published: bool
    tags: list[str] = []
    updated_by_user_id: str
    updated_by_email: str | None = None
    updated_at: datetime
    history: list[AdminChangeOut] = []


class AdminModelsResponse(BaseModel):
    """Response schema for GET /v1/admin/models."""

    models: list[AdminModelOut]


class AdminModelCreate(BaseModel):
    """POST body for /v1/admin/models. The defaults land the model Hidden."""

    modality: Benchmark
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    voice: str | None = Field(default=None, min_length=1)
    voices: list[Voice] = []
    creator: str | None = Field(default=None, min_length=1)
    source: Source = Source.OFFICIAL_API
    licensing: Licensing = Licensing.PROPRIETARY
    on_prem: bool = False
    region: Literal["us", "eu", "asia"] | None = None
    arena_enabled: bool = True
    collected: bool = True
    published: bool = False
    tags: list[str] = []


class AdminModelPatch(BaseModel):
    """PATCH body for /v1/admin/models/{id}; absent fields stay unchanged."""

    provider: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    voice: str | None = Field(default=None, min_length=1)
    voices: list[Voice] | None = None
    creator: str | None = Field(default=None, min_length=1)
    source: Source | None = None
    licensing: Licensing | None = None
    on_prem: bool | None = None
    region: Literal["us", "eu", "asia"] | None = None
    arena_enabled: bool | None = None
    collected: bool | None = None
    published: bool | None = None
    tags: list[str] | None = None


class AdminModelUpdateResponse(BaseModel):
    """PATCH response: the updated model plus warnings a rename leaves behind."""

    model: AdminModelOut
    warnings: list[str] = []
