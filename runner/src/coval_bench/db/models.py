# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic domain models for the persistence layer.

These are **persistence-layer** models only. The API layer has its own
``schemas.py``; do not couple them to these.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from coval_bench.registries.benchmarks import Benchmark

__all__ = [
    "Battle",
    "Benchmark",
    "BillingUnit",
    "LeaderboardSnapshot",
    "PairingRating",
    "PriceRow",
    "Result",
    "ResultStatus",
    "Run",
    "RunStatus",
    "SnapshotStatus",
    "Vote",
    "VoteOutcome",
    "VoterType",
]


class RunStatus(StrEnum):
    """Lifecycle status of a benchmark run."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class ResultStatus(StrEnum):
    """Outcome of a single result row."""

    SUCCESS = "success"
    FAILED = "failed"


class Run(BaseModel):
    """Domain model for a row in ``benchmarks_v2.runs``."""

    id: int | None = None  # set by DB (bigserial)
    started_at: datetime | None = None  # set by DB default (now())
    finished_at: datetime | None = None
    scheduled_at: datetime | None = None  # cron trigger time, floored to the scheduler period
    runner_sha: str
    dataset_id: str
    dataset_sha256: str
    status: RunStatus
    error: str | None = None
    # whisper-1 judge spend for the run; null before migration 0015.
    judge_input_tokens: int | None = None
    judge_output_tokens: int | None = None
    judge_audio_seconds: float | None = None


class Result(BaseModel):
    """Domain model for a row in ``benchmarks_v2.results``."""

    id: int | None = None  # set by DB (bigserial)
    run_id: int
    provider: str
    model: str
    voice: str | None = None
    benchmark: Benchmark
    metric_type: str  # values come from coval_bench.registries.Metric
    metric_value: float | None
    metric_units: str | None
    audio_filename: str | None = None
    transcript: str | None = None
    status: ResultStatus
    error: str | None = None
    http_version: str | None = None
    submit_to_headers_ms: float | None = None
    # WER only: metric_value split in percentage points; null before migration 0014.
    wer_insertions_pct: float | None = None
    wer_deletions_pct: float | None = None
    wer_substitutions_pct: float | None = None
    # Raw usage in native billing units; null before migration 0015 or when
    # the provider reports nothing.
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    billable_seconds: float | None = None
    characters_in: int | None = None
    audio_seconds_in: float | None = None
    audio_seconds_out: float | None = None


class BillingUnit(StrEnum):
    """Native billing unit a provider publishes its rate in.

    Rates are stored raw in these units; normalization to $/1k min or
    $/1M chars happens at read time.
    """

    PER_MINUTE = "per_minute"
    PER_SECOND = "per_second"
    PER_HOUR = "per_hour"
    PER_1M_CHARS = "per_1m_chars"
    PER_1M_TOKENS_INPUT = "per_1m_tokens_input"
    PER_1M_TOKENS_OUTPUT = "per_1m_tokens_output"
    PER_REQUEST = "per_request"


class PriceRow(BaseModel):
    """Domain model for a row in ``benchmarks_v2.model_pricing`` (append-only).

    ``superseded_at IS NULL`` marks the currently effective rate; token-billed
    models hold two effective rows (input + output units).
    """

    id: int | None = None  # set by DB (bigserial)
    provider: str
    model: str
    benchmark: Benchmark
    billing_unit: BillingUnit
    rate_usd: Decimal
    plan_assumption: str | None = None
    effective_at: datetime
    superseded_at: datetime | None = None
    source_url: str
    as_of: date
    evidence: str | None = None
    updated_by: str  # 'human' | 'bot'
    created_at: datetime | None = None  # set by DB default (now())


class VoteOutcome(StrEnum):
    """Outcome of a single A/B battle vote."""

    A_WIN = "A_WIN"
    B_WIN = "B_WIN"
    TIE = "TIE"


class VoterType(StrEnum):
    """Who cast the vote. ``external`` is reserved for future public voting."""

    LABELER = "labeler"
    EXTERNAL = "external"


class Battle(BaseModel):
    """Domain model for a row in ``arena.battles`` — one A vs B matchup."""

    id: UUID | None = None  # set by DB (gen_random_uuid)
    provider_a: str
    model_a: str
    provider_b: str
    model_b: str
    domain: str | None = None
    prompt_text: str
    audio_a_url: str
    audio_b_url: str
    created_at: datetime | None = None  # set by DB default (now())


class Vote(BaseModel):
    """Domain model for a row in ``arena.votes`` — one human judgment."""

    id: UUID | None = None  # set by DB (gen_random_uuid)
    battle_id: UUID
    outcome: VoteOutcome
    voter_type: VoterType
    voter_id: str
    created_at: datetime | None = None  # set by DB default (now())
    updated_at: datetime | None = None  # maintained by the BEFORE UPDATE trigger


class SnapshotStatus(StrEnum):
    """Confidence tier of a leaderboard rating, gated on the CI half-width."""

    PRELIMINARY = "preliminary"
    USABLE = "usable"
    ESTABLISHED = "established"


class LeaderboardSnapshot(BaseModel):
    """Domain model for one ``arena.leaderboard_snapshots`` row — one model in one board.

    A board is every row sharing ``computed_at`` + ``metric_name`` +
    ``methodology_version`` + ``domain``. ``computed_at`` is left to the DB
    default so a board written in one transaction shares a single timestamp.
    """

    computed_at: datetime | None = None  # set by DB default (now())
    metric_name: str
    methodology_version: str
    domain: str = "all"
    provider: str
    model: str
    rating_elo: float
    rating_bt: float
    ci_low: float | None = None
    ci_high: float | None = None
    ci_half_width: float | None = None
    votes_total: int
    wins: float
    losses: float
    ties: float
    status: SnapshotStatus


class PairingRating(BaseModel):
    """Minimal rating view the pairing heuristic needs from a leaderboard board."""

    rating_elo: float
    ci_half_width: float | None = None
