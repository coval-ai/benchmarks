# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic domain models for the persistence layer.

These are **persistence-layer** models only. The API layer has its own
``schemas.py``; do not couple them to these.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from coval_bench.registries.benchmarks import Benchmark

__all__ = [
    "Battle",
    "Benchmark",
    "LeaderboardSnapshot",
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
