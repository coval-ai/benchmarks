# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic domain models for the persistence layer.

These are **persistence-layer** models only. The API layer has its own
``schemas.py``; do not couple them to these.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, JsonValue, field_validator, model_validator

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import Gender

__all__ = [
    "Battle",
    "Benchmark",
    "Gender",
    "LeaderboardSnapshot",
    "PairingRating",
    "Result",
    "ResultStatus",
    "Run",
    "RunStatus",
    "SnapshotStatus",
    "Vote",
    "VoteOutcome",
    "VoterType",
    "Observation",
    "ObservationArtifact",
    "ObservationArtifactType",
    "ObservationFailureOrigin",
    "ObservationSourceKind",
    "ObservationStatus",
    "PreprocessingArtifact",
    "ProcessingStatus",
    "MetricArtifact",
    "MetricEvaluation",
    "MetricExecutor",
    "MetricValue",
    "MetricValueBucket",
    "TimestampArtifactSchema",
    "TimestampArtifactName",
    "MetricEvaluationInput",
]


class ProcessingStatus(StrEnum):
    """Shared lifecycle status for normalized work rows."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ObservationStatus(StrEnum):
    """Capture outcome of one benchmark sample/provider/model observation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ObservationSourceKind(StrEnum):
    """Source category; dataset_id/sha256 identify the concrete dataset."""

    DATASET_AUDIO = "dataset_audio"
    GENERATED_AUDIO = "generated_audio"
    CONVERSATION_AUDIO = "conversation_audio"


class ObservationFailureOrigin(StrEnum):
    PROVIDER = "provider"
    RUNNER = "runner"


class ObservationArtifactType(StrEnum):
    PROVIDER_TRANSCRIPT = "provider_transcript"
    GENERATED_AUDIO = "generated_audio"
    CONVERSATION_AUDIO = "conversation_audio"
    CONVERSATION_TRACE = "conversation_trace"
    TIMING_EVENTS = "timing_events"


class MetricExecutor(StrEnum):
    INLINE = "inline"
    COVAL_API = "coval_api"


class TimestampArtifactSchema(StrEnum):
    """Canonical schemas emitted by timestamp preprocessing pipelines."""

    WORD_TIMESTAMPS_V1 = "WordTimestampsV1"
    PHONEME_TIMESTAMPS_V1 = "PhonemeTimestampsV1"


class TimestampArtifactName(StrEnum):
    WORD_TIMESTAMPS = "word_timestamps"
    PHONEME_TIMESTAMPS = "phoneme_timestamps"


def _validate_processing_lifecycle(
    status: ProcessingStatus,
    started_at: datetime | None,
    finished_at: datetime | None,
    error: str | None,
) -> None:
    """Keep job/evaluation state transitions internally consistent."""
    if finished_at is not None and (started_at is None or finished_at < started_at):
        raise ValueError("finished_at requires started_at and must not precede it")
    if status is ProcessingStatus.QUEUED:
        if started_at is not None or finished_at is not None or error is not None:
            raise ValueError("queued work cannot have timestamps or error")
    elif status is ProcessingStatus.RUNNING:
        if started_at is None or finished_at is not None or error is not None:
            raise ValueError("running work requires started_at and no terminal state")
    elif status is ProcessingStatus.SUCCEEDED:
        if started_at is None or finished_at is None or error is not None:
            raise ValueError("succeeded work requires timestamps and no error")
    elif started_at is None or finished_at is None or not error:
        raise ValueError("failed work requires timestamps and an error")


def _private_gs_uri(value: str) -> str:
    if re.fullmatch(r"gs://[^/?#]+/[^/?#][^?#]*", value) is None:
        raise ValueError("must be a private gs:// object URI without query or fragment")
    return value


class ObservationArtifact(BaseModel):
    """One immutable raw artifact emitted while capturing an observation."""

    id: UUID | None = None
    observation_id: UUID | None = None
    artifact_type: ObservationArtifactType
    schema_name: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    gcs_uri: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    duration_ms: float | None = Field(default=None, gt=0)
    created_at: datetime | None = None

    _validate_uri = field_validator("gcs_uri")(_private_gs_uri)

    @field_validator("duration_ms")
    @classmethod
    def _finite_duration(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("duration_ms must be finite")
        return value


class Observation(BaseModel):
    """Normalized raw benchmark capture, independently of a metric result."""

    id: UUID | None = None
    run_id: int
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    voice: str | None = None
    benchmark: Benchmark
    source_kind: ObservationSourceKind
    transport_protocol: str | None = None
    submit_to_headers_ms: float | None = Field(default=None, ge=0)
    provider_extras: dict[str, JsonValue] | None = None
    artifacts: list[ObservationArtifact] = Field(default_factory=list)
    captured_at: datetime | None = None
    status: ObservationStatus
    error: str | None = None
    failure_origin: ObservationFailureOrigin | None = None

    @model_validator(mode="after")
    def _outcome_matches_error(self) -> Observation:
        if self.submit_to_headers_ms is not None and not math.isfinite(self.submit_to_headers_ms):
            raise ValueError("submit_to_headers_ms must be finite")
        if self.status is ObservationStatus.SUCCEEDED:
            if self.error is not None or self.failure_origin is not None:
                raise ValueError("succeeded observation cannot have error or failure_origin")
        elif not self.error or self.failure_origin is None:
            raise ValueError("failed observation requires error and failure_origin")
        artifact_types = [artifact.artifact_type for artifact in self.artifacts]
        if len(artifact_types) != len(set(artifact_types)):
            raise ValueError("observation artifacts cannot repeat an artifact_type")
        return self


class PreprocessingArtifact(BaseModel):
    id: UUID | None = None
    observation_id: UUID
    pipeline: str = Field(min_length=1)
    pipeline_version: str = Field(min_length=1)
    artifact_name: TimestampArtifactName
    schema_name: TimestampArtifactSchema
    schema_version: Literal["v1"] = "v1"
    producer_name: str = Field(min_length=1)
    producer_provider: str = Field(min_length=1)
    producer_model: str = Field(min_length=1)
    producer_version: str = Field(min_length=1)
    gcs_uri: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime | None = None

    _validate_uri = field_validator("gcs_uri")(_private_gs_uri)

    @model_validator(mode="after")
    def _artifact_identity(self) -> PreprocessingArtifact:
        expected = {
            TimestampArtifactName.WORD_TIMESTAMPS: TimestampArtifactSchema.WORD_TIMESTAMPS_V1,
            TimestampArtifactName.PHONEME_TIMESTAMPS: TimestampArtifactSchema.PHONEME_TIMESTAMPS_V1,
        }[self.artifact_name]
        if self.schema_name is not expected:
            raise ValueError("artifact name and schema must be a supported pair")
        return self


class MetricEvaluation(BaseModel):
    id: UUID | None = None
    observation_id: UUID
    metric_type: str = Field(min_length=1)
    metric_version: str = Field(min_length=1)
    evaluation_variant: str = Field(default="default", min_length=1)
    executor: MetricExecutor
    external_request_id: str | None = None
    status: ProcessingStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @model_validator(mode="after")
    def _lifecycle(self) -> MetricEvaluation:
        _validate_processing_lifecycle(self.status, self.started_at, self.finished_at, self.error)
        return self


class MetricEvaluationInput(BaseModel):
    """One immutable raw or preprocessing input frozen when work is queued."""

    observation_artifact_id: UUID | None = None
    preprocessing_artifact_id: UUID | None = None
    input_role: str = Field(min_length=1)
    input_order: int = Field(ge=0)

    @model_validator(mode="after")
    def _exactly_one_artifact(self) -> MetricEvaluationInput:
        if (self.observation_artifact_id is None) == (self.preprocessing_artifact_id is None):
            raise ValueError("exactly one observation or preprocessing artifact is required")
        return self


class MetricValue(BaseModel):
    metric_evaluation_id: UUID
    value_key: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    value: float
    is_primary: bool = False

    @field_validator("value")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value


class MetricArtifact(BaseModel):
    id: UUID | None = None
    metric_evaluation_id: UUID
    artifact_type: str = Field(min_length=1)
    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    created_at: datetime | None = None

    _validate_uri = field_validator("uri")(_private_gs_uri)


class MetricValueBucket(BaseModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    benchmark: Benchmark
    dataset_id: str = Field(min_length=1)
    metric_type: str = Field(min_length=1)
    metric_version: str = Field(min_length=1)
    evaluation_variant: str = Field(min_length=1)
    value_key: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    bucket_at: datetime
    min_value: float
    p25: float
    p50: float
    p75: float
    max_value: float
    value_sum: float
    sample_count: int = Field(gt=0)

    @model_validator(mode="after")
    def _valid_percentiles(self) -> MetricValueBucket:
        values = (self.min_value, self.p25, self.p50, self.p75, self.max_value, self.value_sum)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bucket values must be finite")
        if not self.min_value <= self.p25 <= self.p50 <= self.p75 <= self.max_value:
            raise ValueError("bucket percentiles must be ordered")
        return self


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
    persona_id: str | None = None  # S2S caller persona; provenance only, never aggregated


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
    # Which voice sang each side, and the gender both share. Stored rather than
    # derived from the registry: voices get retired, and a retired id would
    # leave historical rows unreadable. NULL on pre-gendered battles.
    voice_a: str | None = None
    voice_b: str | None = None
    gender: Gender | None = None
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
