# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Database layer: async psycopg3 pool, Pydantic domain models, RunWriter."""

from coval_bench.db.arena_store import ArenaStore
from coval_bench.db.conn import get_pool, lifespan_pool
from coval_bench.db.models import (
    Battle,
    Benchmark,
    MetricArtifact,
    MetricEvaluation,
    MetricEvaluationInput,
    MetricExecutor,
    MetricValue,
    MetricValueBucket,
    Observation,
    ObservationSourceKind,
    ObservationStatus,
    PreprocessingArtifact,
    ProcessingStatus,
    Result,
    ResultStatus,
    Run,
    RunStatus,
    TimestampArtifactName,
    TimestampArtifactSchema,
    Vote,
    VoteOutcome,
    VoterType,
)
from coval_bench.db.writer import RunWriter

__all__ = [
    "get_pool",
    "lifespan_pool",
    "ArenaStore",
    "Battle",
    "Benchmark",
    "MetricArtifact",
    "MetricEvaluation",
    "MetricEvaluationInput",
    "MetricExecutor",
    "MetricValue",
    "MetricValueBucket",
    "Observation",
    "ObservationSourceKind",
    "ObservationStatus",
    "PreprocessingArtifact",
    "ProcessingStatus",
    "Result",
    "ResultStatus",
    "Run",
    "RunStatus",
    "RunWriter",
    "TimestampArtifactName",
    "TimestampArtifactSchema",
    "Vote",
    "VoteOutcome",
    "VoterType",
]
