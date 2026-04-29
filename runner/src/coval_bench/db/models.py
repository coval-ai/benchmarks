# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic domain models for the persistence layer.

These are **persistence-layer** models only. The API layer has its own
``schemas.py``; do not couple them to these.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


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


class Benchmark(StrEnum):
    """Which benchmark a result belongs to."""

    STT = "STT"
    TTS = "TTS"


class Run(BaseModel):
    """Domain model for a row in ``benchmarks_v2.runs``."""

    id: int | None = None  # set by DB (bigserial)
    started_at: datetime | None = None  # set by DB default (now())
    finished_at: datetime | None = None
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
    metric_type: str  # "WER" | "TTFT" | "TTFA" | "RTF" | "AudioToFinal"
    metric_value: float | None
    metric_units: str | None
    audio_filename: str | None = None
    transcript: str | None = None
    status: ResultStatus
    error: str | None = None
