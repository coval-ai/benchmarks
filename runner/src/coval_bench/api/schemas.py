# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic v2 response schemas for the Coval Benchmarks API.

Note: ``transcript`` is intentionally not exposed — keeps payloads small and
avoids leaking dataset content. A ``?include_transcript=true`` flag can be
added later if needed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


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
    """API response schema for a single benchmark result row."""

    id: int
    run_id: int
    provider: str
    model: str
    voice: str | None
    benchmark: Literal["STT", "TTS"]
    metric_type: str
    metric_value: float
    metric_units: str
    audio_filename: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class LeaderboardEntry(BaseModel):
    """A single entry in the leaderboard response."""

    provider: str
    model: str
    avg: float
    p50: float
    p95: float
    n: int


class ProviderInfo(BaseModel):
    """Information about a single provider's models."""

    provider: str
    models: list[str]
    modes: list[str] | None = None  # only TTS uses this


class ProvidersResponse(BaseModel):
    """Response schema for GET /v1/providers."""

    stt: list[ProviderInfo]
    tts: list[ProviderInfo]
