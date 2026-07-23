# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable JSONL artifacts for benchmark runs.

The database is the source of truth for public aggregates. These artifacts are
for run-level debugging: one file contains the reproducibility metadata, every
metric row that was produced, and a compact failure summary.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from coval_bench.db.models import Result


class ArtifactRunHeader(BaseModel):
    """First JSONL record in a run artifact."""

    record_type: Literal["run"] = "run"
    schema_version: int = 1
    run_id: int
    runner_sha: str
    dataset_id: str
    dataset_sha256: str
    benchmark_kind: str
    smoke: bool
    scheduled_at: datetime | None
    started_at: datetime
    finished_at: datetime


class ArtifactResultRow(BaseModel):
    """Sanitized metric row copied from the in-memory Result model."""

    model_config = ConfigDict(use_enum_values=True)

    record_type: Literal["result"] = "result"
    provider: str
    model: str
    voice: str | None
    benchmark: str
    metric_type: str
    metric_value: float | None
    metric_units: str | None
    audio_filename: str | None
    status: str
    error: str | None
    transcript_sha256: str | None
    transcript_chars: int | None
    http_version: str | None
    submit_to_headers_ms: float | None


class ArtifactFailureBucket(BaseModel):
    """One grouped failure reason in the artifact summary."""

    provider: str
    model: str
    metric_type: str
    error: str
    count: int


class ArtifactSummary(BaseModel):
    """Last JSONL record in a run artifact."""

    record_type: Literal["summary"] = "summary"
    status: str
    total_results: int
    success_count: int
    fail_count: int
    failure_buckets: list[ArtifactFailureBucket]


def write_run_artifact(
    *,
    artifact_dir: Path,
    run_id: int,
    runner_sha: str,
    dataset_id: str,
    dataset_sha256: str,
    benchmark_kind: str,
    smoke: bool,
    scheduled_at: datetime | None,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    results: list[Result],
) -> Path:
    """Write a single atomic JSONL artifact for a completed or partial run."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    final_path = artifact_dir / f"run-{run_id}.jsonl"
    tmp_path = artifact_dir / f".run-{run_id}.jsonl.tmp"

    rows = [_result_row(r) for r in results]
    summary = ArtifactSummary(
        status=status,
        total_results=len(rows),
        success_count=sum(1 for r in rows if r.status == "success"),
        fail_count=sum(1 for r in rows if r.status == "failed"),
        failure_buckets=_failure_buckets(rows),
    )
    header = ArtifactRunHeader(
        run_id=run_id,
        runner_sha=runner_sha,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        benchmark_kind=benchmark_kind,
        smoke=smoke,
        scheduled_at=scheduled_at,
        started_at=started_at,
        finished_at=finished_at,
    )

    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(header.model_dump_json() + "\n")
        for row in rows:
            fh.write(row.model_dump_json() + "\n")
        fh.write(summary.model_dump_json() + "\n")
    tmp_path.replace(final_path)
    return final_path


def _result_row(result: Result) -> ArtifactResultRow:
    transcript = getattr(result, "transcript", None)
    return ArtifactResultRow(
        provider=str(result.provider),
        model=str(result.model),
        voice=result.voice,
        benchmark=str(result.benchmark),
        metric_type=str(result.metric_type),
        metric_value=result.metric_value,
        metric_units=result.metric_units,
        audio_filename=result.audio_filename,
        status=str(result.status),
        error=result.error,
        transcript_sha256=_transcript_sha256(transcript),
        transcript_chars=len(transcript) if isinstance(transcript, str) else None,
        http_version=result.http_version,
        submit_to_headers_ms=result.submit_to_headers_ms,
    )


def _transcript_sha256(transcript: str | None) -> str | None:
    if not isinstance(transcript, str):
        return None
    return hashlib.sha256(transcript.encode("utf-8")).hexdigest()


def _failure_buckets(rows: list[ArtifactResultRow]) -> list[ArtifactFailureBucket]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        if row.status != "failed" or not row.error:
            continue
        counts[(row.provider, row.model, row.metric_type, row.error)] += 1

    return [
        ArtifactFailureBucket(
            provider=provider,
            model=model,
            metric_type=metric_type,
            error=error,
            count=count,
        )
        for (provider, model, metric_type, error), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
