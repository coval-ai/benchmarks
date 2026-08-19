# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN401 -- adapter composes lazy runtime collaborators from orchestrator.
"""Best-effort additive persistence of legacy STT/TTS results."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from coval_bench.db.models import (
    MetricEvaluation,
    MetricEvaluationInput,
    MetricExecutor,
    MetricValue,
    Observation,
    ObservationArtifactType,
    ObservationFailureOrigin,
    ObservationSourceKind,
    ObservationStatus,
    ProcessingStatus,
)
from coval_bench.observation_artifacts import (
    snapshot_generated_audio,
    upload_generated_audio,
    upload_provider_transcript,
    upload_timing_events,
)
from coval_bench.registries import Metric


def _inputs(metric: str, artifacts: dict[Any, Any], benchmark: Any) -> list[MetricEvaluationInput]:
    """Freeze the raw artifact lineage used for one legacy metric."""
    wanted: list[tuple[Any, str]]
    if benchmark.value.upper() == "STT":
        wanted = (
            [(ObservationArtifactType.PROVIDER_TRANSCRIPT, "raw")]
            if metric == Metric.WER
            else [(ObservationArtifactType.TIMING_EVENTS, "timing")]
        )
    elif metric == Metric.WER:
        wanted = [(ObservationArtifactType.GENERATED_AUDIO, "raw")]
    elif metric == Metric.TTFA:
        wanted = [
            (ObservationArtifactType.TIMING_EVENTS, "timing"),
            (ObservationArtifactType.GENERATED_AUDIO, "raw"),
        ]
    else:
        wanted = [(ObservationArtifactType.TIMING_EVENTS, "timing")]
    return [
        MetricEvaluationInput(
            observation_artifact_id=artifacts[kind], input_role=role, input_order=0
        )
        for kind, role in wanted
        if artifacts.get(kind) is not None
    ]


def _values(metric: str, rows: Sequence[Any], evaluation_id: Any) -> list[MetricValue]:
    primary = next(
        row for row in rows if row.metric_value is not None and str(row.status) == "success"
    )
    values = [
        MetricValue(
            metric_evaluation_id=evaluation_id,
            value_key="primary",
            unit=primary.metric_units,
            value=primary.metric_value,
            value_role="primary",
        )
    ]
    if metric == Metric.WER:
        components = (
            ("insertions", primary.wer_insertions_pct),
            ("deletions", primary.wer_deletions_pct),
            ("substitutions", primary.wer_substitutions_pct),
        )
        values.extend(
            MetricValue(
                metric_evaluation_id=evaluation_id, value_key=key, unit="percent", value=value
            )
            for key, value in components
            if value is not None
        )
    if metric == Metric.TTFA:
        component_rows = {str(row.metric_type): row for row in rows}
        roundtrip = component_rows.get(str(Metric.TTFA_ROUNDTRIP))
        silence = component_rows.get(str(Metric.TTFA_LEADING_SILENCE))
        if (
            roundtrip is not None
            and silence is not None
            and roundtrip.metric_value is not None
            and silence.metric_value is not None
        ):
            values.extend(
                (
                    MetricValue(
                        metric_evaluation_id=evaluation_id,
                        value_key="roundtrip",
                        unit="milliseconds",
                        value=roundtrip.metric_value,
                    ),
                    MetricValue(
                        metric_evaluation_id=evaluation_id,
                        value_key="leading_silence",
                        unit="milliseconds",
                        value=silence.metric_value,
                    ),
                )
            )
    return values


async def dual_write(
    *,
    writer: Any,
    storage_client: Any,
    bucket: str,
    run_id: int,
    dataset_id: str,
    dataset_sha256: str,
    sample_id: str,
    entry: Any,
    benchmark: Any,
    results: Sequence[Any],
    provider_error: str | None,
    transcript: str | None = None,
    timing_events: dict[str, Any] | None = None,
    audio_path: Any = None,
    voice: str | None = None,
) -> None:
    """Persist one observation and its grouped normalized evaluations."""
    audio_snapshot = snapshot_generated_audio(audio_path) if audio_path is not None else None
    artifacts = []
    if transcript is not None:
        artifacts.append(
            await asyncio.to_thread(upload_provider_transcript, storage_client, bucket, transcript)
        )
    if timing_events:
        artifacts.append(
            await asyncio.to_thread(upload_timing_events, storage_client, bucket, timing_events)
        )
    if audio_snapshot is not None:
        audio_payload, audio_duration_ms = audio_snapshot
        artifacts.append(
            await asyncio.to_thread(
                upload_generated_audio,
                storage_client,
                bucket,
                audio_payload,
                audio_duration_ms,
            )
        )
    source_kind = (
        ObservationSourceKind.DATASET_AUDIO
        if benchmark.value.upper() == "STT"
        else ObservationSourceKind.GENERATED_AUDIO
    )
    observation = await writer.insert_observation(
        Observation(
            run_id=run_id,
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256,
            sample_id=sample_id,
            provider=entry.provider,
            model=entry.model,
            voice=voice,
            benchmark=benchmark,
            source_kind=source_kind,
            status=ObservationStatus.FAILED if provider_error else ObservationStatus.SUCCEEDED,
            error=provider_error,
            failure_origin=ObservationFailureOrigin.PROVIDER if provider_error else None,
            artifacts=artifacts,
        )
    )
    artifact_ids = {artifact.artifact_type: artifact.id for artifact in observation.artifacts}
    grouped: dict[str, list[Any]] = {}
    for row in results:
        metric = (
            str(Metric.TTFA)
            if row.metric_type in (Metric.TTFA_ROUNDTRIP, Metric.TTFA_LEADING_SILENCE)
            else str(row.metric_type)
        )
        grouped.setdefault(metric, []).append(row)
    for metric, rows in grouped.items():
        evaluation = await writer.insert_metric_evaluation(
            MetricEvaluation(
                observation_id=observation.id,
                metric_type=metric,
                metric_version="v1",
                executor=MetricExecutor.INLINE,
                status=ProcessingStatus.QUEUED,
            ),
            inputs=_inputs(metric, artifact_ids, benchmark),
        )
        if (
            evaluation.status is ProcessingStatus.SUCCEEDED
            or evaluation.status is ProcessingStatus.FAILED
        ):
            continue
        if evaluation.status is ProcessingStatus.QUEUED:
            evaluation = await writer.start_metric_evaluation(
                evaluation.id, started_at=datetime.now(UTC)
            )
        primary = next(
            (row for row in rows if row.metric_value is not None and str(row.status) == "success"),
            None,
        )
        finished_at = datetime.now(UTC)
        if primary is None:
            await writer.fail_metric_evaluation(
                evaluation.id,
                finished_at=finished_at,
                error=next(
                    (row.error for row in rows if row.error), "legacy metric produced no value"
                ),
            )
        else:
            await writer.complete_metric_evaluation(
                evaluation.id, finished_at=finished_at, values=_values(metric, rows, evaluation.id)
            )
