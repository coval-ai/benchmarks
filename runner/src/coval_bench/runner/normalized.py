# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN401 -- adapter composes lazy runtime collaborators from orchestrator.
"""Best-effort dual writes and durable replay for normalized benchmark results."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

import psycopg
import structlog
from google.api_core.exceptions import GoogleAPIError
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, Field, model_validator

from coval_bench.db.models import (
    MetricEvaluation,
    MetricEvaluationInput,
    MetricExecutor,
    MetricValue,
    Observation,
    ObservationArtifact,
    ObservationArtifactType,
    ObservationFailureOrigin,
    ObservationSourceKind,
    ObservationStatus,
    ProcessingStatus,
    Result,
)
from coval_bench.observation_artifacts import (
    prepare_provider_transcript,
    prepare_timing_events,
    snapshot_generated_audio,
    upload_generated_audio,
    upload_prepared_observation_artifact,
    upload_provider_transcript,
    upload_timing_events,
)
from coval_bench.registries import Metric
from coval_bench.runner.capture import (
    CaptureEnvelope,
    build_capture_identity,
    read_legacy_result_allocation,
    upload_claim,
    upload_envelope,
    upload_legacy_result_allocation,
    upload_receipt,
)

logger = structlog.get_logger("coval_bench.runner")


class FrozenEvaluation(BaseModel):
    """Schema-checked normalized evaluation payload used during replay."""

    model_config = ConfigDict(extra="forbid")

    metric_type: str = Field(min_length=1)
    metric_version: str = Field(min_length=1)
    evaluation_variant: str | None = None
    executor: str = Field(min_length=1)
    status: str = Field(min_length=1)
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    external_request_id: str | None = None
    values: list[FrozenValue] = Field(default_factory=list)
    inputs: list[FrozenInput] = Field(default_factory=list)

    @model_validator(mode="after")
    def _terminal_shape(self) -> Self:
        if self.status not in {str(ProcessingStatus.SUCCEEDED), str(ProcessingStatus.FAILED)}:
            raise ValueError("frozen evaluations must be terminal")
        if self.started_at is None or self.finished_at is None:
            raise ValueError("frozen terminal evaluations require timestamps")
        if self.finished_at < self.started_at:
            raise ValueError("frozen evaluation finished_at precedes started_at")
        if self.status == str(ProcessingStatus.SUCCEEDED):
            if self.error is not None or not self.values:
                raise ValueError("frozen succeeded evaluations require values and no error")
        elif not self.error or self.values:
            raise ValueError("frozen failed evaluations require an error and no values")
        return self


class FrozenValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value_key: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    value: float
    value_role: str = Field(min_length=1)


class FrozenInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_name: str = Field(min_length=1)
    input_role: str = Field(min_length=1)
    input_order: int = Field(ge=0)


class FrozenArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    schema_name: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    extension: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    duration_ms: float | None = Field(default=None, gt=0)


class FrozenCapture(BaseModel):
    """Complete database replay shape, independent of current metric mappers."""

    model_config = ConfigDict(extra="forbid")

    run_id: int = Field(gt=0)
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: str = Field(min_length=1)
    benchmark: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    voice: str | None = None
    captured_at: datetime
    observation_status: str = Field(min_length=1)
    observation_error: str | None = None
    source_kind: str = Field(min_length=1)
    transport_protocol: str | None = None
    submit_to_headers_ms: float | None = None
    provider_extras: dict[str, Any] | None = None
    legacy_rows: list[dict[str, Any]] = Field(default_factory=list)
    evaluations: list[FrozenEvaluation] = Field(default_factory=list)
    artifacts: list[FrozenArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def _observation_shape(self) -> Self:
        failed = self.observation_status == str(ObservationStatus.FAILED)
        if self.observation_status not in {
            str(ObservationStatus.SUCCEEDED),
            str(ObservationStatus.FAILED),
        }:
            raise ValueError("frozen observation has an invalid status")
        if failed != (self.observation_error is not None):
            raise ValueError("frozen observation status and error disagree")
        return self


def _frozen_values(metric: str, rows: Sequence[Any], primary: Any) -> list[FrozenValue]:
    values = [
        FrozenValue(
            value_key="primary",
            unit=primary.metric_units,
            value=primary.metric_value,
            value_role="primary",
        )
    ]
    if metric == str(Metric.WER):
        for key, value in (
            ("insertions", primary.wer_insertions_pct),
            ("deletions", primary.wer_deletions_pct),
            ("substitutions", primary.wer_substitutions_pct),
        ):
            if value is not None:
                values.append(
                    FrozenValue(value_key=key, unit="percent", value=value, value_role="component")
                )
        counts = (
            ("substitution_count", primary.wer_substitutions),
            ("deletion_count", primary.wer_deletions),
            ("insertion_count", primary.wer_insertions),
            ("reference_words", primary.wer_reference_words),
        )
        if all(value is not None for _, value in counts):
            values.extend(
                FrozenValue(
                    value_key=key,
                    unit="count",
                    value=float(value),
                    value_role="component",
                )
                for key, value in counts
            )
    if metric == str(Metric.TTFA):
        components = {str(row.metric_type): row for row in rows}
        for metric_type, key in (
            (str(Metric.TTFA_ROUNDTRIP), "roundtrip"),
            (str(Metric.TTFA_LEADING_SILENCE), "leading_silence"),
        ):
            row = components.get(metric_type)
            if row is not None and row.metric_value is not None:
                values.append(
                    FrozenValue(
                        value_key=key,
                        unit="milliseconds",
                        value=row.metric_value,
                        value_role="component",
                    )
                )
    return values


def _frozen_inputs(metric: str, benchmark: str, available: set[str]) -> list[FrozenInput]:
    if benchmark == "STT":
        wanted = [("transcript", "raw")] if metric == str(Metric.WER) else [("timing", "timing")]
    elif metric == str(Metric.WER):
        wanted = [("audio", "raw")]
    elif metric == str(Metric.TTFA):
        wanted = [("timing", "timing"), ("audio", "raw")]
    else:
        wanted = [("timing", "timing")]
    return [
        FrozenInput(artifact_name=name, input_role=role, input_order=0)
        for name, role in wanted
        if name in available
    ]


def _freeze_artifact(
    *,
    name: str,
    descriptor: tuple[ObservationArtifactType, bytes, str, str, str],
    duration_ms: float | None = None,
) -> tuple[FrozenArtifact, bytes]:
    artifact_type, payload, extension, content_type, schema_name = descriptor
    return (
        FrozenArtifact(
            name=name,
            artifact_type=str(artifact_type),
            schema_name=schema_name,
            schema_version="v1",
            extension=extension,
            content_type=content_type,
            content_sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            duration_ms=duration_ms,
        ),
        payload,
    )


def prepare_capture_envelope(
    *,
    run_id: int,
    dataset_id: str,
    dataset_sha256: str,
    sample_id: str,
    entry: Any,
    benchmark: Any,
    results: Sequence[Any],
    provider_error: str | None,
    captured_at: datetime,
    voice: str | None = None,
    transcript: str | None = None,
    timing_events: Mapping[str, Any] | None = None,
    audio_snapshot: tuple[bytes, float] | None = None,
    executor: MetricExecutor = MetricExecutor.INLINE,
    capture_id: str | None = None,
    provider_extras: Mapping[str, Any] | None = None,
) -> CaptureEnvelope:
    """Freeze producer output before any database write.

    Callers provide the already mapped rows and artifact descriptors.  The
    resulting envelope is replayable without invoking providers or remapping
    current metric definitions.
    """
    if provider_error is not None and not provider_error.strip():
        provider_error = None
    benchmark_value = str(getattr(benchmark, "value", benchmark)).upper()
    legacy_rows = [
        Result.model_validate(row).model_dump(mode="json", exclude={"id"}) for row in results
    ]
    frozen_artifacts: list[FrozenArtifact] = []
    frozen_bytes: dict[str, bytes] = {}
    if transcript is not None:
        artifact, payload = _freeze_artifact(
            name="transcript", descriptor=prepare_provider_transcript(transcript)
        )
        frozen_artifacts.append(artifact)
        frozen_bytes[artifact.name] = payload
    if timing_events:
        artifact, payload = _freeze_artifact(
            name="timing", descriptor=prepare_timing_events(dict(timing_events))
        )
        frozen_artifacts.append(artifact)
        frozen_bytes[artifact.name] = payload
    if audio_snapshot is not None:
        audio_payload, duration_ms = audio_snapshot
        artifact, payload = _freeze_artifact(
            name="audio",
            descriptor=(
                ObservationArtifactType.GENERATED_AUDIO,
                audio_payload,
                "wav",
                "audio/wav",
                "GeneratedAudio",
            ),
            duration_ms=duration_ms,
        )
        frozen_artifacts.append(artifact)
        frozen_bytes[artifact.name] = payload
    grouped: dict[str, list[Any]] = {}
    for row in results:
        metric = str(row.metric_type)
        if metric in (str(Metric.TTFA_ROUNDTRIP), str(Metric.TTFA_LEADING_SILENCE)):
            metric = str(Metric.TTFA)
        grouped.setdefault(metric, []).append(row)
    available_artifacts = {artifact.name for artifact in frozen_artifacts}
    evaluations: list[FrozenEvaluation] = []
    for metric, rows in grouped.items():
        primary = next(
            (
                r
                for r in rows
                if getattr(r, "metric_value", None) is not None
                and str(getattr(r, "status", "")) == "success"
            ),
            None,
        )
        if primary is None and any(str(row.status) == "success" for row in rows):
            continue
        evaluations.append(
            FrozenEvaluation(
                metric_type=metric,
                metric_version="v1",
                evaluation_variant="default",
                executor=str(executor),
                status="succeeded" if primary is not None else "failed",
                error=None
                if primary is not None
                else next(
                    (row.error for row in rows if row.error),
                    "legacy metric produced no value",
                ),
                started_at=captured_at,
                finished_at=captured_at,
                values=_frozen_values(metric, rows, primary) if primary is not None else [],
                inputs=_frozen_inputs(metric, benchmark_value, available_artifacts),
            )
        )
    capture = FrozenCapture(
        run_id=run_id,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        sample_id=sample_id,
        benchmark=benchmark_value,
        provider=str(entry.provider),
        model=str(entry.model),
        voice=voice,
        captured_at=captured_at,
        observation_status="failed" if provider_error else "succeeded",
        observation_error=provider_error,
        source_kind=str(
            {
                "STT": ObservationSourceKind.DATASET_AUDIO,
                "TTS": ObservationSourceKind.GENERATED_AUDIO,
                "S2S": ObservationSourceKind.CONVERSATION_AUDIO,
                "LLM": ObservationSourceKind.CONVERSATION_TEXT,
            }[benchmark_value]
        ),
        provider_extras=dict(provider_extras) if provider_extras is not None else None,
        legacy_rows=legacy_rows,
        evaluations=evaluations,
        artifacts=frozen_artifacts,
    )
    identity = build_capture_identity(
        run_id=run_id,
        benchmark=capture.benchmark,
        dataset_id=dataset_id,
        sample_id=sample_id,
        provider=capture.provider,
        model=capture.model,
        voice=voice,
        capture_id=capture_id,
    )
    return CaptureEnvelope.freeze(
        identity, capture.model_dump(mode="json"), artifact_bytes=frozen_bytes
    )


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


class CaptureOutcome(StrEnum):
    COMPLETED = "completed"
    PENDING = "pending"
    CONFLICT = "conflict"
    UNACKNOWLEDGED = "unacknowledged"


def _upload_frozen_artifact(
    storage_client: Any,
    bucket: str,
    artifact: FrozenArtifact,
    encoded_payload: str,
) -> ObservationArtifact:
    raw = base64.b64decode(encoded_payload, validate=True)
    if (
        len(raw) != artifact.size_bytes
        or hashlib.sha256(raw).hexdigest() != artifact.content_sha256
    ):
        raise ValueError(f"frozen artifact {artifact.name!r} failed validation")
    for attempt in range(3):
        try:
            return upload_prepared_observation_artifact(
                storage_client,
                bucket,
                ObservationArtifactType(artifact.artifact_type),
                raw,
                extension=artifact.extension,
                content_type=artifact.content_type,
                schema_name=artifact.schema_name,
                schema_version=artifact.schema_version,
                duration_ms=artifact.duration_ms,
            )
        except (GoogleAPIError, OSError, TimeoutError):
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


async def persist_capture(
    *, writer: Any, storage_client: Any, bucket: str, envelope: CaptureEnvelope
) -> CaptureOutcome:
    """Persist a previously frozen capture without providers or remapping."""
    durable = False
    try:
        await asyncio.to_thread(upload_envelope, storage_client, bucket, envelope)
        durable = True
        await asyncio.to_thread(upload_claim, storage_client, bucket, envelope)
        payload = FrozenCapture.model_validate(envelope.payload)
        results = [Result.model_validate(row) for row in payload.legacy_rows]
        allocation = await asyncio.to_thread(
            read_legacy_result_allocation, storage_client, bucket, envelope
        )
        if allocation is None:
            reserved_ids = await writer.reserve_result_ids(len(results))
            allocation = await asyncio.to_thread(
                upload_legacy_result_allocation,
                storage_client,
                bucket,
                envelope,
                reserved_ids,
            )
        uploaded: dict[str, ObservationArtifact] = {}
        for artifact in payload.artifacts:
            encoded = envelope.artifact_bytes.get(artifact.name)
            if encoded is None:
                raise ValueError(f"frozen artifact {artifact.name!r} has no payload")
            uploaded[artifact.name] = await asyncio.to_thread(
                _upload_frozen_artifact,
                storage_client,
                bucket,
                artifact,
                encoded,
            )
        await writer.record_results_exact(
            results,
            created_at=payload.captured_at,
            capture_identity=envelope.envelope_digest(),
            result_ids=allocation.result_ids,
        )
        observation = await writer.insert_observation(
            Observation(
                run_id=payload.run_id,
                dataset_id=payload.dataset_id,
                dataset_sha256=payload.dataset_sha256,
                sample_id=payload.sample_id,
                provider=payload.provider,
                model=payload.model,
                voice=payload.voice,
                benchmark=payload.benchmark,
                source_kind=ObservationSourceKind(payload.source_kind),
                transport_protocol=payload.transport_protocol,
                submit_to_headers_ms=payload.submit_to_headers_ms,
                provider_extras=payload.provider_extras,
                captured_at=payload.captured_at,
                status=ObservationStatus.FAILED
                if payload.observation_error
                else ObservationStatus.SUCCEEDED,
                error=payload.observation_error,
                failure_origin=ObservationFailureOrigin.PROVIDER
                if payload.observation_error
                else None,
                artifacts=list(uploaded.values()),
            )
        )
        artifact_ids = {artifact.artifact_type: artifact.id for artifact in observation.artifacts}
        artifact_specs = {artifact.name: artifact for artifact in payload.artifacts}
        for frozen in payload.evaluations:
            inputs: list[MetricEvaluationInput] = []
            for frozen_input in frozen.inputs:
                artifact_spec = artifact_specs.get(frozen_input.artifact_name)
                if artifact_spec is None:
                    raise ValueError(
                        f"unknown frozen artifact input {frozen_input.artifact_name!r}"
                    )
                artifact_id = artifact_ids.get(ObservationArtifactType(artifact_spec.artifact_type))
                if artifact_id is None:
                    raise ValueError(
                        f"frozen artifact input {frozen_input.artifact_name!r} was not stored"
                    )
                inputs.append(
                    MetricEvaluationInput(
                        observation_artifact_id=artifact_id,
                        input_role=frozen_input.input_role,
                        input_order=frozen_input.input_order,
                    )
                )
            evaluation = await writer.insert_metric_evaluation(
                MetricEvaluation(
                    observation_id=observation.id,
                    metric_type=frozen.metric_type,
                    metric_version=frozen.metric_version,
                    evaluation_variant=frozen.evaluation_variant or "default",
                    executor=MetricExecutor(frozen.executor),
                    external_request_id=frozen.external_request_id,
                    status=ProcessingStatus.QUEUED,
                ),
                inputs=inputs,
                validate_contract=False,
            )
            if frozen.started_at is None or frozen.finished_at is None:
                raise ValueError("frozen terminal evaluation requires timestamps")
            await writer.start_metric_evaluation_exact(evaluation.id, started_at=frozen.started_at)
            if frozen.status == str(ProcessingStatus.FAILED):
                await writer.fail_metric_evaluation_exact(
                    evaluation.id,
                    started_at=frozen.started_at,
                    finished_at=frozen.finished_at,
                    error=frozen.error or "frozen evaluation failed",
                )
            else:
                values = [
                    MetricValue(metric_evaluation_id=evaluation.id, **value.model_dump())
                    for value in frozen.values
                ]
                await writer.complete_metric_evaluation(
                    evaluation.id,
                    values=values,
                    finished_at=frozen.finished_at,
                    validate_contract=False,
                )
        await asyncio.to_thread(upload_receipt, storage_client, bucket, envelope)
        return CaptureOutcome.COMPLETED
    except ValueError as exc:
        logger.warning(
            "normalized_capture_conflict",
            run_id=envelope.identity.run_id,
            capture_id=envelope.identity.capture_id,
            exception_type=type(exc).__name__,
            exc_info=True,
        )
        return CaptureOutcome.CONFLICT
    except asyncio.CancelledError:
        raise
    except (GoogleAPIError, OSError, TimeoutError, psycopg.Error, PoolTimeout) as exc:
        outcome = CaptureOutcome.PENDING if durable else CaptureOutcome.UNACKNOWLEDGED
        logger.warning(
            "normalized_capture_transport_failed",
            run_id=envelope.identity.run_id,
            capture_id=envelope.identity.capture_id,
            outcome=str(outcome),
            exception_type=type(exc).__name__,
            exc_info=True,
        )
        return outcome


async def replay_capture(**kwargs: Any) -> CaptureOutcome:
    """Alias used by recovery workers; replay is identical to persistence."""
    return await persist_capture(**kwargs)


def _values(
    metric: str, rows: Sequence[Any], evaluation_id: Any, primary: Any
) -> list[MetricValue]:
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
        counts = (
            ("substitution_count", primary.wer_substitutions),
            ("deletion_count", primary.wer_deletions),
            ("insertion_count", primary.wer_insertions),
            ("reference_words", primary.wer_reference_words),
        )
        if all(count is not None for _, count in counts):
            values.extend(
                MetricValue(
                    metric_evaluation_id=evaluation_id, value_key=key, unit="count", value=count
                )
                for key, count in counts
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
    captured_at: datetime | None = None,
    transcript: str | None = None,
    timing_events: dict[str, Any] | None = None,
    audio_path: Any = None,
    voice: str | None = None,
    executor: MetricExecutor = MetricExecutor.INLINE,
    db_retry_attempts: int = 1,
) -> None:
    """Persist one observation and its grouped normalized evaluations."""
    if provider_error is not None and not provider_error.strip():
        provider_error = None
    captured_at = captured_at or datetime.now(UTC)
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
                upload_generated_audio, storage_client, bucket, audio_payload, audio_duration_ms
            )
        )
    source_kind = {
        "STT": ObservationSourceKind.DATASET_AUDIO,
        "TTS": ObservationSourceKind.GENERATED_AUDIO,
        "S2S": ObservationSourceKind.CONVERSATION_AUDIO,
        "LLM": ObservationSourceKind.CONVERSATION_TEXT,
    }[benchmark.value.upper()]

    async def persist_db() -> None:
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
                captured_at=captured_at,
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
            primary = next(
                (
                    row
                    for row in rows
                    if row.metric_value is not None and str(row.status) == "success"
                ),
                None,
            )
            if primary is None and any(str(row.status) == "success" for row in rows):
                continue
            evaluation = await writer.insert_metric_evaluation(
                MetricEvaluation(
                    observation_id=observation.id,
                    metric_type=metric,
                    metric_version="v1",
                    executor=executor,
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
                    evaluation.id,
                    finished_at=finished_at,
                    values=_values(metric, rows, evaluation.id, primary),
                )

    if db_retry_attempts <= 1:
        await persist_db()
        return
    from coval_bench.runner.retry import with_retry

    retry_on = (PoolTimeout, psycopg.OperationalError)
    await with_retry(
        persist_db,
        max_attempts=db_retry_attempts,
        retry_on=retry_on,
        retry_event="normalized_persistence_retry",
        exhaustion_event="normalized_persistence_exhausted",
        retry_state=writer.pool_diagnostics,
    )
