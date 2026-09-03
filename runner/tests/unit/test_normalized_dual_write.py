# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import wave
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg_pool import PoolTimeout

from coval_bench.db.models import (
    Benchmark,
    MetricArtifact,
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
    ResultStatus,
)
from coval_bench.registries import Metric
from coval_bench.registries.metrics import validate_metric_values
from coval_bench.runner import normalized


class _Writer:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.evaluations: dict[UUID, MetricEvaluation] = {}
        self.inputs: dict[UUID, list[MetricEvaluationInput]] = {}
        self.completed: dict[UUID, list[MetricValue]] = {}
        self.failed: dict[UUID, str] = {}

    async def insert_observation(self, observation: Observation) -> Observation:
        observation_id = uuid4()
        artifacts = [
            artifact.model_copy(update={"id": uuid4(), "observation_id": observation_id})
            for artifact in observation.artifacts
        ]
        stored = observation.model_copy(update={"id": observation_id, "artifacts": artifacts})
        self.observations.append(stored)
        return stored

    async def insert_metric_evaluation(
        self,
        evaluation: MetricEvaluation,
        *,
        inputs: Sequence[MetricEvaluationInput] = (),
    ) -> MetricEvaluation:
        evaluation_id = uuid4()
        stored = evaluation.model_copy(update={"id": evaluation_id})
        self.evaluations[evaluation_id] = stored
        self.inputs[evaluation_id] = list(inputs)
        return stored

    async def start_metric_evaluation(
        self, evaluation_id: UUID, *, started_at: datetime
    ) -> MetricEvaluation:
        stored = self.evaluations[evaluation_id].model_copy(
            update={"status": ProcessingStatus.RUNNING, "started_at": started_at}
        )
        self.evaluations[evaluation_id] = stored
        return stored

    async def complete_metric_evaluation(
        self,
        evaluation_id: UUID,
        *,
        finished_at: datetime,
        values: Sequence[MetricValue],
        artifacts: Sequence[MetricArtifact] = (),
    ) -> None:
        del artifacts
        validate_metric_values(
            self.evaluations[evaluation_id].metric_type,
            "v1",
            tuple((value.value_key, value.unit, value.value, value.value_role) for value in values),
        )
        self.completed[evaluation_id] = list(values)
        stored = self.evaluations[evaluation_id].model_copy(
            update={
                "status": ProcessingStatus.SUCCEEDED,
                "finished_at": finished_at,
            }
        )
        self.evaluations[evaluation_id] = stored

    async def fail_metric_evaluation(
        self, evaluation_id: UUID, *, finished_at: datetime, error: str
    ) -> MetricEvaluation:
        self.failed[evaluation_id] = error
        stored = self.evaluations[evaluation_id].model_copy(
            update={
                "status": ProcessingStatus.FAILED,
                "finished_at": finished_at,
                "error": error,
            }
        )
        self.evaluations[evaluation_id] = stored
        return stored


def _artifact(artifact_type: ObservationArtifactType) -> ObservationArtifact:
    return ObservationArtifact(
        artifact_type=artifact_type,
        schema_name="TestArtifact",
        schema_version="v1",
        gcs_uri=f"gs://private/{artifact_type}.bin",
        content_sha256="a" * 64,
        size_bytes=1,
        duration_ms=1 if artifact_type is ObservationArtifactType.GENERATED_AUDIO else None,
    )


@pytest.fixture(autouse=True)
def _fake_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        normalized,
        "upload_provider_transcript",
        lambda *_args: _artifact(ObservationArtifactType.PROVIDER_TRANSCRIPT),
    )
    monkeypatch.setattr(
        normalized,
        "upload_timing_events",
        lambda *_args: _artifact(ObservationArtifactType.TIMING_EVENTS),
    )
    monkeypatch.setattr(
        normalized,
        "upload_generated_audio",
        lambda *_args: _artifact(ObservationArtifactType.GENERATED_AUDIO),
    )


def _result(
    benchmark: Benchmark,
    metric: Metric,
    value: float | None,
    unit: str | None,
    *,
    status: ResultStatus = ResultStatus.SUCCESS,
    error: str | None = None,
    **components: float,
) -> Result:
    return Result(
        run_id=1,
        provider="provider",
        model="model",
        benchmark=benchmark,
        metric_type=metric,
        metric_value=value,
        metric_units=unit,
        status=status,
        error=error,
        **components,
    )


def _evaluation_by_metric(writer: _Writer, metric: Metric) -> tuple[UUID, MetricEvaluation]:
    return next(
        (evaluation_id, evaluation)
        for evaluation_id, evaluation in writer.evaluations.items()
        if evaluation.metric_type == metric
    )


def _artifact_ids(observation: Observation) -> dict[ObservationArtifactType, UUID]:
    return {
        artifact.artifact_type: artifact.id
        for artifact in observation.artifacts
        if artifact.id is not None
    }


@pytest.mark.asyncio
async def test_stt_groups_wer_and_freezes_transcript_and_timing_lineage() -> None:
    writer = _Writer()
    captured_at = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
    results = [
        _result(Benchmark.STT, Metric.TTFT, 0.25, "seconds"),
        _result(
            Benchmark.STT,
            Metric.WER,
            10,
            "percent",
            wer_insertions_pct=2,
            wer_deletions_pct=3,
            wer_substitutions_pct=5,
        ),
    ]

    await normalized.dual_write(
        writer=writer,
        storage_client=object(),
        bucket="private",
        run_id=1,
        dataset_id="stt-v1",
        dataset_sha256="a" * 64,
        sample_id="sample-1",
        entry=SimpleNamespace(provider="provider", model="model"),
        benchmark=Benchmark.STT,
        results=results,
        provider_error=None,
        captured_at=captured_at,
        transcript="hello",
        timing_events={"ttft_seconds": 0.25},
    )

    assert len(writer.observations) == 1
    observation = writer.observations[0]
    assert observation.source_kind is ObservationSourceKind.DATASET_AUDIO
    assert observation.captured_at == captured_at
    artifacts = _artifact_ids(observation)

    wer_id, _ = _evaluation_by_metric(writer, Metric.WER)
    ttft_id, _ = _evaluation_by_metric(writer, Metric.TTFT)
    assert [value.value_key for value in writer.completed[wer_id]] == [
        "primary",
        "insertions",
        "deletions",
        "substitutions",
    ]
    assert writer.inputs[wer_id] == [
        MetricEvaluationInput(
            observation_artifact_id=artifacts[ObservationArtifactType.PROVIDER_TRANSCRIPT],
            input_role="raw",
            input_order=0,
        )
    ]
    assert writer.inputs[ttft_id] == [
        MetricEvaluationInput(
            observation_artifact_id=artifacts[ObservationArtifactType.TIMING_EVENTS],
            input_role="timing",
            input_order=0,
        )
    ]


@pytest.mark.asyncio
async def test_tts_groups_ttfa_components_and_audio_lineage(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setparams((1, 2, 16_000, 160, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 160)

    writer = _Writer()
    results = [
        _result(Benchmark.TTS, Metric.TTFA, 120, "milliseconds"),
        _result(Benchmark.TTS, Metric.TTFA_ROUNDTRIP, 75, "milliseconds"),
        _result(Benchmark.TTS, Metric.TTFA_LEADING_SILENCE, 45, "milliseconds"),
        _result(
            Benchmark.TTS,
            Metric.WER,
            10,
            "percent",
            wer_insertions_pct=2,
            wer_deletions_pct=3,
            wer_substitutions_pct=5,
        ),
    ]

    await normalized.dual_write(
        writer=writer,
        storage_client=object(),
        bucket="private",
        run_id=1,
        dataset_id="tts-v1",
        dataset_sha256="b" * 64,
        sample_id="tts-1",
        entry=SimpleNamespace(provider="provider", model="model"),
        benchmark=Benchmark.TTS,
        results=results,
        provider_error=None,
        timing_events={"ttfa_ms": 120},
        audio_path=audio_path,
        voice="voice",
    )

    observation = writer.observations[0]
    assert observation.source_kind is ObservationSourceKind.GENERATED_AUDIO
    artifacts = _artifact_ids(observation)
    ttfa_id, _ = _evaluation_by_metric(writer, Metric.TTFA)
    wer_id, _ = _evaluation_by_metric(writer, Metric.WER)
    assert {evaluation.metric_type for evaluation in writer.evaluations.values()} == {
        Metric.TTFA,
        Metric.WER,
    }
    assert [value.value_key for value in writer.completed[ttfa_id]] == [
        "primary",
        "roundtrip",
        "leading_silence",
    ]
    assert writer.inputs[ttfa_id] == [
        MetricEvaluationInput(
            observation_artifact_id=artifacts[ObservationArtifactType.TIMING_EVENTS],
            input_role="timing",
            input_order=0,
        ),
        MetricEvaluationInput(
            observation_artifact_id=artifacts[ObservationArtifactType.GENERATED_AUDIO],
            input_role="raw",
            input_order=0,
        ),
    ]
    assert writer.inputs[wer_id] == [
        MetricEvaluationInput(
            observation_artifact_id=artifacts[ObservationArtifactType.GENERATED_AUDIO],
            input_role="raw",
            input_order=0,
        )
    ]


@pytest.mark.asyncio
async def test_provider_and_metric_failure_map_to_failed_normalized_rows() -> None:
    writer = _Writer()
    await normalized.dual_write(
        writer=writer,
        storage_client=object(),
        bucket="private",
        run_id=1,
        dataset_id="stt-v1",
        dataset_sha256="c" * 64,
        sample_id="sample-1",
        entry=SimpleNamespace(provider="provider", model="model"),
        benchmark=Benchmark.STT,
        results=[
            _result(
                Benchmark.STT,
                Metric.TTFT,
                None,
                "seconds",
                status=ResultStatus.FAILED,
                error="provider failed",
            )
        ],
        provider_error="provider failed",
        timing_events={"ttft_seconds": None},
    )

    observation = writer.observations[0]
    assert observation.status is ObservationStatus.FAILED
    assert observation.failure_origin is ObservationFailureOrigin.PROVIDER
    assert not writer.completed
    assert list(writer.failed.values()) == ["provider failed"]


@pytest.mark.asyncio
async def test_null_success_metric_is_excluded_instead_of_failed() -> None:
    writer = _Writer()
    await normalized.dual_write(
        writer=writer,
        storage_client=object(),
        bucket="private",
        run_id=1,
        dataset_id="tts-v1",
        dataset_sha256="d" * 64,
        sample_id="tts-1",
        entry=SimpleNamespace(provider="provider", model="model"),
        benchmark=Benchmark.TTS,
        results=[
            _result(
                Benchmark.TTS,
                Metric.TTFA,
                None,
                "milliseconds",
                error="TTFA measured over HTTP/1.1; not comparable",
            )
        ],
        provider_error=None,
        timing_events={"ttfa_ms": 120},
    )

    assert writer.observations[0].status is ObservationStatus.SUCCEEDED
    assert not writer.evaluations
    assert not writer.completed
    assert not writer.failed


@pytest.mark.asyncio
async def test_s2s_uses_conversation_source_and_coval_executor() -> None:
    writer = _Writer()

    await normalized.dual_write(
        writer=writer,
        storage_client=None,
        bucket="",
        run_id=1,
        dataset_id="s2s-multiturn-v1",
        dataset_sha256="e" * 64,
        sample_id="R1/s1",
        entry=SimpleNamespace(provider="provider", model="model"),
        benchmark=Benchmark.S2S,
        results=[_result(Benchmark.S2S, Metric.V2V, 500, "milliseconds")],
        provider_error=None,
        executor=MetricExecutor.COVAL_API,
    )

    observation = writer.observations[0]
    assert observation.source_kind is ObservationSourceKind.CONVERSATION_AUDIO
    evaluation_id, evaluation = _evaluation_by_metric(writer, Metric.V2V)
    assert evaluation.executor is MetricExecutor.COVAL_API
    assert writer.inputs[evaluation_id] == []
    assert [(value.value_key, value.value) for value in writer.completed[evaluation_id]] == [
        ("primary", 500.0)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_error", ["", "   "])
async def test_blank_provider_error_is_normalized(provider_error: str) -> None:
    writer = _Writer()
    await normalized.dual_write(
        writer=writer,
        storage_client=object(),
        bucket="private",
        run_id=1,
        dataset_id="stt-v1",
        dataset_sha256="a" * 64,
        sample_id="sample",
        entry=SimpleNamespace(provider="provider", model="model"),
        benchmark=Benchmark.STT,
        results=[_result(Benchmark.STT, Metric.TTFT, 1, "seconds")],
        provider_error=provider_error,
        timing_events={"ttft_seconds": 1},
    )
    observation = writer.observations[0]
    assert observation.error is None
    assert observation.status is ObservationStatus.SUCCEEDED
    assert observation.failure_origin is None


@pytest.mark.asyncio
async def test_nonblank_provider_error_is_preserved() -> None:
    writer = _Writer()
    error = "  provider failed  "
    await normalized.dual_write(
        writer=writer,
        storage_client=object(),
        bucket="private",
        run_id=1,
        dataset_id="stt-v1",
        dataset_sha256="a" * 64,
        sample_id="sample",
        entry=SimpleNamespace(provider="provider", model="model"),
        benchmark=Benchmark.STT,
        results=[
            _result(
                Benchmark.STT, Metric.TTFT, None, "seconds", status=ResultStatus.FAILED, error=error
            )
        ],
        provider_error=error,
        timing_events={"ttft_seconds": None},
    )
    assert writer.observations[0].error == error
    assert writer.observations[0].status is ObservationStatus.FAILED


class _RetryWriter(_Writer):
    def __init__(self, exc: BaseException | None = None) -> None:
        super().__init__()
        self.calls = 0
        self.exc = exc

    async def insert_observation(self, observation: Observation) -> Observation:
        self.calls += 1
        if self.calls == 1 and self.exc is not None:
            raise self.exc
        return await super().insert_observation(observation)


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [PoolTimeout("busy"), psycopg.OperationalError("gone")])
async def test_normalized_db_retry_does_not_repeat_upload(
    exc: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _RetryWriter(exc)
    uploads = 0

    def upload(*_args: object) -> ObservationArtifact:
        nonlocal uploads
        uploads += 1
        return _artifact(ObservationArtifactType.PROVIDER_TRANSCRIPT)

    monkeypatch.setattr(normalized, "upload_provider_transcript", upload)
    await normalized.dual_write(
        writer=writer,
        storage_client=object(),
        bucket="b",
        run_id=1,
        dataset_id="d",
        dataset_sha256="a" * 64,
        sample_id="s",
        entry=SimpleNamespace(provider="p", model="m"),
        benchmark=Benchmark.STT,
        results=[],
        provider_error=None,
        transcript="x",
        db_semaphore=asyncio.Semaphore(3),
        db_retry_attempts=3,
    )
    assert writer.calls == 2 and uploads == 1


@pytest.mark.asyncio
async def test_normalized_runtime_error_and_cancellation_are_not_retried() -> None:
    for exc in (RuntimeError("bad"), asyncio.CancelledError()):
        writer = _RetryWriter(exc)
        with pytest.raises(type(exc)):
            await normalized.dual_write(
                writer=writer,
                storage_client=object(),
                bucket="b",
                run_id=1,
                dataset_id="d",
                dataset_sha256="a" * 64,
                sample_id="s",
                entry=SimpleNamespace(provider="p", model="m"),
                benchmark=Benchmark.STT,
                results=[],
                provider_error=None,
                db_semaphore=asyncio.Semaphore(3),
                db_retry_attempts=3,
            )
        assert writer.calls == 1


@pytest.mark.asyncio
async def test_normalized_db_semaphore_caps_at_three() -> None:
    active = 0
    maximum = 0

    class Writer(_Writer):
        async def insert_observation(self, observation: Observation) -> Observation:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0)
            result = await super().insert_observation(observation)
            active -= 1
            return result

    sem = asyncio.Semaphore(3)
    writer = Writer()

    async def write(index: int) -> None:
        await normalized.dual_write(
            writer=writer,
            storage_client=object(),
            bucket="b",
            run_id=1,
            dataset_id="d",
            dataset_sha256="a" * 64,
            sample_id=str(index),
            entry=SimpleNamespace(provider="p", model="m"),
            benchmark=Benchmark.STT,
            results=[],
            provider_error=None,
            db_semaphore=sem,
            db_retry_attempts=1,
        )

    await asyncio.gather(*(write(index) for index in range(8)))
    assert maximum == 3
