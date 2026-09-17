# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import psycopg
import pytest
from click.testing import CliRunner
from google.api_core.exceptions import NotFound, PreconditionFailed, ServiceUnavailable
from google.cloud import storage

from coval_bench.db.models import (
    Benchmark,
    MetricEvaluation,
    MetricEvaluationInput,
    MetricValue,
    Observation,
    ProcessingStatus,
    Result,
    ResultStatus,
)
from coval_bench.db.writer import RunWriter
from coval_bench.registries import Metric
from coval_bench.runner import capture_cli
from coval_bench.runner.capture import (
    CaptureEnvelope,
    ImportRunClaim,
    ImportRunIdentity,
    RunManifest,
    identity_digest,
    list_envelope_uris,
    payload_digest,
    read_receipt,
    upload_envelope,
    upload_import_run_claim,
    upload_run_state,
)
from coval_bench.runner.normalized import (
    CaptureOutcome,
    persist_capture,
    prepare_capture_envelope,
)


class _BlobRecord:
    def __init__(self) -> None:
        self.payload = b""
        self.metadata: dict[str, str] | None = None
        self.content_type: str | None = None


class _Blob:
    def __init__(self, client: _Storage, name: str) -> None:
        self._client = client
        self.name = name
        self.metadata: dict[str, str] | None = None

    @property
    def _record(self) -> _BlobRecord:
        try:
            return self._client.objects[self.name]
        except KeyError as exc:
            raise cast(type[Exception], NotFound)("missing") from exc

    @property
    def size(self) -> int | None:
        return len(self._record.payload)

    @property
    def content_type(self) -> str | None:
        return self._record.content_type

    def upload_from_string(
        self, payload: bytes, *, content_type: str, if_generation_match: int
    ) -> None:
        assert if_generation_match == 0
        if self._client.fail_uploads:
            self._client.fail_uploads -= 1
            raise cast(type[Exception], ServiceUnavailable)("temporary")
        if self.name in self._client.objects:
            raise cast(type[Exception], PreconditionFailed)("exists")
        record = _BlobRecord()
        record.payload = payload
        record.metadata = self.metadata
        record.content_type = content_type
        self._client.objects[self.name] = record

    def reload(self) -> None:
        self.metadata = self._record.metadata

    def download_as_bytes(self) -> bytes:
        return self._record.payload


class _Bucket:
    def __init__(self, client: _Storage) -> None:
        self._client = client

    def blob(self, key: str) -> _Blob:
        return _Blob(self._client, key)


class _Storage:
    def __init__(self, *, fail_uploads: int = 0) -> None:
        self.objects: dict[str, _BlobRecord] = {}
        self.fail_uploads = fail_uploads

    def bucket(self, _name: str) -> _Bucket:
        return _Bucket(self)

    def list_blobs(
        self,
        _bucket: str,
        *,
        prefix: str,
        max_results: int | None = None,
        start_offset: str | None = None,
    ) -> list[_Blob]:
        names = [name for name in sorted(self.objects) if name.startswith(prefix)]
        if start_offset is not None:
            names = [name for name in names if name >= start_offset]
        if max_results is not None:
            names = names[:max_results]
        return [_Blob(self, name) for name in names]


class _Writer:
    def __init__(self) -> None:
        self.legacy: dict[int, Result] = {}
        self.observation: Observation | None = None
        self.evaluations: dict[str, MetricEvaluation] = {}
        self.values: dict[str, list[MetricValue]] = {}
        self.fail_after_legacy_once = False
        self.cancel_after_legacy_once = False
        self.next_result_id = 1

    async def reserve_result_ids(self, count: int) -> list[int]:
        result = list(range(self.next_result_id, self.next_result_id + count))
        self.next_result_id += count
        return result

    async def record_results_exact(
        self,
        results: Sequence[Result],
        *,
        created_at: datetime,
        capture_identity: str,
        result_ids: Sequence[int],
    ) -> None:
        del created_at, capture_identity
        results = [
            row.model_copy(update={"id": result_ids[index]}) for index, row in enumerate(results)
        ]
        for row in results:
            assert row.id is not None
            existing = self.legacy.get(row.id)
            if existing is not None and existing.model_dump() != row.model_dump():
                raise ValueError("legacy conflict")
            self.legacy[row.id] = row
        if self.cancel_after_legacy_once:
            self.cancel_after_legacy_once = False
            raise asyncio.CancelledError
        if self.fail_after_legacy_once:
            self.fail_after_legacy_once = False
            raise psycopg.OperationalError("ambiguous commit")

    async def insert_observation(self, observation: Observation) -> Observation:
        if self.observation is None:
            observation_id = uuid4()
            artifacts = [
                item.model_copy(update={"id": uuid4(), "observation_id": observation_id})
                for item in observation.artifacts
            ]
            self.observation = observation.model_copy(
                update={"id": observation_id, "artifacts": artifacts}
            )
        return self.observation

    async def insert_metric_evaluation(
        self,
        evaluation: MetricEvaluation,
        *,
        inputs: Sequence[MetricEvaluationInput] = (),
        validate_contract: bool = True,
    ) -> MetricEvaluation:
        del inputs, validate_contract
        current = self.evaluations.get(evaluation.metric_type)
        if current is None:
            current = evaluation.model_copy(update={"id": uuid4()})
            self.evaluations[evaluation.metric_type] = current
        return current

    async def start_metric_evaluation_exact(
        self, evaluation_id: UUID, *, started_at: datetime
    ) -> MetricEvaluation:
        metric, current = next(
            (metric, evaluation)
            for metric, evaluation in self.evaluations.items()
            if evaluation.id == evaluation_id
        )
        if current.status is ProcessingStatus.QUEUED:
            current = current.model_copy(
                update={"status": ProcessingStatus.RUNNING, "started_at": started_at}
            )
            self.evaluations[metric] = current
        return current

    async def complete_metric_evaluation(
        self,
        evaluation_id: UUID,
        *,
        finished_at: datetime,
        values: Sequence[MetricValue],
        validate_contract: bool = True,
    ) -> None:
        del validate_contract
        metric, current = next(
            (metric, evaluation)
            for metric, evaluation in self.evaluations.items()
            if evaluation.id == evaluation_id
        )
        existing = self.values.get(metric)
        if existing is not None and [v.model_dump() for v in existing] != [
            v.model_dump() for v in values
        ]:
            raise ValueError("value conflict")
        self.values[metric] = list(values)
        self.evaluations[metric] = current.model_copy(
            update={"status": ProcessingStatus.SUCCEEDED, "finished_at": finished_at}
        )

    async def fail_metric_evaluation_exact(
        self,
        evaluation_id: UUID,
        *,
        started_at: datetime,
        finished_at: datetime,
        error: str,
    ) -> MetricEvaluation:
        metric, current = next(
            (metric, evaluation)
            for metric, evaluation in self.evaluations.items()
            if evaluation.id == evaluation_id
        )
        current = current.model_copy(
            update={
                "status": ProcessingStatus.FAILED,
                "started_at": started_at,
                "finished_at": finished_at,
                "error": error,
            }
        )
        self.evaluations[metric] = current
        return current


def _client(value: _Storage) -> storage.Client:
    return cast(storage.Client, value)


def _envelope(
    benchmark: Benchmark = Benchmark.STT,
    *,
    value: float | None = 10.0,
    error: str | None = None,
    sample: str = "sample",
) -> CaptureEnvelope:
    status = ResultStatus.FAILED if error else ResultStatus.SUCCESS
    result = Result(
        run_id=1,
        provider="provider",
        model="model",
        benchmark=benchmark,
        metric_type=Metric.WER,
        metric_value=value,
        metric_units="percent" if value is not None else None,
        audio_filename=f"{sample}.wav",
        status=status,
        error=error,
    )
    return prepare_capture_envelope(
        run_id=1,
        dataset_id="dataset",
        dataset_sha256="a" * 64,
        sample_id=sample,
        entry=SimpleNamespace(provider="provider", model="model"),
        benchmark=benchmark,
        results=[result],
        provider_error=error,
        captured_at=datetime(2026, 9, 17, tzinfo=UTC),
        transcript="secret" if benchmark is Benchmark.STT else None,
        timing_events={"latency_ms": 10},
        audio_snapshot=(b"RIFFaudio", 10.0) if benchmark is Benchmark.TTS else None,
    )


@pytest.mark.asyncio
async def test_ambiguous_database_commit_is_discoverable_and_replays_once() -> None:
    storage_value = _Storage()
    writer = _Writer()
    writer.fail_after_legacy_once = True
    envelope = _envelope()

    assert (
        await persist_capture(
            writer=writer,
            storage_client=_client(storage_value),
            bucket="private",
            envelope=envelope,
        )
        is CaptureOutcome.PENDING
    )
    uris, cursor = list_envelope_uris(_client(storage_value), "private", run_id=1)
    assert len(uris) == 1
    assert cursor is None
    assert read_receipt(_client(storage_value), "private", envelope) is None

    assert (
        await persist_capture(
            writer=writer,
            storage_client=_client(storage_value),
            bucket="private",
            envelope=envelope,
        )
        is CaptureOutcome.COMPLETED
    )
    assert (
        await persist_capture(
            writer=writer,
            storage_client=_client(storage_value),
            bucket="private",
            envelope=envelope,
        )
        is CaptureOutcome.COMPLETED
    )
    assert len(writer.legacy) == 1
    assert len(writer.evaluations) == 1
    assert len(writer.values[str(Metric.WER)]) == 1
    assert read_receipt(_client(storage_value), "private", envelope) is not None


@pytest.mark.asyncio
async def test_distinct_same_timestamp_captures_get_distinct_legacy_ids() -> None:
    storage_value = _Storage()
    writer = _Writer()

    first, second = _envelope(sample="one"), _envelope(sample="two")
    assert first.payload["captured_at"] == second.payload["captured_at"]
    assert (
        await persist_capture(
            writer=writer,
            storage_client=_client(storage_value),
            bucket="private",
            envelope=first,
        )
        is CaptureOutcome.COMPLETED
    )
    assert (
        await persist_capture(
            writer=writer,
            storage_client=_client(storage_value),
            bucket="private",
            envelope=second,
        )
        is CaptureOutcome.COMPLETED
    )

    assert set(writer.legacy) == {1, 2}


@pytest.mark.asyncio
async def test_envelope_upload_retries_and_exhaustion_has_no_false_ack() -> None:
    recovered = _Storage(fail_uploads=2)
    envelope = _envelope()
    assert (
        await persist_capture(
            writer=_Writer(), storage_client=_client(recovered), bucket="private", envelope=envelope
        )
        is CaptureOutcome.COMPLETED
    )

    exhausted = _Storage(fail_uploads=3)
    assert (
        await persist_capture(
            writer=_Writer(), storage_client=_client(exhausted), bucket="private", envelope=envelope
        )
        is CaptureOutcome.UNACKNOWLEDGED
    )
    assert not exhausted.objects


@pytest.mark.asyncio
async def test_conflicting_identity_is_rejected_and_both_envelopes_remain_visible() -> None:
    storage_value = _Storage()
    first = _envelope(value=10)
    conflict = _envelope(value=20)
    assert (
        await persist_capture(
            writer=_Writer(),
            storage_client=_client(storage_value),
            bucket="private",
            envelope=first,
        )
        is CaptureOutcome.COMPLETED
    )
    assert (
        await persist_capture(
            writer=_Writer(),
            storage_client=_client(storage_value),
            bucket="private",
            envelope=conflict,
        )
        is CaptureOutcome.CONFLICT
    )
    uris, _ = list_envelope_uris(_client(storage_value), "private", run_id=1)
    assert len(uris) == 2


@pytest.mark.asyncio
async def test_cancellation_propagates_after_durable_capture() -> None:
    storage_value = _Storage()
    writer = _Writer()
    writer.cancel_after_legacy_once = True
    envelope = _envelope()
    with pytest.raises(asyncio.CancelledError):
        await persist_capture(
            writer=writer,
            storage_client=_client(storage_value),
            bucket="private",
            envelope=envelope,
        )
    uris, _ = list_envelope_uris(_client(storage_value), "private", run_id=1)
    assert len(uris) == 1
    assert read_receipt(_client(storage_value), "private", envelope) is None


@pytest.mark.parametrize("benchmark", list(Benchmark))
def test_all_benchmark_kinds_freeze_without_provider_replay(benchmark: Benchmark) -> None:
    envelope = _envelope(benchmark)
    assert envelope.identity.benchmark == str(benchmark)
    assert envelope.payload["captured_at"] == "2026-09-17T00:00:00Z"


@pytest.mark.asyncio
async def test_provider_failure_is_a_completed_failed_observation() -> None:
    storage_value = _Storage()
    writer = _Writer()
    envelope = _envelope(value=None, error="provider unavailable")
    assert (
        await persist_capture(
            writer=writer,
            storage_client=_client(storage_value),
            bucket="private",
            envelope=envelope,
        )
        is CaptureOutcome.COMPLETED
    )
    assert writer.observation is not None
    assert str(writer.observation.status) == "failed"
    assert writer.observation.error == "provider unavailable"


def test_capture_status_is_bounded_and_does_not_print_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_value = _Storage()
    envelope = _envelope()
    upload_envelope(_client(storage_value), "private", envelope)
    monkeypatch.setattr(
        capture_cli,
        "_storage",
        lambda _settings: (_client(storage_value), "private"),
    )

    result = CliRunner().invoke(
        capture_cli.capture_status,
        ["--run-id", "1", "--limit", "1"],
    )

    assert result.exit_code == 0
    assert '"pending": 1' in result.output
    assert envelope.envelope_digest() in result.output
    assert "secret" not in result.output


def test_capture_status_reports_expected_identity_without_an_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_value = _Storage()
    envelope = _envelope()
    expected = identity_digest(envelope.identity)
    upload_run_state(
        _client(storage_value),
        "private",
        1,
        "manifest",
        RunManifest(
            run_id=1,
            scheduled_at=datetime(2026, 9, 17, tzinfo=UTC),
            benchmark_kind="stt",
            source="runner",
            datasets={"dataset": "a" * 64},
            expected_capture_ids=[expected],
        ),
    )
    monkeypatch.setattr(
        capture_cli,
        "_storage",
        lambda _settings: (_client(storage_value), "private"),
    )

    result = CliRunner().invoke(capture_cli.capture_status, ["--run-id", "1"])

    assert result.exit_code == 0
    assert '"missing_expected": 1' in result.output
    assert f'"missing_expected_capture_ids": ["{expected}"]' in result.output


def test_envelope_rejects_payload_bound_to_a_different_identity() -> None:
    envelope = _envelope()
    value = envelope.model_dump(mode="json")
    value["payload"]["sample_id"] = "different"
    value["payload_sha256"] = payload_digest(value["payload"])

    with pytest.raises(ValueError, match="identity conflicts"):
        CaptureEnvelope.model_validate(value)


def test_concurrent_import_claim_adopts_the_first_run_allocation() -> None:
    storage_value = _Storage()
    identity = ImportRunIdentity(
        external_run_id="external-run",
        workspace_id="workspace",
        benchmark="S2S",
        provider="provider",
        model="model",
        dataset_id="dataset",
        dataset_sha256="a" * 64,
    )
    now = datetime(2026, 9, 17, tzinfo=UTC)
    first = ImportRunClaim(
        identity=identity,
        generation=0,
        run_id=10,
        started_at=now,
        scheduled_at=now,
        captured_at=now,
        metric_types=["V2V"],
    )
    contender = first.model_copy(update={"run_id": 11})

    assert upload_import_run_claim(_client(storage_value), "private", first) == first
    assert upload_import_run_claim(_client(storage_value), "private", contender) == first


@pytest.mark.asyncio
async def test_required_database_preflight_rejects_missing_write_privilege() -> None:
    pool = MagicMock()
    schema_cursor = MagicMock()
    schema_cursor.execute = AsyncMock()
    schema_cursor.fetchall = AsyncMock(return_value=[])
    privilege_cursor = MagicMock()
    privilege_cursor.execute = AsyncMock()
    privilege_cursor.fetchall = AsyncMock(return_value=[("results",)])

    def connection(cursor: MagicMock) -> MagicMock:
        cursor_context = MagicMock()
        cursor_context.__aenter__ = AsyncMock(return_value=cursor)
        cursor_context.__aexit__ = AsyncMock(return_value=None)
        conn = MagicMock()
        conn.cursor.return_value = cursor_context
        conn_context = MagicMock()
        conn_context.__aenter__ = AsyncMock(return_value=conn)
        conn_context.__aexit__ = AsyncMock(return_value=None)
        return conn_context

    pool.connection.side_effect = [connection(schema_cursor), connection(privilege_cursor)]
    writer = RunWriter(pool)

    with pytest.raises(RuntimeError, match="privileges unavailable: results"):
        await writer.preflight_required_capture_schema()
