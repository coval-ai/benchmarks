# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Real-Postgres coverage for normalized benchmark storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
import psycopg.errors
import psycopg.rows
import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql

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
    ObservationSourceKind,
    ObservationStatus,
    PreprocessingArtifact,
    ProcessingStatus,
    RunStatus,
)
from coval_bench.db.writer import RunWriter
from coval_bench.registries import Metric, MetricValueRole, validate_metric_values

pg_conn = postgresql("pg_proc")
_INI_PATH = Path(__file__).parents[2] / "alembic.ini"
_NOW = datetime(2026, 8, 12, tzinfo=UTC)
_SHA = "a" * 64


def _required[T](value: T | None) -> T:
    """Narrow values returned by INSERT ... RETURNING and persisted models."""
    assert value is not None
    return value


def _dsn(conn: psycopg.Connection[Any]) -> str:
    info = conn.info
    auth = f"{info.user}:{info.password}@" if info.password else f"{info.user}@"
    return (
        f"postgresql://{auth}{info.host or 'localhost'}:{info.port or 5432}/{info.dbname or 'test'}"
    )


def _migrate(conn: psycopg.Connection[Any], target: str = "head") -> None:
    config = AlembicConfig(str(_INI_PATH))
    config.set_main_option(
        "sqlalchemy.url", _dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    if target == "head":
        config.attributes["allow_metric_code_cleanup"] = True
    alembic_command.upgrade(config, target)


async def _pool(
    conn: psycopg.Connection[Any],
) -> AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]:
    pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]] = AsyncConnectionPool(
        conninfo=_dsn(conn), open=False, kwargs={"row_factory": psycopg.rows.dict_row}
    )
    await pool.open()
    return pool


async def _observation(
    writer: RunWriter,
    *,
    sample: str = "sample",
    run_dataset: str = "run-dataset",
    observation_dataset: str = "observation-dataset",
) -> tuple[int, Observation]:
    run = await writer.start_run(dataset_id=run_dataset, dataset_sha256=_SHA, scheduled_at=_NOW)
    observation = await writer.insert_observation(
        Observation(
            run_id=_required(run.id),
            dataset_id=observation_dataset,
            dataset_sha256="b" * 64,
            sample_id=sample,
            provider="provider",
            model="model",
            benchmark=Benchmark.STT,
            source_kind=ObservationSourceKind.DATASET_AUDIO,
            transport_protocol="HTTP/2",
            submit_to_headers_ms=5.0,
            status=ObservationStatus.SUCCEEDED,
        )
    )
    return _required(run.id), observation


async def _evaluation(
    writer: RunWriter, observation: Observation, *, metric: Metric = Metric.WER
) -> MetricEvaluation:
    queued = await writer.insert_metric_evaluation(
        MetricEvaluation(
            observation_id=_required(observation.id),
            metric_type=str(metric),
            metric_version="v1",
            executor=MetricExecutor.INLINE,
            status=ProcessingStatus.QUEUED,
        )
    )
    return await writer.start_metric_evaluation_exact(_required(queued.id), started_at=_NOW)


def _word_artifact(observation_id: Any, *, sha: str = _SHA) -> PreprocessingArtifact:
    return PreprocessingArtifact(
        observation_id=observation_id,
        pipeline="align",
        pipeline_version="v1",
        artifact_name="word_timestamps",
        schema_name="WordTimestampsV1",
        producer_name="word_aligner",
        producer_provider="google",
        producer_model="latest",
        producer_version="words-v1",
        gcs_uri="gs://private/words",
        content_sha256=sha,
    )


def _phone_artifact(observation_id: Any) -> PreprocessingArtifact:
    return PreprocessingArtifact(
        observation_id=observation_id,
        pipeline="align",
        pipeline_version="v1",
        artifact_name="phoneme_timestamps",
        schema_name="PhonemeTimestampsV1",
        producer_name="phoneme_aligner",
        producer_provider="phoneme-provider",
        producer_model="latest",
        producer_version="phones-v1",
        gcs_uri="gs://private/phones",
        content_sha256="c" * 64,
    )


def _future_artifact(observation_id: Any) -> PreprocessingArtifact:
    return PreprocessingArtifact(
        observation_id=observation_id,
        pipeline="align",
        pipeline_version="v2",
        artifact_name="future_artifact",
        schema_name="FutureArtifactV2",
        schema_version="v2",
        producer_name="future_aligner",
        producer_provider="future-provider",
        producer_model="future-model",
        producer_version="future-v2",
        gcs_uri="gs://private/future",
        content_sha256="d" * 64,
    )


def _raw_artifact(*, sha: str = _SHA) -> ObservationArtifact:
    return ObservationArtifact(
        artifact_type=ObservationArtifactType.PROVIDER_TRANSCRIPT,
        schema_name="ProviderTranscriptV1",
        schema_version="v1",
        gcs_uri="gs://private/provider-transcript",
        content_sha256=sha,
        size_bytes=1,
    )


@pytest.mark.asyncio
async def test_capture_recovery_promotes_pending_run(
    pg_conn: psycopg.Connection[Any],
) -> None:
    _migrate(pg_conn)
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        run = await writer.start_run(dataset_id="stt-v1", dataset_sha256=_SHA, scheduled_at=_NOW)
        run_id = _required(run.id)
        await writer.insert_observation(
            Observation(
                run_id=run_id,
                dataset_id="stt-v1",
                dataset_sha256=_SHA,
                sample_id="sample",
                provider="provider",
                model="model",
                benchmark=Benchmark.STT,
                source_kind=ObservationSourceKind.DATASET_AUDIO,
                status=ObservationStatus.SUCCEEDED,
            )
        )

        finished_at = _NOW + timedelta(minutes=1)
        await writer.finish_run_exact(
            run_id,
            status=RunStatus.PARTIAL,
            finished_at=finished_at,
            error="normalized capture pending",
        )
        await writer.finish_run_exact(
            run_id,
            status=RunStatus.SUCCEEDED,
            finished_at=finished_at,
            allow_capture_recovery=True,
        )
        stored = await writer.get_run(run_id)
        assert stored.status is RunStatus.SUCCEEDED
        assert stored.finished_at == finished_at
        assert stored.error is None

        await writer.mark_run_capture_pending(run_id, finished_at=finished_at)
        pending = await writer.get_run(run_id)
        assert pending.status is RunStatus.PARTIAL
        assert pending.error == "normalized capture pending"
        await writer.finish_run_exact(
            run_id,
            status=RunStatus.SUCCEEDED,
            finished_at=finished_at,
            allow_capture_recovery=True,
        )
    finally:
        await pool.close()


def _wer_values(evaluation_id: Any) -> list[MetricValue]:
    return [
        MetricValue(
            metric_evaluation_id=evaluation_id,
            value_key="primary",
            unit="percent",
            value=10,
            value_role=MetricValueRole.PRIMARY,
        ),
        MetricValue(
            metric_evaluation_id=evaluation_id, value_key="insertions", unit="percent", value=1
        ),
        MetricValue(
            metric_evaluation_id=evaluation_id, value_key="deletions", unit="percent", value=2
        ),
        MetricValue(
            metric_evaluation_id=evaluation_id, value_key="substitutions", unit="percent", value=7
        ),
    ]


def test_observation_check_constraints(pg_conn: psycopg.Connection[Any]) -> None:
    _migrate(pg_conn)
    pg_conn.autocommit = True
    insert = """INSERT INTO benchmarks_v2.benchmark_observations
        (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark,
         source_kind, provider_extras, status, error, failure_origin)
        VALUES (%s, %s, %s, %s, 'p', 'm', %s, %s, %s::jsonb, %s, %s, %s)"""
    with pg_conn.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.runs (runner_sha, dataset_id, dataset_sha256, status)
               VALUES ('sha', 'run-dataset', %s, 'running') RETURNING id""",
            (_SHA,),
        )
        run_id = _required(cur.fetchone())[0]
        # Observations carry their own dataset identity, independent of the run's.
        cur.execute(
            insert,
            (
                run_id,
                "tts-dataset",
                "b" * 64,
                "valid",
                "TTS",
                "generated_audio",
                "{}",
                "succeeded",
                None,
                None,
            ),
        )
        cur.execute(
            insert,
            (
                run_id,
                "dataset",
                _SHA,
                "failed-valid",
                "STT",
                "dataset_audio",
                "{}",
                "failed",
                "provider error",
                "provider",
            ),
        )
        invalid = (
            ("succeeded-origin", "STT", "dataset_audio", "{}", "succeeded", None, "runner"),
            ("failed-no-origin", "STT", "dataset_audio", "{}", "failed", "error", None),
            ("array-extras", "STT", "dataset_audio", "[]", "succeeded", None, None),
            ("unknown-source", "STT", "future_audio_source", "{}", "succeeded", None, None),
            ("lowercase", "stt", "dataset_audio", "{}", "succeeded", None, None),
        )
        for sample, benchmark, source, extras, status, error, origin in invalid:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    insert,
                    (
                        run_id,
                        "dataset",
                        _SHA,
                        sample,
                        benchmark,
                        source,
                        extras,
                        status,
                        error,
                        origin,
                    ),
                )


def test_observation_model_rejects_inconsistent_payloads() -> None:
    base = {
        "run_id": 1,
        "dataset_id": "dataset",
        "dataset_sha256": _SHA,
        "sample_id": "sample",
        "provider": "provider",
        "model": "model",
        "benchmark": Benchmark.STT,
        "source_kind": ObservationSourceKind.DATASET_AUDIO,
        "status": ObservationStatus.SUCCEEDED,
    }
    artifact = _raw_artifact()
    with pytest.raises(ValueError, match="cannot repeat"):
        Observation(**base, artifacts=[artifact, artifact])
    with pytest.raises(ValueError, match="private gs:// object URI"):
        ObservationArtifact.model_validate(artifact.model_dump() | {"gcs_uri": "gs://private"})
    with pytest.raises(ValueError, match="failure_origin"):
        Observation(**(base | {"status": ObservationStatus.FAILED}), error="provider error")


@pytest.mark.asyncio
async def test_preprocessing_artifact_writer_validates_supported_contracts(
    pg_conn: psycopg.Connection[Any],
) -> None:
    _migrate(pg_conn)
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await _observation(writer)
        observation_id = _required(observation.id)

        word = await writer.insert_preprocessing_artifact(_word_artifact(observation_id))
        phoneme = await writer.insert_preprocessing_artifact(_phone_artifact(observation_id))
        assert word.schema_version == "v1"
        assert phoneme.schema_version == "v1"

        with pytest.raises(ValueError, match="unknown preprocessing artifact contract"):
            await writer.insert_preprocessing_artifact(_future_artifact(observation_id))

        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT count(*) AS count FROM benchmarks_v2.preprocessing_artifacts "
                "WHERE observation_id = %s AND artifact_name = 'future_artifact'",
                (observation_id,),
            )
            assert _required(await cur.fetchone())["count"] == 0
    finally:
        await pool.close()


def test_metric_input_requires_exactly_one_artifact_kind() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MetricEvaluationInput(input_role="missing", input_order=0)
    with pytest.raises(ValueError, match="exactly one"):
        MetricEvaluationInput(
            observation_artifact_id=uuid4(),
            preprocessing_artifact_id=uuid4(),
            input_role="both",
            input_order=0,
        )


@pytest.mark.asyncio
async def test_nested_deletes_cannot_bypass_immutable_lineage(
    pg_conn: psycopg.Connection[Any],
) -> None:
    """Unrelated nested triggers are not mistaken for foreign-key cascades."""
    _migrate(pg_conn)
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await _observation(writer)
        observation_id = _required(observation.id)
        artifact = await writer.insert_preprocessing_artifact(_word_artifact(observation_id))
        evaluation = await writer.insert_metric_evaluation(
            MetricEvaluation(
                observation_id=observation_id,
                metric_type=str(Metric.WER),
                metric_version="v1",
                evaluation_variant="nested-delete",
                executor=MetricExecutor.INLINE,
                status=ProcessingStatus.QUEUED,
            ),
            inputs=[
                MetricEvaluationInput(
                    preprocessing_artifact_id=_required(artifact.id),
                    input_role="word",
                    input_order=0,
                )
            ],
        )
        evaluation_id = _required(evaluation.id)
        await writer.fail_metric_evaluation_exact(
            evaluation_id, started_at=_NOW, finished_at=_NOW, error="controlled failure"
        )

        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """CREATE TABLE benchmarks_v2.nested_delete_requests (
                       target_kind TEXT NOT NULL,
                       target_id UUID NOT NULL
                   )"""
            )
            await cur.execute(
                """CREATE FUNCTION benchmarks_v2.run_nested_delete_request()
                   RETURNS trigger AS $$
                   BEGIN
                       IF NEW.target_kind = 'artifact' THEN
                           DELETE FROM benchmarks_v2.preprocessing_artifacts
                           WHERE id = NEW.target_id;
                       ELSIF NEW.target_kind = 'input' THEN
                           DELETE FROM benchmarks_v2.metric_evaluation_inputs
                           WHERE metric_evaluation_id = NEW.target_id;
                       ELSIF NEW.target_kind = 'evaluation' THEN
                           DELETE FROM benchmarks_v2.metric_evaluations
                           WHERE id = NEW.target_id;
                       END IF;
                       RETURN NEW;
                   END; $$ LANGUAGE plpgsql"""
            )
            await cur.execute(
                """CREATE TRIGGER nested_delete_request
                   AFTER INSERT ON benchmarks_v2.nested_delete_requests
                   FOR EACH ROW EXECUTE FUNCTION benchmarks_v2.run_nested_delete_request()"""
            )
            await conn.commit()

            attempts = (
                ("artifact", _required(artifact.id), "artifacts are immutable"),
                ("input", evaluation_id, "inputs are immutable"),
                ("evaluation", evaluation_id, "terminal work rows are immutable"),
            )
            for target_kind, target_id, message in attempts:
                with pytest.raises(psycopg.errors.RaiseException, match=message):
                    await cur.execute(
                        """INSERT INTO benchmarks_v2.nested_delete_requests
                           (target_kind, target_id) VALUES (%s, %s)""",
                        (target_kind, target_id),
                    )
                await conn.rollback()

            await cur.execute(
                "SELECT count(*) AS count FROM benchmarks_v2.preprocessing_artifacts WHERE id = %s",
                (_required(artifact.id),),
            )
            assert _required(await cur.fetchone())["count"] == 1
            await cur.execute(
                "SELECT status FROM benchmarks_v2.metric_evaluations WHERE id = %s",
                (evaluation_id,),
            )
            assert _required(await cur.fetchone())["status"] == "failed"
            await cur.execute(
                """SELECT count(*) AS count FROM benchmarks_v2.metric_evaluation_inputs
                   WHERE metric_evaluation_id = %s""",
                (evaluation_id,),
            )
            assert _required(await cur.fetchone())["count"] == 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_create_get_is_retry_safe_and_strict(pg_conn: psycopg.Connection[Any]) -> None:
    _migrate(pg_conn)
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await _observation(writer)
        raw_artifact = _raw_artifact()
        observation = await writer.insert_observation(
            observation.model_copy(update={"id": None, "artifacts": [raw_artifact]})
        )
        assert len(observation.artifacts) == 1
        raw_id = _required(observation.artifacts[0].id)
        duplicate = await writer.insert_observation(observation.model_copy(update={"id": None}))
        assert duplicate.id == observation.id
        assert _required(duplicate.artifacts[0].id) == raw_id
        with pytest.raises(ValueError, match="artifact retry conflicts"):
            await writer.insert_observation(
                observation.model_copy(
                    update={"id": None, "artifacts": [_raw_artifact(sha="b" * 64)]}
                )
            )
        assert duplicate.transport_protocol == "HTTP/2"
        with pytest.raises(ValueError, match="transport_protocol"):
            await writer.insert_observation(
                observation.model_copy(
                    update={
                        "id": None,
                        "transport_protocol": "HTTP/1.1",
                    }
                )
            )
        with pytest.raises(ValueError, match="dataset_sha256"):
            await writer.insert_observation(
                observation.model_copy(
                    update={
                        "id": None,
                        "dataset_sha256": "d" * 64,
                        "artifacts": [
                            _raw_artifact(sha="d" * 64).model_copy(
                                update={"artifact_type": ObservationArtifactType.TIMING_EVENTS}
                            )
                        ],
                    }
                )
            )
        with pytest.raises(ValueError, match="dataset_id"):
            await writer.insert_observation(
                observation.model_copy(
                    update={"id": None, "dataset_id": "different-dataset", "artifacts": []}
                )
            )
        with pytest.raises(ValueError, match="benchmark"):
            await writer.insert_observation(
                observation.model_copy(
                    update={"id": None, "benchmark": Benchmark.TTS, "artifacts": []}
                )
            )
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT count(*) AS count FROM benchmarks_v2.observation_artifacts "
                "WHERE observation_id = %s",
                (_required(observation.id),),
            )
            assert _required(await cur.fetchone())["count"] == 1

        observation_id = _required(observation.id)
        artifact = await writer.insert_preprocessing_artifact(_word_artifact(observation_id))
        assert (
            await writer.insert_preprocessing_artifact(_word_artifact(observation_id))
        ).id == artifact.id
        with pytest.raises(ValueError, match="immutable fields"):
            await writer.insert_preprocessing_artifact(_word_artifact(observation_id, sha="e" * 64))

        evaluation = await writer.insert_metric_evaluation(
            MetricEvaluation(
                observation_id=observation_id,
                metric_type=str(Metric.WER),
                metric_version="v1",
                executor=MetricExecutor.INLINE,
                status=ProcessingStatus.QUEUED,
            )
        )
        with pytest.raises(ValueError, match="unknown metric/version"):
            await writer.insert_metric_evaluation(
                MetricEvaluation(
                    observation_id=observation_id,
                    metric_type=str(Metric.WER),
                    metric_version="v2",
                    evaluation_variant="future",
                    executor=MetricExecutor.INLINE,
                    status=ProcessingStatus.QUEUED,
                )
            )
        with pytest.raises(ValueError, match="executor"):
            await writer.insert_metric_evaluation(
                MetricEvaluation(
                    observation_id=observation_id,
                    metric_type=str(Metric.WER),
                    metric_version="v1",
                    executor=MetricExecutor.COVAL_API,
                    status=ProcessingStatus.QUEUED,
                )
            )
        assert evaluation.status is ProcessingStatus.QUEUED
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_ensemble_variants_and_frozen_inputs(pg_conn: psycopg.Connection[Any]) -> None:
    """Providers/models and ordered preprocessing lineage remain independently addressable."""
    _migrate(pg_conn)
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await _observation(writer)
        observation = await writer.insert_observation(
            observation.model_copy(update={"id": None, "artifacts": [_raw_artifact()]})
        )
        observation_id = _required(observation.id)
        raw_artifact_id = _required(observation.artifacts[0].id)
        google = await writer.insert_preprocessing_artifact(_word_artifact(observation_id))
        deepgram = await writer.insert_preprocessing_artifact(
            _word_artifact(observation_id).model_copy(
                update={
                    "producer_provider": "deepgram",
                    "producer_model": "nova",
                    "gcs_uri": "gs://private/deepgram",
                }
            )
        )
        phone_a = await writer.insert_preprocessing_artifact(_phone_artifact(observation_id))
        phone_b = await writer.insert_preprocessing_artifact(
            _phone_artifact(observation_id).model_copy(
                update={"producer_model": "model-b", "gcs_uri": "gs://private/phones-b"}
            )
        )
        assert len({google.id, deepgram.id, phone_a.id, phone_b.id}) == 4

        def queued(variant: str) -> MetricEvaluation:
            return MetricEvaluation(
                observation_id=observation_id,
                metric_type=str(Metric.WER),
                metric_version="v1",
                evaluation_variant=variant,
                executor=MetricExecutor.INLINE,
                status=ProcessingStatus.QUEUED,
            )

        inputs = [
            MetricEvaluationInput(
                observation_artifact_id=raw_artifact_id, input_role="raw", input_order=0
            ),
            MetricEvaluationInput(
                preprocessing_artifact_id=_required(google.id), input_role="word", input_order=0
            ),
            MetricEvaluationInput(
                preprocessing_artifact_id=_required(deepgram.id), input_role="word", input_order=1
            ),
            MetricEvaluationInput(
                preprocessing_artifact_id=_required(phone_a.id), input_role="phoneme", input_order=0
            ),
        ]
        ensemble = await writer.insert_metric_evaluation(queued("ensemble"), inputs=inputs)
        assert await writer.insert_metric_evaluation(queued("ensemble"), inputs=inputs) == ensemble
        variants = {"ensemble": ensemble}
        for variant in ("google", "deepgram"):
            variants[variant] = await writer.insert_metric_evaluation(queued(variant))
            assert variants[variant].evaluation_variant == variant
        with pytest.raises(ValueError, match="immutable inputs"):
            await writer.insert_metric_evaluation(queued("ensemble"), inputs=inputs[:-1])
        with pytest.raises(ValueError, match="immutable inputs"):
            await writer.insert_metric_evaluation(
                queued("ensemble"),
                inputs=[*inputs[:2], inputs[2].model_copy(update={"input_order": 1})],
            )

        _, other = await _observation(writer, sample="other")
        other = await writer.insert_observation(
            other.model_copy(update={"id": None, "artifacts": [_raw_artifact()]})
        )
        with pytest.raises(psycopg.errors.RaiseException, match="share the evaluation observation"):
            async with pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO benchmarks_v2.metric_evaluation_inputs
                       (metric_evaluation_id, observation_artifact_id, input_role, input_order)
                       VALUES (%s, %s, 'other', 0)""",
                    (_required(ensemble.id), _required(other.artifacts[0].id)),
                )
        async with pool.connection() as conn, conn.cursor() as cur:
            with pytest.raises(psycopg.errors.RaiseException, match="immutable"):
                await cur.execute(
                    "DELETE FROM benchmarks_v2.metric_evaluation_inputs "
                    "WHERE metric_evaluation_id = %s",
                    (_required(ensemble.id),),
                )
            await conn.rollback()
        started = await writer.start_metric_evaluation_exact(
            _required(ensemble.id), started_at=_NOW
        )
        assert started.evaluation_variant == "ensemble"
        async with pool.connection() as conn, conn.cursor() as cur:
            with pytest.raises(
                psycopg.errors.RaiseException, match="only be inserted while queued"
            ):
                await cur.execute(
                    """INSERT INTO benchmarks_v2.metric_evaluation_inputs
                       (metric_evaluation_id, observation_artifact_id, input_role, input_order)
                       VALUES (%s, %s, 'late', 2)""",
                    (_required(ensemble.id), raw_artifact_id),
                )
            await conn.rollback()
        finished = _NOW + timedelta(seconds=1)
        await writer.complete_metric_evaluation(
            _required(ensemble.id), values=_wer_values(_required(ensemble.id)), finished_at=finished
        )
        assert (
            await writer.insert_metric_evaluation(queued("ensemble"), inputs=inputs)
        ).status is ProcessingStatus.SUCCEEDED
        for changed in (
            inputs[:-1],
            [*inputs[:2], inputs[2].model_copy(update={"input_role": "changed"})],
        ):
            with pytest.raises(ValueError, match="immutable inputs"):
                await writer.insert_metric_evaluation(queued("ensemble"), inputs=changed)
        for variant in ("google", "deepgram"):
            running = await writer.start_metric_evaluation_exact(
                _required(variants[variant].id), started_at=_NOW
            )
            await writer.complete_metric_evaluation(
                _required(running.id),
                values=_wer_values(_required(running.id)),
                finished_at=finished,
            )
        await writer.finish_run(_required(observation.run_id), status=RunStatus.SUCCEEDED)
        await writer.rebuild_run_rollup(_required(observation.run_id))
        await writer.rebuild_run_rollup(_required(observation.run_id))
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """SELECT dataset_id, evaluation_variant, sample_count
                   FROM benchmarks_v2.dashboard_rollups
                   WHERE grain = 'run' AND value_key = 'primary'"""
            )
            assert {
                (row["dataset_id"], row["evaluation_variant"], row["sample_count"])
                for row in await cur.fetchall()
            } == {
                ("__all__", "deepgram", 1),
                ("__all__", "ensemble", 1),
                ("__all__", "google", 1),
                ("observation-dataset", "deepgram", 1),
                ("observation-dataset", "ensemble", 1),
                ("observation-dataset", "google", 1),
            }
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_metric_input_freeze_serializes_with_lifecycle_updates(
    pg_conn: psycopg.Connection[Any],
) -> None:
    """The input trigger locks an evaluation before checking its queued state."""
    _migrate(pg_conn)
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await _observation(writer)
        observation_id = _required(observation.id)
        artifact = await writer.insert_preprocessing_artifact(_word_artifact(observation_id))

        def queued(variant: str) -> MetricEvaluation:
            return MetricEvaluation(
                observation_id=observation_id,
                metric_type=str(Metric.WER),
                metric_version="v1",
                evaluation_variant=variant,
                executor=MetricExecutor.INLINE,
                status=ProcessingStatus.QUEUED,
            )

        input_first = await writer.insert_metric_evaluation(queued("input-first"))
        input_first_id = _required(input_first.id)
        async with pool.connection() as conn_a, conn_a.cursor() as cur_a:
            await cur_a.execute(
                """INSERT INTO benchmarks_v2.metric_evaluation_inputs
                   (metric_evaluation_id, preprocessing_artifact_id, input_role, input_order)
                   VALUES (%s, %s, 'word', 0)""",
                (input_first_id, _required(artifact.id)),
            )
            async with pool.connection() as conn_b, conn_b.cursor() as cur_b:
                await cur_b.execute("SET LOCAL lock_timeout = '100ms'")
                # A server lock timeout proves the input trigger holds the evaluation lock.
                with pytest.raises(psycopg.errors.LockNotAvailable):
                    await cur_b.execute(
                        """UPDATE benchmarks_v2.metric_evaluations
                           SET status = 'running', started_at = %s WHERE id = %s""",
                        (_NOW, input_first_id),
                    )
                await conn_b.rollback()
            await conn_a.commit()
        await writer.start_metric_evaluation_exact(input_first_id, started_at=_NOW)

        lifecycle_first = await writer.insert_metric_evaluation(queued("lifecycle-first"))
        lifecycle_first_id = _required(lifecycle_first.id)
        async with pool.connection() as conn_b, conn_b.cursor() as cur_b:
            await cur_b.execute(
                """UPDATE benchmarks_v2.metric_evaluations
                   SET status = 'running', started_at = %s WHERE id = %s""",
                (_NOW, lifecycle_first_id),
            )
            async with pool.connection() as conn_a, conn_a.cursor() as cur_a:
                await cur_a.execute("SET LOCAL lock_timeout = '100ms'")
                # The input trigger's row lock must wait behind the lifecycle update.
                with pytest.raises(psycopg.errors.LockNotAvailable):
                    await cur_a.execute(
                        """INSERT INTO benchmarks_v2.metric_evaluation_inputs
                           (metric_evaluation_id, preprocessing_artifact_id,
                            input_role, input_order)
                           VALUES (%s, %s, 'word', 0)""",
                        (lifecycle_first_id, _required(artifact.id)),
                    )
                await conn_a.rollback()
            await conn_b.commit()
        async with pool.connection() as conn_a, conn_a.cursor() as cur_a:
            with pytest.raises(
                psycopg.errors.RaiseException, match="only be inserted while queued"
            ):
                await cur_a.execute(
                    """INSERT INTO benchmarks_v2.metric_evaluation_inputs
                       (metric_evaluation_id, preprocessing_artifact_id, input_role, input_order)
                       VALUES (%s, %s, 'word', 0)""",
                    (lifecycle_first_id, _required(artifact.id)),
                )
            await conn_a.rollback()

        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT status FROM benchmarks_v2.metric_evaluations WHERE id = %s",
                (input_first_id,),
            )
            assert _required(await cur.fetchone())["status"] == "running"
            await cur.execute(
                "SELECT count(*) AS count FROM benchmarks_v2.metric_evaluation_inputs "
                "WHERE metric_evaluation_id = %s",
                (input_first_id,),
            )
            assert _required(await cur.fetchone())["count"] == 1
            await cur.execute(
                "SELECT count(*) AS count FROM benchmarks_v2.metric_evaluation_inputs "
                "WHERE metric_evaluation_id = %s",
                (lifecycle_first_id,),
            )
            assert _required(await cur.fetchone())["count"] == 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_explicit_lifecycle_and_failed_terminal_state(
    pg_conn: psycopg.Connection[Any],
) -> None:
    _migrate(pg_conn)
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await _observation(writer)
        observation_id = _required(observation.id)
        evaluation = await writer.insert_metric_evaluation(
            MetricEvaluation(
                observation_id=observation_id,
                metric_type=str(Metric.TTFT),
                metric_version="v1",
                executor=MetricExecutor.INLINE,
                status=ProcessingStatus.QUEUED,
            )
        )
        evaluation_id = _required(evaluation.id)
        failed_evaluation = await writer.fail_metric_evaluation_exact(
            evaluation_id, started_at=_NOW, finished_at=_NOW, error="request failed"
        )
        assert failed_evaluation.started_at == failed_evaluation.finished_at == _NOW
        with pytest.raises(ValueError, match="conflicts with stored result"):
            await writer.fail_metric_evaluation_exact(
                evaluation_id, started_at=_NOW, finished_at=_NOW, error="retry failed"
            )
        missing_id = uuid4()
        with pytest.raises(ValueError, match=str(missing_id)):
            await writer.complete_metric_evaluation(
                missing_id,
                values=[
                    MetricValue(
                        metric_evaluation_id=missing_id,
                        value_key="primary",
                        unit="seconds",
                        value=0,
                        value_role=MetricValueRole.PRIMARY,
                    )
                ],
                finished_at=_NOW,
            )
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_metric_completion_replay_and_rollback(pg_conn: psycopg.Connection[Any]) -> None:
    _migrate(pg_conn)
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await _observation(writer)
        evaluation = await _evaluation(writer, observation)
        evaluation_id = _required(evaluation.id)
        values = _wer_values(evaluation_id)
        artifact = MetricArtifact(
            metric_evaluation_id=evaluation_id,
            artifact_type="details",
            uri="gs://private/details",
            sha256=_SHA,
            size_bytes=1,
        )
        finished = _NOW + timedelta(seconds=1)
        await writer.complete_metric_evaluation(
            evaluation_id, values=values, artifacts=[artifact], finished_at=finished
        )
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """SELECT has_primary_role, value, wer_insertions_pct,
                          wer_deletions_pct, wer_substitutions_pct
                   FROM benchmarks_v2.dashboard_metric_values
                   WHERE evaluation_id = %s""",
                (evaluation_id,),
            )
            assert await cur.fetchone() == {
                "has_primary_role": True,
                "value": 10.0,
                "wer_insertions_pct": 1.0,
                "wer_deletions_pct": 2.0,
                "wer_substitutions_pct": 7.0,
            }
        await writer.complete_metric_evaluation(
            evaluation_id, values=values, artifacts=[artifact], finished_at=finished
        )
        changed = [*values]
        changed[0] = changed[0].model_copy(update={"value": 11})
        changed[3] = changed[3].model_copy(update={"value": 8})
        with pytest.raises(ValueError, match="replay conflicts"):
            await writer.complete_metric_evaluation(
                evaluation_id, values=changed, artifacts=[artifact], finished_at=finished
            )
        async with pool.connection() as conn, conn.cursor() as cur:
            with pytest.raises(psycopg.errors.RaiseException, match="payloads are immutable"):
                await cur.execute(
                    """UPDATE benchmarks_v2.metric_values SET value = value + 1
                       WHERE metric_evaluation_id = %s AND value_key = 'primary'""",
                    (evaluation_id,),
                )
            await conn.rollback()

        invalid = await _evaluation(writer, observation, metric=Metric.TTFA)
        invalid_id = _required(invalid.id)
        with pytest.raises(psycopg.errors.CheckViolation):
            await writer.complete_metric_evaluation(
                invalid_id,
                values=[
                    MetricValue(
                        metric_evaluation_id=invalid_id,
                        value_key="primary",
                        unit="milliseconds",
                        value=1,
                        value_role=MetricValueRole.PRIMARY,
                    )
                ],
                finished_at=_NOW - timedelta(seconds=1),
            )
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT count(*) AS count FROM benchmarks_v2.metric_values "
                "WHERE metric_evaluation_id = %s",
                (invalid_id,),
            )
            assert _required(await cur.fetchone())["count"] == 0
            await cur.execute(
                "SELECT count(*) AS count FROM benchmarks_v2.dashboard_metric_values "
                "WHERE evaluation_id = %s",
                (invalid_id,),
            )
            assert _required(await cur.fetchone())["count"] == 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_metric_payload_insert_waits_for_completion_then_rejects(
    pg_conn: psycopg.Connection[Any],
) -> None:
    """A payload writer serializes behind completion and then sees terminal state."""
    _migrate(pg_conn)
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await _observation(writer, sample="payload-lock")
        evaluation = await _evaluation(writer, observation)
        evaluation_id = _required(evaluation.id)
        async with pool.connection() as conn_a, conn_a.cursor() as cur_a:
            await cur_a.execute(
                "SELECT id FROM benchmarks_v2.metric_evaluations WHERE id = %s FOR UPDATE",
                (evaluation_id,),
            )
            async with pool.connection() as conn_b, conn_b.cursor() as cur_b:
                await cur_b.execute("SET LOCAL lock_timeout = '100ms'")
                with pytest.raises(psycopg.errors.LockNotAvailable):
                    await cur_b.execute(
                        "INSERT INTO benchmarks_v2.metric_values "
                        "(metric_evaluation_id, value_key, unit, value, value_role) "
                        "VALUES (%s, 'primary', 'percent', 10, 'primary')",
                        (evaluation_id,),
                    )
                await conn_b.rollback()
            await cur_a.execute(
                "INSERT INTO benchmarks_v2.metric_values "
                "(metric_evaluation_id, value_key, unit, value, value_role) "
                "VALUES (%s, 'primary', 'percent', 10, 'primary')",
                (evaluation_id,),
            )
            await cur_a.execute(
                "UPDATE benchmarks_v2.metric_evaluations SET status = 'succeeded', "
                "finished_at = %s WHERE id = %s",
                (_NOW + timedelta(seconds=1), evaluation_id),
            )
            await conn_a.commit()
        async with pool.connection() as conn, conn.cursor() as cur:
            with pytest.raises(psycopg.errors.RaiseException, match="payloads are immutable"):
                await cur.execute(
                    "INSERT INTO benchmarks_v2.metric_values "
                    "(metric_evaluation_id, value_key, unit, value, value_role) "
                    "VALUES (%s, 'insertions', 'percent', 1, 'component')",
                    (evaluation_id,),
                )
            await conn.rollback()
    finally:
        await pool.close()


def test_metric_value_contracts_cover_wer_and_optional_ttfa_components() -> None:
    validate_metric_values(
        Metric.WER,
        "v1",
        (
            ("primary", "percent", 3, MetricValueRole.PRIMARY),
            ("insertions", "percent", 1, MetricValueRole.COMPONENT),
            ("deletions", "percent", 1, MetricValueRole.COMPONENT),
            ("substitutions", "percent", 1, MetricValueRole.COMPONENT),
        ),
    )
    validate_metric_values(
        Metric.WER,
        "v1",
        (
            ("primary", "percent", 3, MetricValueRole.PRIMARY),
            ("insertions", "percent", 1, MetricValueRole.COMPONENT),
            ("deletions", "percent", 1, MetricValueRole.COMPONENT),
            ("substitutions", "percent", 1, MetricValueRole.COMPONENT),
            ("substitution_count", "count", 1, MetricValueRole.COMPONENT),
            ("deletion_count", "count", 1, MetricValueRole.COMPONENT),
            ("insertion_count", "count", 1, MetricValueRole.COMPONENT),
            ("reference_words", "count", 100, MetricValueRole.COMPONENT),
        ),
    )
    with pytest.raises(ValueError, match="optional metric value group"):
        validate_metric_values(
            Metric.WER,
            "v1",
            (
                ("primary", "percent", 3, MetricValueRole.PRIMARY),
                ("insertions", "percent", 1, MetricValueRole.COMPONENT),
                ("deletions", "percent", 1, MetricValueRole.COMPONENT),
                ("substitutions", "percent", 1, MetricValueRole.COMPONENT),
                ("reference_words", "count", 100, MetricValueRole.COMPONENT),
            ),
        )
    validate_metric_values(
        Metric.TTFA, "v1", (("primary", "milliseconds", 12, MetricValueRole.PRIMARY),)
    )
    validate_metric_values(
        Metric.TTFA,
        "v1",
        (
            ("primary", "milliseconds", 12, MetricValueRole.PRIMARY),
            ("roundtrip", "milliseconds", 10, MetricValueRole.COMPONENT),
            ("leading_silence", "milliseconds", 2, MetricValueRole.COMPONENT),
        ),
    )
    with pytest.raises(ValueError, match="optional metric value group"):
        validate_metric_values(
            Metric.TTFA,
            "v1",
            (
                ("primary", "milliseconds", 12, MetricValueRole.PRIMARY),
                ("roundtrip", "milliseconds", 10, MetricValueRole.COMPONENT),
            ),
        )
    with pytest.raises(ValueError, match="wrong value role"):
        validate_metric_values(
            Metric.WER,
            "v1",
            (
                ("primary", "percent", 3, MetricValueRole.COMPONENT),
                ("insertions", "percent", 1, MetricValueRole.COMPONENT),
                ("deletions", "percent", 1, MetricValueRole.COMPONENT),
                ("substitutions", "percent", 1, MetricValueRole.COMPONENT),
            ),
        )


def test_database_enforces_queued_creation_and_success_outputs(
    pg_conn: psycopg.Connection[Any],
) -> None:
    _migrate(pg_conn)
    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.runs (runner_sha, dataset_id, dataset_sha256, status)
               VALUES ('sha', 'dataset', %s, 'running') RETURNING id""",
            (_SHA,),
        )
        run_id = _required(cur.fetchone())[0]
        cur.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
               (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark,
                source_kind, status)
               VALUES (%s, 'dataset', %s, 'sample', 'p', 'm', 'STT', 'dataset_audio',
                'succeeded') RETURNING id""",
            (run_id, _SHA),
        )
        observation_id = _required(cur.fetchone())[0]
        with pytest.raises(psycopg.errors.RaiseException, match="work rows must be created queued"):
            cur.execute(
                """INSERT INTO benchmarks_v2.metric_evaluations
                   (observation_id, metric_id, metric_version, executor, status)
                   VALUES (%s, benchmarks_v2.metric_id_for_code('WER'),
                    'v1', 'inline', 'partial')""",
                (observation_id,),
            )
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations
               (observation_id, metric_id, metric_version, executor, status)
               VALUES (%s, benchmarks_v2.metric_id_for_code('WER'), 'v1', 'inline', 'queued')
               RETURNING id""",
            (observation_id,),
        )
        evaluation_id = _required(cur.fetchone())[0]
        with pytest.raises(psycopg.errors.RaiseException, match="identity is immutable"):
            cur.execute(
                """UPDATE benchmarks_v2.metric_evaluations
                   SET status = 'running', started_at = now(),
                       evaluation_variant = 'mutated', executor = 'coval_api'
                   WHERE id = %s""",
                (evaluation_id,),
            )
        cur.execute(
            """SELECT status, evaluation_variant, executor
               FROM benchmarks_v2.metric_evaluations WHERE id = %s""",
            (evaluation_id,),
        )
        assert _required(cur.fetchone()) == ("queued", "default", "inline")
        cur.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET status = 'running', started_at = now() WHERE id = %s""",
            (evaluation_id,),
        )
        cur.execute("BEGIN")
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_values
               (metric_evaluation_id, value_key, unit, value, value_role)
               VALUES (%s, 'primary', 'percent', 1, 'component')""",
            (evaluation_id,),
        )
        cur.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET status = 'succeeded', finished_at = now() WHERE id = %s""",
            (evaluation_id,),
        )
        with pytest.raises(psycopg.errors.RaiseException, match="exactly one primary"):
            cur.execute("COMMIT")
        cur.execute("ROLLBACK")


@pytest.mark.asyncio
async def test_rollup_is_idempotent_and_cascades(pg_conn: psycopg.Connection[Any]) -> None:
    _migrate(pg_conn)
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        run_id, observation = await _observation(writer)
        evaluation = await _evaluation(writer, observation)
        evaluation_id = _required(evaluation.id)
        await writer.complete_metric_evaluation(
            evaluation_id,
            values=_wer_values(evaluation_id),
            finished_at=_NOW + timedelta(seconds=1),
        )
        failed_run_id, failed_observation = await _observation(writer, sample="failed-sample")
        failed_evaluation = await _evaluation(writer, failed_observation)
        failed_evaluation_id = _required(failed_evaluation.id)
        await writer.complete_metric_evaluation(
            failed_evaluation_id,
            values=_wer_values(failed_evaluation_id),
            finished_at=_NOW + timedelta(seconds=1),
        )
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """UPDATE benchmarks_v2.benchmark_observations
                   SET status = 'failed', error = 'capture failed', failure_origin = 'provider'
                   WHERE id = %s""",
                (_required(failed_observation.id),),
            )
            await conn.commit()
        await writer.finish_run(run_id, status=RunStatus.SUCCEEDED)
        await writer.finish_run(failed_run_id, status=RunStatus.SUCCEEDED)
        await writer.rebuild_run_rollup(run_id)
        await writer.rebuild_run_rollup(run_id)
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """SELECT dataset_id, sample_count
                   FROM benchmarks_v2.dashboard_rollups
                   WHERE grain = 'run' ORDER BY dataset_id"""
            )
            rows = await cur.fetchall()
            datasets = [row["dataset_id"] for row in rows]
            assert datasets.count("__all__") == datasets.count("observation-dataset") == 1
            assert {row["sample_count"] for row in rows} == {1}
            await cur.execute(
                "DELETE FROM benchmarks_v2.benchmark_observations WHERE id = %s",
                (_required(observation.id),),
            )
            await cur.execute("SELECT count(*) AS count FROM benchmarks_v2.metric_values")
            # Deleting the successful observation cascades only its four WER values;
            # the failed observation created above still owns the other four.
            assert _required(await cur.fetchone())["count"] == 4
            await cur.execute(
                "DELETE FROM benchmarks_v2.benchmark_observations WHERE id = %s",
                (_required(failed_observation.id),),
            )
            await cur.execute("SELECT count(*) AS count FROM benchmarks_v2.metric_values")
            assert _required(await cur.fetchone())["count"] == 0
            await conn.commit()
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_evaluation_delete_lifecycle_and_observation_cascade(
    pg_conn: psycopg.Connection[Any],
) -> None:
    _migrate(pg_conn)
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        run_id, observation = await _observation(writer)
        observation = await writer.insert_observation(
            observation.model_copy(update={"id": None, "artifacts": [_raw_artifact()]})
        )
        observation_id = _required(observation.id)
        raw_artifact_id = _required(observation.artifacts[0].id)
        queued_artifact = await writer.insert_preprocessing_artifact(
            _phone_artifact(observation_id)
        )
        queued = await writer.insert_metric_evaluation(
            MetricEvaluation(
                observation_id=observation_id,
                metric_type=str(Metric.TTFT),
                metric_version="v1",
                executor=MetricExecutor.INLINE,
                status=ProcessingStatus.QUEUED,
            ),
            inputs=[
                MetricEvaluationInput(
                    preprocessing_artifact_id=_required(queued_artifact.id),
                    input_role="timing",
                    input_order=0,
                )
            ],
        )
        running = await _evaluation(writer, observation, metric=Metric.TTFA)
        queued_id = _required(queued.id)
        running_id = _required(running.id)
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM benchmarks_v2.metric_evaluations WHERE id = %s", (queued_id,)
            )
            await cur.execute(
                "DELETE FROM benchmarks_v2.metric_evaluations WHERE id = %s", (running_id,)
            )
            await cur.execute(
                """SELECT count(*) AS count FROM benchmarks_v2.metric_evaluation_inputs
                   WHERE metric_evaluation_id = %s""",
                (queued_id,),
            )
            assert _required(await cur.fetchone())["count"] == 0
            await conn.commit()
        terminal_artifact = await writer.insert_preprocessing_artifact(
            _word_artifact(observation_id)
        )
        queued_terminal = await writer.insert_metric_evaluation(
            MetricEvaluation(
                observation_id=observation_id,
                metric_type=str(Metric.WER),
                metric_version="v1",
                executor=MetricExecutor.INLINE,
                status=ProcessingStatus.QUEUED,
            ),
            inputs=[
                MetricEvaluationInput(
                    observation_artifact_id=raw_artifact_id,
                    input_role="raw",
                    input_order=0,
                ),
                MetricEvaluationInput(
                    preprocessing_artifact_id=_required(terminal_artifact.id),
                    input_role="word",
                    input_order=0,
                ),
            ],
        )
        evaluation = await writer.start_metric_evaluation_exact(
            _required(queued_terminal.id), started_at=_NOW
        )
        evaluation_id = _required(evaluation.id)
        await writer.complete_metric_evaluation(
            evaluation_id,
            values=_wer_values(evaluation_id),
            artifacts=[
                MetricArtifact(
                    metric_evaluation_id=evaluation_id,
                    artifact_type="details",
                    uri="gs://private/details",
                    sha256=_SHA,
                    size_bytes=1,
                )
            ],
            finished_at=_NOW + timedelta(seconds=1),
        )
        async with pool.connection() as conn, conn.cursor() as cur:
            with pytest.raises(
                psycopg.errors.RaiseException, match="observation artifacts are immutable"
            ):
                await cur.execute(
                    "UPDATE benchmarks_v2.observation_artifacts SET size_bytes = 2 WHERE id = %s",
                    (raw_artifact_id,),
                )
            await conn.rollback()
            with pytest.raises(
                psycopg.errors.RaiseException, match="observation artifacts are immutable"
            ):
                await cur.execute(
                    "DELETE FROM benchmarks_v2.observation_artifacts WHERE id = %s",
                    (raw_artifact_id,),
                )
            await conn.rollback()
            with pytest.raises(psycopg.errors.RaiseException, match="terminal work rows"):
                await cur.execute(
                    "DELETE FROM benchmarks_v2.metric_evaluations WHERE id = %s", (evaluation_id,)
                )
            await conn.rollback()
            await cur.execute(
                """SELECT count(*) AS count FROM benchmarks_v2.metric_evaluation_inputs
                   WHERE metric_evaluation_id = %s""",
                (evaluation_id,),
            )
            assert _required(await cur.fetchone())["count"] == 2
            await cur.execute("DELETE FROM benchmarks_v2.runs WHERE id = %s", (run_id,))
            await cur.execute(
                "SELECT count(*) AS count FROM benchmarks_v2.metric_evaluation_inputs"
            )
            assert _required(await cur.fetchone())["count"] == 0
            await cur.execute("SELECT count(*) AS count FROM benchmarks_v2.preprocessing_artifacts")
            assert _required(await cur.fetchone())["count"] == 0
            await cur.execute("SELECT count(*) AS count FROM benchmarks_v2.observation_artifacts")
            assert _required(await cur.fetchone())["count"] == 0
            await cur.execute("SELECT count(*) AS count FROM benchmarks_v2.metric_values")
            assert _required(await cur.fetchone())["count"] == 0
            await cur.execute("SELECT count(*) AS count FROM benchmarks_v2.metric_artifacts")
            assert _required(await cur.fetchone())["count"] == 0
            await conn.commit()
    finally:
        await pool.close()


def test_output_writers_are_not_public() -> None:
    assert hasattr(RunWriter, "insert_preprocessing_artifact")
    assert not hasattr(RunWriter, "insert_metric_values")
    assert not hasattr(RunWriter, "insert_metric_artifacts")
