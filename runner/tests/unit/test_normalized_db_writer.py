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
    ObservationFailureOrigin,
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
    run = await writer.start_run(
        runner_sha="sha", dataset_id=run_dataset, dataset_sha256=_SHA, scheduled_at=_NOW
    )
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
    return await writer.start_metric_evaluation(_required(queued.id), started_at=_NOW)


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


def test_migration_is_additive_and_reversible(pg_conn: psycopg.Connection[Any]) -> None:
    _migrate(pg_conn)
    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'benchmarks_v2'"
        )
        names = {row[0] for row in cur.fetchall()}
        assert {
            "runs",
            "results",
            "benchmark_observations",
            "observation_artifacts",
            "preprocessing_artifacts",
            "metric_evaluations",
            "metric_evaluation_inputs",
        } <= names
        assert "preprocessing_jobs" not in names
        assert "metric_evaluation_inputs" in names
        expected_columns = {
            "benchmark_observations": {
                "id",
                "run_id",
                "dataset_id",
                "dataset_sha256",
                "sample_id",
                "provider",
                "model",
                "voice",
                "benchmark",
                "source_kind",
                "transport_protocol",
                "submit_to_headers_ms",
                "provider_extras",
                "captured_at",
                "status",
                "error",
                "failure_origin",
            },
            "preprocessing_artifacts": {
                "id",
                "observation_id",
                "pipeline",
                "pipeline_version",
                "artifact_name",
                "schema_name",
                "schema_version",
                "producer_name",
                "producer_provider",
                "producer_model",
                "producer_version",
                "gcs_uri",
                "content_sha256",
                "created_at",
            },
            "observation_artifacts": {
                "id",
                "observation_id",
                "artifact_type",
                "schema_name",
                "schema_version",
                "gcs_uri",
                "content_sha256",
                "size_bytes",
                "duration_ms",
                "created_at",
            },
            "metric_evaluations": {
                "id",
                "observation_id",
                "metric_type",
                "metric_version",
                "evaluation_variant",
                "executor",
                "external_request_id",
                "status",
                "started_at",
                "finished_at",
                "error",
                "created_at",
                "updated_at",
            },
            "metric_values": {
                "metric_evaluation_id",
                "value_key",
                "unit",
                "value",
                "value_role",
            },
            "metric_artifacts": {
                "id",
                "metric_evaluation_id",
                "artifact_type",
                "uri",
                "sha256",
                "size_bytes",
                "created_at",
            },
            "metric_evaluation_inputs": {
                "metric_evaluation_id",
                "observation_artifact_id",
                "preprocessing_artifact_id",
                "input_role",
                "input_order",
            },
            "metric_values_by_bucket": {
                "provider",
                "model",
                "benchmark",
                "dataset_id",
                "metric_type",
                "metric_version",
                "evaluation_variant",
                "value_key",
                "unit",
                "bucket_at",
                "min_value",
                "p25",
                "p50",
                "p75",
                "max_value",
                "value_sum",
                "sample_count",
            },
        }
        for table_name, expected in expected_columns.items():
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'benchmarks_v2' AND table_name = %s",
                (table_name,),
            )
            assert {row[0] for row in cur.fetchall()} == expected
        cur.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'benchmarks_v2' AND data_type = 'boolean'"
        )
        normalized_tables = set(expected_columns)
        assert not {(row[0], row[1]) for row in cur.fetchall() if row[0] in normalized_tables}
        cur.execute(
            "SELECT indexdef FROM pg_indexes "
            "WHERE schemaname = 'benchmarks_v2' "
            "AND indexname = 'metric_values_by_bucket_bucket_at'"
        )
        assert _required(cur.fetchone())[0].endswith(
            "ON benchmarks_v2.metric_values_by_bucket USING btree (bucket_at)"
        )
    config = AlembicConfig(str(_INI_PATH))
    config.set_main_option(
        "sqlalchemy.url", _dsn(pg_conn).replace("postgresql://", "postgresql+psycopg://")
    )
    alembic_command.downgrade(config, "20260812_0017")
    with pg_conn.cursor() as cur:
        for table_name in (
            "benchmark_observations",
            "observation_artifacts",
            "preprocessing_artifacts",
            "metric_evaluations",
            "metric_evaluation_inputs",
            "metric_values",
            "metric_artifacts",
            "metric_values_by_bucket",
        ):
            cur.execute("SELECT to_regclass(%s)", (f"benchmarks_v2.{table_name}",))
            assert cur.fetchone() == (None,)
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'benchmarks_v2' AND table_name = 'runs' "
            "AND column_name = 'persona_id'"
        )
        assert cur.fetchone() == ("persona_id",)
        cur.execute("SELECT to_regclass('benchmarks_v2.results')")
        assert cur.fetchone() == ("benchmarks_v2.results",)


def test_observation_contract_and_independent_dataset_identity(
    pg_conn: psycopg.Connection[Any],
) -> None:
    _migrate(pg_conn)
    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.runs (runner_sha, dataset_id, dataset_sha256, status)
               VALUES ('sha', 'run-dataset', %s, 'running') RETURNING id""",
            (_SHA,),
        )
        run_id = _required(cur.fetchone())[0]
        cur.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
               (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark,
                source_kind, status)
               VALUES (%s, 'different-tts-dataset', %s, 'valid', 'p', 'm', 'TTS',
                'generated_audio', 'succeeded')""",
            (run_id, "b" * 64),
        )
        cur.execute(
            """INSERT INTO benchmarks_v2.benchmark_observations
               (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark,
                source_kind, status, error, failure_origin)
               VALUES (%s, 'dataset', %s, 'failed-valid', 'p', 'm', 'STT',
                'dataset_audio', 'failed', 'provider error', 'provider')""",
            (run_id, _SHA),
        )
        for sample, status, error, origin in (
            ("succeeded-origin", "succeeded", None, "runner"),
            ("failed-no-origin", "failed", "runner error", None),
        ):
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    """INSERT INTO benchmarks_v2.benchmark_observations
                       (run_id, dataset_id, dataset_sha256, sample_id, provider, model,
                        benchmark, source_kind, status, error, failure_origin)
                       VALUES (%s, 'dataset', %s, %s, 'p', 'm', 'STT', 'dataset_audio',
                        %s, %s, %s)""",
                    (run_id, _SHA, sample, status, error, origin),
                )
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """INSERT INTO benchmarks_v2.benchmark_observations
                   (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark,
                    source_kind, provider_extras, status)
                   VALUES (%s, 'dataset', %s, 'partial-audio', 'p', 'm', 'STT',
                    'dataset_audio', '[]'::jsonb, 'succeeded')""",
                (run_id, _SHA),
            )
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """INSERT INTO benchmarks_v2.benchmark_observations
               (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark,
                source_kind, status)
               VALUES (%s, 'dataset', %s, 'future-source', 'p', 'm', 'STT',
                'future_audio_source', 'succeeded')""",
                (run_id, _SHA),
            )
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """INSERT INTO benchmarks_v2.benchmark_observations
                   (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark,
                    source_kind, status)
                   VALUES (%s, 'dataset', %s, 'lower', 'p', 'm', 'stt',
                    'dataset_audio', 'succeeded')""",
                (run_id, _SHA),
            )


def test_database_uri_checks_require_bucket_and_object(pg_conn: psycopg.Connection[Any]) -> None:
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
               VALUES (%s, 'dataset', %s, 'valid-uri', 'p', 'm', 'STT', 'dataset_audio',
                'succeeded') RETURNING id""",
            (run_id, _SHA),
        )
        observation_id = _required(cur.fetchone())[0]
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """INSERT INTO benchmarks_v2.preprocessing_artifacts
                   (observation_id, pipeline, pipeline_version, artifact_name, schema_name,
                    schema_version, producer_name, producer_provider, producer_model,
                    producer_version, gcs_uri, content_sha256)
                   VALUES (%s, 'align', 'v1', 'word_timestamps', 'WordTimestampsV1', 'v1',
                    'word_aligner', 'google', 'latest', 'words-v1', 'gs://private', %s)""",
                (observation_id, _SHA),
            )
        cur.execute(
            """INSERT INTO benchmarks_v2.preprocessing_artifacts
               (observation_id, pipeline, pipeline_version, artifact_name, schema_name,
                schema_version, producer_name, producer_provider, producer_model,
                producer_version, gcs_uri, content_sha256)
               VALUES (%s, 'align', 'v1', 'future_artifact', 'FutureArtifactV2', 'v2',
                'word_aligner', 'google', 'latest', 'words-v1', 'gs://private/words', %s)""",
            (observation_id, _SHA),
        )
        with pg_conn.cursor(row_factory=psycopg.rows.dict_row) as dict_cur:
            dict_cur.execute(
                """SELECT id, observation_id, pipeline, pipeline_version, artifact_name,
                          schema_name, schema_version, producer_name, producer_provider,
                          producer_model, producer_version, gcs_uri, content_sha256, created_at
                   FROM benchmarks_v2.preprocessing_artifacts
                   WHERE observation_id = %s AND artifact_name = 'future_artifact'""",
                (observation_id,),
            )
            future_artifact = PreprocessingArtifact.model_validate(
                dict(_required(dict_cur.fetchone()))
            )
        assert future_artifact.artifact_name == "future_artifact"
        assert future_artifact.schema_name == "FutureArtifactV2"
        assert future_artifact.schema_version == "v2"
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations
               (observation_id, metric_type, metric_version, executor, status)
               VALUES (%s, 'TTFT', 'v1', 'inline', 'queued') RETURNING id""",
            (observation_id,),
        )
        evaluation_id = _required(cur.fetchone())[0]
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """INSERT INTO benchmarks_v2.metric_artifacts
                   (metric_evaluation_id, artifact_type, uri, sha256, size_bytes)
                   VALUES (%s, 'details', 'gs://private', %s, 1)""",
                (evaluation_id, _SHA),
            )


def test_observation_model_validates_artifacts_and_provider_extras() -> None:
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
    assert Observation(**base).artifacts == []
    artifact = ObservationArtifact(
        artifact_type=ObservationArtifactType.GENERATED_AUDIO,
        schema_name="AudioV1",
        schema_version="v1",
        gcs_uri="gs://private/audio",
        content_sha256=_SHA,
        size_bytes=1,
        duration_ms=1,
    )
    complete = Observation(**base, provider_extras={"provider_flag": True}, artifacts=[artifact])
    assert complete.artifacts[0].duration_ms == 1
    assert complete.provider_extras == {"provider_flag": True}
    with pytest.raises(ValueError, match="cannot repeat"):
        Observation(**base, artifacts=[artifact, artifact])
    with pytest.raises(ValueError, match="private gs:// object URI"):
        ObservationArtifact(
            artifact_type=ObservationArtifactType.GENERATED_AUDIO,
            schema_name="AudioV1",
            schema_version="v1",
            gcs_uri="gs://private",
            content_sha256=_SHA,
            size_bytes=1,
        )
    with pytest.raises(ValueError, match="failure_origin"):
        Observation(**(base | {"status": ObservationStatus.FAILED}), error="provider error")
    failed = Observation(
        **(base | {"status": ObservationStatus.FAILED}),
        error="provider error",
        failure_origin=ObservationFailureOrigin.PROVIDER,
    )
    assert failed.failure_origin is ObservationFailureOrigin.PROVIDER


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


def test_metric_input_models_freeze_one_tagged_artifact_kind() -> None:
    artifact_id = uuid4()
    raw = MetricEvaluationInput(
        observation_artifact_id=artifact_id, input_role="raw", input_order=0
    )
    preprocessed = MetricEvaluationInput(
        preprocessing_artifact_id=artifact_id, input_role="preprocessed", input_order=0
    )
    assert raw.observation_artifact_id == artifact_id
    assert raw.preprocessing_artifact_id is None
    assert preprocessed.preprocessing_artifact_id == artifact_id
    assert preprocessed.observation_artifact_id is None
    with pytest.raises(ValueError, match="exactly one"):
        MetricEvaluationInput(input_role="missing", input_order=0)
    with pytest.raises(ValueError, match="exactly one"):
        MetricEvaluationInput(
            observation_artifact_id=uuid4(),
            preprocessing_artifact_id=uuid4(),
            input_role="both",
            input_order=0,
        )
    with pytest.raises(ValueError):
        ProcessingStatus("partial")
    assert RunStatus.PARTIAL.value == "partial"


def test_nested_preprocessing_artifact_update_is_rejected(
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
               VALUES (%s, 'dataset', %s, 'nested-update', 'p', 'm', 'STT',
                'dataset_audio', 'succeeded') RETURNING id""",
            (run_id, _SHA),
        )
        observation_id = _required(cur.fetchone())[0]
        cur.execute(
            """INSERT INTO benchmarks_v2.preprocessing_artifacts
               (observation_id, pipeline, pipeline_version, artifact_name, schema_name,
                schema_version, producer_name, producer_provider, producer_model,
                producer_version, gcs_uri, content_sha256)
               VALUES (%s, 'align', 'v1', 'word_timestamps', 'WordTimestampsV1', 'v1',
                'word_aligner', 'google', 'latest', 'words-v1', 'gs://private/words', %s)""",
            (observation_id, _SHA),
        )
        cur.execute(
            """CREATE FUNCTION benchmarks_v2.update_preprocessing_artifact_from_observation()
               RETURNS trigger AS $$
               BEGIN
                   UPDATE benchmarks_v2.preprocessing_artifacts
                   SET producer_version = producer_version || '-changed'
                   WHERE observation_id = NEW.id;
                   RETURN NEW;
               END; $$ LANGUAGE plpgsql"""
        )
        cur.execute(
            """CREATE TRIGGER observation_updates_preprocessing_artifact
               AFTER UPDATE OF model ON benchmarks_v2.benchmark_observations
               FOR EACH ROW EXECUTE FUNCTION
               benchmarks_v2.update_preprocessing_artifact_from_observation()"""
        )
        with pytest.raises(psycopg.errors.RaiseException, match="artifacts are immutable"):
            cur.execute(
                "UPDATE benchmarks_v2.benchmark_observations SET model = 'updated' WHERE id = %s",
                (observation_id,),
            )
        cur.execute(
            "SELECT model FROM benchmarks_v2.benchmark_observations WHERE id = %s",
            (observation_id,),
        )
        assert _required(cur.fetchone())[0] == "m"
        cur.execute(
            "SELECT producer_version FROM benchmarks_v2.preprocessing_artifacts "
            "WHERE observation_id = %s",
            (observation_id,),
        )
        assert _required(cur.fetchone())[0] == "words-v1"


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
        await writer.fail_metric_evaluation(
            evaluation_id, finished_at=_NOW, error="controlled failure"
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
        started = await writer.start_metric_evaluation(_required(ensemble.id), started_at=_NOW)
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
            running = await writer.start_metric_evaluation(
                _required(variants[variant].id), started_at=_NOW
            )
            await writer.complete_metric_evaluation(
                _required(running.id),
                values=_wer_values(_required(running.id)),
                finished_at=finished,
            )
        await writer.finish_run(_required(observation.run_id), status=RunStatus.SUCCEEDED)
        await writer.refresh_metric_values_bucket(_required(observation.run_id))
        await writer.refresh_metric_values_bucket(_required(observation.run_id))
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """SELECT dataset_id, evaluation_variant, sample_count
                   FROM benchmarks_v2.metric_values_by_bucket WHERE value_key = 'primary'"""
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
        await writer.start_metric_evaluation(input_first_id, started_at=_NOW)

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
        failed_evaluation = await writer.fail_metric_evaluation(
            evaluation_id, finished_at=_NOW, error="request failed"
        )
        assert failed_evaluation.started_at == failed_evaluation.finished_at == _NOW
        with pytest.raises(ValueError, match=str(evaluation_id)):
            await writer.start_metric_evaluation(evaluation_id, started_at=_NOW)
        with pytest.raises(ValueError, match=str(evaluation_id)):
            await writer.fail_metric_evaluation(
                evaluation_id, finished_at=_NOW, error="retry failed"
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
                   (observation_id, metric_type, metric_version, executor, status)
                   VALUES (%s, 'WER', 'v1', 'inline', 'partial')""",
                (observation_id,),
            )
        with pytest.raises(psycopg.errors.RaiseException, match="created queued"):
            cur.execute(
                """INSERT INTO benchmarks_v2.metric_evaluations
                   (observation_id, metric_type, metric_version, executor, status, started_at)
                   VALUES (%s, 'WER', 'v1', 'inline', 'running', now())""",
                (observation_id,),
            )
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations
               (observation_id, metric_type, metric_version, executor, status)
               VALUES (%s, 'WER', 'v1', 'inline', 'queued') RETURNING id""",
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
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations
               (observation_id, metric_type, metric_version, evaluation_variant, executor, status)
               VALUES (%s, 'WER', 'v1', 'constraint-checks', 'inline', 'queued') RETURNING id""",
            (observation_id,),
        )
        constraint_evaluation_id = _required(cur.fetchone())[0]
        cur.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET status = 'running', started_at = now() WHERE id = %s""",
            (constraint_evaluation_id,),
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """INSERT INTO benchmarks_v2.metric_values
                   (metric_evaluation_id, value_key, unit, value, value_role)
                   VALUES (%s, 'invalid', 'percent', 1, 'unsupported')""",
                (constraint_evaluation_id,),
            )
        cur.execute("BEGIN")
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_values
               (metric_evaluation_id, value_key, unit, value, value_role)
               VALUES (%s, 'primary', 'percent', 1, 'primary')""",
            (constraint_evaluation_id,),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(
                """INSERT INTO benchmarks_v2.metric_values
                   (metric_evaluation_id, value_key, unit, value, value_role)
                   VALUES (%s, 'second-primary', 'percent', 1, 'primary')""",
                (constraint_evaluation_id,),
            )
        cur.execute("ROLLBACK")


def test_database_success_validation_is_metric_agnostic(
    pg_conn: psycopg.Connection[Any],
) -> None:
    """SQL enforces generic output integrity without freezing application metric options."""
    _migrate(pg_conn)
    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute(
            """INSERT INTO benchmarks_v2.runs (runner_sha, dataset_id, dataset_sha256, status)
               VALUES ('sha', 'dataset', %s, 'running') RETURNING id""",
            (_SHA,),
        )
        run_id = _required(cur.fetchone())[0]

        def new_observation_id() -> Any:
            cur.execute(
                """INSERT INTO benchmarks_v2.benchmark_observations
                   (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark,
                    source_kind, status)
                   VALUES (%s, 'dataset', %s, %s, 'p', 'm', 'STT', 'dataset_audio',
                    'succeeded') RETURNING id""",
                (run_id, _SHA, str(uuid4())),
            )
            return _required(cur.fetchone())[0]

        def complete_direct(
            metric: str, version: str, values: tuple[tuple[str, str, float, MetricValueRole], ...]
        ) -> None:
            cur.execute("BEGIN")
            observation_id = new_observation_id()
            cur.execute(
                """INSERT INTO benchmarks_v2.metric_evaluations
                   (observation_id, metric_type, metric_version, executor, status)
                   VALUES (%s, %s, %s, 'inline', 'queued') RETURNING id""",
                (observation_id, metric, version),
            )
            evaluation_id = _required(cur.fetchone())[0]
            cur.execute(
                """UPDATE benchmarks_v2.metric_evaluations
                   SET status = 'running', started_at = now() WHERE id = %s""",
                (evaluation_id,),
            )
            cur.executemany(
                """INSERT INTO benchmarks_v2.metric_values
                   (metric_evaluation_id, value_key, unit, value, value_role)
                   VALUES (%s, %s, %s, %s, %s)""",
                [(evaluation_id, *value) for value in values],
            )
            cur.execute(
                """UPDATE benchmarks_v2.metric_evaluations
                   SET status = 'succeeded', finished_at = now() WHERE id = %s""",
                (evaluation_id,),
            )
            cur.execute("COMMIT")

        complete_direct(
            "FutureMetric", "v9", (("score", "custom_unit", 1.0, MetricValueRole.PRIMARY),)
        )

        with pytest.raises(psycopg.errors.RaiseException, match="exactly one primary"):
            complete_direct(
                "FutureMetric", "v9", (("score", "custom_unit", 1.0, MetricValueRole.COMPONENT),)
            )
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
        await writer.finish_run(run_id, status=RunStatus.SUCCEEDED)
        await writer.refresh_metric_values_bucket(run_id)
        await writer.refresh_metric_values_bucket(run_id)
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT dataset_id FROM benchmarks_v2.metric_values_by_bucket ORDER BY dataset_id"
            )
            datasets = [row["dataset_id"] for row in await cur.fetchall()]
            assert datasets.count("__all__") == datasets.count("observation-dataset") == 4
            await cur.execute(
                "DELETE FROM benchmarks_v2.benchmark_observations WHERE id = %s",
                (_required(observation.id),),
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
        evaluation = await writer.start_metric_evaluation(
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


def test_dashboard_read_indexes_migrate_and_downgrade(pg_conn: psycopg.Connection[Any]) -> None:
    """The dashboard indexes are additive and retain the bucket-writer index."""
    _migrate(pg_conn, "20260818_0018")
    config = AlembicConfig(str(_INI_PATH))
    config.set_main_option(
        "sqlalchemy.url", _dsn(pg_conn).replace("postgresql://", "postgresql+psycopg://")
    )
    pg_conn.autocommit = True

    def index_names() -> set[str]:
        with pg_conn.cursor() as cur:
            cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'benchmarks_v2'")
            return {row[0] for row in cur.fetchall()}

    new_indexes = {
        "benchmark_observations_recent_results_idx",
        "metric_values_by_bucket_series_idx",
    }
    assert not new_indexes & index_names()
    assert "metric_values_by_bucket_bucket_at" in index_names()

    alembic_command.upgrade(config, "20260824_0019")
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexrel.relname,
                   array_agg(attribute.attname ORDER BY key.ordinality),
                   pg_get_expr(index_data.indpred, index_data.indrelid)
            FROM pg_index AS index_data
            JOIN pg_class AS indexrel ON indexrel.oid = index_data.indexrelid
            JOIN pg_class AS table_rel ON table_rel.oid = index_data.indrelid
            JOIN pg_namespace AS namespace ON namespace.oid = table_rel.relnamespace
            CROSS JOIN LATERAL unnest(index_data.indkey) WITH ORDINALITY AS key(attnum, ordinality)
            JOIN pg_attribute AS attribute
              ON attribute.attrelid = table_rel.oid AND attribute.attnum = key.attnum
            WHERE namespace.nspname = 'benchmarks_v2'
              AND indexrel.relname IN (%s, %s)
            GROUP BY indexrel.relname, index_data.indpred, index_data.indrelid
            """,
            tuple(sorted(new_indexes)),
        )
        indexes = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    assert indexes == {
        "benchmark_observations_recent_results_idx": (
            ["benchmark", "dataset_id", "captured_at", "id"],
            "(status = 'succeeded'::text)",
        ),
        "metric_values_by_bucket_series_idx": (
            [
                "benchmark",
                "dataset_id",
                "metric_version",
                "evaluation_variant",
                "value_key",
                "bucket_at",
            ],
            None,
        ),
    }
    assert "metric_values_by_bucket_bucket_at" in index_names()

    alembic_command.downgrade(config, "20260818_0018")
    assert not new_indexes & index_names()
    assert "metric_values_by_bucket_bucket_at" in index_names()


def test_migration_conditionally_revokes_api_access(pg_conn: psycopg.Connection[Any]) -> None:
    _migrate(pg_conn, "20260807_0015")
    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute("CREATE ROLE api")
        cur.execute("ALTER DEFAULT PRIVILEGES GRANT SELECT ON TABLES TO api")
    try:
        _migrate(pg_conn)
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT has_table_privilege("
                "'api', 'benchmarks_v2.benchmark_observations', 'SELECT')"
            )
            assert cur.fetchone() == (False,)
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("DROP OWNED BY api")
            cur.execute("DROP ROLE api")


def test_output_writers_are_not_public() -> None:
    assert hasattr(RunWriter, "insert_preprocessing_artifact")
    assert not hasattr(RunWriter, "insert_metric_values")
    assert not hasattr(RunWriter, "insert_metric_artifacts")
