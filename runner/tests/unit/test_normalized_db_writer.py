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
    MetricExecutor,
    MetricValue,
    Observation,
    ObservationSourceKind,
    ObservationStatus,
    PreprocessingArtifact,
    ProcessingStatus,
    RunStatus,
)
from coval_bench.db.writer import RunWriter
from coval_bench.registries import METRIC_VALUE_CONTRACTS, Metric, validate_metric_values

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
        producer_version="words-v1",
        gcs_uri="gs://private/words",
        content_sha256=sha,
        size_bytes=10,
        duration_ms=100.0,
    )


def _phone_artifact(observation_id: Any) -> PreprocessingArtifact:
    return PreprocessingArtifact(
        observation_id=observation_id,
        pipeline="align",
        pipeline_version="v1",
        artifact_name="phoneme_timestamps",
        schema_name="PhonemeTimestampsV1",
        producer_name="phoneme_aligner",
        producer_version="phones-v1",
        gcs_uri="gs://private/phones",
        content_sha256="c" * 64,
        size_bytes=20,
    )


def _wer_values(evaluation_id: Any) -> list[MetricValue]:
    return [
        MetricValue(
            metric_evaluation_id=evaluation_id,
            value_key="primary",
            unit="percent",
            value=10,
            is_primary=True,
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
            "preprocessing_artifacts",
            "metric_evaluations",
        } <= names
        assert {"preprocessing_jobs", "metric_evaluation_inputs"}.isdisjoint(names)
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
                "audio_filename",
                "transport_protocol",
                "submit_to_headers_ms",
                "audio_uri",
                "audio_sha256",
                "audio_size_bytes",
                "audio_duration_ms",
                "captured_at",
                "status",
                "error",
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
                "producer_version",
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
                "is_primary",
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
            "metric_values_by_bucket": {
                "provider",
                "model",
                "benchmark",
                "dataset_id",
                "metric_type",
                "metric_version",
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
            "preprocessing_artifacts",
            "metric_evaluations",
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
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """INSERT INTO benchmarks_v2.benchmark_observations
                   (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark,
                    source_kind, audio_uri, status)
                   VALUES (%s, 'dataset', %s, 'partial-audio', 'p', 'm', 'STT',
                    'dataset_audio', 'gs://private/audio', 'succeeded')""",
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
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """INSERT INTO benchmarks_v2.benchmark_observations
                   (run_id, dataset_id, dataset_sha256, sample_id, provider, model, benchmark,
                    source_kind, audio_uri, audio_sha256, audio_size_bytes, audio_duration_ms,
                    status)
                   VALUES (%s, 'dataset', %s, 'invalid-uri', 'p', 'm', 'STT', 'dataset_audio',
                    'gs://private', %s, 1, 1, 'succeeded')""",
                (run_id, _SHA, _SHA),
            )
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
                    schema_version, producer_name, producer_version, gcs_uri, content_sha256,
                    size_bytes)
                   VALUES (%s, 'align', 'v1', 'word_timestamps', 'WordTimestampsV1', 'v1',
                    'word_aligner', 'words-v1', 'gs://private', %s, 1)""",
                (observation_id, _SHA),
            )
        cur.execute(
            """INSERT INTO benchmarks_v2.preprocessing_artifacts
               (observation_id, pipeline, pipeline_version, artifact_name, schema_name,
                schema_version, producer_name, producer_version, gcs_uri, content_sha256,
                size_bytes)
               VALUES (%s, 'align', 'v1', 'word_timestamps', 'WordTimestampsV1', 'v1',
                'word_aligner', 'words-v1', 'gs://private/words', %s, 1)""",
            (observation_id, _SHA),
        )
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


def test_observation_model_requires_complete_audio_tuple() -> None:
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
    assert Observation(**base).audio_uri is None
    with pytest.raises(ValueError, match="audio URI"):
        Observation(**base, audio_uri="gs://private/audio")
    complete = Observation(
        **base,
        audio_uri="gs://private/audio",
        audio_sha256=_SHA,
        audio_size_bytes=1,
        audio_duration_ms=1,
    )
    assert complete.audio_duration_ms == 1
    with pytest.raises(ValueError, match="private gs:// object URI"):
        Observation(
            **base,
            audio_uri="gs://private",
            audio_sha256=_SHA,
            audio_size_bytes=1,
            audio_duration_ms=1,
        )


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
                schema_version, producer_name, producer_version, gcs_uri, content_sha256,
                size_bytes)
               VALUES (%s, 'align', 'v1', 'word_timestamps', 'WordTimestampsV1', 'v1',
                'word_aligner', 'words-v1', 'gs://private/words', %s, 1)""",
            (observation_id, _SHA),
        )
        cur.execute(
            """CREATE FUNCTION benchmarks_v2.update_preprocessing_artifact_from_observation()
               RETURNS trigger AS $$
               BEGIN
                   UPDATE benchmarks_v2.preprocessing_artifacts
                   SET size_bytes = size_bytes + 1 WHERE observation_id = NEW.id;
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
            "SELECT size_bytes FROM benchmarks_v2.preprocessing_artifacts "
            "WHERE observation_id = %s",
            (observation_id,),
        )
        assert _required(cur.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_create_get_is_retry_safe_and_strict(pg_conn: psycopg.Connection[Any]) -> None:
    _migrate(pg_conn)
    pool = await _pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await _observation(writer)
        duplicate = await writer.insert_observation(observation.model_copy(update={"id": None}))
        assert duplicate.id == observation.id
        with pytest.raises(ValueError, match="dataset_sha256"):
            await writer.insert_observation(
                observation.model_copy(update={"id": None, "dataset_sha256": "d" * 64})
            )

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
                        is_primary=True,
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
                        is_primary=True,
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
            ("primary", "percent", 3, True),
            ("insertions", "percent", 1, False),
            ("deletions", "percent", 1, False),
            ("substitutions", "percent", 1, False),
        ),
    )
    validate_metric_values(Metric.TTFA, "v1", (("primary", "milliseconds", 12, True),))
    validate_metric_values(
        Metric.TTFA,
        "v1",
        (
            ("primary", "milliseconds", 12, True),
            ("roundtrip", "milliseconds", 10, False),
            ("leading_silence", "milliseconds", 2, False),
        ),
    )
    with pytest.raises(ValueError, match="optional metric value group"):
        validate_metric_values(
            Metric.TTFA,
            "v1",
            (("primary", "milliseconds", 12, True), ("roundtrip", "milliseconds", 10, False)),
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
        cur.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET status = 'running', started_at = now() WHERE id = %s""",
            (evaluation_id,),
        )
        cur.execute("BEGIN")
        cur.execute(
            """INSERT INTO benchmarks_v2.metric_values
               (metric_evaluation_id, value_key, unit, value, is_primary)
               VALUES (%s, 'primary', 'percent', 1, true)""",
            (evaluation_id,),
        )
        cur.execute(
            """UPDATE benchmarks_v2.metric_evaluations
               SET status = 'succeeded', finished_at = now() WHERE id = %s""",
            (evaluation_id,),
        )
        with pytest.raises(psycopg.errors.RaiseException, match="all components"):
            cur.execute("COMMIT")
        cur.execute("ROLLBACK")


def test_database_metric_contracts_match_python_validation(
    pg_conn: psycopg.Connection[Any],
) -> None:
    """Every v1 registry contract succeeds through independent deferred SQL validation."""
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
        _required(cur.fetchone())

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
            metric: str, version: str, values: tuple[tuple[str, str, float, bool], ...]
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
                   (metric_evaluation_id, value_key, unit, value, is_primary)
                   VALUES (%s, %s, %s, %s, %s)""",
                [(evaluation_id, *value) for value in values],
            )
            cur.execute(
                """UPDATE benchmarks_v2.metric_evaluations
                   SET status = 'succeeded', finished_at = now() WHERE id = %s""",
                (evaluation_id,),
            )
            cur.execute("COMMIT")

        for (registry_metric, version), contract in METRIC_VALUE_CONTRACTS.items():
            if version != "v1":
                continue
            values = tuple(
                (definition.key, definition.unit, 0.0, definition.primary)
                for definition in contract.values
                if definition.required
            )
            validate_metric_values(registry_metric, version, values)
            complete_direct(str(registry_metric), version, values)

        for metric_type, version, values in (
            (str(Metric.TTFT), "v2", (("primary", "seconds", 0.0, True),)),
            (str(Metric.TTFT), "v1", (("primary", "milliseconds", 0.0, True),)),
            (str(Metric.TTFT), "v1", (("primary", "seconds", -1.0, True),)),
            (str(Metric.INSTRUCTION_FOLLOWING), "v1", (("primary", "percent", 101.0, True),)),
        ):
            with pytest.raises((psycopg.errors.RaiseException, psycopg.errors.CheckViolation)):
                complete_direct(metric_type, version, values)
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
        await writer.refresh_metric_values_bucket(run_id, period_seconds=1800)
        await writer.refresh_metric_values_bucket(run_id, period_seconds=1800)
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
        observation_id = _required(observation.id)
        queued = await writer.insert_metric_evaluation(
            MetricEvaluation(
                observation_id=observation_id,
                metric_type=str(Metric.TTFT),
                metric_version="v1",
                executor=MetricExecutor.INLINE,
                status=ProcessingStatus.QUEUED,
            )
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
            await conn.commit()
        evaluation = await _evaluation(writer, observation)
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
        await writer.insert_preprocessing_artifact(_word_artifact(observation_id))
        async with pool.connection() as conn, conn.cursor() as cur:
            with pytest.raises(psycopg.errors.RaiseException, match="terminal work rows"):
                await cur.execute(
                    "DELETE FROM benchmarks_v2.metric_evaluations WHERE id = %s", (evaluation_id,)
                )
            await conn.rollback()
            await cur.execute("DELETE FROM benchmarks_v2.runs WHERE id = %s", (run_id,))
            await cur.execute("SELECT count(*) AS count FROM benchmarks_v2.preprocessing_artifacts")
            assert _required(await cur.fetchone())["count"] == 0
            await cur.execute("SELECT count(*) AS count FROM benchmarks_v2.metric_values")
            assert _required(await cur.fetchone())["count"] == 0
            await cur.execute("SELECT count(*) AS count FROM benchmarks_v2.metric_artifacts")
            assert _required(await cur.fetchone())["count"] == 0
            await conn.commit()
    finally:
        await pool.close()


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
