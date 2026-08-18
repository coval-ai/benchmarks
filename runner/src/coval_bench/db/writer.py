# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""``RunWriter`` — typed insert helpers for the benchmark persistence layer.

All SQL uses parameterised queries (psycopg ``%s`` style).  No string
interpolation with user data is performed anywhere in this module.

Transaction semantics
---------------------
``record_results`` inserts all rows in a single transaction.  If any single
insert fails (e.g. a check-constraint violation) the entire batch is rolled
back and the exception propagates to the caller.  The orchestrator is
responsible for retry logic.

``record_result`` (singular) delegates to ``record_results`` and shares the
same single-transaction guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

import psycopg
import psycopg.rows
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from coval_bench.db.models import (
    MetricArtifact,
    MetricEvaluation,
    MetricEvaluationInput,
    MetricValue,
    Observation,
    ObservationArtifact,
    PreprocessingArtifact,
    ProcessingStatus,
    Result,
    Run,
    RunStatus,
)
from coval_bench.registries import (
    SERIES_EXCLUDED_METRICS,
    Metric,
    validate_metric_contract,
    validate_metric_values,
    validate_preprocessing_artifact_contract,
)

STATS_MATVIEWS: tuple[str, ...] = ("results_24h", "results_7d", "results_30d")


class RunWriter:
    """Per-run persistence helper.

    Lifecycle::

        writer = RunWriter(pool)
        run = await writer.start_run(runner_sha=..., dataset_id=..., dataset_sha256=...)
        await writer.record_result(result)
        await writer.record_results([result1, result2, ...])
        await writer.finish_run(run.id, status=RunStatus.SUCCEEDED)

    All methods raise on error; exceptions are never swallowed.
    """

    def __init__(
        self,
        pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]],
    ) -> None:
        self._pool = pool

    async def start_run(
        self,
        *,
        runner_sha: str,
        dataset_id: str,
        dataset_sha256: str,
        scheduled_at: datetime | None = None,
        persona_id: str | None = None,
    ) -> Run:
        """Insert a ``running`` row into ``benchmarks_v2.runs``.

        Returns a ``Run`` with ``id`` and ``started_at`` populated from the DB.
        """
        sql = """
            INSERT INTO benchmarks_v2.runs
                (runner_sha, dataset_id, dataset_sha256, status, scheduled_at, persona_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, started_at, finished_at, scheduled_at, runner_sha,
                      dataset_id, dataset_sha256, status, error, persona_id
        """
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    sql,
                    (
                        runner_sha,
                        dataset_id,
                        dataset_sha256,
                        RunStatus.RUNNING,
                        scheduled_at,
                        persona_id,
                    ),
                )
                row = await cur.fetchone()
                if row is None:  # pragma: no cover — unreachable after INSERT RETURNING
                    raise RuntimeError("INSERT INTO runs returned no row")
            await conn.commit()
        return Run.model_validate(dict(row))

    async def record_result(self, result: Result) -> None:
        """Insert a single ``benchmarks_v2.results`` row in its own transaction."""
        await self.record_results([result])

    async def record_results(self, results: Sequence[Result]) -> None:
        """Batch-insert ``results`` in a single transaction.

        All rows are inserted via ``executemany``.  If any row fails (e.g. a
        check-constraint violation), the whole batch is rolled back and the
        exception propagates.  The caller decides whether to retry.

        Every ``metric_type`` must be a known ``Metric`` value; an unknown
        value rejects the whole batch before any SQL is executed.
        """
        if not results:
            return

        for r in results:
            try:
                Metric(r.metric_type)
            except ValueError as exc:
                raise ValueError(
                    f"unknown metric_type {r.metric_type!r} (run_id={r.run_id}); "
                    "expected a coval_bench.registries.Metric value"
                ) from exc

        sql = """
            INSERT INTO benchmarks_v2.results
                (run_id, provider, model, voice, benchmark, metric_type,
                 metric_value, metric_units, audio_filename, transcript,
                 status, error, http_version, submit_to_headers_ms,
                 wer_insertions_pct, wer_deletions_pct, wer_substitutions_pct)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s)
        """
        params = [
            (
                r.run_id,
                r.provider,
                r.model,
                r.voice,
                r.benchmark,
                r.metric_type,
                r.metric_value,
                r.metric_units,
                r.audio_filename,
                r.transcript,
                r.status,
                r.error,
                r.http_version,
                r.submit_to_headers_ms,
                r.wer_insertions_pct,
                r.wer_deletions_pct,
                r.wer_substitutions_pct,
            )
            for r in results
        ]

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(sql, params)
            await conn.commit()

    async def insert_observation(self, observation: Observation) -> Observation:
        """Create or retrieve an observation, rejecting conflicting retries."""
        sql = """
            INSERT INTO benchmarks_v2.benchmark_observations
            (run_id, dataset_id, dataset_sha256, sample_id, provider, model, voice,
             benchmark, source_kind, transport_protocol, submit_to_headers_ms,
             provider_extras, captured_at, status, error, failure_origin)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, COALESCE(%s, now()), %s, %s, %s)
            ON CONFLICT (run_id, sample_id, provider, model, voice)
            DO NOTHING
            RETURNING id, run_id, dataset_id, dataset_sha256, sample_id, provider, model,
                      voice, benchmark, source_kind, transport_protocol, submit_to_headers_ms,
                      provider_extras, captured_at, status, error, failure_origin
        """
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    sql,
                    (
                        observation.run_id,
                        observation.dataset_id,
                        observation.dataset_sha256,
                        observation.sample_id,
                        observation.provider,
                        observation.model,
                        observation.voice,
                        observation.benchmark,
                        observation.source_kind,
                        observation.transport_protocol,
                        observation.submit_to_headers_ms,
                        Jsonb(observation.provider_extras)
                        if observation.provider_extras is not None
                        else None,
                        observation.captured_at,
                        observation.status,
                        observation.error,
                        observation.failure_origin,
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    await cur.execute(
                        """SELECT id, run_id, dataset_id, dataset_sha256, sample_id,
                                  provider, model,
                                  voice, benchmark, source_kind, transport_protocol,
                                  submit_to_headers_ms, provider_extras, captured_at, status, error,
                                  failure_origin
                           FROM benchmarks_v2.benchmark_observations
                           WHERE run_id = %s AND sample_id = %s AND provider = %s
                             AND model = %s AND voice IS NOT DISTINCT FROM %s""",
                        (
                            observation.run_id,
                            observation.sample_id,
                            observation.provider,
                            observation.model,
                            observation.voice,
                        ),
                    )
                    row = await cur.fetchone()
                if row is None:  # pragma: no cover
                    raise RuntimeError("INSERT INTO benchmark_observations returned no row")
                stored_parent = Observation.model_validate(dict(row))
                compared_fields = observation.model_fields_set - {"id", "artifacts"}
                if observation.captured_at is None:
                    compared_fields.discard("captured_at")
                mismatches = sorted(
                    field
                    for field in compared_fields
                    if getattr(observation, field) != getattr(stored_parent, field)
                )
                if mismatches:
                    raise ValueError(
                        "observation retry conflicts with stored immutable fields: "
                        + ", ".join(mismatches)
                    )
                observation_id = row["id"]
                for artifact in observation.artifacts:
                    if (
                        artifact.observation_id is not None
                        and artifact.observation_id != observation_id
                    ):
                        raise ValueError(
                            "nested artifact observation_id conflicts with observation"
                        )
                    await cur.execute(
                        """INSERT INTO benchmarks_v2.observation_artifacts
                           (observation_id, artifact_type, schema_name, schema_version, gcs_uri,
                            content_sha256, size_bytes, duration_ms)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (observation_id, artifact_type) DO NOTHING
                           RETURNING id, observation_id, artifact_type, schema_name,
                                     schema_version, gcs_uri, content_sha256, size_bytes,
                                     duration_ms, created_at""",
                        (
                            observation_id,
                            artifact.artifact_type,
                            artifact.schema_name,
                            artifact.schema_version,
                            artifact.gcs_uri,
                            artifact.content_sha256,
                            artifact.size_bytes,
                            artifact.duration_ms,
                        ),
                    )
                    stored_artifact = await cur.fetchone()
                    if stored_artifact is None:
                        await cur.execute(
                            """SELECT id, observation_id, artifact_type, schema_name,
                                      schema_version, gcs_uri, content_sha256, size_bytes,
                                      duration_ms, created_at
                               FROM benchmarks_v2.observation_artifacts
                               WHERE observation_id = %s AND artifact_type = %s""",
                            (observation_id, artifact.artifact_type),
                        )
                        stored_artifact = await cur.fetchone()
                    if stored_artifact is None:  # pragma: no cover
                        raise RuntimeError("INSERT INTO observation_artifacts returned no row")
                    stored_model = ObservationArtifact.model_validate(dict(stored_artifact))
                    artifact_fields = (
                        "artifact_type",
                        "schema_name",
                        "schema_version",
                        "gcs_uri",
                        "content_sha256",
                        "size_bytes",
                        "duration_ms",
                    )
                    mismatches = [
                        field
                        for field in artifact_fields
                        if getattr(artifact, field) != getattr(stored_model, field)
                    ]
                    if mismatches:
                        raise ValueError(
                            "observation artifact retry conflicts with stored immutable fields: "
                            + ", ".join(mismatches)
                        )
                await cur.execute(
                    """SELECT id, observation_id, artifact_type, schema_name, schema_version,
                              gcs_uri, content_sha256, size_bytes, duration_ms, created_at
                       FROM benchmarks_v2.observation_artifacts
                       WHERE observation_id = %s ORDER BY artifact_type""",
                    (observation_id,),
                )
                artifact_rows = await cur.fetchall()
            await conn.commit()
        stored = Observation.model_validate(
            {**dict(row), "artifacts": [dict(item) for item in artifact_rows]}
        )
        return stored

    async def insert_metric_evaluation(
        self, evaluation: MetricEvaluation, *, inputs: Sequence[MetricEvaluationInput] = ()
    ) -> MetricEvaluation:
        """Create a queued evaluation or retrieve the same evaluation on retry."""
        if (
            evaluation.status is not ProcessingStatus.QUEUED
            or evaluation.started_at is not None
            or evaluation.finished_at is not None
            or evaluation.error is not None
        ):
            raise ValueError("metric evaluations must be created queued")
        validate_metric_contract(evaluation.metric_type, evaluation.metric_version)
        input_keys = [(item.input_role, item.input_order) for item in inputs]
        if len(input_keys) != len(set(input_keys)):
            raise ValueError("metric evaluation inputs must have unique role/order pairs")
        artifact_ids = [
            ("observation", item.observation_artifact_id)
            if item.observation_artifact_id is not None
            else ("preprocessing", item.preprocessing_artifact_id)
            for item in inputs
        ]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("metric evaluation inputs must not repeat an artifact")
        sql = """
            INSERT INTO benchmarks_v2.metric_evaluations
            (observation_id, metric_type, metric_version, evaluation_variant, executor,
             external_request_id, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (observation_id, metric_type, metric_version, evaluation_variant)
            DO NOTHING
            RETURNING id, observation_id, metric_type, metric_version,
                      evaluation_variant, executor,
                      external_request_id,
                      status, started_at, finished_at, error, created_at, updated_at
        """
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    sql,
                    (
                        evaluation.observation_id,
                        evaluation.metric_type,
                        evaluation.metric_version,
                        evaluation.evaluation_variant,
                        evaluation.executor,
                        evaluation.external_request_id,
                        evaluation.status,
                    ),
                )
                row = await cur.fetchone()
                created = row is not None
                if row is None:
                    await cur.execute(
                        """SELECT id, observation_id, metric_type, metric_version,
                                  evaluation_variant, executor,
                                  external_request_id, status, started_at, finished_at, error,
                                  created_at, updated_at
                           FROM benchmarks_v2.metric_evaluations
                           WHERE observation_id = %s
                             AND metric_type = %s AND metric_version = %s
                             AND evaluation_variant = %s""",
                        (
                            evaluation.observation_id,
                            evaluation.metric_type,
                            evaluation.metric_version,
                            evaluation.evaluation_variant,
                        ),
                    )
                    row = await cur.fetchone()
                if row is not None:
                    await cur.execute(
                        """SELECT observation_artifact_id, preprocessing_artifact_id,
                                  input_role, input_order
                           FROM benchmarks_v2.metric_evaluation_inputs
                           WHERE metric_evaluation_id = %s ORDER BY input_role, input_order""",
                        (row["id"],),
                    )
                    stored_inputs = await cur.fetchall()
                    expected_inputs = sorted(
                        (
                            (
                                item.observation_artifact_id,
                                item.preprocessing_artifact_id,
                                item.input_role,
                                item.input_order,
                            )
                            for item in inputs
                        ),
                        key=lambda item: (item[2], item[3]),
                    )
                    actual_inputs = [
                        (
                            item["observation_artifact_id"],
                            item["preprocessing_artifact_id"],
                            item["input_role"],
                            item["input_order"],
                        )
                        for item in stored_inputs
                    ]
                    if not created and actual_inputs != expected_inputs:
                        raise ValueError(
                            "metric evaluation retry conflicts with stored immutable inputs"
                        )
                    # Only the successful INSERT creates links. Existing rows were checked
                    # above and are intentionally frozen at queue time.
                    if created and inputs:
                        await cur.executemany(
                            """INSERT INTO benchmarks_v2.metric_evaluation_inputs
                               (metric_evaluation_id, observation_artifact_id,
                                preprocessing_artifact_id, input_role, input_order)
                               VALUES (%s, %s, %s, %s, %s)""",
                            [
                                (
                                    row["id"],
                                    item.observation_artifact_id,
                                    item.preprocessing_artifact_id,
                                    item.input_role,
                                    item.input_order,
                                )
                                for item in inputs
                            ],
                        )
            await conn.commit()
        if row is None:  # pragma: no cover
            raise RuntimeError("INSERT INTO metric_evaluations returned no row")
        stored = MetricEvaluation.model_validate(dict(row))
        immutable_fields = (
            "observation_id",
            "metric_type",
            "metric_version",
            "evaluation_variant",
            "executor",
            "external_request_id",
        )
        mismatches = [
            field
            for field in immutable_fields
            if getattr(evaluation, field) != getattr(stored, field)
        ]
        if mismatches:
            raise ValueError(
                "metric evaluation retry conflicts with stored immutable fields: "
                + ", ".join(mismatches)
            )
        return stored

    # Database transitions are guarded by validate_metric_transition() in the normalized migration.
    async def start_metric_evaluation(
        self, evaluation_id: UUID, *, started_at: datetime
    ) -> MetricEvaluation:
        """Transition one queued metric evaluation to running."""
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    """UPDATE benchmarks_v2.metric_evaluations
                       SET status = %s, started_at = %s, updated_at = now()
                       WHERE id = %s AND status = %s
                       RETURNING id, observation_id, metric_type, metric_version,
                                 evaluation_variant, executor,
                                 external_request_id, status, started_at, finished_at, error,
                                 created_at, updated_at""",
                    (ProcessingStatus.RUNNING, started_at, evaluation_id, ProcessingStatus.QUEUED),
                )
                row = await cur.fetchone()
            await conn.commit()
        if row is None:
            raise ValueError(f"metric evaluation {evaluation_id} is not queued")
        return MetricEvaluation.model_validate(dict(row))

    async def fail_metric_evaluation(
        self, evaluation_id: UUID, *, finished_at: datetime, error: str
    ) -> MetricEvaluation:
        """Atomically mark a queued or running metric evaluation failed."""
        if not error:
            raise ValueError("failed metric evaluations require an error")
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    """UPDATE benchmarks_v2.metric_evaluations
                       SET status = %s, started_at = COALESCE(started_at, %s), finished_at = %s,
                           error = %s, updated_at = now()
                       WHERE id = %s AND status IN (%s, %s)
                       RETURNING id, observation_id, metric_type, metric_version,
                                 evaluation_variant, executor,
                                 external_request_id, status, started_at, finished_at, error,
                                 created_at, updated_at""",
                    (
                        ProcessingStatus.FAILED,
                        finished_at,
                        finished_at,
                        error,
                        evaluation_id,
                        ProcessingStatus.QUEUED,
                        ProcessingStatus.RUNNING,
                    ),
                )
                row = await cur.fetchone()
            await conn.commit()
        if row is None:
            raise ValueError(f"metric evaluation {evaluation_id} is not queued or running")
        return MetricEvaluation.model_validate(dict(row))

    async def insert_preprocessing_artifact(
        self, artifact: PreprocessingArtifact
    ) -> PreprocessingArtifact:
        """Create or retrieve one immutable, versioned preprocessing artifact."""
        validate_preprocessing_artifact_contract(
            artifact.artifact_name, artifact.schema_name, artifact.schema_version
        )
        fields = (
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
        )
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    """INSERT INTO benchmarks_v2.preprocessing_artifacts
                       (observation_id, pipeline, pipeline_version, artifact_name, schema_name,
                        schema_version, producer_name, producer_provider, producer_model,
                        producer_version, gcs_uri, content_sha256)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (observation_id, pipeline, pipeline_version,
                                    artifact_name, schema_name, producer_name, producer_provider,
                                    producer_model, producer_version, schema_version)
                       DO NOTHING
                       RETURNING id, observation_id, pipeline, pipeline_version,
                                 artifact_name, schema_name, schema_version, producer_name,
                                 producer_provider, producer_model,
                                 producer_version, gcs_uri, content_sha256, created_at""",
                    tuple(getattr(artifact, field) for field in fields),
                )
                row = await cur.fetchone()
                if row is None:
                    await cur.execute(
                        """SELECT id, observation_id, pipeline, pipeline_version,
                                  artifact_name, schema_name, schema_version, producer_name,
                                  producer_provider, producer_model,
                                  producer_version, gcs_uri, content_sha256, created_at
                           FROM benchmarks_v2.preprocessing_artifacts
                           WHERE observation_id = %s AND pipeline = %s AND pipeline_version = %s
                             AND artifact_name = %s AND schema_name = %s
                             AND schema_version = %s AND producer_name = %s
                             AND producer_provider = %s AND producer_model = %s
                             AND producer_version = %s""",
                        (
                            artifact.observation_id,
                            artifact.pipeline,
                            artifact.pipeline_version,
                            artifact.artifact_name,
                            artifact.schema_name,
                            artifact.schema_version,
                            artifact.producer_name,
                            artifact.producer_provider,
                            artifact.producer_model,
                            artifact.producer_version,
                        ),
                    )
                    row = await cur.fetchone()
            await conn.commit()
        if row is None:  # pragma: no cover
            raise RuntimeError("INSERT INTO preprocessing_artifacts returned no row")
        stored = PreprocessingArtifact.model_validate(dict(row))
        mismatches = [
            field for field in fields if getattr(artifact, field) != getattr(stored, field)
        ]
        if mismatches:
            raise ValueError(
                "preprocessing artifact retry conflicts with stored immutable fields: "
                + ", ".join(mismatches)
            )
        return stored

    async def complete_metric_evaluation(
        self,
        evaluation_id: UUID,
        *,
        values: Sequence[MetricValue],
        artifacts: Sequence[MetricArtifact] = (),
        finished_at: datetime,
    ) -> None:
        """Atomically succeed a running evaluation; exact replays are harmless."""
        if not values:
            raise ValueError("succeeded metric evaluations require metric values")
        if any(value.metric_evaluation_id != evaluation_id for value in values):
            raise ValueError("all metric values must belong to the completed evaluation")
        if any(artifact.metric_evaluation_id != evaluation_id for artifact in artifacts):
            raise ValueError("all metric artifacts must belong to the completed evaluation")

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    """SELECT metric_type, metric_version, status, finished_at
                       FROM benchmarks_v2.metric_evaluations WHERE id = %s FOR UPDATE""",
                    (evaluation_id,),
                )
                evaluation = await cur.fetchone()
                if evaluation is None:
                    raise ValueError(f"metric evaluation {evaluation_id} does not exist")
                validate_metric_values(
                    evaluation["metric_type"],
                    evaluation["metric_version"],
                    tuple(
                        (value.value_key, value.unit, value.value, value.value_role)
                        for value in values
                    ),
                )
                if evaluation["status"] == ProcessingStatus.SUCCEEDED:
                    if evaluation["finished_at"] != finished_at:
                        raise ValueError("metric completion replay conflicts with stored result")
                    await cur.execute(
                        """SELECT value_key, unit, value, value_role
                           FROM benchmarks_v2.metric_values
                           WHERE metric_evaluation_id = %s""",
                        (evaluation_id,),
                    )
                    stored_values = await cur.fetchall()
                    await cur.execute(
                        """SELECT artifact_type, uri, sha256, size_bytes
                           FROM benchmarks_v2.metric_artifacts WHERE metric_evaluation_id = %s""",
                        (evaluation_id,),
                    )
                    stored_artifacts = await cur.fetchall()
                    value_fields = ("value_key", "unit", "value", "value_role")
                    expected_values = sorted(
                        (
                            tuple(getattr(value, field) for field in value_fields)
                            for value in values
                        ),
                        key=repr,
                    )
                    actual_values = sorted(
                        (tuple(value[field] for field in value_fields) for value in stored_values),
                        key=repr,
                    )
                    artifact_fields = ("artifact_type", "uri", "sha256", "size_bytes")
                    expected_artifacts = sorted(
                        (
                            tuple(getattr(item, field) for field in artifact_fields)
                            for item in artifacts
                        ),
                        key=repr,
                    )
                    actual_artifacts = sorted(
                        (
                            tuple(item[field] for field in artifact_fields)
                            for item in stored_artifacts
                        ),
                        key=repr,
                    )
                    if actual_values != expected_values or actual_artifacts != expected_artifacts:
                        raise ValueError("metric completion replay conflicts with stored result")
                    return
                if evaluation["status"] != ProcessingStatus.RUNNING:
                    raise ValueError("only running metric evaluations may be completed")
                await cur.executemany(
                    """INSERT INTO benchmarks_v2.metric_values
                       (metric_evaluation_id, value_key, unit, value, value_role)
                       VALUES (%s, %s, %s, %s, %s)""",
                    [
                        (
                            value.metric_evaluation_id,
                            value.value_key,
                            value.unit,
                            value.value,
                            value.value_role,
                        )
                        for value in values
                    ],
                )
                if artifacts:
                    await cur.executemany(
                        """INSERT INTO benchmarks_v2.metric_artifacts
                           (metric_evaluation_id, artifact_type, uri, sha256, size_bytes)
                           VALUES (%s, %s, %s, %s, %s)""",
                        [
                            (
                                artifact.metric_evaluation_id,
                                artifact.artifact_type,
                                artifact.uri,
                                artifact.sha256,
                                artifact.size_bytes,
                            )
                            for artifact in artifacts
                        ],
                    )
                await cur.execute(
                    """UPDATE benchmarks_v2.metric_evaluations
                       SET status = %s, finished_at = %s, error = NULL, updated_at = now()
                       WHERE id = %s""",
                    (ProcessingStatus.SUCCEEDED, finished_at, evaluation_id),
                )
            await conn.commit()

    async def refresh_metric_values_bucket(self, run_id: int) -> None:
        """Idempotently recompute normalized metric rollups for a run's bucket."""
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    "SELECT scheduled_at FROM benchmarks_v2.runs WHERE id = %s", (run_id,)
                )
                row = await cur.fetchone()
                bucket_at = row["scheduled_at"] if row is not None else None
                if bucket_at is None:
                    return
                params = {"bucket": bucket_at}
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended('metric_values_by_bucket',"
                    " extract(epoch FROM %(bucket)s::timestamptz)::bigint))",
                    params,
                )
                await cur.execute(
                    "DELETE FROM benchmarks_v2.metric_values_by_bucket "
                    "WHERE bucket_at = %(bucket)s",
                    params,
                )
                await cur.execute(
                    """
                    INSERT INTO benchmarks_v2.metric_values_by_bucket
                    (provider, model, benchmark, dataset_id, metric_type, metric_version,
                     evaluation_variant, value_key,
                     unit, bucket_at, min_value, p25, p50, p75, max_value, value_sum, sample_count)
                    SELECT observation.provider, observation.model, observation.benchmark,
                           COALESCE(observation.dataset_id, '__all__'), evaluation.metric_type,
                           evaluation.metric_version, evaluation.evaluation_variant,
                           value.value_key, value.unit, %(bucket)s,
                           MIN(value.value)::float8,
                           PERCENTILE_CONT(.25) WITHIN GROUP (ORDER BY value.value)::float8,
                           PERCENTILE_CONT(.5) WITHIN GROUP (ORDER BY value.value)::float8,
                           PERCENTILE_CONT(.75) WITHIN GROUP (ORDER BY value.value)::float8,
                           MAX(value.value)::float8, SUM(value.value)::float8, COUNT(*)::int
                    FROM benchmarks_v2.metric_values value
                    JOIN benchmarks_v2.metric_evaluations evaluation
                      ON evaluation.id = value.metric_evaluation_id
                    JOIN benchmarks_v2.benchmark_observations observation
                      ON observation.id = evaluation.observation_id
                    JOIN benchmarks_v2.runs run ON run.id = observation.run_id
                    WHERE evaluation.status = 'succeeded' AND run.status IN ('succeeded', 'partial')
                      AND run.scheduled_at = %(bucket)s
                    GROUP BY GROUPING SETS (
                      (observation.provider, observation.model, observation.benchmark,
                       observation.dataset_id, evaluation.metric_type,
                       evaluation.metric_version, evaluation.evaluation_variant,
                       value.value_key, value.unit),
                      (observation.provider, observation.model, observation.benchmark,
                       evaluation.metric_type, evaluation.metric_version,
                       evaluation.evaluation_variant,
                       value.value_key, value.unit)
                    )
                    """,
                    params,
                )
            await conn.commit()

    async def refresh_bucket(self, run_id: int, *, period_seconds: int) -> None:
        """Recompute the series rollup bucket for this run's scheduled_at slot.

        Delete-then-insert the whole bucket from raw result rows in one
        transaction, serialized per bucket by an advisory lock. Recomputing
        the full bucket (not just this run's rows) keeps it correct when runs
        share a slot, and makes the call idempotent. Runs without a
        ``scheduled_at`` are skipped — the migration backfill owns those.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    "SELECT scheduled_at FROM benchmarks_v2.runs WHERE id = %s",
                    (run_id,),
                )
                row = await cur.fetchone()
                bucket_at = row["scheduled_at"] if row is not None else None
                if bucket_at is None:
                    return

                # Serializes concurrent refreshes of one bucket. An empty
                # bucket gives DELETE nothing to lock, so two refreshes can
                # interleave such that the staler recompute commits and the
                # fresher one aborts on the primary key, dropping a run from
                # the slot. Released on commit/abort.
                params = {
                    "bucket": bucket_at,
                    "period": period_seconds,
                    # Window-aggregate-only metrics stay out of the series
                    # rollup; see SERIES_EXCLUDED_METRICS for the why.
                    "series_excluded": [str(m) for m in SERIES_EXCLUDED_METRICS],
                }
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended('results_by_bucket',"
                    " extract(epoch FROM %(bucket)s::timestamptz)::bigint))",
                    params,
                )
                # Two statements on purpose: a data-modifying CTE
                # (WITH deleted AS (DELETE ...) INSERT) collides on the primary
                # key — the INSERT cannot see the CTE's deletes.
                await cur.execute(
                    "DELETE FROM benchmarks_v2.results_by_bucket WHERE bucket_at = %(bucket)s",
                    params,
                )
                # Bucket membership: scheduled_at matches exactly, or a legacy
                # null-scheduled row whose created_at falls in
                # [bucket_at, bucket_at + period).
                await cur.execute(
                    """
                    INSERT INTO benchmarks_v2.results_by_bucket
                        (provider, model, benchmark, dataset_id, metric_type, bucket_at,
                         min_value, p25, p50, p75, max_value, value_sum, sample_count)
                    SELECT r.provider, r.model, r.benchmark,
                           COALESCE(
                               CASE WHEN r.benchmark = 'TTS' THEN 'tts-v1'
                                    ELSE rn.dataset_id END,
                               '__all__'
                           ),
                           r.metric_type, %(bucket)s,
                           MIN(r.metric_value)::float8,
                           PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY r.metric_value)::float8,
                           PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY r.metric_value)::float8,
                           PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.metric_value)::float8,
                           MAX(r.metric_value)::float8,
                           SUM(r.metric_value)::float8,
                           COUNT(*)::int
                    FROM benchmarks_v2.results r
                    JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
                    WHERE r.status = 'success'
                      AND rn.status IN ('succeeded', 'partial')
                      AND r.metric_value IS NOT NULL
                      AND r.metric_type != ALL(%(series_excluded)s)
                      AND (
                          rn.scheduled_at = %(bucket)s
                          OR (
                              rn.scheduled_at IS NULL
                              AND r.created_at >= %(bucket)s
                              AND r.created_at < %(bucket)s
                                  + (%(period)s::double precision) * INTERVAL '1 second'
                          )
                      )
                    GROUP BY GROUPING SETS (
                        (r.provider, r.model, r.benchmark, r.metric_type,
                         CASE WHEN r.benchmark = 'TTS' THEN 'tts-v1' ELSE rn.dataset_id END),
                        (r.provider, r.model, r.benchmark, r.metric_type)
                    )
                    """,
                    params,
                )
            await conn.commit()

    async def finish_run(
        self,
        run_id: int,
        *,
        status: RunStatus,
        error: str | None = None,
    ) -> None:
        """Set ``finished_at = now()`` and update ``status`` / ``error`` on a run row."""
        sql = """
            UPDATE benchmarks_v2.runs
            SET finished_at = now(),
                status = %s,
                error  = %s
            WHERE id = %s
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (status, error, run_id))
            await conn.commit()

    async def coval_run_ingested(self, *, provider: str, coval_run_id: str) -> bool:
        """True if a succeeded or partial run already holds rows for this Coval run.

        S2S rows store ``audio_filename = '<coval_run_id>/<sim_id>'``. Lets the
        fetch job skip a re-pulled run so a retry or stale re-pull doesn't
        double-write the day's bucket. Rows from failed runs don't count: they
        never reach the bucket, so a retry must stay free to re-ingest the run.
        """
        sql = """
            SELECT 1
            FROM benchmarks_v2.results r
            JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
            WHERE r.provider = %s
              AND r.benchmark = 'S2S'
              AND split_part(r.audio_filename, '/', 1) = %s
              AND rn.status IN ('succeeded', 'partial')
            LIMIT 1
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (provider, coval_run_id))
                row = await cur.fetchone()
            await conn.commit()
        return row is not None

    async def coval_metric_ingested(
        self, *, provider: str, coval_run_id: str, metric_type: str
    ) -> bool:
        """True if a succeeded/partial run already holds this Coval run's ``metric_type`` rows.

        Metric-aware counterpart of :meth:`coval_run_ingested`: lets the fetch
        backfill one metric (e.g. instruction) onto a run whose other metric
        (latency) already landed, instead of skipping the whole run.
        """
        sql = """
            SELECT 1
            FROM benchmarks_v2.results r
            JOIN benchmarks_v2.runs rn ON rn.id = r.run_id
            WHERE r.provider = %s
              AND r.benchmark = 'S2S'
              AND r.metric_type = %s
              AND split_part(r.audio_filename, '/', 1) = %s
              AND rn.status IN ('succeeded', 'partial')
            LIMIT 1
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (provider, metric_type, coval_run_id))
                row = await cur.fetchone()
            await conn.commit()
        return row is not None

    async def refresh_stats_matviews(self) -> None:
        """Concurrently refresh the per-window stats materialized views.

        ``CONCURRENTLY`` relies on each view's unique group-key index and does
        not block API reads. Raises on error like the rest of ``RunWriter``.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                for view in STATS_MATVIEWS:
                    await cur.execute(  # noqa: S608 — view names are constants
                        f"REFRESH MATERIALIZED VIEW CONCURRENTLY benchmarks_v2.{view}"
                    )
            await conn.commit()
