# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""``RunWriter`` — typed insert helpers for the benchmark persistence layer.

All SQL uses parameterised queries (psycopg ``%s`` style).  No string
interpolation with user data is performed anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

import psycopg
import psycopg.errors
import psycopg.rows
import structlog
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from coval_bench.db.llm_turns import fetch_conversation_ttft
from coval_bench.db.models import (
    MetricArtifact,
    MetricEvaluation,
    MetricEvaluationInput,
    MetricValue,
    Observation,
    ObservationArtifact,
    PreprocessingArtifact,
    ProcessingStatus,
    Run,
    RunStatus,
)
from coval_bench.registries import (
    Metric,
    validate_metric_contract,
    validate_metric_values,
    validate_preprocessing_artifact_contract,
)
from coval_bench.registries.metrics import METRIC_SPECS

logger = structlog.get_logger(__name__)


class RunWriter:
    """Per-run persistence helper.

    Lifecycle::

        writer = RunWriter(pool)
        run = await writer.start_run(dataset_id=..., dataset_sha256=...)
        await writer.finish_run(run.id, status=RunStatus.SUCCEEDED)
    """

    def __init__(
        self,
        pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]],
    ) -> None:
        self._pool = pool

    async def start_run(
        self,
        *,
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
            VALUES ('untracked', %s, %s, %s, %s, %s)
            RETURNING id, started_at, finished_at, scheduled_at,
                      dataset_id, dataset_sha256, status, error, persona_id
        """
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    sql,
                    (
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

    async def get_run(self, run_id: int) -> Run:
        """Return one run for recovery decisions without changing it."""
        async with (
            self._pool.connection() as conn,
            conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
        ):
            await cur.execute(
                """SELECT id, started_at, finished_at, scheduled_at, dataset_id,
                          dataset_sha256, status, error, persona_id
                   FROM benchmarks_v2.runs WHERE id = %s""",
                (run_id,),
            )
            row = await cur.fetchone()
        if row is None:
            raise ValueError(f"run {run_id} does not exist")
        return Run.model_validate(dict(row))

    async def reserve_run_id(self) -> int:
        """Reserve a run primary key without creating a visible run row."""
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT nextval(pg_get_serial_sequence('benchmarks_v2.runs', 'id'))")
            row = await cur.fetchone()
            await conn.commit()
        if row is None:
            raise RuntimeError("run ID sequence returned no value")
        return int(row[0] if not isinstance(row, dict) else next(iter(row.values())))

    async def ensure_capture_run(
        self,
        run_id: int,
        *,
        started_at: datetime,
        dataset_id: str,
        dataset_sha256: str,
        scheduled_at: datetime,
        persona_id: str | None,
    ) -> Run:
        """Insert or exactly retrieve a run elected by a durable import claim."""
        fields = (
            "id",
            "started_at",
            "scheduled_at",
            "dataset_id",
            "dataset_sha256",
            "persona_id",
        )
        expected = (
            run_id,
            started_at,
            scheduled_at,
            dataset_id,
            dataset_sha256,
            persona_id or None,
        )
        async with (
            self._pool.connection() as conn,
            conn.transaction(),
            conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
        ):
            await cur.execute(
                """INSERT INTO benchmarks_v2.runs
                           (id, runner_sha, started_at, dataset_id, dataset_sha256,
                            status, scheduled_at, persona_id)
                       VALUES (%s, 'untracked', %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (id) DO NOTHING""",
                (
                    run_id,
                    started_at,
                    dataset_id,
                    dataset_sha256,
                    RunStatus.RUNNING,
                    scheduled_at,
                    persona_id or None,
                ),
            )
            await cur.execute(
                """SELECT id, started_at, finished_at, scheduled_at, dataset_id,
                              dataset_sha256, status, error, persona_id
                       FROM benchmarks_v2.runs WHERE id = %s FOR UPDATE""",
                (run_id,),
            )
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("capture run insert returned no row")
            actual = tuple(row[field] for field in fields)
            if actual != expected:
                raise ValueError("capture run conflicts with durable import claim")
        return Run.model_validate(dict(row))

    async def preflight_required_capture_schema(self) -> None:
        """Fail before provider work when normalized capture/publication is unavailable."""
        required = (
            "runs",
            "metrics",
            "benchmark_observations",
            "observation_artifacts",
            "metric_evaluations",
            "metric_evaluation_inputs",
            "metric_values",
            "dashboard_rollups",
            "dashboard_rollup_queue",
        )
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """SELECT name
                   FROM unnest(%s::text[]) AS name
                   WHERE to_regclass('benchmarks_v2.' || name) IS NULL""",
                (list(required),),
            )
            missing = [
                str(row[0] if not isinstance(row, dict) else row["name"])
                for row in await cur.fetchall()
            ]
        if missing:
            raise RuntimeError(
                "required normalized capture schema is unavailable: " + ", ".join(missing)
            )
        await self._assert_required_capture_privileges()

    async def _assert_required_capture_privileges(self) -> None:
        """Check write/publication privileges without attempting a mutating probe."""
        tables = (
            "metrics",
            "benchmark_observations",
            "observation_artifacts",
            "metric_evaluations",
            "metric_evaluation_inputs",
            "metric_values",
            "runs",
            "dashboard_rollups",
            "dashboard_rollup_queue",
        )
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """SELECT table_name
                   FROM unnest(%s::text[]) AS table_name
                   WHERE NOT has_table_privilege(
                       current_user,
                       'benchmarks_v2.' || table_name,
                       'SELECT,INSERT,UPDATE'
                   )""",
                (list(tables),),
            )
            denied = [
                str(row[0] if not isinstance(row, dict) else row["table_name"])
                for row in await cur.fetchall()
            ]
        if denied:
            raise RuntimeError(
                "required normalized capture privileges unavailable: " + ", ".join(denied)
            )
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """SELECT table_name
                   FROM unnest(%s::text[]) AS table_name
                   WHERE NOT has_sequence_privilege(
                       current_user,
                       pg_get_serial_sequence('benchmarks_v2.' || table_name, 'id'),
                       'USAGE'
                   )""",
                (["runs", "metrics"],),
            )
            denied_sequences = [
                str(row[0] if not isinstance(row, dict) else row["table_name"])
                for row in await cur.fetchall()
            ]
        if denied_sequences:
            raise RuntimeError(
                "required normalized capture sequence privileges unavailable: "
                + ", ".join(denied_sequences)
            )

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
        self,
        evaluation: MetricEvaluation,
        *,
        inputs: Sequence[MetricEvaluationInput] = (),
        validate_contract: bool = True,
    ) -> MetricEvaluation:
        """Create a queued evaluation or retrieve the same evaluation on retry."""
        if (
            evaluation.status is not ProcessingStatus.QUEUED
            or evaluation.started_at is not None
            or evaluation.finished_at is not None
            or evaluation.error is not None
        ):
            raise ValueError("metric evaluations must be created queued")
        if validate_contract:
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
            (observation_id, metric_id, metric_version, evaluation_variant, executor,
             external_request_id, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (observation_id, metric_id, metric_version, evaluation_variant)
            DO NOTHING
            RETURNING id, observation_id, metric_id,
                      (SELECT m.code FROM benchmarks_v2.metrics m
                       WHERE m.id = metric_evaluations.metric_id) AS metric_type,
                      metric_version,
                      evaluation_variant, executor,
                      external_request_id,
                      status, started_at, finished_at, error, created_at, updated_at
        """
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    "SELECT id FROM benchmarks_v2.metrics WHERE code = %s",
                    (evaluation.metric_type,),
                )
                metric_row = await cur.fetchone()
                if metric_row is None:
                    # This path is for a newly registered known metric only;
                    # normal evaluations resolve the existing row directly.
                    try:
                        Metric(evaluation.metric_type)
                    except ValueError as exc:
                        raise ValueError(f"unknown metric_type {evaluation.metric_type!r}") from exc
                    await cur.execute(
                        """INSERT INTO benchmarks_v2.metrics (code, display_name)
                           VALUES (%s, %s) ON CONFLICT (code) DO NOTHING RETURNING id""",
                        (
                            evaluation.metric_type,
                            METRIC_SPECS[Metric(evaluation.metric_type)].display_name,
                        ),
                    )
                    metric_row = await cur.fetchone()
                    if metric_row is None:
                        await cur.execute(
                            "SELECT id FROM benchmarks_v2.metrics WHERE code = %s",
                            (evaluation.metric_type,),
                        )
                        metric_row = await cur.fetchone()
                if metric_row is None:
                    raise RuntimeError(
                        f"metric definition is unavailable: {evaluation.metric_type}"
                    )
                metric_id = int(metric_row["id"])
                if evaluation.metric_id is not None and evaluation.metric_id != metric_id:
                    raise ValueError("metric code and id must refer to the same definition")
                async with conn.transaction():
                    await cur.execute(
                        sql,
                        (
                            evaluation.observation_id,
                            metric_id,
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
                        """SELECT id, observation_id, metric_id,
                                  (SELECT m.code FROM benchmarks_v2.metrics m
                                   WHERE m.id = metric_evaluations.metric_id) AS metric_type,
                                  metric_version,
                                  evaluation_variant, executor,
                                  external_request_id, status, started_at, finished_at, error,
                                  created_at, updated_at
                           FROM benchmarks_v2.metric_evaluations
                           WHERE observation_id = %s
                             AND metric_id = %s AND metric_version = %s
                             AND evaluation_variant = %s""",
                        (
                            evaluation.observation_id,
                            metric_id,
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
    async def start_metric_evaluation_exact(
        self, evaluation_id: UUID, *, started_at: datetime
    ) -> MetricEvaluation:
        """Start or verify a replayed evaluation without timestamp drift."""
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    """SELECT id, observation_id, metric_id,
                              (SELECT m.code FROM benchmarks_v2.metrics m
                               WHERE m.id = metric_evaluations.metric_id) AS metric_type,
                              metric_version,
                              evaluation_variant, executor, external_request_id, status,
                              started_at, finished_at, error, created_at, updated_at
                       FROM benchmarks_v2.metric_evaluations WHERE id = %s FOR UPDATE""",
                    (evaluation_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    raise ValueError(f"metric evaluation {evaluation_id} does not exist")
                if row["status"] == ProcessingStatus.QUEUED:
                    await cur.execute(
                        """UPDATE benchmarks_v2.metric_evaluations
                           SET status = %s, started_at = %s, updated_at = now()
                           WHERE id = %s""",
                        (ProcessingStatus.RUNNING, started_at, evaluation_id),
                    )
                    row["status"] = ProcessingStatus.RUNNING
                    row["started_at"] = started_at
                elif row["started_at"] != started_at:
                    raise ValueError("metric evaluation replay conflicts with started_at")
            await conn.commit()
        return MetricEvaluation.model_validate(dict(row))

    async def fail_metric_evaluation_exact(
        self, evaluation_id: UUID, *, started_at: datetime, finished_at: datetime, error: str
    ) -> MetricEvaluation:
        """Fail or verify a replayed evaluation using its frozen timestamps."""
        if not error:
            raise ValueError("failed metric evaluations require an error")
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    """SELECT id, observation_id, metric_id,
                              (SELECT m.code FROM benchmarks_v2.metrics m
                               WHERE m.id = metric_evaluations.metric_id) AS metric_type,
                              metric_version,
                              evaluation_variant, executor, external_request_id, status,
                              started_at, finished_at, error, created_at, updated_at
                       FROM benchmarks_v2.metric_evaluations WHERE id = %s FOR UPDATE""",
                    (evaluation_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    raise ValueError(f"metric evaluation {evaluation_id} does not exist")
                if row["status"] == ProcessingStatus.FAILED:
                    if (
                        row["started_at"] != started_at
                        or row["finished_at"] != finished_at
                        or row["error"] != error
                    ):
                        raise ValueError("metric failure replay conflicts with stored result")
                elif row["status"] == ProcessingStatus.SUCCEEDED:
                    raise ValueError("metric failure replay conflicts with succeeded result")
                else:
                    await cur.execute(
                        """UPDATE benchmarks_v2.metric_evaluations
                           SET status = %s, started_at = %s, finished_at = %s,
                               error = %s, updated_at = now() WHERE id = %s""",
                        (ProcessingStatus.FAILED, started_at, finished_at, error, evaluation_id),
                    )
                    row.update(
                        status=ProcessingStatus.FAILED,
                        started_at=started_at,
                        finished_at=finished_at,
                        error=error,
                    )
            await conn.commit()
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
        validate_contract: bool = True,
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
                    """SELECT m.code AS metric_type, e.metric_version, e.status, e.finished_at
                       FROM benchmarks_v2.metric_evaluations e
                       JOIN benchmarks_v2.metrics m ON m.id = e.metric_id
                       WHERE e.id = %s FOR UPDATE OF e""",
                    (evaluation_id,),
                )
                evaluation = await cur.fetchone()
                if evaluation is None:
                    raise ValueError(f"metric evaluation {evaluation_id} does not exist")
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
                if validate_contract:
                    validate_metric_values(
                        evaluation["metric_type"],
                        evaluation["metric_version"],
                        tuple(
                            (value.value_key, value.unit, value.value, value.value_role)
                            for value in values
                        ),
                    )
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

    async def rebuild_run_rollup(self, run_id: int) -> None:
        """Rebuild the run's run-grain rows now; its slot is already queued for the job."""
        from coval_bench.db.dashboard_rollups import RUN_SLOT, fill_rollup

        async with (
            self._pool.connection() as conn,
            conn.transaction(),
            conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
        ):
            await cur.execute(
                "SELECT scheduled_at FROM benchmarks_v2.runs WHERE id = %s", (run_id,)
            )
            row = await cur.fetchone()
            if row is not None and row["scheduled_at"] is not None:
                await fill_rollup(conn, grain=RUN_SLOT, bucket_at=row["scheduled_at"])

    async def refresh_window_views(self, run_id: int | None = None) -> str:
        """Publish normalized summaries independently of legacy maintenance."""
        from coval_bench.db.dashboard_windows import refresh_window_views

        result = await refresh_window_views(self._pool, run_id=run_id)
        return result.status

    async def finish_run(
        self,
        run_id: int,
        *,
        status: RunStatus,
        error: str | None = None,
    ) -> None:
        """Commit the run's terminal status and queue its slot for the rollup job."""
        from coval_bench.db.dashboard_rollups import ENQUEUE_SLOT_SQL

        sql = """
            UPDATE benchmarks_v2.runs
            SET finished_at = now(),
                status = %s,
                error  = %s
            WHERE id = %s
        """
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(sql, (status, error, run_id))
            await cur.execute(ENQUEUE_SLOT_SQL, {"run_id": run_id})

    async def finish_run_exact(
        self,
        run_id: int,
        *,
        status: RunStatus,
        finished_at: datetime,
        error: str | None = None,
        allow_capture_recovery: bool = False,
    ) -> None:
        """Replay run completion while preserving the original finish time."""
        from coval_bench.db.dashboard_rollups import ENQUEUE_SLOT_SQL

        async with (
            self._pool.connection() as conn,
            conn.transaction(),
            conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
        ):
            await cur.execute(
                """SELECT status, finished_at, error FROM benchmarks_v2.runs
                       WHERE id = %s FOR UPDATE""",
                (run_id,),
            )
            row = await cur.fetchone()
            if row is None:
                raise ValueError(f"run {run_id} does not exist")
            if row["finished_at"] is not None:
                exact = (
                    row["status"] == status
                    and row["finished_at"] == finished_at
                    and row["error"] == error
                )
                recoverable_partial = (
                    allow_capture_recovery
                    and row["status"] == RunStatus.PARTIAL
                    and row["finished_at"] == finished_at
                    and row["error"] == "normalized capture pending"
                )
                if recoverable_partial:
                    await cur.execute(
                        "UPDATE benchmarks_v2.runs SET status = %s, error = %s WHERE id = %s",
                        (status, error, run_id),
                    )
                elif not exact:
                    raise ValueError("run completion replay conflicts with stored result")
            else:
                await cur.execute(
                    """UPDATE benchmarks_v2.runs SET finished_at = %s, status = %s,
                                  error = %s WHERE id = %s""",
                    (finished_at, status, error, run_id),
                )
            await cur.execute(ENQUEUE_SLOT_SQL, {"run_id": run_id})

    async def mark_run_capture_pending(self, run_id: int, *, finished_at: datetime) -> None:
        """Downgrade a finalized non-failed run when its final receipt is missing."""
        from coval_bench.db.dashboard_rollups import ENQUEUE_SLOT_SQL

        async with (
            self._pool.connection() as conn,
            conn.transaction(),
            conn.cursor(row_factory=psycopg.rows.dict_row) as cur,
        ):
            await cur.execute(
                """SELECT status, finished_at, error FROM benchmarks_v2.runs
                   WHERE id = %s FOR UPDATE""",
                (run_id,),
            )
            row = await cur.fetchone()
            if row is None:
                raise ValueError(f"run {run_id} does not exist")
            if row["finished_at"] != finished_at:
                raise ValueError("capture-pending downgrade conflicts with finished_at")
            if row["status"] == RunStatus.FAILED:
                raise ValueError("failed provider runs cannot be relabeled as capture pending")
            await cur.execute(
                """UPDATE benchmarks_v2.runs
                   SET status = %s, error = 'normalized capture pending'
                   WHERE id = %s""",
                (RunStatus.PARTIAL, run_id),
            )
            await cur.execute(ENQUEUE_SLOT_SQL, {"run_id": run_id})

    async def conversation_ttft(self, simulation_ids: Sequence[str]) -> dict[str, float]:
        """Mean proxy-measured TTFT in seconds per Coval conversation that has turns."""
        return await fetch_conversation_ttft(self._pool, simulation_ids)

    async def coval_metric_ingested(
        self,
        *,
        provider: str,
        coval_run_id: str,
        metric_type: str,
        benchmark: str = "S2S",
    ) -> bool:
        """Check normalized observation/evaluation storage for an import key.

        The parent run status is the only lifecycle filter.  In particular, a
        failed evaluation still counts as ingested so a missing metric can be
        imported independently without requiring a value or successful child
        status.  ``model`` and evaluation version/variant are deliberately
        absent: the import identity is provider/benchmark/sample prefix/metric.
        """
        sql = """
            SELECT 1
            FROM benchmarks_v2.benchmark_observations o
            JOIN benchmarks_v2.metric_evaluations e ON e.observation_id = o.id
            JOIN benchmarks_v2.runs rn ON rn.id = o.run_id
            JOIN benchmarks_v2.metrics m ON m.id = e.metric_id
            WHERE o.provider = %s
              AND o.benchmark = %s
              AND split_part(o.sample_id, '/', 1) = %s
              AND m.code = %s
              AND rn.status IN ('succeeded', 'partial')
            LIMIT 1
        """
        params = (provider, benchmark, coval_run_id, metric_type)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                row = await cur.fetchone()
            await conn.commit()
        return row is not None
