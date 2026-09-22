from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from psycopg_pool import AsyncConnectionPool
from pytest_postgresql.factories import postgresql

from coval_bench.db.models import (
    Benchmark,
    MetricEvaluation,
    MetricExecutor,
    Observation,
    ObservationSourceKind,
    ObservationStatus,
    ProcessingStatus,
    Result,
    ResultStatus,
    RunStatus,
)
from coval_bench.db.writer import RunWriter
from coval_bench.migrations.audit_normalized_s2s import S2SAuditReport, report_json
from coval_bench.registries import Metric

pg_conn = postgresql("pg_proc")


def _apply_migrations(conn: psycopg.Connection[Any]) -> None:
    info = conn.info
    dsn = f"postgresql://{info.user}@{info.host}:{info.port}/{info.dbname}"
    cfg = AlembicConfig(str(Path(__file__).parents[2] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", dsn.replace("postgresql://", "postgresql+psycopg://"))
    alembic_command.upgrade(cfg, "head")


def _async_dsn(conn: psycopg.Connection[Any]) -> str:
    info = conn.info
    return f"postgresql://{info.user}@{info.host}:{info.port}/{info.dbname}"


async def _pool(
    conn: psycopg.Connection[Any],
) -> AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]]:
    pool: AsyncConnectionPool[psycopg.AsyncConnection[psycopg.rows.DictRow]] = AsyncConnectionPool(
        _async_dsn(conn),
        min_size=1,
        max_size=1,
        open=False,
        kwargs={"row_factory": psycopg.rows.dict_row},
    )
    await pool.open()
    return pool


def _report(**overrides: object) -> S2SAuditReport:
    values: dict[str, object] = {
        "bounds": {"legacy_result_id_min": 1, "legacy_result_id_max": 2},
        "row_counts": {"legacy_s2s_results": 2},
        "key_counts": {"legacy_eligible_keys": 2, "normalized_eligible_keys": 2},
        "mismatches": {
            "legacy_missing_from_normalized_keys": 0,
            "normalized_extra_keys": 0,
            "same_key_count_mismatches": 0,
            "absolute_key_row_delta": 0,
            "samples": [],
        },
        "failed_null_metric_coverage": {"legacy_failed_status_rows": 1},
        "unusable_identities": {
            "legacy_null_empty_or_missing_prefix": 0,
            "normalized_null_empty_or_missing_prefix": 0,
        },
    }
    values.update(overrides)
    return S2SAuditReport(**values)  # type: ignore[arg-type]


def test_audit_is_ready_only_when_identity_sets_are_exact() -> None:
    assert _report().ready
    assert _report(
        mismatches={
            "legacy_missing_from_normalized_keys": 0,
            "normalized_extra_keys": 0,
            "same_key_count_mismatches": 1,
            "absolute_key_row_delta": 1,
            "samples": [],
        }
    ).ready
    assert not _report(
        mismatches={
            "legacy_missing_from_normalized_keys": 1,
            "normalized_extra_keys": 0,
            "same_key_count_mismatches": 0,
            "absolute_key_row_delta": 1,
            "samples": [],
        }
    ).ready
    assert not _report(
        unusable_identities={
            "legacy_null_empty_or_missing_prefix": 0,
            "normalized_null_empty_or_missing_prefix": 1,
        }
    ).ready


def test_report_json_includes_ready_and_bounded_samples() -> None:
    payload = json.loads(report_json(_report()))
    assert payload["ready"] is True
    assert payload["sample_limit"] == 100


def test_audit_presence_ignores_multiplicity_and_reports_metadata(
    pg_conn: psycopg.Connection[Any],
) -> None:
    _apply_migrations(pg_conn)

    async def seed() -> None:
        pool = await _pool(pg_conn)
        try:
            writer = RunWriter(pool)
            legacy_run = await writer.start_run(dataset_id="s2s", dataset_sha256="a" * 64)
            assert legacy_run.id is not None
            legacy = Result(
                run_id=legacy_run.id,
                provider="provider-a",
                model="model-a",
                benchmark=Benchmark.S2S,
                metric_type=Metric.INSTRUCTION_FOLLOWING,
                metric_value=100.0,
                metric_units="percent",
                audio_filename="COVAL-1/sim-1",
                status=ResultStatus.SUCCESS,
            )
            await writer.record_results([legacy, legacy])
            await writer.finish_run(legacy_run.id, status=RunStatus.SUCCEEDED)

            run = await writer.start_run(dataset_id="s2s", dataset_sha256="b" * 64)
            assert run.id is not None
            observation = await writer.insert_observation(
                Observation(
                    run_id=run.id,
                    dataset_id="s2s",
                    dataset_sha256="b" * 64,
                    sample_id="COVAL-1/sim-1",
                    provider="provider-a",
                    model="model-a",
                    benchmark=Benchmark.S2S,
                    source_kind=ObservationSourceKind.CONVERSATION_AUDIO,
                    status=ObservationStatus.SUCCEEDED,
                )
            )
            evaluation = await writer.insert_metric_evaluation(
                MetricEvaluation(
                    observation_id=observation.id,
                    metric_type=Metric.INSTRUCTION_FOLLOWING,
                    metric_version="v1",
                    evaluation_variant="reevaluated",
                    executor=MetricExecutor.INLINE,
                    status=ProcessingStatus.QUEUED,
                )
            )
            assert evaluation.id is not None
            await writer.fail_metric_evaluation(
                evaluation.id, finished_at=datetime.now(UTC), error="no value"
            )
            await writer.finish_run(run.id, status=RunStatus.SUCCEEDED)
        finally:
            await pool.close()

    asyncio.run(seed())
    from coval_bench.migrations.audit_normalized_s2s import audit_s2s

    report = audit_s2s(pg_conn, sample_limit=5)
    assert report.ready
    assert report.mismatches["same_key_count_mismatches"] == 1
    assert report.mismatches["legacy_missing_from_normalized_keys"] == 0
    assert report.failed_null_metric_coverage["normalized_failed_status_evaluations"] == 1
    assert report.bounds["isolation"] == "repeatable read"
    assert report.bounds["read_only"] == "on"
    assert report.to_dict()["complete"] is True


def test_audit_reports_bidirectional_gaps_and_unusable_identities(
    pg_conn: psycopg.Connection[Any],
) -> None:
    _apply_migrations(pg_conn)

    async def seed() -> None:
        pool = await _pool(pg_conn)
        try:
            writer = RunWriter(pool)
            legacy_run = await writer.start_run(dataset_id="s2s", dataset_sha256="c" * 64)
            assert legacy_run.id is not None
            legacy = Result(
                run_id=legacy_run.id,
                provider="provider-a",
                model="model-a",
                benchmark=Benchmark.S2S,
                metric_type=Metric.INSTRUCTION_FOLLOWING,
                metric_value=None,
                metric_units="percent",
                audio_filename=None,
                status=ResultStatus.FAILED,
                error="provider failed",
            )
            await writer.record_results([legacy])
            valid_legacy = legacy.model_copy(
                update={
                    "audio_filename": "LEGACY-ONLY/sim-1",
                    "metric_value": 1.0,
                    "status": ResultStatus.SUCCESS,
                    "error": None,
                }
            )
            await writer.record_results([valid_legacy])
            await writer.finish_run(legacy_run.id, status=RunStatus.SUCCEEDED)

            for sample_id in ("EXTRA/sim-1", "/sim-2", "slashless"):
                run = await writer.start_run(dataset_id="s2s", dataset_sha256="d" * 64)
                assert run.id is not None
                observation = await writer.insert_observation(
                    Observation(
                        run_id=run.id,
                        dataset_id="s2s",
                        dataset_sha256="d" * 64,
                        sample_id=sample_id,
                        provider="provider-a",
                        model="model-a",
                        benchmark=Benchmark.S2S,
                        source_kind=ObservationSourceKind.CONVERSATION_AUDIO,
                        status=ObservationStatus.SUCCEEDED,
                    )
                )
                await writer.insert_metric_evaluation(
                    MetricEvaluation(
                        observation_id=observation.id,
                        metric_type=Metric.INSTRUCTION_FOLLOWING,
                        metric_version="v1",
                        executor=MetricExecutor.INLINE,
                        status=ProcessingStatus.QUEUED,
                    )
                )
                await writer.finish_run(run.id, status=RunStatus.SUCCEEDED)
        finally:
            await pool.close()

    asyncio.run(seed())
    from coval_bench.migrations.audit_normalized_s2s import audit_s2s

    report = audit_s2s(pg_conn, sample_limit=1)
    assert not report.ready
    assert report.mismatches["legacy_missing_from_normalized_keys"] >= 1
    assert report.mismatches["normalized_extra_keys"] >= 1
    assert report.unusable_identities["legacy_null_empty_or_missing_prefix"] >= 1
    assert report.unusable_identities["normalized_null_empty_or_missing_prefix"] >= 2
    assert report.failed_null_metric_coverage["legacy_failed_status_rows"] == 1
    assert report.failed_null_metric_coverage["legacy_null_value_rows"] == 1
