# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Migration and transaction coverage for the dashboard value projection."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import psycopg
import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from pytest_postgresql.factories import postgresql
from sqlalchemy.exc import OperationalError

from coval_bench.db.writer import RunWriter
from tests.unit import test_normalized_db_writer as storage

pg_conn = postgresql("pg_proc")


@pytest.mark.asyncio
async def test_projection_seeds_existing_values_and_downgrades(
    pg_conn: psycopg.Connection[Any],
) -> None:
    storage._migrate(pg_conn, "20260910_0030")
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await storage._observation(writer)
        evaluation = await storage._evaluation(writer, observation)
        evaluation_id = storage._required(evaluation.id)
        await writer.complete_metric_evaluation(
            evaluation_id,
            values=storage._wer_values(evaluation_id),
            finished_at=storage._NOW + timedelta(seconds=1),
        )
    finally:
        await pool.close()
    storage._migrate(pg_conn)
    pg_conn.autocommit = True
    row = pg_conn.execute(
        "SELECT value, wer_insertions_pct, wer_deletions_pct, wer_substitutions_pct, "
        "reference_words, has_primary_role FROM benchmarks_v2.dashboard_metric_values "
        "WHERE evaluation_id = %s",
        (evaluation_id,),
    ).fetchone()
    assert row == (10, 1, 2, 7, None, True)
    config = AlembicConfig(str(storage._INI_PATH))
    config.set_main_option(
        "sqlalchemy.url", storage._dsn(pg_conn).replace("postgresql://", "postgresql+psycopg://")
    )
    alembic_command.downgrade(config, "20260910_0030")
    assert pg_conn.execute(
        "SELECT to_regclass('benchmarks_v2.dashboard_metric_values')"
    ).fetchone() == (None,)
    assert pg_conn.execute(
        "SELECT count(*) FROM benchmarks_v2.metric_values WHERE metric_evaluation_id = %s",
        (evaluation_id,),
    ).fetchone() == (4,)
    with pytest.raises(psycopg.errors.RaiseException, match="payloads are immutable"):
        pg_conn.execute(
            "UPDATE benchmarks_v2.metric_values SET value = 0 "
            "WHERE metric_evaluation_id = %s AND value_key = 'primary'",
            (evaluation_id,),
        )


@pytest.mark.asyncio
async def test_projection_cascades_and_evaluation_uuid_can_be_reused(
    pg_conn: psycopg.Connection[Any],
) -> None:
    storage._migrate(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await storage._observation(writer)
        evaluation = await storage._evaluation(writer, observation)
        evaluation_id = storage._required(evaluation.id)
        await writer.complete_metric_evaluation(
            evaluation_id,
            values=storage._wer_values(evaluation_id),
            finished_at=storage._NOW + timedelta(seconds=1),
        )
        async with pool.connection() as conn:
            await conn.execute(
                "DELETE FROM benchmarks_v2.benchmark_observations WHERE id = %s",
                (observation.id,),
            )
            row = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM benchmarks_v2.dashboard_metric_values"
                )
            ).fetchone()
            assert row == {"n": 0}
        _, replacement = await storage._observation(writer, sample="replacement")
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO benchmarks_v2.metric_evaluations "
                "(id, observation_id, metric_type, metric_version, evaluation_variant, "
                "executor, status) VALUES (%s, %s, 'WER', 'v1', 'default', 'inline', 'queued')",
                (evaluation_id, replacement.id),
            )
        await writer.start_metric_evaluation(evaluation_id, started_at=storage._NOW)
        values = storage._wer_values(evaluation_id)
        values[0] = values[0].model_copy(update={"value": 11})
        values[3] = values[3].model_copy(update={"value": 8})
        await writer.complete_metric_evaluation(
            evaluation_id, values=values, finished_at=storage._NOW + timedelta(seconds=1)
        )
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT observation_id, value FROM benchmarks_v2.dashboard_metric_values "
                    "WHERE evaluation_id = %s",
                    (evaluation_id,),
                )
            ).fetchone()
            assert row == {"observation_id": replacement.id, "value": 11}
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_deferred_validation_failure_rolls_back_projection(
    pg_conn: psycopg.Connection[Any],
) -> None:
    storage._migrate(pg_conn)
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await storage._observation(writer)
        evaluation = await storage._evaluation(writer, observation)
        evaluation_id = storage._required(evaluation.id)
        async with pool.connection() as conn:
            with pytest.raises(psycopg.errors.RaiseException, match="exactly one primary"):
                async with conn.transaction():
                    await conn.execute(
                        "INSERT INTO benchmarks_v2.metric_values VALUES "
                        "(%s, 'roundtrip', 'milliseconds', 42, 'component')",
                        (evaluation_id,),
                    )
                    await conn.execute(
                        "UPDATE benchmarks_v2.metric_evaluations "
                        "SET status = 'succeeded', finished_at = %s, updated_at = now() "
                        "WHERE id = %s",
                        (storage._NOW + timedelta(seconds=1), evaluation_id),
                    )
                    row = await (
                        await conn.execute(
                            "SELECT roundtrip FROM benchmarks_v2.dashboard_metric_values "
                            "WHERE evaluation_id = %s",
                            (evaluation_id,),
                        )
                    ).fetchone()
                    assert row == {"roundtrip": 42}
            row = await (
                await conn.execute(
                    "SELECT status FROM benchmarks_v2.metric_evaluations WHERE id = %s",
                    (evaluation_id,),
                )
            ).fetchone()
            assert row == {"status": "running"}
            row = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM benchmarks_v2.dashboard_metric_values"
                )
            ).fetchone()
            assert row == {"n": 0}
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_projection_upgrade_nowait_does_not_interrupt_completion(
    pg_conn: psycopg.Connection[Any],
) -> None:
    """A partial lock set never makes migration and an active writer wait on each other."""
    storage._migrate(pg_conn, "20260910_0030")
    pool = await storage._pool(pg_conn)
    try:
        writer = RunWriter(pool)
        _, observation = await storage._observation(writer)
        evaluation = await storage._evaluation(writer, observation)
        evaluation_id = storage._required(evaluation.id)
        async with pool.connection() as conn:
            await conn.execute(
                "SELECT id FROM benchmarks_v2.metric_evaluations WHERE id = %s FOR UPDATE",
                (evaluation_id,),
            )
            await conn.execute(
                "INSERT INTO benchmarks_v2.metric_values VALUES "
                "(%s, 'primary', 'percent', 10, 'primary')",
                (evaluation_id,),
            )
            # The writer holds evaluations ROW SHARE and values ROW EXCLUSIVE.
            # An ordered blocking migration lock could deadlock its final UPDATE.
            with pytest.raises(OperationalError, match="could not obtain lock") as error:
                storage._migrate(pg_conn)
            assert isinstance(error.value.orig, psycopg.errors.LockNotAvailable)
            await conn.execute(
                "UPDATE benchmarks_v2.metric_evaluations "
                "SET status = 'succeeded', finished_at = %s WHERE id = %s",
                (storage._NOW + timedelta(seconds=1), evaluation_id),
            )
            await conn.commit()
        storage._migrate(pg_conn)
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT d.value, v.value AS raw_value "
                    "FROM benchmarks_v2.dashboard_metric_values d "
                    "JOIN benchmarks_v2.metric_values v "
                    "ON v.metric_evaluation_id = d.evaluation_id "
                    "AND v.value_key = 'primary' WHERE d.evaluation_id = %s",
                    (evaluation_id,),
                )
            ).fetchone()
            assert row == {"value": 10, "raw_value": 10}
    finally:
        await pool.close()
