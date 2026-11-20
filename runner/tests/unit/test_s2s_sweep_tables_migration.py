# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""The S2S sweep tables hold a call from queued to scored, and come back out."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from pytest_postgresql.factories import postgresql

from .conftest import _INI_PATH, async_dsn

s2s_pg = postgresql("pg_proc")

CALL = ("email-1", "quiet room", 1)


def _config(conn: psycopg.Connection[Any]) -> Config:
    conn.commit()
    config = Config(str(_INI_PATH))
    config.set_main_option(
        "sqlalchemy.url", async_dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    config.attributes["allow_metric_code_cleanup"] = True
    return config


def _sweep(conn: psycopg.Connection[Any]) -> int:
    model = conn.execute(
        """INSERT INTO benchmarks_v2.models
               (modality, provider, model, collected, updated_by_user_id)
           VALUES ('S2S', 'host', 'one', true, 'tests') RETURNING id"""
    ).fetchone()
    assert model is not None
    sweep = conn.execute(
        """INSERT INTO benchmarks_v2.sweeps (model_id, model_version, dataset_id, dataset_version)
           VALUES (%s, 1, 'id_roulette', 'abc') RETURNING id""",
        (model[0],),
    ).fetchone()
    assert sweep is not None
    return int(sweep[0])


def test_a_call_goes_from_queued_to_scored(s2s_pg: psycopg.Connection[Any]) -> None:
    command.upgrade(_config(s2s_pg), "20261120_0071")
    sweep = _sweep(s2s_pg)

    s2s_pg.execute(
        "INSERT INTO benchmarks_v2.s2s_queue (sweep_id, test_case, persona, repeat)"
        " VALUES (%s, %s, %s, %s)",
        (sweep, *CALL),
    )
    s2s_pg.execute(
        "UPDATE benchmarks_v2.s2s_queue SET claimed_at = now(), simulation_id = 'sim-1'"
        " WHERE sweep_id = %s",
        (sweep,),
    )
    result = s2s_pg.execute(
        "INSERT INTO benchmarks_v2.s2s_results"
        " (sweep_id, test_case, persona, repeat, simulation_id)"
        " VALUES (%s, %s, %s, %s, 'sim-1') RETURNING id",
        (sweep, *CALL),
    ).fetchone()
    assert result is not None
    s2s_pg.execute(
        "INSERT INTO benchmarks_v2.s2s_scores"
        " (s2s_result_id, metric, metric_version, value, reference_units)"
        " VALUES (%s, 'CallLength', '1', 41.5, 1)",
        (result[0],),
    )
    s2s_pg.commit()

    queued = s2s_pg.execute("SELECT tries_left, last_error FROM benchmarks_v2.s2s_queue").fetchone()
    assert queued == (3, None)
    with pytest.raises(psycopg.errors.UniqueViolation):
        s2s_pg.execute(
            "INSERT INTO benchmarks_v2.s2s_results"
            " (sweep_id, test_case, persona, repeat, simulation_id)"
            " VALUES (%s, %s, %s, %s, 'sim-2')",
            (sweep, *CALL),
        )


def test_a_simulation_id_needs_a_claim(s2s_pg: psycopg.Connection[Any]) -> None:
    command.upgrade(_config(s2s_pg), "20261120_0071")
    sweep = _sweep(s2s_pg)

    with pytest.raises(psycopg.errors.CheckViolation):
        s2s_pg.execute(
            "INSERT INTO benchmarks_v2.s2s_queue"
            " (sweep_id, test_case, persona, repeat, simulation_id)"
            " VALUES (%s, %s, %s, %s, 'sim-1')",
            (sweep, *CALL),
        )


def test_the_tables_come_back_out(s2s_pg: psycopg.Connection[Any]) -> None:
    config = _config(s2s_pg)
    command.upgrade(config, "20261120_0071")

    command.downgrade(config, "20261120_0070")

    left = s2s_pg.execute(
        "SELECT count(*) FROM information_schema.tables"
        " WHERE table_schema = 'benchmarks_v2'"
        " AND table_name IN ('s2s_queue', 's2s_results', 's2s_scores')"
    ).fetchone()
    assert left == (0,)
