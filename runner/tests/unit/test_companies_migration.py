# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""The companies migration makes a company for every provider and creator on a model."""

from __future__ import annotations

from typing import Any

import psycopg
from alembic import command
from alembic.config import Config
from pytest_postgresql.factories import postgresql

from .conftest import _INI_PATH, async_dsn

companies_pg = postgresql("pg_proc")


def _migrate(conn: psycopg.Connection[Any], revision: str) -> None:
    conn.commit()
    config = Config(str(_INI_PATH))
    config.set_main_option(
        "sqlalchemy.url", async_dsn(conn).replace("postgresql://", "postgresql+psycopg://")
    )
    config.attributes["allow_metric_code_cleanup"] = True
    command.upgrade(config, revision)


def test_every_provider_and_creator_becomes_a_company(
    companies_pg: psycopg.Connection[Any],
) -> None:
    _migrate(companies_pg, "20261117_0065")
    companies_pg.execute(
        """INSERT INTO benchmarks_v2.models
               (modality, provider, model, creator, collected, published, updated_by_user_id)
           VALUES ('STT', 'host', 'one', 'lab', true, true, 'tests'),
                  ('STT', 'host', 'two', NULL, true, false, 'tests'),
                  ('TTS', 'solo', 'one', NULL, true, true, 'tests')"""
    )

    _migrate(companies_pg, "20261117_0066")

    expected = companies_pg.execute(
        """SELECT provider FROM benchmarks_v2.models
           UNION SELECT creator FROM benchmarks_v2.models WHERE creator IS NOT NULL"""
    ).fetchall()
    companies = companies_pg.execute("SELECT slug, name FROM benchmarks_v2.companies").fetchall()
    assert {"host", "lab", "solo"} <= {slug for slug, _ in companies}
    assert sorted(slug for slug, _ in companies) == sorted(slug for (slug,) in expected)
    assert all(slug == name for slug, name in companies)
