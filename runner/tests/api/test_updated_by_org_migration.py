# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Migration 0069: a company or model changed by an org with no user."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import psycopg
import pytest
from httpx import AsyncClient

from tests.api.conftest import _make_db_url

_INSERT = (
    "INSERT INTO benchmarks_v2.companies (slug, name, updated_by_user_id, updated_by_org_id)"
    " VALUES (%s, %s, %s, %s)"
)


def test_a_row_names_a_user_or_an_org_and_downgrade_keeps_the_org(
    client: AsyncClient, postgresql: Any
) -> None:
    migration = import_module("coval_bench.db.migrations.versions.20261119_0069_updated_by_org")
    conn: Any
    with psycopg.connect(_make_db_url(postgresql), autocommit=True) as conn:
        conn.execute(_INSERT, ("acme", "Acme", None, "org_acme"))
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(_INSERT, ("nobody", "Nobody", None, None))

        with patch.object(migration, "op", SimpleNamespace(execute=conn.execute)):
            migration.downgrade()
        row = conn.execute(
            "SELECT updated_by_user_id FROM benchmarks_v2.companies WHERE slug = 'acme'"
        ).fetchone()
    assert row == ("org_acme",)
