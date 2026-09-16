# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The 20260916_0038 seed: AssemblyAI universal-3.6-pro joins the model registry.

Uses ``pytest-postgresql`` (embedded ``pg_ctl``, no Docker) to spin up a real
Postgres and run every migration to head. No remote DB is ever contacted.
"""

from __future__ import annotations

from typing import Any

import psycopg
from pytest_postgresql.factories import postgresql

from .conftest import apply_migrations

migration_pg = postgresql("pg_proc")  # shared server from conftest, own per-test DB

_MODEL_KEY = ("STT", "assemblyai", "universal-3.6-pro")


def _model_row(conn: psycopg.Connection[Any]) -> tuple[Any, ...] | None:
    return conn.execute(
        """
        SELECT collected, published, arena_enabled, source, licensing, on_prem, region
        FROM benchmarks_v2.models
        WHERE modality = %s AND provider = %s AND model = %s
        """,
        _MODEL_KEY,
    ).fetchone()


def _tags(conn: psycopg.Connection[Any], model: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT mt.tag
        FROM benchmarks_v2.model_tags mt
        JOIN benchmarks_v2.models m ON m.id = mt.model_id
        WHERE m.modality = 'STT' AND m.provider = 'assemblyai' AND m.model = %s
        """,
        (model,),
    ).fetchall()
    return {row[0] for row in rows}


def test_universal_3_6_pro_registered_but_not_collected(
    migration_pg: psycopg.Connection[Any],
) -> None:
    """Registered only: neither collected nor published until flipped after release."""
    apply_migrations(migration_pg)

    row = _model_row(migration_pg)
    assert row is not None, "universal-3.6-pro missing from benchmarks_v2.models"
    collected, published, arena_enabled, source, licensing, on_prem, region = row
    assert (collected, published, arena_enabled) == (False, False, False)
    # Same vendor/deployment facts as the 3.5-pro row it sits beside.
    assert (source, licensing, on_prem, region) == ("official-api", "proprietary", True, "us")


def test_universal_3_6_pro_carries_the_3_5_pro_feature_tags(
    migration_pg: psycopg.Connection[Any],
) -> None:
    apply_migrations(migration_pg)

    assert _tags(migration_pg, "universal-3.6-pro") == _tags(migration_pg, "universal-3.5-pro")
    assert _tags(migration_pg, "universal-3.6-pro") == {
        "code-switching",
        "diarization",
        "keyterm-biasing",
        "multilingual",
        "vad",
    }
