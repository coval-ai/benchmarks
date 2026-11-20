# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""The benchmark registry tables keep one dataset per benchmark and condition."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from pytest_postgresql.factories import postgresql

from .conftest import apply_migrations

registry_pg = postgresql("pg_proc")


@pytest.fixture
def conn(registry_pg: psycopg.Connection[Any]) -> psycopg.Connection[Any]:
    apply_migrations(registry_pg)
    registry_pg.autocommit = True
    registry_pg.execute(
        """INSERT INTO benchmarks_v2.benchmarks (slug, modality, name, updated_by_user_id)
           VALUES ('stt-wildasr', 'STT', 'WildASR', 'tests'),
                  ('tts-v2', 'TTS', 'TTS v2', 'tests')"""
    )
    return registry_pg


def _add_dataset(
    conn: psycopg.Connection[Any], dataset_id: str, benchmark: str, condition: str | None
) -> None:
    conn.execute(
        "INSERT INTO benchmarks_v2.benchmark_datasets (dataset_id, benchmark, condition)"
        " VALUES (%s, %s, %s)",
        (dataset_id, benchmark, condition),
    )


def test_a_condition_maps_to_one_dataset(conn: psycopg.Connection[Any]) -> None:
    _add_dataset(conn, "stt-wildasr-clean", "stt-wildasr", "clean")
    _add_dataset(conn, "stt-wildasr-farfield", "stt-wildasr", "farfield")

    with pytest.raises(psycopg.errors.UniqueViolation):
        _add_dataset(conn, "stt-wildasr-farfield-2", "stt-wildasr", "farfield")


def test_a_benchmark_without_conditions_has_one_dataset(conn: psycopg.Connection[Any]) -> None:
    _add_dataset(conn, "tts-v2", "tts-v2", None)

    with pytest.raises(psycopg.errors.UniqueViolation):
        _add_dataset(conn, "tts-v2-copy", "tts-v2", None)
