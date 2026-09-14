# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Database identity for metrics used by saved dashboard aggregates."""

from __future__ import annotations

from typing import Any

from psycopg import AsyncConnection

from coval_bench.registries.metrics import METRIC_SPECS


async def register_metric_definitions(conn: AsyncConnection[Any]) -> dict[str, int]:
    """Register known codes without replacing existing IDs or display labels."""
    definitions_by_code = {metric.value: spec.display_name for metric, spec in METRIC_SPECS.items()}
    definitions = sorted(
        definitions_by_code.items(),
        key=lambda item: item[0],
    )
    async with conn.cursor() as cur:
        await cur.executemany(
            """INSERT INTO benchmarks_v2.metrics (code, display_name)
               VALUES (%(code)s, %(display_name)s)
               ON CONFLICT (code) DO NOTHING""",
            [{"code": code, "display_name": display_name} for code, display_name in definitions],
        )
    rows = await (
        await conn.execute(
            """SELECT code, id
               FROM benchmarks_v2.metrics
               WHERE code = ANY(%(codes)s::text[])
               ORDER BY code""",
            {"codes": [code for code, _display_name in definitions]},
        )
    ).fetchall()
    mapping = {str(row["code"]): int(row["id"]) for row in rows}
    expected = {code for code, _display_name in definitions}
    if set(mapping) != expected:
        missing = ", ".join(sorted(expected - set(mapping)))
        raise RuntimeError(f"metric definition registry is incomplete: {missing}")
    return mapping
