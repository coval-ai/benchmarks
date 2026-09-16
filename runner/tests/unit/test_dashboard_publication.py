"""Transaction and lock guarantees for published dashboard summaries."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from pytest_postgresql.factories import postgresql
from structlog.testing import capture_logs

from coval_bench.db import dashboard_summaries
from coval_bench.db.dashboard_summaries import SUMMARY_VIEWS

from .conftest import apply_migrations, open_pool

summary_publication_pg = postgresql("pg_proc")


def test_failed_view_refresh_rolls_back_state_and_prior_views(
    summary_publication_pg: psycopg.Connection[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    apply_migrations(summary_publication_pg)
    first_as_of = datetime(2026, 9, 14, 12, tzinfo=UTC)
    second_as_of = datetime(2026, 9, 14, 13, tzinfo=UTC)

    async def scenario() -> None:
        pool = await open_pool(summary_publication_pg)
        try:
            first = await dashboard_summaries.refresh_summary_snapshots(pool, as_of=first_as_of)
            assert first.status == "published"
            async with pool.connection() as conn:
                before = await (
                    await conn.execute(
                        "SELECT generation, as_of FROM benchmarks_v2.dashboard_summary_state"
                    )
                ).fetchone()
            broken = dict(SUMMARY_VIEWS)
            broken["30d"] = "benchmarks_v2.missing_summary_view"
            monkeypatch.setattr(dashboard_summaries, "SUMMARY_VIEWS", broken)
            with capture_logs() as logs, pytest.raises(psycopg.errors.UndefinedTable):
                await dashboard_summaries.refresh_summary_snapshots(pool, as_of=second_as_of)
            failed = [
                event for event in logs if event["event"] == "dashboard_summary_view_refresh_failed"
            ]
            assert len(failed) == 1
            assert failed[0]["window"] == "30d"
            assert failed[0]["view"] == "benchmarks_v2.missing_summary_view"
            assert failed[0]["elapsed_seconds"] >= 0
            monkeypatch.setattr(dashboard_summaries, "SUMMARY_VIEWS", SUMMARY_VIEWS)
            async with pool.connection() as conn:
                after = await (
                    await conn.execute(
                        "SELECT generation, as_of FROM benchmarks_v2.dashboard_summary_state"
                    )
                ).fetchone()
                assert after == before
                for view in SUMMARY_VIEWS.values():
                    assert (
                        await (
                            await conn.execute(  # noqa: S608
                                f"SELECT count(*) AS n FROM {view}"  # noqa: S608
                            )
                        ).fetchone()
                    )["n"] == 0  # noqa: S608
        finally:
            await pool.close()

    asyncio.run(scenario())


def test_summary_refresh_reports_skipped_lock(
    summary_publication_pg: psycopg.Connection[Any],
) -> None:
    apply_migrations(summary_publication_pg)

    async def scenario() -> None:
        pool = await open_pool(summary_publication_pg)
        try:
            async with pool.connection() as held, held.transaction():
                await held.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended('dashboard_summary', 0))"
                )
                result = await dashboard_summaries.refresh_summary_snapshots(pool)
                assert result.status == "skipped_lock"
        finally:
            await pool.close()

    asyncio.run(scenario())
