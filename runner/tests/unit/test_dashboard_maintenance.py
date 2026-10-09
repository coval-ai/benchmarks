"""The hourly maintenance job: fill closed buckets, then publish summaries."""

import os
import subprocess
import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.db import dashboard_aggregates
from coval_bench.db.dashboard_rollups import DrainResult
from coval_bench.db.dashboard_windows import RefreshResult

pg_conn = postgresql("pg_proc")


@pytest.mark.asyncio
async def test_reconciliation_fills_then_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = MagicMock()
    fill = AsyncMock(return_value=DrainResult(3, 1))
    refresh = AsyncMock(return_value=RefreshResult("published", 7))
    monkeypatch.setattr(dashboard_aggregates, "drain_rollup_queue", fill)
    monkeypatch.setattr(dashboard_aggregates, "refresh_window_views", refresh)
    at = datetime(2026, 9, 16, tzinfo=UTC)

    result = await dashboard_aggregates.refresh_dashboard_aggregates(pool, as_of=at)

    assert (result.filled, result.remaining, result.summary) == (
        3,
        1,
        RefreshResult("published", 7),
    )
    assert fill.await_args is not None and fill.await_args.kwargs["as_of"] == at
    refresh.assert_awaited_once_with(pool, as_of=at)


@pytest.mark.asyncio
async def test_fill_failure_still_publishes_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = MagicMock()
    monkeypatch.setattr(
        dashboard_aggregates, "drain_rollup_queue", AsyncMock(side_effect=RuntimeError("boom"))
    )
    refresh = AsyncMock(return_value=RefreshResult("published", 1))
    monkeypatch.setattr(dashboard_aggregates, "refresh_window_views", refresh)

    with pytest.raises(RuntimeError, match="missing buckets are retained"):
        await dashboard_aggregates.refresh_dashboard_aggregates(
            pool, as_of=datetime(2026, 9, 16, tzinfo=UTC)
        )
    refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconciliation_requires_timezone() -> None:
    with pytest.raises(ValueError, match="timezone"):
        await dashboard_aggregates.refresh_dashboard_aggregates(
            MagicMock(), as_of=datetime(2026, 9, 16)
        )


def test_aggregation_fingerprint_is_hash_seed_independent() -> None:
    code = (
        "from coval_bench.db.dashboard_contracts import aggregation_fingerprint; "
        "print(aggregation_fingerprint())"
    )
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.append(
            subprocess.check_output(  # noqa: S603
                [sys.executable, "-c", code], env=env, text=True
            ).strip()
        )
    assert outputs[0] == outputs[1]
