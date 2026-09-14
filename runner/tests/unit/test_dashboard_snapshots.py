"""Pure readiness checks for normalized dashboard readers."""

from __future__ import annotations

import datetime as dt
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from psycopg import AsyncConnection

from coval_bench.api import dashboard_snapshots


def _connection(row: object) -> AsyncConnection[Any]:
    connection = Mock(spec=AsyncConnection)
    cursor = Mock()
    cursor.fetchone = AsyncMock(return_value=row)
    connection.execute = AsyncMock(return_value=cursor)
    return cast(AsyncConnection[Any], connection)


def _row(*, generation: int = 3, revision: int = 1, fingerprint: str = "fp") -> dict[str, object]:
    now = dt.datetime.now(dt.UTC)
    return {
        "generation": generation,
        "as_of": now,
        "published_at": now,
        "definition_revision": revision,
        "definition_fingerprint": fingerprint,
    }


@pytest.mark.asyncio
async def test_snapshot_accepts_initialized_current_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dashboard_snapshots, "aggregation_fingerprint", lambda: "fp")
    snapshot = await dashboard_snapshots.require_snapshot(_connection(_row()))
    assert snapshot.generation == 3
    assert snapshot.stale is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [None, _row(generation=0), _row(revision=99), _row(fingerprint="old")],
    ids=["missing", "unpublished", "revision", "fingerprint"],
)
async def test_snapshot_rejects_unusable_state(
    monkeypatch: pytest.MonkeyPatch, row: object
) -> None:
    monkeypatch.setattr(dashboard_snapshots, "aggregation_fingerprint", lambda: "fp")
    with pytest.raises(HTTPException) as error:
        await dashboard_snapshots.require_snapshot(_connection(row))
    assert error.value.status_code == 503
    assert error.value.detail == "dashboard_snapshot_not_ready"


@pytest.mark.asyncio
async def test_snapshot_marks_old_publication_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dashboard_snapshots, "aggregation_fingerprint", lambda: "fp")
    row = _row()
    row["published_at"] = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=31)
    snapshot = await dashboard_snapshots.require_snapshot(_connection(row))
    assert snapshot.stale is True
