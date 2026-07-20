# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the S2S sample backfill."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from coval_bench.s2s import backfill
from coval_bench.s2s.samples import SampleRun

B1 = datetime(2026, 7, 18, tzinfo=UTC)
B2 = datetime(2026, 7, 19, tzinfo=UTC)


def _row(provider: str, model: str, bucket: datetime, coval_run_id: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "scheduled_at": bucket,
        "coval_run_id": coval_run_id,
    }


def test_group_rows_buckets_by_scheduled_at_and_skips_empty() -> None:
    rows = [
        _row("openai", "gpt-realtime", B1, "R1"),
        _row("google", "gemini-live", B1, "R2"),
        _row("xai", "grok-realtime", B1, ""),  # no coval_run_id -> dropped
        _row("openai", "gpt-realtime", B2, "R3"),
    ]

    grouped = backfill._group_rows(rows)

    assert set(grouped) == {B1, B2}
    assert {r.provider for r in grouped[B1]} == {"openai", "google"}
    assert grouped[B1][0].bucket_at == B1
    assert [r.coval_run_id for r in grouped[B2]] == ["R3"]


@contextlib.asynccontextmanager
async def _noop_cm(*_args: Any, **_kwargs: Any) -> AsyncIterator[MagicMock]:
    yield MagicMock()


@pytest.mark.asyncio
async def test_backfill_copies_each_bucket_oldest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = {
        B2: [SampleRun(provider="openai", model="m", coval_run_id="R2", bucket_at=B2)],
        B1: [SampleRun(provider="openai", model="m", coval_run_id="R1", bucket_at=B1)],
    }
    monkeypatch.setattr(backfill, "_runs_by_bucket", AsyncMock(return_value=runs))
    monkeypatch.setattr(backfill, "_client", lambda _s: _noop_cm())
    monkeypatch.setattr(backfill, "lifespan_pool", lambda _s: _noop_cm())

    order: list[datetime] = []

    async def fake_copy(_client: Any, *, bucket_name: str, runs: list[SampleRun], rng: Any) -> int:
        order.append(runs[0].bucket_at)
        return 3

    monkeypatch.setattr(backfill, "copy_tick_samples", fake_copy)

    settings = MagicMock()
    settings.s2s_samples_bucket = "bkt"
    result = await backfill.backfill_v2v_samples(settings, days=7)

    assert order == [B1, B2]  # oldest first, so the index ends up newest-first
    assert result == {"buckets": 2, "stored": 2, "skipped": 0}


@pytest.mark.asyncio
async def test_backfill_counts_skipped_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = {B1: [SampleRun(provider="openai", model="m", coval_run_id="R1", bucket_at=B1)]}
    monkeypatch.setattr(backfill, "_runs_by_bucket", AsyncMock(return_value=runs))
    monkeypatch.setattr(backfill, "_client", lambda _s: _noop_cm())
    monkeypatch.setattr(backfill, "lifespan_pool", lambda _s: _noop_cm())
    # 0 recordings => the bucket already existed or had no shared clip.
    monkeypatch.setattr(backfill, "copy_tick_samples", AsyncMock(return_value=0))

    settings = MagicMock()
    settings.s2s_samples_bucket = "bkt"
    result = await backfill.backfill_v2v_samples(settings, days=7)

    assert result == {"buckets": 1, "stored": 0, "skipped": 1}


@pytest.mark.asyncio
async def test_backfill_requires_bucket() -> None:
    settings = MagicMock()
    settings.s2s_samples_bucket = ""
    with pytest.raises(RuntimeError, match="s2s_samples_bucket"):
        await backfill.backfill_v2v_samples(settings, days=7)
