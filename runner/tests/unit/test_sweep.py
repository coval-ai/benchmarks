# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweeps against PostgreSQL, with a fake provider and dataset."""

from __future__ import annotations

import wave
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pytest_postgresql.factories import postgresql

from coval_bench.config import Settings
from coval_bench.db.registry_store import NewModel, RegistryStore
from coval_bench.providers.base import TranscriptionResult
from coval_bench.registries import Benchmark
from coval_bench.runner import sweep
from tests.unit.conftest import apply_migrations, open_pool

pg_conn = postgresql("pg_proc")

_SETTINGS = cast(Settings, SimpleNamespace())
_ONLY = frozenset({"acme/fast-1"})


async def _pool_with_model(conn: Any) -> Any:
    apply_migrations(conn)
    pool = await open_pool(conn)
    new = NewModel(modality=Benchmark.STT, provider="acme", model="fast-1")
    await RegistryStore(pool).insert_model(new, user_id="user_editor", org_id=None, email=None)
    return pool


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pool: Any,
    versions: dict[str, str],
    transcribe: Callable[[int], Any] = lambda _n: "hello world",
) -> list[int]:
    """Datasets of two clips each. *transcribe* gets the call number and returns the
    transcript, or an exception to raise."""
    calls: list[int] = []
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    clip = SimpleNamespace(path=path, duration_sec=0.1, speech_end_offset_ms=None)
    items = [SimpleNamespace(sample_id=name, path=f"audio/{name}.wav") for name in ("s1", "s2")]

    class Provider:
        def __init__(self, settings: Any, model: str) -> None:
            pass

        async def measure_ttft(self, *args: Any) -> TranscriptionResult:
            calls.append(len(calls) + 1)
            outcome = transcribe(calls[-1])
            if isinstance(outcome, BaseException):
                raise outcome
            return TranscriptionResult(
                provider="acme",
                ttft_seconds=0.2,
                audio_to_final_seconds=0.5,
                complete_transcript=outcome,
            )

    @asynccontextmanager
    async def fake_pool(_settings: Any) -> Any:
        yield pool

    async def nothing() -> None:
        return None

    monkeypatch.setattr(sweep, "lifespan_pool", fake_pool)
    monkeypatch.setattr(sweep, "SWEEP_STT_DATASETS", tuple(versions))
    monkeypatch.setattr(
        sweep,
        "dataset_manifest",
        lambda dataset_id: SimpleNamespace(version=versions[dataset_id], items=items),
    )
    monkeypatch.setattr(sweep, "stt_item_fetcher", lambda _dataset_id, settings: lambda _raw: clip)
    monkeypatch.setattr(sweep, "_get_stt_providers", lambda: {"acme": Provider})
    monkeypatch.setattr(sweep, "_close_http_clients", nothing)
    return calls


async def _run(max_minutes: float = 1) -> sweep.SweepSummary:
    return await sweep.run_sweeps(settings=_SETTINGS, only=_ONLY, max_minutes=max_minutes)


def _query(conn: Any, sql: str) -> list[Any]:
    conn.autocommit = True
    return list(conn.execute(sql).fetchall())


_SWEEPS = (
    "SELECT model_version, dataset_id, dataset_version, completed_at IS NOT NULL"
    " FROM benchmarks_v2.sweeps ORDER BY id"
)
_RESULTS = "SELECT sample_id, transcript, error FROM benchmarks_v2.sweep_results ORDER BY id"


async def test_a_run_opens_one_sweep_per_dataset_and_a_rerun_opens_none(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = await _pool_with_model(pg_conn)
    try:
        _wire(monkeypatch, tmp_path, pool, {"ds-1": "1.0.0", "ds-2": "1.0.0"})
        first = await _run(max_minutes=0)
        second = await _run(max_minutes=0)
    finally:
        await pool.close()

    assert (first.models, first.opened, second.opened) == (1, 2, 0)
    assert _query(pg_conn, _SWEEPS) == [(1, "ds-1", "1.0.0", False), (1, "ds-2", "1.0.0", False)]


async def test_a_model_version_bump_or_a_new_dataset_version_opens_a_new_sweep(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = await _pool_with_model(pg_conn)
    try:
        versions = {"ds-1": "1.0.0"}
        _wire(monkeypatch, tmp_path, pool, versions)
        await _run(max_minutes=0)
        versions["ds-1"] = "1.1.0"
        redone = await _run(max_minutes=0)
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE benchmarks_v2.models SET version = 2 WHERE provider = 'acme'"
            )
        bumped = await _run(max_minutes=0)
    finally:
        await pool.close()

    assert (redone.opened, bumped.opened) == (1, 1)
    assert [row[:3] for row in _query(pg_conn, _SWEEPS)] == [
        (1, "ds-1", "1.0.0"),
        (1, "ds-1", "1.1.0"),
        (2, "ds-1", "1.1.0"),
    ]


async def test_a_sweep_runs_each_item_once_stores_it_and_completes(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = await _pool_with_model(pg_conn)
    try:
        calls = _wire(monkeypatch, tmp_path, pool, {"ds-1": "1.0.0"})
        first = await _run()
        second = await _run()
    finally:
        await pool.close()

    assert (first.called, first.failed, first.completed) == (2, 0, 1)
    assert (second.called, second.completed) == (0, 0)
    assert len(calls) == 2
    assert _query(pg_conn, _SWEEPS) == [(1, "ds-1", "1.0.0", True)]
    assert _query(pg_conn, _RESULTS) == [
        ("s1", "hello world", None),
        ("s2", "hello world", None),
    ]
    assert _query(pg_conn, "SELECT timings FROM benchmarks_v2.sweep_results LIMIT 1") == [
        ({"ttft_seconds": 0.2, "audio_to_final_seconds": 0.5},)
    ]


async def test_a_failed_call_is_stored_as_failed_and_the_sweep_still_completes(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = await _pool_with_model(pg_conn)
    try:
        _wire(
            monkeypatch,
            tmp_path,
            pool,
            {"ds-1": "1.0.0"},
            lambda n: RuntimeError("upstream 503") if n == 1 else "hello world",
        )
        summary = await _run()
    finally:
        await pool.close()

    assert (summary.called, summary.failed, summary.completed) == (2, 1, 1)
    assert _query(pg_conn, _RESULTS) == [
        ("s1", None, "upstream 503"),
        ("s2", "hello world", None),
    ]


async def test_a_run_out_of_time_leaves_the_rest_for_the_next_run(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = await _pool_with_model(pg_conn)
    try:
        calls = _wire(monkeypatch, tmp_path, pool, {"ds-1": "1.0.0"})
        stopped = await _run(max_minutes=0)
        resumed = await _run()
    finally:
        await pool.close()

    assert (stopped.called, stopped.completed) == (0, 0)
    assert (resumed.called, resumed.completed) == (2, 1)
    assert len(calls) == 2


async def test_a_clip_that_fails_to_load_is_left_for_the_next_run(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(_dataset_id: str, settings: Any) -> Any:
        def fetch(_raw: Any) -> Any:
            raise OSError("storage unavailable")

        return fetch

    pool = await _pool_with_model(pg_conn)
    try:
        calls = _wire(monkeypatch, tmp_path, pool, {"ds-1": "1.0.0"})
        monkeypatch.setattr(sweep, "stt_item_fetcher", broken)
        summary = await _run()
    finally:
        await pool.close()

    assert (summary.called, summary.failed, summary.completed, len(calls)) == (0, 0, 0, 0)
    assert _query(pg_conn, _RESULTS) == []


async def test_an_item_whose_model_is_busy_is_left_for_the_next_run(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    @asynccontextmanager
    async def busy(_pool: Any, _key: str, *, timeout_s: float) -> Any:
        yield False

    pool = await _pool_with_model(pg_conn)
    try:
        calls = _wire(monkeypatch, tmp_path, pool, {"ds-1": "1.0.0"})
        monkeypatch.setattr(sweep, "model_lock", busy)
        summary = await _run()
    finally:
        await pool.close()

    assert (summary.called, summary.completed, len(calls)) == (0, 0, 0)
    assert _query(pg_conn, _SWEEPS) == [(1, "ds-1", "1.0.0", False)]
