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
from coval_bench.providers.base import TranscriptionResult, TTSResult
from coval_bench.registries import Benchmark, Gender, Voice
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
    """Datasets of two clips each, both with the reference "hello world". *transcribe*
    gets the call number and returns the transcript, or an exception to raise."""
    calls: list[int] = []
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    clip = SimpleNamespace(path=path, duration_sec=0.1, speech_end_offset_ms=None)
    items = [
        SimpleNamespace(sample_id=name, path=f"audio/{name}.wav", transcript="hello world")
        for name in ("s1", "s2")
    ]

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


async def _run(max_minutes: float = 1, kind: sweep.Kind = "stt") -> sweep.SweepSummary:
    return await sweep.run_sweeps(
        settings=_SETTINGS, kind=kind, only=_ONLY, max_minutes=max_minutes
    )


def _query(conn: Any, sql: str) -> list[Any]:
    conn.autocommit = True
    return list(conn.execute(sql).fetchall())


_SWEEPS = (
    "SELECT model_version, dataset_id, dataset_version, completed_at IS NOT NULL"
    " FROM benchmarks_v2.sweeps ORDER BY id"
)
_RESULTS = "SELECT sample_id, transcript FROM benchmarks_v2.sweep_results ORDER BY id"
_QUEUE = (
    "SELECT sample_id, tries_left, last_error FROM benchmarks_v2.sweep_queue"
    " ORDER BY sweep_id, sample_id"
)


async def test_a_run_opens_one_sweep_per_dataset_queues_its_items_and_a_rerun_opens_none(
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
    assert _query(pg_conn, _QUEUE) == [("s1", 3, None), ("s2", 3, None)] * 2


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
    assert _query(pg_conn, _RESULTS) == [("s1", "hello world"), ("s2", "hello world")]
    assert _query(pg_conn, _QUEUE) == []
    assert _query(pg_conn, "SELECT timings FROM benchmarks_v2.sweep_results LIMIT 1") == [
        ({"ttft_seconds": 0.2, "audio_to_final_seconds": 0.5},)
    ]


async def test_an_item_the_provider_fails_is_run_again_on_the_next_run(
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
        first = await _run()
        queue_after_first = _query(pg_conn, _QUEUE)
        second = await _run()
    finally:
        await pool.close()

    assert (first.called, first.failed, first.completed) == (2, 1, 0)
    assert queue_after_first == [("s1", 2, "upstream 503")]
    assert (second.called, second.failed, second.scored, second.completed) == (1, 0, 1, 1)
    assert _query(pg_conn, _RESULTS) == [("s2", "hello world"), ("s1", "hello world")]
    assert _query(pg_conn, _QUEUE) == []


async def test_an_item_out_of_tries_stays_queued_at_zero_and_the_sweep_completes(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = await _pool_with_model(pg_conn)
    try:
        calls = _wire(
            monkeypatch,
            tmp_path,
            pool,
            {"ds-1": "1.0.0"},
            lambda n: "hello world" if n == 2 else RuntimeError("upstream 503"),
        )
        runs = [await _run() for _ in range(4)]
    finally:
        await pool.close()

    assert [run.called for run in runs] == [2, 1, 1, 0]
    assert [run.completed for run in runs] == [0, 0, 1, 0]
    assert len(calls) == 4
    assert _query(pg_conn, _RESULTS) == [("s2", "hello world")]
    assert _query(pg_conn, _QUEUE) == [("s1", 0, "upstream 503")]


async def test_a_clean_transcript_is_scored_and_a_failed_call_is_not(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = await _pool_with_model(pg_conn)
    try:
        _wire(
            monkeypatch,
            tmp_path,
            pool,
            {"ds-1": "1.0.0", "ds-2": "1.0.0"},
            lambda n: ("hello world", "hello", RuntimeError("upstream 503"), "hello world")[n - 1],
        )
        summary = await _run()
    finally:
        await pool.close()

    assert (summary.called, summary.failed, summary.scored) == (4, 1, 3)
    scores = _query(
        pg_conn,
        """SELECT w.dataset_id, r.sample_id, s.metric, s.metric_version, s.errors,
                  s.reference_units
           FROM benchmarks_v2.sweep_scores s
           JOIN benchmarks_v2.sweep_results r ON r.id = s.sweep_result_id
           JOIN benchmarks_v2.sweeps w ON w.id = r.sweep_id
           ORDER BY w.dataset_id, r.sample_id""",
    )
    assert scores == [
        ("ds-1", "s1", "WER", "4", 0, 2),
        ("ds-1", "s2", "WER", "4", 1, 2),
        ("ds-2", "s2", "WER", "4", 0, 2),
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


async def test_a_clip_that_fails_to_load_keeps_its_tries(
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
    assert _query(pg_conn, _QUEUE) == [("s1", 3, None), ("s2", 3, None)]


async def test_an_item_whose_model_is_busy_keeps_its_tries(
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
    assert _query(pg_conn, _QUEUE) == [("s1", 3, None), ("s2", 3, None)]


def _stored(key: str, _path: Path) -> str:
    return f"gs://artifacts/{key}"


async def _tts(
    conn: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    whisper: Any,
    upload: Any = _stored,
) -> tuple[Any, list[tuple[str | None, str]]]:
    """A TTS model with two voices and a dataset of one prompt. Returns the pool and
    the (voice, text) pairs the provider was asked to speak."""
    apply_migrations(conn)
    pool = await open_pool(conn)
    voices = (Voice(id="f1", gender=Gender.FEMALE), Voice(id="m1", gender=Gender.MALE))
    new = NewModel(modality=Benchmark.TTS, provider="acme", model="fast-1", voices=voices)
    await RegistryStore(pool).insert_model(new, user_id="user_editor", org_id=None, email=None)
    prompts = [SimpleNamespace(testcase_id="t1", transcript="Good morning.", spoken_reference=None)]
    spoken: list[tuple[str | None, str]] = []

    class Provider:
        def __init__(self, *, settings: Any, model: str, voice: str | None) -> None:
            self.voice = voice

        async def synthesize(self, text: str) -> TTSResult:
            spoken.append((self.voice, text))
            path = tmp_path / f"{self.voice}.wav"
            path.write_bytes(b"audio")
            return TTSResult(
                provider="acme",
                model="fast-1",
                voice=self.voice or "",
                ttfa_ms=180.0,
                audio_path=path,
                error=None,
            )

    @asynccontextmanager
    async def fake_pool(_settings: Any) -> Any:
        yield pool

    async def nothing() -> None:
        return None

    monkeypatch.setattr(sweep, "lifespan_pool", fake_pool)
    monkeypatch.setattr(sweep, "SWEEP_TTS_DATASETS", ("tts-1",))
    monkeypatch.setattr(
        sweep, "dataset_manifest", lambda _id: SimpleNamespace(version="1.0.0", items=prompts)
    )
    monkeypatch.setattr(sweep, "_get_tts_providers", lambda: {"acme": Provider})
    monkeypatch.setattr(sweep, "_transcribe_with_whisper", whisper)
    monkeypatch.setattr(sweep, "_uploader", lambda _settings: upload)
    monkeypatch.setattr(sweep, "_close_http_clients", nothing)
    return pool, spoken


_TTS_QUEUE = "SELECT sample_id, voice, tries_left FROM benchmarks_v2.sweep_queue ORDER BY 1, 2"


async def test_a_tts_sweep_runs_each_prompt_in_each_voice_and_scores_the_whisper_transcript(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def whisper(_path: Path, _settings: Any) -> str:
        return "good morning"

    pool, spoken = await _tts(pg_conn, monkeypatch, tmp_path, whisper)
    try:
        queued = await _run(max_minutes=0, kind="tts")
        queue_before = _query(pg_conn, _TTS_QUEUE)
        summary = await _run(kind="tts")
    finally:
        await pool.close()

    assert (queued.opened, queued.called) == (1, 0)
    assert queue_before == [("t1", "f1", 3), ("t1", "m1", 3)]
    assert (summary.called, summary.failed, summary.scored, summary.completed) == (2, 0, 2, 1)
    assert sorted(spoken) == [("f1", "Good morning."), ("m1", "Good morning.")]
    assert not list(tmp_path.glob("*.wav"))
    assert _query(pg_conn, _TTS_QUEUE) == []
    assert _query(
        pg_conn,
        """SELECT r.voice, r.transcript, s.errors, s.reference_units, r.audio_uri
           FROM benchmarks_v2.sweep_results r
           JOIN benchmarks_v2.sweep_scores s ON s.sweep_result_id = r.id ORDER BY 1""",
    ) == [
        ("f1", "good morning", 0, 2, "gs://artifacts/sweeps/1/t1/f1.wav"),
        ("m1", "good morning", 0, 2, "gs://artifacts/sweeps/1/t1/m1.wav"),
    ]


async def test_a_tts_item_whisper_cannot_transcribe_loses_a_try(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def whisper(_path: Path, _settings: Any) -> str:
        raise RuntimeError("whisper unavailable")

    pool, spoken = await _tts(pg_conn, monkeypatch, tmp_path, whisper)
    try:
        summary = await _run(kind="tts")
    finally:
        await pool.close()

    assert (summary.called, summary.failed, summary.completed, len(spoken)) == (2, 2, 0, 2)
    assert not list(tmp_path.glob("*.wav"))
    assert _query(pg_conn, _QUEUE) == [
        ("t1", 2, "whisper: whisper unavailable"),
        ("t1", 2, "whisper: whisper unavailable"),
    ]
    assert _query(pg_conn, _RESULTS) == []


async def test_a_tts_item_whose_audio_cannot_be_kept_loses_a_try(
    pg_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def whisper(_path: Path, _settings: Any) -> str:
        return "good morning"

    def upload(_key: str, _path: Path) -> str:
        raise OSError("bucket unavailable")

    pool, _ = await _tts(pg_conn, monkeypatch, tmp_path, whisper, upload)
    try:
        summary = await _run(kind="tts")
    finally:
        await pool.close()

    assert (summary.called, summary.failed, summary.completed) == (2, 2, 0)
    assert not list(tmp_path.glob("*.wav"))
    assert _query(pg_conn, _QUEUE) == [("t1", 2, "upload: bucket unavailable")] * 2
    assert _query(pg_conn, _RESULTS) == []
