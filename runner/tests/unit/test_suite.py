from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib import resources
from typing import Any

import pytest

from coval_bench.config import Settings
from coval_bench.datasets.suite import (
    DEDICATED_STT_SUITE,
    DEDICATED_TTS_SAMPLE_SIZE,
    DEFAULT_SAMPLE_SIZE,
    SHARED_TTS_SUITE,
    tts_sample_size,
)
from coval_bench.runner import orchestrator
from coval_bench.runner.orchestrator import RunSummary, run_suite

_SETTINGS = Settings(
    database_url="postgresql://runner:password@localhost:5432/benchmarks",
    posthog_disabled=True,
)


def _summary(*, sigterm: bool = False) -> RunSummary:
    now = datetime.now(tz=UTC)
    return RunSummary(
        run_id=1,
        started_at=now,
        finished_at=now,
        status="succeeded",
        total_results=0,
        success_count=0,
        fail_count=0,
        sigterm=sigterm,
    )


def test_suite_sizes_fit_their_manifests() -> None:
    manifests = resources.files("coval_bench.datasets.manifests")
    for dataset_id, size in DEDICATED_STT_SUITE.items():
        items = json.loads(manifests.joinpath(f"{dataset_id}.json").read_text())["items"]
        assert size <= len(items), dataset_id


@pytest.mark.parametrize(
    ("source", "dataset_id", "override", "expected"),
    [
        ("shared", "tts-v2", None, SHARED_TTS_SUITE["tts-v2"]),
        ("shared", "tts-v1", None, DEFAULT_SAMPLE_SIZE),
        ("dedicated", "tts-v2", None, DEDICATED_TTS_SAMPLE_SIZE),
        ("shared", "tts-v2", 7, 7),
    ],
)
def test_tts_sample_size_owned_by_suite(
    source: str, dataset_id: str, override: int | None, expected: int
) -> None:
    assert tts_sample_size(source, dataset_id, override) == expected


@pytest.mark.asyncio
async def test_run_suite_walks_datasets_on_one_tick_past_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_run(**kwargs: Any) -> RunSummary:
        calls.append(kwargs)
        if kwargs["dataset_id"] == "stt-v3":
            raise RuntimeError("boom")
        return _summary()

    monkeypatch.setattr(orchestrator, "run_benchmarks", fake_run)
    summaries = await run_suite(settings=_SETTINGS, benchmark_kind="stt", source="dedicated")

    assert [c["dataset_id"] for c in calls] == list(DEDICATED_STT_SUITE)
    assert len({c["scheduled_at"] for c in calls}) == 1
    assert len(summaries) == len(DEDICATED_STT_SUITE) - 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "kind", "dataset_id"),
    [("shared", "stt", None), ("dedicated", "tts", None), ("dedicated", "stt", "stt-v1")],
)
async def test_run_suite_single_run_otherwise(
    monkeypatch: pytest.MonkeyPatch, source: str, kind: str, dataset_id: str | None
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_run(**kwargs: Any) -> RunSummary:
        calls.append(kwargs)
        return _summary()

    monkeypatch.setattr(orchestrator, "run_benchmarks", fake_run)
    pinned = _SETTINGS.model_copy(update={"dataset_id": dataset_id})
    await run_suite(settings=pinned, benchmark_kind=kind, source=source)  # type: ignore[arg-type]

    assert len(calls) == 1
    assert "dataset_id" not in calls[0]


@pytest.mark.asyncio
async def test_run_suite_stops_after_sigterm(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(**kwargs: Any) -> RunSummary:
        return _summary(sigterm=True)

    monkeypatch.setattr(orchestrator, "run_benchmarks", fake_run)
    summaries = await run_suite(settings=_SETTINGS, benchmark_kind="stt", source="dedicated")

    assert len(summaries) == 1
