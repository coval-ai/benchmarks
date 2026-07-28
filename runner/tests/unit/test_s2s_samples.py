# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the multi-turn S2S sample publisher."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from google.api_core.exceptions import NotFound

from coval_bench.s2s import samples
from coval_bench.s2s.samples import PREFIX, SampleRun, publish_tick_sample

BUCKET_AT = datetime(2026, 7, 17, tzinfo=UTC)
TICK = "2026-07-17T00:00:00Z"
# The fetch window spans two periods, so a run set can carry the previous day too.
PREV_BUCKET_AT = datetime(2026, 7, 16, tzinfo=UTC)
PREV_TICK = "2026-07-16T00:00:00Z"
TEST_SET = "DvAqQ4md"

# Real bench persona ids so the label map is exercised too.
FEMALE = "PN3xgmsqeLDjsNNEA2e55e"
MALE = "9ATy64zKXxSUaVWb5YnQtd"


def _run(
    provider: str,
    model: str,
    run_id: str,
    persona_id: str,
    agent_id: str,
    bucket_at: datetime = BUCKET_AT,
) -> SampleRun:
    return SampleRun(
        provider=provider,
        model=model,
        coval_run_id=run_id,
        bucket_at=bucket_at,
        persona_id=persona_id,
        agent_id=agent_id,
    )


# One day's worth: two providers x two personas, ids suffixed so each day's runs
# stay distinct when a set spans two buckets.
def _runs_on(bucket_at: datetime, suffix: str) -> list[SampleRun]:
    return [
        _run("openai", "gpt-realtime", f"RO_F{suffix}", FEMALE, "AO", bucket_at),
        _run("openai", "gpt-realtime", f"RO_M{suffix}", MALE, "AO", bucket_at),
        _run("google", "gemini-live", f"RG_F{suffix}", FEMALE, "AG", bucket_at),
        _run("google", "gemini-live", f"RG_M{suffix}", MALE, "AG", bucket_at),
    ]


# Two providers, each run once per persona (mirrors run = agent + persona + test_set).
def _runs(*, include_google_male: bool = True) -> list[SampleRun]:
    runs = [
        _run("openai", "gpt-realtime", "RO_F", FEMALE, "AO"),
        _run("openai", "gpt-realtime", "RO_M", MALE, "AO"),
        _run("google", "gemini-live", "RG_F", FEMALE, "AG"),
    ]
    if include_google_male:
        runs.append(_run("google", "gemini-live", "RG_M", MALE, "AG"))
    return runs


def _sims_payload(run_id: str, test_cases: list[str]) -> dict[str, Any]:
    return {
        "simulations": [
            {"simulation_id": f"{run_id}-{tc}", "test_case_id": tc} for tc in test_cases
        ]
    }


def _fake_client(
    test_cases_by_run: dict[str, list[str]],
    *,
    audio_404: frozenset[str] = frozenset(),
    empty_turns: frozenset[str] = frozenset(),
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/simulations"):
            run_id = request.url.params["filter"].split('"')[1]
            return httpx.Response(200, json=_sims_payload(run_id, test_cases_by_run[run_id]))
        if "blobs.test" in str(request.url):
            return httpx.Response(200, content=b"RIFFfake")
        if path.endswith("/audio"):
            sim_id = path.rsplit("/", 2)[-2]
            if sim_id in audio_404:
                return httpx.Response(404)
            return httpx.Response(200, json={"audio_url": "https://blobs.test/x.wav"})
        # sim detail: /simulations/{sim_id}
        sim_id = path.rsplit("/", 1)[-1]
        turns = (
            []
            if sim_id in empty_turns
            else [
                {"role": "user", "content": "hi there", "start_offset": 1.5, "end_offset": 2.25},
                {"role": "assistant", "content": "how can i help", "start_offset": 3.0},
                # The persona's end_conversation tool record, as Coval returns it.
                {"role": "tool", "content": '{"function": "end_conversation"}'},
            ]
        )
        return httpx.Response(200, json={"simulation": {"transcript": turns}})

    return httpx.AsyncClient(base_url="https://api.test/v1", transport=httpx.MockTransport(handler))


class _FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_reads: set[str] = set()

    def blob(self, key: str) -> MagicMock:
        blob = MagicMock()
        blob.upload_from_string = lambda data, content_type: self.objects.__setitem__(
            key, data if isinstance(data, bytes) else data.encode()
        )

        def _download() -> bytes:
            if key in self.fail_reads:
                raise RuntimeError("transient read failure")
            if key not in self.objects:
                raise NotFound(key)  # type: ignore[no-untyped-call]
            return self.objects[key]

        blob.download_as_bytes = _download
        blob.exists = lambda: key in self.objects
        return blob


def _fake_storage() -> tuple[MagicMock, _FakeBucket]:
    bucket = _FakeBucket()
    client = MagicMock()
    client.bucket.return_value = bucket
    return client, bucket


async def _publish(
    client: httpx.AsyncClient,
    storage_client: MagicMock,
    runs: list[SampleRun],
    *,
    rng_seed: int = 0,
    expected_providers: set[str] | None = None,
) -> int:
    return await publish_tick_sample(
        client,
        bucket_name="bkt",
        test_set_id=TEST_SET,
        runs=runs,
        rng=random.Random(rng_seed),
        storage_client=storage_client,
        download_client=client,
        expected_providers=expected_providers,
    )


@pytest.mark.asyncio
async def test_publishes_complete_conversation_for_all_providers() -> None:
    storage_client, bucket = _fake_storage()
    cases = {"RO_F": ["b", "c"], "RG_F": ["b", "c"], "RO_M": ["b", "c"], "RG_M": ["b", "c"]}
    async with _fake_client(cases) as client:
        stored = await _publish(client, storage_client, _runs())

    assert stored == 2
    assert f"{PREFIX}/{TICK}/openai.wav" in bucket.objects
    assert f"{PREFIX}/{TICK}/google.wav" in bucket.objects

    manifest = json.loads(bucket.objects[f"{PREFIX}/{TICK}/manifest.json"])
    assert manifest["schema_version"] == 2
    assert manifest["bucket_at"] == TICK
    assert manifest["test_set_id"] == TEST_SET
    assert manifest["persona_name"] in {"Standard Female", "Standard Male"}
    assert "input_audio_url" not in manifest  # dropped in v2
    assert "transcript" not in manifest  # dropped in v2
    assert {r["provider"] for r in manifest["recordings"]} == {"openai", "google"}
    for rec in manifest["recordings"]:
        assert rec["agent_id"] in {"AO", "AG"}
        assert [t["index"] for t in rec["turns"]] == [0, 1]
        assert [t["role"] for t in rec["turns"]] == ["user", "assistant"]
        # Per-agent offsets flow through; a missing offset coerces to None.
        assert [t["start_offset"] for t in rec["turns"]] == [1.5, 3.0]
        assert [t["end_offset"] for t in rec["turns"]] == [2.25, None]

    assert json.loads(bucket.objects[samples.INDEX_KEY]) == [TICK]


@pytest.mark.asyncio
async def test_conversation_key_never_mixes_personas() -> None:
    # Female shares only "b"; male shares only "d" — the pick must stay within one persona.
    storage_client, bucket = _fake_storage()
    cases = {"RO_F": ["b"], "RG_F": ["b"], "RO_M": ["d"], "RG_M": ["d"]}
    async with _fake_client(cases) as client:
        stored = await _publish(client, storage_client, _runs())

    assert stored == 2
    manifest = json.loads(bucket.objects[f"{PREFIX}/{TICK}/manifest.json"])
    assert (manifest["persona_name"], manifest["test_case_id"]) in {
        ("Standard Female", "b"),
        ("Standard Male", "d"),
    }


@pytest.mark.asyncio
async def test_incomplete_persona_is_skipped_and_other_is_published() -> None:
    # Female's openai audio is missing → female can never complete → male is published.
    storage_client, bucket = _fake_storage()
    cases = {"RO_F": ["b"], "RG_F": ["b"], "RO_M": ["c"], "RG_M": ["c"]}
    async with _fake_client(cases, audio_404=frozenset({"RO_F-b"})) as client:
        stored = await _publish(client, storage_client, _runs())

    assert stored == 2
    manifest = json.loads(bucket.objects[f"{PREFIX}/{TICK}/manifest.json"])
    assert manifest["persona_name"] == "Standard Male"
    assert manifest["test_case_id"] == "c"
    assert {r["provider"] for r in manifest["recordings"]} == {"openai", "google"}


@pytest.mark.asyncio
async def test_no_complete_conversation_publishes_nothing() -> None:
    # Every candidate has a provider missing audio → no partial sample, nothing stored.
    storage_client, bucket = _fake_storage()
    cases = {"RO_F": ["b"], "RG_F": ["b"], "RO_M": ["c"], "RG_M": ["c"]}
    async with _fake_client(cases, audio_404=frozenset({"RO_F-b", "RO_M-c"})) as client:
        stored = await _publish(client, storage_client, _runs())

    assert stored == 0
    assert f"{PREFIX}/{TICK}/manifest.json" not in bucket.objects


@pytest.mark.asyncio
async def test_empty_transcript_counts_as_incomplete() -> None:
    # Only the female persona is eligible, but one provider's transcript is empty → skip.
    storage_client, bucket = _fake_storage()
    runs = [
        _run("openai", "gpt-realtime", "RO_F", FEMALE, "AO"),
        _run("google", "gemini-live", "RG_F", FEMALE, "AG"),
    ]
    async with _fake_client(
        {"RO_F": ["b"], "RG_F": ["b"]}, empty_turns=frozenset({"RO_F-b"})
    ) as client:
        stored = await _publish(client, storage_client, runs)

    assert stored == 0
    assert bucket.objects == {}


@pytest.mark.asyncio
async def test_persona_missing_a_provider_is_skipped() -> None:
    # No google male run → male persona lacks a provider → only female is eligible.
    storage_client, bucket = _fake_storage()
    cases = {"RO_F": ["b"], "RG_F": ["b"], "RO_M": ["b"]}
    async with _fake_client(cases) as client:
        stored = await _publish(client, storage_client, _runs(include_google_male=False))

    assert stored == 2
    manifest = json.loads(bucket.objects[f"{PREFIX}/{TICK}/manifest.json"])
    assert manifest["persona_name"] == "Standard Female"


@pytest.mark.asyncio
async def test_no_shared_conversation_stores_nothing() -> None:
    storage_client, bucket = _fake_storage()
    cases = {"RO_F": ["a"], "RG_F": ["b"], "RO_M": ["c"], "RG_M": ["d"]}
    async with _fake_client(cases) as client:
        stored = await _publish(client, storage_client, _runs())

    assert stored == 0
    assert bucket.objects == {}


@pytest.mark.asyncio
async def test_existing_manifest_is_never_overwritten() -> None:
    storage_client, bucket = _fake_storage()
    bucket.objects[f"{PREFIX}/{TICK}/manifest.json"] = b'{"sentinel": true}'
    cases = {"RO_F": ["b"], "RG_F": ["b"], "RO_M": ["b"], "RG_M": ["b"]}
    async with _fake_client(cases) as client:
        stored = await _publish(client, storage_client, _runs())

    assert stored == 0
    assert bucket.objects[f"{PREFIX}/{TICK}/manifest.json"] == b'{"sentinel": true}'
    assert json.loads(bucket.objects[samples.INDEX_KEY]) == [TICK]  # index repaired


@pytest.mark.asyncio
async def test_index_keeps_existing_history() -> None:
    storage_client, bucket = _fake_storage()
    bucket.objects[samples.INDEX_KEY] = json.dumps([PREV_TICK]).encode()
    cases = {"RO_F": ["b"], "RG_F": ["b"], "RO_M": ["b"], "RG_M": ["b"]}
    async with _fake_client(cases) as client:
        await _publish(client, storage_client, _runs())

    assert json.loads(bucket.objects[samples.INDEX_KEY]) == [TICK, PREV_TICK]


@pytest.mark.asyncio
async def test_index_stays_newest_first_when_an_older_day_lands_later() -> None:
    """A recovered day is older than the history, and must not become index[0].

    The dashboard treats index[0] as the latest tick, so ordering has to come
    from the timestamps rather than from the order days happened to be written.
    """
    storage_client, bucket = _fake_storage()
    newer = ["2026-07-27T00:00:00Z", TICK]
    bucket.objects[samples.INDEX_KEY] = json.dumps(newer).encode()
    runs = _runs_on(PREV_BUCKET_AT, "_Y")
    cases = {r.coval_run_id: ["b"] for r in runs}
    async with _fake_client(cases) as client:
        await _publish(client, storage_client, runs)

    assert json.loads(bucket.objects[samples.INDEX_KEY]) == [*newer, PREV_TICK]


@pytest.mark.asyncio
async def test_never_raises_on_total_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    storage_client, bucket = _fake_storage()
    async with httpx.AsyncClient(
        base_url="https://api.test/v1", transport=httpx.MockTransport(handler)
    ) as client:
        stored = await _publish(client, storage_client, _runs())

    assert stored == 0
    assert bucket.objects == {}


@pytest.mark.asyncio
async def test_missed_day_publishes_alongside_the_current_tick() -> None:
    """A run set spanning two buckets publishes BOTH — this is the day-gap recovery."""
    storage_client, bucket = _fake_storage()
    runs = _runs_on(PREV_BUCKET_AT, "_Y") + _runs_on(BUCKET_AT, "_T")
    cases = {r.coval_run_id: ["b", "c"] for r in runs}
    async with _fake_client(cases) as client:
        stored = await _publish(client, storage_client, runs)

    assert stored == 4  # two providers x two days
    for tick in (PREV_TICK, TICK):
        manifest = json.loads(bucket.objects[f"{PREFIX}/{tick}/manifest.json"])
        assert manifest["bucket_at"] == tick
        assert {r["provider"] for r in manifest["recordings"]} == {"openai", "google"}
        assert f"{PREFIX}/{tick}/openai.wav" in bucket.objects
        assert f"{PREFIX}/{tick}/google.wav" in bucket.objects
    # Published oldest-first, so the index ends up newest-first.
    assert json.loads(bucket.objects[f"{PREFIX}/index.json"]) == [TICK, PREV_TICK]


@pytest.mark.asyncio
async def test_bucket_missing_an_expected_provider_publishes_nothing() -> None:
    """A provider can be present today and absent yesterday.

    Completeness has to be judged against the configured provider set, not
    against whichever providers happen to appear in that one bucket — otherwise
    yesterday calls itself complete and ships a one-provider comparison.
    """
    storage_client, bucket = _fake_storage()
    openai_only = [
        _run("openai", "gpt-realtime", "RO_F_Y", FEMALE, "AO", PREV_BUCKET_AT),
        _run("openai", "gpt-realtime", "RO_M_Y", MALE, "AO", PREV_BUCKET_AT),
    ]
    runs = openai_only + _runs_on(BUCKET_AT, "_T")
    cases = {r.coval_run_id: ["b", "c"] for r in runs}
    async with _fake_client(cases) as client:
        stored = await _publish(
            client, storage_client, runs, expected_providers={"openai", "google"}
        )

    # Today is complete and publishes; yesterday is refused outright.
    assert stored == 2
    assert f"{PREFIX}/{TICK}/manifest.json" in bucket.objects
    assert f"{PREFIX}/{PREV_TICK}/manifest.json" not in bucket.objects
    assert f"{PREFIX}/{PREV_TICK}/openai.wav" not in bucket.objects
    assert json.loads(bucket.objects[samples.INDEX_KEY]) == [TICK]


@pytest.mark.asyncio
async def test_published_day_is_reindexed_even_when_a_provider_is_absent() -> None:
    """An index write that once failed must still be repairable.

    A published day whose current runs no longer cover every provider still owns
    its manifest, so it has to be put back in the index. Checking completeness
    first would strand it: its runs age out of the window, and no later fetch
    would revisit the bucket.
    """
    storage_client, bucket = _fake_storage()
    already_there = b'{"schema_version": 2, "bucket_at": "2026-07-16T00:00:00Z"}'
    bucket.objects[f"{PREFIX}/{PREV_TICK}/manifest.json"] = already_there
    # Only openai remains for that day, so the provider gate would otherwise fire.
    openai_only = [_run("openai", "gpt-realtime", "RO_F_Y", FEMALE, "AO", PREV_BUCKET_AT)]
    cases = {r.coval_run_id: ["b"] for r in openai_only}
    async with _fake_client(cases) as client:
        stored = await _publish(
            client, storage_client, openai_only, expected_providers={"openai", "google"}
        )

    assert stored == 0
    assert bucket.objects[f"{PREFIX}/{PREV_TICK}/manifest.json"] == already_there
    assert json.loads(bucket.objects[samples.INDEX_KEY]) == [PREV_TICK]


@pytest.mark.asyncio
async def test_recovery_never_overwrites_an_existing_manifest() -> None:
    storage_client, bucket = _fake_storage()
    already_there = b'{"schema_version": 2, "bucket_at": "2026-07-16T00:00:00Z"}'
    bucket.objects[f"{PREFIX}/{PREV_TICK}/manifest.json"] = already_there
    runs = _runs_on(PREV_BUCKET_AT, "_Y") + _runs_on(BUCKET_AT, "_T")
    cases = {r.coval_run_id: ["b", "c"] for r in runs}
    async with _fake_client(cases) as client:
        stored = await _publish(client, storage_client, runs)

    assert stored == 2  # only the current tick
    assert bucket.objects[f"{PREFIX}/{PREV_TICK}/manifest.json"] == already_there
    assert f"{PREFIX}/{PREV_TICK}/openai.wav" not in bucket.objects
    assert f"{PREFIX}/{TICK}/manifest.json" in bucket.objects
    # The existing day is still repaired into the index.
    assert PREV_TICK in json.loads(bucket.objects[f"{PREFIX}/index.json"])


@pytest.mark.asyncio
async def test_an_unusable_day_does_not_block_the_other() -> None:
    storage_client, bucket = _fake_storage()
    stale = _runs_on(PREV_BUCKET_AT, "_Y")
    runs = stale + _runs_on(BUCKET_AT, "_T")
    cases = {r.coval_run_id: ["b", "c"] for r in runs}
    # Every recording for the previous day 404s.
    gone = frozenset(f"{r.coval_run_id}-{tc}" for r in stale for tc in ("b", "c"))
    async with _fake_client(cases, audio_404=gone) as client:
        stored = await _publish(client, storage_client, runs)

    assert stored == 2
    assert f"{PREFIX}/{PREV_TICK}/manifest.json" not in bucket.objects
    assert f"{PREFIX}/{TICK}/manifest.json" in bucket.objects


def test_conversation_turns_coerces_offsets() -> None:
    sim = {
        "transcript": [
            {"role": "user", "content": "a", "start_offset": 1, "end_offset": 2.5},
            {"role": "assistant", "content": "b", "start_offset": "x", "end_offset": True},
            {"role": "user", "content": "c"},
        ]
    }
    turns = samples._conversation_turns(sim)
    assert [(t["start_offset"], t["end_offset"]) for t in turns] == [
        (1.0, 2.5),
        (None, None),
        (None, None),
    ]
