# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the S2S sample-recording copier."""

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
from coval_bench.s2s.samples import PREFIX, SampleRun, copy_tick_samples

BUCKET_AT = datetime(2026, 7, 17, tzinfo=UTC)

RUNS = [
    SampleRun(provider="openai", model="gpt-realtime", coval_run_id="RO", bucket_at=BUCKET_AT),
    SampleRun(provider="google", model="gemini-live", coval_run_id="RG", bucket_at=BUCKET_AT),
]


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
    transcript: str = "is my alarm set for tomorrow morning",
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/simulations"):
            flt = request.url.params["filter"]
            run_id = flt.split('"')[1]
            return httpx.Response(200, json=_sims_payload(run_id, test_cases_by_run[run_id]))
        if path.endswith("/audio"):
            sim_id = path.rsplit("/", 2)[-2]
            if sim_id in audio_404:
                return httpx.Response(404)
            return httpx.Response(200, json={"audio_url": "https://blobs.test/x.wav"})
        if "blobs.test" in str(request.url):
            return httpx.Response(200, content=b"RIFFfake")
        return httpx.Response(
            200,
            json={"transcript": [{"role": "user", "content": transcript}], "test_case_id": "tc"},
        )

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


@pytest.mark.asyncio
async def test_copies_shared_clip_for_all_providers() -> None:
    storage_client, bucket = _fake_storage()
    async with _fake_client({"RO": ["a", "b", "c"], "RG": ["b", "c", "d"]}) as client:
        stored = await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(7),
            storage_client=storage_client,
            download_client=client,
        )

    assert stored == 2
    tick = "2026-07-17T00:00:00Z"
    assert f"{PREFIX}/{tick}/openai.wav" in bucket.objects
    assert f"{PREFIX}/{tick}/google.wav" in bucket.objects

    manifest = json.loads(bucket.objects[f"{PREFIX}/{tick}/manifest.json"])
    assert manifest["bucket_at"] == tick
    assert manifest["test_case_id"] in {"b", "c"}  # only shared clips eligible
    assert manifest["transcript"] == "is my alarm set for tomorrow morning"
    assert manifest["input_audio_url"].endswith("/s2s-v1/audio/s2s-v1-0001.wav")
    assert {r["provider"] for r in manifest["recordings"]} == {"openai", "google"}

    index = json.loads(bucket.objects[samples.INDEX_KEY])
    assert index == [tick]


@pytest.mark.asyncio
async def test_deterministic_pick_under_seeded_rng() -> None:
    picks = set()
    for _ in range(3):
        storage_client, bucket = _fake_storage()
        async with _fake_client({"RO": ["a", "b", "c"], "RG": ["a", "b", "c"]}) as client:
            await copy_tick_samples(
                client,
                bucket_name="bkt",
                runs=RUNS,
                rng=random.Random(42),
                storage_client=storage_client,
                download_client=client,
            )
        manifest = json.loads(bucket.objects[f"{PREFIX}/2026-07-17T00:00:00Z/manifest.json"])
        picks.add(manifest["test_case_id"])
    assert len(picks) == 1


@pytest.mark.asyncio
async def test_no_shared_clip_stores_nothing() -> None:
    storage_client, bucket = _fake_storage()
    async with _fake_client({"RO": ["a"], "RG": ["b"]}) as client:
        stored = await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(0),
            storage_client=storage_client,
            download_client=client,
        )
    assert stored == 0
    assert bucket.objects == {}


@pytest.mark.asyncio
async def test_missing_audio_ships_partial_manifest() -> None:
    storage_client, bucket = _fake_storage()
    async with _fake_client({"RO": ["b"], "RG": ["b"]}, audio_404=frozenset({"RO-b"})) as client:
        stored = await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(0),
            storage_client=storage_client,
            download_client=client,
        )
    assert stored == 1
    manifest = json.loads(bucket.objects[f"{PREFIX}/2026-07-17T00:00:00Z/manifest.json"])
    assert [r["provider"] for r in manifest["recordings"]] == ["google"]


@pytest.mark.asyncio
async def test_unknown_transcript_omits_input_audio_url() -> None:
    storage_client, bucket = _fake_storage()
    async with _fake_client({"RO": ["b"], "RG": ["b"]}, transcript="not in the manifest") as client:
        await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(0),
            storage_client=storage_client,
            download_client=client,
        )
    manifest = json.loads(bucket.objects[f"{PREFIX}/2026-07-17T00:00:00Z/manifest.json"])
    assert manifest["input_audio_url"] is None
    assert manifest["transcript"] == "not in the manifest"


@pytest.mark.asyncio
async def test_index_prepends_and_dedupes() -> None:
    storage_client, bucket = _fake_storage()
    bucket.objects[samples.INDEX_KEY] = json.dumps(["2026-07-16T00:00:00Z"]).encode()
    async with _fake_client({"RO": ["b"], "RG": ["b"]}) as client:
        await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(0),
            storage_client=storage_client,
            download_client=client,
        )
        await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(0),
            storage_client=storage_client,
            download_client=client,
        )
    index = json.loads(bucket.objects[samples.INDEX_KEY])
    assert index == ["2026-07-17T00:00:00Z", "2026-07-16T00:00:00Z"]


@pytest.mark.asyncio
async def test_never_raises_on_total_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    storage_client, _ = _fake_storage()
    async with httpx.AsyncClient(
        base_url="https://api.test/v1", transport=httpx.MockTransport(handler)
    ) as client:
        stored = await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(0),
            storage_client=storage_client,
            download_client=client,
        )
    assert stored == 0


@pytest.mark.asyncio
async def test_existing_manifest_is_never_overwritten() -> None:
    storage_client, bucket = _fake_storage()
    tick = "2026-07-17T00:00:00Z"
    bucket.objects[f"{PREFIX}/{tick}/manifest.json"] = b'{"sentinel": true}'
    async with _fake_client({"RO": ["b"], "RG": ["b"]}) as client:
        stored = await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(0),
            storage_client=storage_client,
            download_client=client,
        )
    assert stored == 0
    assert bucket.objects[f"{PREFIX}/{tick}/manifest.json"] == b'{"sentinel": true}'


@pytest.mark.asyncio
async def test_ambiguous_transcript_omits_input_audio_url() -> None:
    storage_client, bucket = _fake_storage()
    async with _fake_client(
        {"RO": ["b"], "RG": ["b"]}, transcript="what is the weather now"
    ) as client:
        await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(0),
            storage_client=storage_client,
            download_client=client,
        )
    manifest = json.loads(bucket.objects[f"{PREFIX}/2026-07-17T00:00:00Z/manifest.json"])
    assert manifest["input_audio_url"] is None


@pytest.mark.asyncio
async def test_index_read_failure_preserves_history() -> None:
    storage_client, bucket = _fake_storage()
    bucket.objects[samples.INDEX_KEY] = json.dumps(["2026-07-16T00:00:00Z"]).encode()
    bucket.fail_reads.add(samples.INDEX_KEY)
    async with _fake_client({"RO": ["b"], "RG": ["b"]}) as client:
        stored = await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(0),
            storage_client=storage_client,
            download_client=client,
        )
    assert stored == 2
    assert json.loads(bucket.objects[samples.INDEX_KEY]) == ["2026-07-16T00:00:00Z"]
    assert f"{PREFIX}/2026-07-17T00:00:00Z/manifest.json" in bucket.objects


@pytest.mark.asyncio
async def test_fetch_failure_retries_once_and_succeeds() -> None:
    attempts = {"audio": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/simulations"):
            run_id = request.url.params["filter"].split('"')[1]
            return httpx.Response(200, json=_sims_payload(run_id, ["b"]))
        if path.endswith("/audio"):
            attempts["audio"] += 1
            if attempts["audio"] == 1:
                return httpx.Response(500)
            return httpx.Response(200, json={"audio_url": "https://blobs.test/x.wav"})
        if "blobs.test" in str(request.url):
            return httpx.Response(200, content=b"RIFFfake")
        return httpx.Response(
            200, json={"transcript": [{"role": "user", "content": "hi"}], "test_case_id": "b"}
        )

    storage_client, bucket = _fake_storage()
    async with httpx.AsyncClient(
        base_url="https://api.test/v1", transport=httpx.MockTransport(handler)
    ) as client:
        stored = await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(0),
            storage_client=storage_client,
            download_client=client,
        )
    assert stored == 2
    assert attempts["audio"] == 3  # provider one: fail+retry; provider two: first try


@pytest.mark.asyncio
async def test_tick_exists_repairs_missing_index() -> None:
    storage_client, bucket = _fake_storage()
    tick = "2026-07-17T00:00:00Z"
    bucket.objects[f"{PREFIX}/{tick}/manifest.json"] = b'{"sentinel": true}'
    async with _fake_client({"RO": ["b"], "RG": ["b"]}) as client:
        stored = await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(0),
            storage_client=storage_client,
            download_client=client,
        )
    assert stored == 0
    assert json.loads(bucket.objects[samples.INDEX_KEY]) == [tick]
    assert bucket.objects[f"{PREFIX}/{tick}/manifest.json"] == b'{"sentinel": true}'


@pytest.mark.asyncio
async def test_malformed_index_is_preserved() -> None:
    storage_client, bucket = _fake_storage()
    bucket.objects[samples.INDEX_KEY] = b'{"not": "a list"}'
    async with _fake_client({"RO": ["b"], "RG": ["b"]}) as client:
        stored = await copy_tick_samples(
            client,
            bucket_name="bkt",
            runs=RUNS,
            rng=random.Random(0),
            storage_client=storage_client,
            download_client=client,
        )
    assert stored == 2
    assert bucket.objects[samples.INDEX_KEY] == b'{"not": "a list"}'
