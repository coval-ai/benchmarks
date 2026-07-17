# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy one shared utterance's conversation recordings into the samples bucket.

Each fetch tick picks ONE dataset clip present in every provider's newest
ingested run and copies that clip's reconstructed conversation recording from
each provider into the public samples bucket, next to a ``manifest.json``
keyed by the timeline bucket timestamp. The dashboard reads the manifests
directly (public bucket, no API hop); a rolling ``index.json`` lists the
available ticks newest-first. The bucket's 30-day TTL prunes old ticks.

Sampling failures never fail the fetch — the metric rows are the product.
"""

from __future__ import annotations

import importlib.resources
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import httpx
import structlog
from google.api_core.exceptions import NotFound

from coval_bench.runner.retry import with_retry

if TYPE_CHECKING:
    from random import Random

    from google.cloud import storage

logger = structlog.get_logger("coval_bench.s2s.samples")

PREFIX = "s2s-samples"
INDEX_KEY = f"{PREFIX}/index.json"
_INDEX_MAX_ENTRIES = 60
_PUBLIC_DATASET_BASE = "https://storage.googleapis.com/coval-benchmarks-datasets/s2s-v1"
_DOWNLOAD_TIMEOUT = 120.0


@dataclass(frozen=True)
class SampleRun:
    """One provider's newest ingested run, eligible for this tick's sample."""

    provider: str
    model: str
    coval_run_id: str
    bucket_at: datetime


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _clips_by_transcript() -> dict[str, str]:
    """Packaged-manifest transcript -> public dataset clip path, unambiguous only.

    Several transcripts appear on multiple clips (different speakers); those
    can't be resolved by text, so they're dropped and the sample ships with
    input_audio_url = null rather than possibly the wrong recording.
    """
    ref = importlib.resources.files("coval_bench.datasets.manifests").joinpath("s2s-v1.json")
    manifest = json.loads(ref.read_bytes())
    paths_by_transcript: dict[str, set[str]] = {}
    for item in manifest["items"]:
        if item.get("transcript") and item.get("path"):
            paths_by_transcript.setdefault(_norm(item["transcript"]), set()).add(item["path"])
    return {t: next(iter(paths)) for t, paths in paths_by_transcript.items() if len(paths) == 1}


def _input_transcript(sim: dict[str, Any]) -> str | None:
    """First user turn of the conversation — the utterance the model heard."""
    for message in cast("list[dict[str, Any]]", sim.get("transcript") or []):
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


async def _fetch_retry[T](fn: Callable[[], Awaitable[T]], *, provider: str, what: str) -> T:
    """with_retry for Coval fetches, alerting (error log) on the FIRST failure."""
    first = True

    async def attempt() -> T:
        nonlocal first
        try:
            return await fn()
        except httpx.HTTPError:
            if first:
                first = False
                logger.error("sample_fetch_failed", provider=provider, what=what)
            raise

    return await with_retry(attempt, max_attempts=2, retry_on=(httpx.HTTPError,))


async def _sims_by_test_case(client: httpx.AsyncClient, coval_run_id: str) -> dict[str, str]:
    """Map test_case_id -> simulation id for one run (run_id is AIP-160-filterable)."""
    resp = await client.get(
        "/simulations", params={"filter": f'run_id="{coval_run_id}"', "page_size": 100}
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("next_page_token"):
        # 100 covers today's 50-clip dataset twice over; page instead of
        # truncating silently if a future dataset outgrows it.
        logger.error("samples_sims_truncated", coval_run_id=coval_run_id)
    items = next((v for v in payload.values() if isinstance(v, list)), [])
    return {
        cast("str", s["test_case_id"]): cast("str", s["simulation_id"])
        for s in cast("list[dict[str, Any]]", items)
        if s.get("test_case_id") and s.get("simulation_id")
    }


async def _download_recording(
    client: httpx.AsyncClient, download_client: httpx.AsyncClient, sim_id: str
) -> bytes | None:
    """Recording bytes via the presigned URL; None when the object is gone."""
    url_resp = await client.get(f"/simulations/{sim_id}/audio")
    if url_resp.status_code == 404:
        return None
    url_resp.raise_for_status()
    audio_url = cast("str", url_resp.json()["audio_url"])
    blob = await download_client.get(audio_url)
    blob.raise_for_status()
    return blob.content


def _upload(bucket: storage.Bucket, key: str, data: bytes, content_type: str) -> None:
    bucket.blob(key).upload_from_string(data, content_type=content_type)


def _update_index(bucket: storage.Bucket, tick_key: str) -> None:
    """Prepend the tick to index.json (single daily writer — no race to guard).

    Only a confirmed missing object starts an empty index; any other read
    failure leaves the existing index untouched so a transient error can't
    erase the history.
    """
    blob = bucket.blob(INDEX_KEY)
    ticks: list[str]
    try:
        decoded = json.loads(blob.download_as_bytes())
        if not isinstance(decoded, list) or not all(isinstance(t, str) for t in decoded):
            raise ValueError("index.json must be a list of tick strings")
        ticks = decoded
    except NotFound:
        ticks = []
    except Exception:
        logger.error("samples_index_read_failed", tick=tick_key, exc_info=True)
        return
    ticks = [tick_key, *(t for t in ticks if t != tick_key)][:_INDEX_MAX_ENTRIES]
    _upload(bucket, INDEX_KEY, json.dumps(ticks).encode(), "application/json")


async def copy_tick_samples(
    client: httpx.AsyncClient,
    *,
    bucket_name: str,
    runs: list[SampleRun],
    rng: Random,
    storage_client: storage.Client | None = None,
    download_client: httpx.AsyncClient | None = None,
) -> int:
    """Copy this tick's shared-clip recordings; return how many were stored.

    Never raises: any failure is logged and the tick is simply skipped or
    shipped with fewer providers.
    """
    if not bucket_name or not runs:
        return 0
    try:
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT) as default_download:
            return await _copy_tick_samples(
                client,
                download_client or default_download,
                bucket_name=bucket_name,
                runs=runs,
                rng=rng,
                storage_client=storage_client,
            )
    except Exception:
        logger.error("samples_tick_failed", exc_info=True)
        return 0


async def _copy_tick_samples(
    client: httpx.AsyncClient,
    download_client: httpx.AsyncClient,
    *,
    bucket_name: str,
    runs: list[SampleRun],
    rng: Random,
    storage_client: storage.Client | None,
) -> int:
    sims_per_run: dict[str, dict[str, str]] = {}
    for run in runs:

        async def list_sims(run: SampleRun = run) -> dict[str, str]:
            return await _sims_by_test_case(client, run.coval_run_id)

        sims_per_run[run.coval_run_id] = await _fetch_retry(
            list_sims, provider=run.provider, what="sims_list"
        )

    shared = set.intersection(*(set(m) for m in sims_per_run.values()))
    if not shared:
        logger.error("samples_no_shared_clip", runs=[r.coval_run_id for r in runs])
        return 0
    test_case_id = rng.choice(sorted(shared))

    if storage_client is None:  # pragma: no cover -- real client only outside tests
        from google.cloud import storage as gcs

        storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)

    # Providers of one Coval round land in the same daily bucket; if they ever
    # straddle midnight, key the folder by the newest so the timeline join
    # points at data that exists.
    tick_key = max(r.bucket_at for r in runs).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest_key = f"{PREFIX}/{tick_key}/manifest.json"
    if bucket.blob(manifest_key).exists():
        # Idempotent repair: a prior run may have published the manifest but
        # failed the index update, leaving the tick invisible.
        _update_index(bucket, tick_key)
        logger.info("samples_tick_exists", tick=tick_key)
        return 0

    transcript: str | None = None
    recordings: list[dict[str, Any]] = []
    for run in runs:
        sim_id = sims_per_run[run.coval_run_id][test_case_id]
        try:
            if transcript is None:

                async def fetch_detail(sim_id: str = sim_id) -> httpx.Response:
                    resp = await client.get(f"/simulations/{sim_id}")
                    resp.raise_for_status()
                    return resp

                detail = await _fetch_retry(fetch_detail, provider=run.provider, what="sim_detail")
                transcript = _input_transcript(cast("dict[str, Any]", detail.json()))

            async def fetch_recording(sim_id: str = sim_id) -> bytes | None:
                return await _download_recording(client, download_client, sim_id)

            audio = await _fetch_retry(fetch_recording, provider=run.provider, what="recording")
            if audio is None:
                logger.error("sample_audio_missing", provider=run.provider, sim_id=sim_id)
                continue
            key = f"{PREFIX}/{tick_key}/{run.provider}.wav"
            _upload(bucket, key, audio, "audio/wav")
            recordings.append(
                {
                    "provider": run.provider,
                    "model": run.model,
                    "object": key,
                    "coval_run_id": run.coval_run_id,
                    "sim_id": sim_id,
                }
            )
        except Exception:
            logger.error("sample_copy_failed", provider=run.provider, sim_id=sim_id, exc_info=True)

    if not recordings:
        return 0

    clip_path = _clips_by_transcript().get(_norm(transcript)) if transcript else None
    manifest = {
        "bucket_at": tick_key,
        "test_case_id": test_case_id,
        "transcript": transcript,
        "input_audio_url": f"{_PUBLIC_DATASET_BASE}/{clip_path}" if clip_path else None,
        "recordings": recordings,
    }
    _upload(
        bucket,
        f"{PREFIX}/{tick_key}/manifest.json",
        json.dumps(manifest).encode(),
        "application/json",
    )
    _update_index(bucket, tick_key)
    logger.info("samples_tick_stored", tick=tick_key, recordings=len(recordings))
    return len(recordings)
