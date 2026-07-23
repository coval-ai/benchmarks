# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish one multi-turn conversation sample per fetch tick.

Each tick picks ONE (scenario, persona) present for every provider, uploads
each provider's full-conversation recording plus its per-agent transcript into
the public samples bucket, and writes a ``manifest.json`` keyed by the timeline
bucket timestamp. Only a fully-complete sample (audio + transcript for every
provider) is published; otherwise the tick is skipped. The dashboard reads the
manifests directly (public bucket, no API hop); a rolling ``index.json`` lists
the available ticks newest-first. The bucket's 30-day TTL prunes old ticks.

Sampling failures never fail the fetch — the metric rows are the product.
"""

from __future__ import annotations

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
_DOWNLOAD_TIMEOUT = 120.0

# Multi-turn bench personas, matched by id — raw Coval names are unreliable (the
# male persona's name carries a trailing space). If the personas churn, update
# here (or lift into Settings).
_PERSONA_LABELS: dict[str, str] = {
    "PN3xgmsqeLDjsNNEA2e55e": "Standard Female",
    "9ATy64zKXxSUaVWb5YnQtd": "Standard Male",
}


def _persona_label(persona_id: str) -> str:
    return _PERSONA_LABELS.get(persona_id, persona_id)


@dataclass(frozen=True)
class SampleRun:
    """One provider's ingested run, eligible for this tick's sample.

    ``persona_id``/``agent_id`` are populated for multi-turn (where a provider
    has one run per persona); the single-turn path leaves them empty.
    """

    provider: str
    model: str
    coval_run_id: str
    bucket_at: datetime
    persona_id: str = ""
    agent_id: str = ""


def _conversation_turns(sim: dict[str, Any]) -> list[dict[str, Any]]:
    """Full ordered conversation as ``{index, role, content}`` turns.

    Non-string content is skipped so a malformed message can't poison the
    manifest; an empty list means the transcript couldn't be resolved and the
    sample is treated as incomplete.
    """
    turns: list[dict[str, Any]] = []
    for message in cast("list[dict[str, Any]]", sim.get("transcript") or []):
        role = message.get("role")
        content = message.get("content")
        if isinstance(role, str) and isinstance(content, str):
            turns.append({"index": len(turns), "role": role, "content": content})
    return turns


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
        # 100 covers today's 50-clip dataset twice over; if a dataset outgrows
        # it we proceed on the first page but alert, so paging gets added
        # instead of shared clips silently going missing.
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


async def publish_tick_sample(
    client: httpx.AsyncClient,
    *,
    bucket_name: str,
    test_set_id: str,
    runs: list[SampleRun],
    rng: Random,
    storage_client: storage.Client | None = None,
    download_client: httpx.AsyncClient | None = None,
) -> int:
    """Copy one multi-turn conversation (every provider, one persona) as a v2 sample.

    Never raises: any failure is logged and the tick is simply skipped.
    """
    if not bucket_name or not runs:
        return 0
    try:
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT) as default_download:
            return await _publish_tick_sample(
                client,
                download_client or default_download,
                bucket_name=bucket_name,
                test_set_id=test_set_id,
                runs=runs,
                rng=rng,
                storage_client=storage_client,
            )
    except Exception:
        logger.error("samples_tick_failed", exc_info=True)
        return 0


async def _publish_tick_sample(
    client: httpx.AsyncClient,
    download_client: httpx.AsyncClient,
    *,
    bucket_name: str,
    test_set_id: str,
    runs: list[SampleRun],
    rng: Random,
    storage_client: storage.Client | None,
) -> int:
    providers = {r.provider for r in runs}

    # A "conversation" is one (test_case, persona) pair — the same scenario and
    # the same simulated caller, which is what makes the sample comparable across
    # providers. Group runs by persona; only a persona present for EVERY provider
    # can yield a conversation. sims[(persona, provider)] = {test_case_id: sim_id}.
    runs_by_persona: dict[str, dict[str, SampleRun]] = {}
    for run in runs:
        runs_by_persona.setdefault(run.persona_id, {})[run.provider] = run

    sims: dict[tuple[str, str], dict[str, str]] = {}
    pool: list[tuple[str, str]] = []
    for persona_id, prov_runs in runs_by_persona.items():
        if set(prov_runs) != providers:
            logger.error(
                "samples_persona_incomplete",
                persona=persona_id,
                missing=sorted(providers - set(prov_runs)),
            )
            continue
        for provider, run in prov_runs.items():

            async def list_sims(run: SampleRun = run) -> dict[str, str]:
                return await _sims_by_test_case(client, run.coval_run_id)

            try:
                sims[(persona_id, provider)] = await _fetch_retry(
                    list_sims, provider=provider, what="sims_list"
                )
            except Exception:
                logger.error("samples_provider_missing", missing=[provider], exc_info=True)
                sims[(persona_id, provider)] = {}
        shared = set.intersection(*(set(sims[(persona_id, p)]) for p in providers))
        pool.extend((persona_id, tc) for tc in sorted(shared))

    if not pool:
        logger.error("samples_no_shared_clip", runs=[r.coval_run_id for r in runs])
        return 0

    if storage_client is None:  # pragma: no cover -- real client only outside tests
        from google.cloud import storage as gcs

        storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)

    # Uniform draw over the shared conversations of both personas; repick on any
    # incomplete provider so only a fully-complete conversation (audio + turns for
    # EVERY provider) is ever surfaced.
    rng.shuffle(pool)
    for persona_id, test_case_id in pool:
        prov_runs = runs_by_persona[persona_id]
        # Key the sample by the CHOSEN conversation's own bucket, not the max across
        # all personas: personas can straddle bucket boundaries, and the global max
        # would store this sample under an unrelated (newer) tick and mislabel it.
        tick_key = max(prov_runs[p].bucket_at for p in providers).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest_key = f"{PREFIX}/{tick_key}/manifest.json"
        if bucket.blob(manifest_key).exists():
            # Idempotent repair: manifest published but the index update failed.
            _update_index(bucket, tick_key)
            logger.info("samples_tick_exists", tick=tick_key)
            return 0
        staged: list[tuple[SampleRun, str, bytes, list[dict[str, Any]]]] = []
        for provider in sorted(providers):
            run = prov_runs[provider]
            sim_id = sims[(persona_id, provider)][test_case_id]
            audio: bytes | None = None
            turns: list[dict[str, Any]] = []
            try:

                async def fetch_detail(sim_id: str = sim_id) -> httpx.Response:
                    resp = await client.get(f"/simulations/{sim_id}")
                    resp.raise_for_status()
                    return resp

                detail = await _fetch_retry(fetch_detail, provider=provider, what="sim_detail")
                turns = _conversation_turns(cast("dict[str, Any]", detail.json()))

                async def fetch_recording(sim_id: str = sim_id) -> bytes | None:
                    return await _download_recording(client, download_client, sim_id)

                audio = await _fetch_retry(fetch_recording, provider=provider, what="recording")
            except Exception:
                logger.error("sample_copy_failed", provider=provider, sim_id=sim_id, exc_info=True)
            if audio is None or not turns:
                break
            staged.append((run, sim_id, audio, turns))

        if len(staged) != len(providers):
            logger.info("sample_incomplete_skipped", persona=persona_id, test_case_id=test_case_id)
            continue

        recordings: list[dict[str, Any]] = []
        for s_run, s_sim_id, s_audio, s_turns in staged:
            key = f"{PREFIX}/{tick_key}/{s_run.provider}.wav"
            _upload(bucket, key, s_audio, "audio/wav")
            recordings.append(
                {
                    "provider": s_run.provider,
                    "model": s_run.model,
                    "object": key,
                    "coval_run_id": s_run.coval_run_id,
                    "sim_id": s_sim_id,
                    "agent_id": s_run.agent_id,
                    "turns": s_turns,
                }
            )
        manifest = {
            "schema_version": 2,
            "bucket_at": tick_key,
            "test_set_id": test_set_id,
            "test_case_id": test_case_id,
            "persona_name": _persona_label(persona_id),
            "input_audio_url": None,
            "recordings": recordings,
        }
        _upload(bucket, manifest_key, json.dumps(manifest).encode(), "application/json")
        _update_index(bucket, tick_key)
        logger.info(
            "samples_tick_stored",
            tick=tick_key,
            recordings=len(recordings),
            persona=persona_id,
            test_case_id=test_case_id,
        )
        return len(recordings)

    logger.error("samples_no_complete_sample")
    return 0
