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
_TICK_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
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


_SPOKEN_ROLES = frozenset({"user", "assistant"})


def _offset_seconds(value: object) -> float | None:
    """A Coval transcript offset as float seconds, or ``None`` when absent/non-numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _conversation_turns(sim: dict[str, Any]) -> list[dict[str, Any]]:
    """Spoken conversation as ``{index, role, content, start_offset, end_offset}`` turns.

    Only ``user`` (the persona) and ``assistant`` (the agent) messages are
    spoken. Coval's transcript also carries the persona's tool records — the
    ``end_conversation`` call arrives as ``role="tool"`` with raw JSON content —
    and those are dropped rather than rendered as a caller turn. Non-string
    content is skipped so a malformed message can't poison the manifest; an
    empty list means the transcript couldn't be resolved and the sample is
    treated as incomplete. ``start_offset``/``end_offset`` are full-recording
    positions in seconds (``None`` when Coval omits them) that the dashboard
    uses to sync the playhead to each turn.
    """
    turns: list[dict[str, Any]] = []
    for message in cast("list[dict[str, Any]]", sim.get("transcript") or []):
        role = message.get("role")
        content = message.get("content")
        if role in _SPOKEN_ROLES and isinstance(content, str):
            turns.append(
                {
                    "index": len(turns),
                    "role": role,
                    "content": content,
                    "start_offset": _offset_seconds(message.get("start_offset")),
                    "end_offset": _offset_seconds(message.get("end_offset")),
                }
            )
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
    """Add the tick to index.json, kept newest-first (single writer — no race to guard).

    Sorted rather than prepended because ticks are not always written in
    chronological order: one fetch publishes every day in its window oldest
    first, and a recovery can add a day older than everything already there.
    The dashboard reads index[0] as the latest tick, so write order must not
    decide it. Fixed-width ISO-8601 UTC makes a reverse string sort
    chronological, and it keeps the newest entries when truncating.

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
    ticks = sorted({tick_key, *ticks}, reverse=True)[:_INDEX_MAX_ENTRIES]
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
    expected_providers: set[str] | None = None,
) -> int:
    """Copy one multi-turn conversation (every provider, one persona) as a v2 sample.

    ``expected_providers`` is the configured provider set a sample must cover.
    Pass it: without it completeness is judged against whichever providers turned
    up, so a day one provider sat out would publish a lopsided comparison.

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
                expected_providers=expected_providers,
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
    expected_providers: set[str] | None,
) -> int:
    """Publish a sample for EVERY bucket present in ``runs``, oldest first.

    The fetch window spans two periods, so a run set can carry both yesterday and
    today. Publishing per bucket is what recovers a missed day: yesterday
    republishes if it never landed, and is skipped cheaply if it did (the
    manifest-exists check below). One bucket failing never blocks the others.

    Each bucket must cover ``expected_providers`` on its own. A provider can be
    present for today and absent for yesterday, and judging each bucket against
    only its own runs would call that yesterday complete.
    """
    if storage_client is None:  # pragma: no cover -- real client only outside tests
        from google.cloud import storage as gcs

        storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)

    by_bucket: dict[datetime, list[SampleRun]] = {}
    for run in runs:
        by_bucket.setdefault(run.bucket_at, []).append(run)

    published = 0
    for bucket_at in sorted(by_bucket):
        try:
            published += await _publish_one_bucket(
                client,
                download_client,
                bucket=bucket,
                test_set_id=test_set_id,
                runs=by_bucket[bucket_at],
                bucket_at=bucket_at,
                rng=rng,
                expected_providers=expected_providers,
            )
        except Exception:
            logger.error(
                "samples_bucket_failed",
                tick=bucket_at.strftime(_TICK_FORMAT),
                exc_info=True,
            )
    return published


async def _publish_one_bucket(
    client: httpx.AsyncClient,
    download_client: httpx.AsyncClient,
    *,
    bucket: storage.Bucket,
    test_set_id: str,
    runs: list[SampleRun],
    bucket_at: datetime,
    rng: Random,
    expected_providers: set[str] | None,
) -> int:
    tick_key = bucket_at.strftime(_TICK_FORMAT)
    manifest_key = f"{PREFIX}/{tick_key}/manifest.json"
    # Ahead of every other gate: an already-published day needs no runs at all,
    # only its index entry restored if that write once failed. Gating this behind
    # the checks below would strand such a day for good — its runs age out of the
    # window, so no later fetch would revisit the bucket to repair it.
    if bucket.blob(manifest_key).exists():
        _update_index(bucket, tick_key)
        logger.info("samples_tick_exists", tick=tick_key)
        return 0

    providers = expected_providers or {r.provider for r in runs}
    absent = providers - {r.provider for r in runs}
    if absent:
        logger.error("samples_bucket_provider_absent", tick=tick_key, absent=sorted(absent))
        return 0

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
        logger.error(
            "samples_no_shared_clip",
            tick=tick_key,
            providers=sorted(providers),
            personas=sorted(runs_by_persona),
            runs=[r.coval_run_id for r in runs],
        )
        return 0

    # Uniform draw over the shared conversations of both personas; repick on any
    # incomplete provider so only a fully-complete conversation (audio + turns for
    # EVERY provider) is ever surfaced.
    rng.shuffle(pool)
    for persona_id, test_case_id in pool:
        prov_runs = runs_by_persona[persona_id]
        blocked_by: str | None = None
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
                # /simulations/{id} wraps the object in a "simulation" envelope.
                turns = _conversation_turns(cast("dict[str, Any]", detail.json()["simulation"]))

                async def fetch_recording(sim_id: str = sim_id) -> bytes | None:
                    return await _download_recording(client, download_client, sim_id)

                audio = await _fetch_retry(fetch_recording, provider=provider, what="recording")
            except Exception:
                logger.error("sample_copy_failed", provider=provider, sim_id=sim_id, exc_info=True)
            if audio is None or not turns:
                blocked_by = provider
                break
            staged.append((run, sim_id, audio, turns))

        if len(staged) != len(providers):
            logger.info(
                "sample_incomplete_skipped",
                tick=tick_key,
                persona=persona_id,
                test_case_id=test_case_id,
                provider=blocked_by,
            )
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

    logger.error(
        "samples_no_complete_sample",
        tick=tick_key,
        providers=sorted(providers),
        attempted=len(pool),
        test_case_ids=sorted({tc for _, tc in pool}),
    )
    return 0
