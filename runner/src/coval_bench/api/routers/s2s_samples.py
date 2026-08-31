# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/s2s/samples — conversation samples from the private samples bucket.

The bucket allows no anonymous read. The API reads each manifest as its own
service account, drops the recordings this caller may not see (their transcripts
go with them, since both live in the same object), and hands back an API route
per surviving recording. Asking for one returns a short-lived signed URL minted at
that moment, so no signature is ever stored or cached.

The URL comes back as a body rather than a redirect: a browser cannot carry its
early-access header through a cross-origin redirect to storage, so a redirect
would be unplayable for exactly the callers the embargo exists to serve.

Every route resolves visibility through the same ``hidden_early_access``
dependency: the audio route re-checks rather than trusting that the caller got
its path from a filtered manifest.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from urllib.parse import quote

import structlog
from fastapi import APIRouter, Depends, HTTPException, Path, Response
from posthog import Posthog
from starlette.requests import Request

from coval_bench.api.deps import capture_api_event, get_models, get_posthog, get_settings
from coval_bench.api.internal import hidden_early_access, never_shared
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import S2SSampleAudioOut, S2SSampleOut, S2SSampleRecordingOut
from coval_bench.config import Settings
from coval_bench.gcs import signed_url
from coval_bench.registries import Benchmark, RegisteredModel
from coval_bench.s2s.samples import (
    AUDIO_URL_TTL,
    audio_object_key,
    load_sample,
    load_sample_ids,
)

logger = structlog.get_logger("coval_bench.api.s2s_samples")

router = APIRouter(tags=["s2s"])

_SAMPLE_ID = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"


def _sees_any_s2s_model(
    models: Sequence[RegisteredModel], hidden: frozenset[tuple[str, str]]
) -> bool:
    return any(m.benchmark is Benchmark.S2S and (m.provider, m.model) not in hidden for m in models)


def _audio_path(sample_id: str, provider: str, model: str) -> str:
    return (
        f"/v1/s2s/samples/{quote(sample_id, safe='')}"
        f"/{quote(provider, safe='')}/{quote(model, safe='')}/audio"
    )


@router.get("/s2s/samples", response_model=list[str])
@limiter.limit("60/minute")
async def list_s2s_samples(
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
    models: Sequence[RegisteredModel] = Depends(get_models),
) -> list[str]:
    """Sample ids newest-first, empty when the caller may see no S2S model at all."""
    never_shared(response)
    if not settings.s2s_samples_bucket or not _sees_any_s2s_model(models, hidden):
        return []
    return await asyncio.to_thread(load_sample_ids, settings.s2s_samples_bucket)


@router.get("/s2s/samples/{sample_id}", response_model=S2SSampleOut)
@limiter.limit("60/minute")
async def get_s2s_sample(
    request: Request,
    response: Response,
    sample_id: str = Path(pattern=_SAMPLE_ID),
    settings: Settings = Depends(get_settings),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
    posthog_client: Posthog | None = Depends(get_posthog),
) -> S2SSampleOut:
    """One sample's manifest, carrying only the recordings this caller may see.

    A sample whose every recording is embargoed for this caller is a 404 rather
    than an empty shell: the surrounding metadata would otherwise confirm that a
    tick exists and name the scenario and persona behind it.
    """
    never_shared(response)
    if not settings.s2s_samples_bucket:
        raise HTTPException(404, "s2s samples are not configured")

    sample = await asyncio.to_thread(
        load_sample, settings.s2s_samples_bucket, sample_id, hidden=hidden
    )
    if sample is None or not sample["recordings"]:
        raise HTTPException(404, f"no s2s sample for {sample_id}")

    recordings = [
        S2SSampleRecordingOut(
            provider=rec["provider"],
            model=rec["model"],
            audio_path=_audio_path(sample_id, rec["provider"], rec["model"]),
            coval_run_id=rec["coval_run_id"],
            sim_id=rec["sim_id"],
            agent_id=rec.get("agent_id"),
            turns=rec.get("turns", []),
        )
        for rec in sample["recordings"]
    ]
    out = S2SSampleOut(
        schema_version=sample.get("schema_version"),
        sample_id=sample_id,
        test_case_id=sample["test_case_id"],
        test_set_id=sample.get("test_set_id"),
        persona_name=sample.get("persona_name"),
        transcript=sample.get("transcript"),
        recordings=recordings,
    )
    capture_api_event(
        posthog_client,
        "s2s_sample_served",
        {
            "sample_id": sample_id,
            "recording_count": len(recordings),
            "$process_person_profile": False,
        },
    )
    return out


@router.get(
    "/s2s/samples/{sample_id}/{provider}/{model}/audio",
    response_model=S2SSampleAudioOut,
)
@limiter.limit("120/minute")
async def get_s2s_sample_audio(
    request: Request,
    response: Response,
    provider: str,
    model: str,
    sample_id: str = Path(pattern=_SAMPLE_ID),
    settings: Settings = Depends(get_settings),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
) -> S2SSampleAudioOut:
    """A freshly signed URL for one recording, minted only if this caller may hear it."""
    never_shared(response)
    if not settings.s2s_samples_bucket:
        raise HTTPException(404, "s2s samples are not configured")

    key = await asyncio.to_thread(
        audio_object_key,
        settings.s2s_samples_bucket,
        sample_id,
        provider,
        model,
        hidden=hidden,
    )
    if key is None:
        raise HTTPException(404, "no such s2s sample recording")

    url = await asyncio.to_thread(signed_url, settings.s2s_samples_bucket, key, ttl=AUDIO_URL_TTL)
    return S2SSampleAudioOut(url=url, expires_at=datetime.now(UTC) + AUDIO_URL_TTL)
