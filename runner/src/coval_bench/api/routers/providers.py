# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/providers — catalogue of benchmarked providers and models.

The catalogue is sourced from the runner's enabled provider matrices
(``DEFAULT_STT_MATRIX`` / ``DEFAULT_TTS_MATRIX``) — same source of truth the
orchestrator uses to decide what to actually run, so the website can never
drift from the runner's reality. Only ``enabled=True`` entries are exposed.

No DB hit is made by this endpoint.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import structlog
from fastapi import APIRouter
from starlette.requests import Request

from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import ProviderInfo, ProvidersResponse
from coval_bench.runner.config import DEFAULT_STT_MATRIX, DEFAULT_TTS_MATRIX

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["providers"])


def _describe() -> ProvidersResponse:
    stt_models: dict[str, list[str]] = defaultdict(list)
    for entry in DEFAULT_STT_MATRIX:
        if entry.enabled and entry.model not in stt_models[entry.provider]:
            stt_models[entry.provider].append(entry.model)

    tts_models: dict[str, list[str]] = defaultdict(list)
    for entry in DEFAULT_TTS_MATRIX:
        if entry.enabled and entry.model not in tts_models[entry.provider]:
            tts_models[entry.provider].append(entry.model)

    return ProvidersResponse(
        stt=[ProviderInfo(provider=p, models=m) for p, m in sorted(stt_models.items())],
        tts=[ProviderInfo(provider=p, models=m) for p, m in sorted(tts_models.items())],
    )


@router.get("/providers")
@limiter.limit("60/minute")
async def get_providers(request: Request) -> dict[str, Any]:
    """Return the catalogue of benchmarked providers and models.

    Sourced from the runner's enabled matrix — the website's view always
    reflects what the runner is actually executing.
    """
    return _describe().model_dump()
