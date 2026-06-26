# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/providers — catalogue of benchmarked providers and models.

The catalogue is sourced from the model registry
(``coval_bench.registries.MODEL_REGISTRY``) — the same source of truth the
orchestrator runs from, so the website can never drift from the runner's
reality. Every registered model is exposed; ``disabled`` is true for
``RETIRED`` and ``PENDING`` models so the frontend keeps them off the site
even when historical result rows exist.

No DB hit is made by this endpoint.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends
from posthog import Posthog
from starlette.requests import Request

from coval_bench.api.deps import capture_api_event, get_posthog
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import ModelInfo, ProviderInfo, ProvidersResponse
from coval_bench.registries import MODEL_REGISTRY, Benchmark, ModelStatus

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["providers"])

_HIDDEN_STATUSES = frozenset({ModelStatus.RETIRED, ModelStatus.PENDING})


def _build_provider_map(benchmark: Benchmark) -> dict[str, list[ModelInfo]]:
    """Build an ordered {provider: [ModelInfo, ...]} map from the model registry.

    Every registered model for *benchmark* is included (whatever its status)
    so the frontend knows about retired models and can keep them hidden.
    Providers and models keep registry order.
    """
    result: dict[str, list[ModelInfo]] = {}
    for m in MODEL_REGISTRY:
        if m.benchmark is benchmark:
            result.setdefault(m.provider, []).append(
                ModelInfo(model=m.model, disabled=m.status in _HIDDEN_STATUSES)
            )
    return result


def _describe() -> ProvidersResponse:
    stt_map = _build_provider_map(Benchmark.STT)
    tts_map = _build_provider_map(Benchmark.TTS)

    return ProvidersResponse(
        stt=[ProviderInfo(provider=p, models=m) for p, m in sorted(stt_map.items())],
        tts=[ProviderInfo(provider=p, models=m) for p, m in sorted(tts_map.items())],
    )


@router.get("/providers", response_model=ProvidersResponse)
@limiter.limit("60/minute")
async def get_providers(
    request: Request,
    posthog_client: Posthog | None = Depends(get_posthog),
) -> ProvidersResponse:
    """Return the catalogue of benchmarked providers and models.

    Sourced from the model registry (all entries, not just actively run ones).
    Each model includes a ``disabled`` flag that the frontend can use to
    hide or grey out models that are known but not actively benchmarked.
    """
    response = _describe()
    capture_api_event(
        posthog_client,
        "providers_listed",
        {
            "stt_provider_count": len(response.stt),
            "tts_provider_count": len(response.tts),
            "$process_person_profile": False,
        },
    )
    return response
