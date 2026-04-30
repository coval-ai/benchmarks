# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/providers — catalogue of benchmarked providers and models.

The catalogue is sourced from the runner's provider matrices
(``DEFAULT_STT_MATRIX`` / ``DEFAULT_TTS_MATRIX``) — same source of truth the
orchestrator uses to decide what to actually run, so the website can never
drift from the runner's reality. All matrix entries are exposed (including
``enabled=False`` and ``disabled=True``) so the frontend can decide whether to
show or grey out disabled models. The ``disabled`` flag indicates a model the
runner intentionally skips (configured in ``runner/config.py``).

No DB hit is made by this endpoint.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from starlette.requests import Request

from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import ModelInfo, ProviderInfo, ProvidersResponse
from coval_bench.runner.config import DEFAULT_STT_MATRIX, DEFAULT_TTS_MATRIX, ProviderEntry

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["providers"])


def _build_provider_map(
    matrix: list[ProviderEntry],
) -> dict[str, list[ModelInfo]]:
    """Build an ordered {provider: [ModelInfo, ...]} map from a matrix.

    Every entry in the matrix is included (regardless of ``enabled``) so the
    frontend knows about disabled models and can hide or grey them out.
    Entries are deduped by ``(provider, model)``; if any matrix row for a given
    ``(provider, model)`` pair is disabled, the resulting ``ModelInfo`` is
    marked ``disabled=True`` (safer for the FE — never hides a live model).
    """
    # Track per-(provider, model) disabled state; use insertion-order dict for stable ordering.
    # Key: (provider, model) → tracks (seen_order, any_disabled)
    seen: dict[tuple[str, str], bool] = {}
    provider_order: list[str] = []

    for entry in matrix:
        key = (entry.provider, entry.model)
        if entry.provider not in provider_order:
            provider_order.append(entry.provider)
        if key not in seen:
            seen[key] = entry.disabled
        else:
            # any disabled entry "wins"
            seen[key] = seen[key] or entry.disabled

    # Group by provider in insertion order
    result: dict[str, list[ModelInfo]] = {}
    for provider in provider_order:
        models: list[ModelInfo] = []
        for (prov, model), is_disabled in seen.items():
            if prov == provider:
                models.append(ModelInfo(model=model, disabled=is_disabled))
        result[provider] = models

    return result


def _describe() -> ProvidersResponse:
    stt_map = _build_provider_map(DEFAULT_STT_MATRIX)
    tts_map = _build_provider_map(DEFAULT_TTS_MATRIX)

    return ProvidersResponse(
        stt=[ProviderInfo(provider=p, models=m) for p, m in sorted(stt_map.items())],
        tts=[ProviderInfo(provider=p, models=m) for p, m in sorted(tts_map.items())],
    )


@router.get("/providers", response_model=ProvidersResponse)
@limiter.limit("60/minute")
async def get_providers(request: Request) -> ProvidersResponse:
    """Return the catalogue of benchmarked providers and models.

    Sourced from the full runner matrix (all entries, not just enabled ones).
    Each model includes a ``disabled`` flag that the frontend can use to
    hide or grey out models that are known but not actively benchmarked.
    """
    return _describe()
