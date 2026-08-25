# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/providers — catalogue of benchmarked providers and models.

The catalogue is sourced from the model registry
(``coval_bench.registries.MODEL_REGISTRY``) — the same source of truth the
orchestrator runs from, so the website can never drift from the runner's
reality. ``RETIRED``/``PENDING`` models come back ``disabled=true``;
``EARLY_ACCESS`` models are omitted for public callers (a disabled entry
would still leak existence) and included enabled for internal ones.

No DB hit is made by this endpoint.
"""

from __future__ import annotations

from collections.abc import Sequence

import structlog
from fastapi import APIRouter, Depends
from posthog import Posthog
from starlette.requests import Request

from coval_bench.api.deps import capture_api_event, get_posthog
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    ModelInfo,
    ModelTagOut,
    ProviderInfo,
    ProvidersResponse,
    TagCategoryOut,
)
from coval_bench.registries import (
    CATEGORY_LABELS,
    MODEL_REGISTRY,
    PROVIDER_VALUED_CATEGORIES,
    TAG_CATEGORIES,
    Benchmark,
    ModelStatus,
    RegisteredModel,
    TagCategory,
    tag_value_label,
)

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["providers"])

_HIDDEN_STATUSES = frozenset({ModelStatus.RETIRED, ModelStatus.PENDING})


def _tag(category: TagCategory, value: str) -> ModelTagOut:
    """Build a facet tag with its display label resolved from the registry."""
    return ModelTagOut(category=category, value=value, label=tag_value_label(category, value))


def _model_tags(m: RegisteredModel) -> list[ModelTagOut]:
    """Flatten a model's facets: derived columns/attributes plus curated tags."""
    deployment = "on-prem" if m.on_prem else "cloud"
    return [
        _tag(TagCategory.TYPE, m.benchmark),
        _tag(TagCategory.HOST, m.provider),
        _tag(TagCategory.CREATOR, m.creator or m.provider),
        _tag(TagCategory.SOURCE, m.source),
        _tag(TagCategory.LICENSING, m.licensing),
        _tag(TagCategory.DEPLOYMENT, deployment),
        *([_tag(TagCategory.REGION, m.region)] if m.region else []),
        *(_tag(TAG_CATEGORIES[t], t) for t in m.tags),
    ]


def _tag_categories() -> list[TagCategoryOut]:
    """The facet vocabulary in display order (TagCategory definition order)."""
    return [
        TagCategoryOut(
            category=c,
            label=CATEGORY_LABELS[c],
            provider_valued=c in PROVIDER_VALUED_CATEGORIES,
        )
        for c in TagCategory
    ]


def _build_provider_map(
    models: Sequence[RegisteredModel],
    benchmark: Benchmark,
    hidden: frozenset[tuple[str, str]],
) -> dict[str, list[ModelInfo]]:
    """Build an ordered {provider: [ModelInfo, ...]} map from the model registry.

    Registry order throughout; an EARLY_ACCESS model appears only for a caller
    whose allowlist names it.
    """
    result: dict[str, list[ModelInfo]] = {}
    for m in models:
        if m.benchmark is not benchmark:
            continue
        if (m.provider, m.model) in hidden:
            continue
        result.setdefault(m.provider, []).append(
            ModelInfo(
                model=m.model,
                disabled=m.status in _HIDDEN_STATUSES,
                early_access=m.status is ModelStatus.EARLY_ACCESS,
                tags=_model_tags(m),
            )
        )
    return result


def _describe(
    models: Sequence[RegisteredModel], hidden: frozenset[tuple[str, str]]
) -> ProvidersResponse:
    stt_map = _build_provider_map(models, Benchmark.STT, hidden)
    tts_map = _build_provider_map(models, Benchmark.TTS, hidden)
    s2s_map = _build_provider_map(models, Benchmark.S2S, hidden)

    return ProvidersResponse(
        stt=[ProviderInfo(provider=p, models=m) for p, m in sorted(stt_map.items())],
        tts=[ProviderInfo(provider=p, models=m) for p, m in sorted(tts_map.items())],
        s2s=[ProviderInfo(provider=p, models=m) for p, m in sorted(s2s_map.items())],
        tag_categories=_tag_categories(),
    )


@router.get("/providers", response_model=ProvidersResponse)
@limiter.limit("60/minute")
async def get_providers(
    request: Request,
    posthog_client: Posthog | None = Depends(get_posthog),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
) -> ProvidersResponse:
    """Return the catalogue of benchmarked providers and models.

    Sourced from the model registry (all entries, not just actively run ones).
    Each model includes a ``disabled`` flag that the frontend can use to
    hide or grey out models that are known but not actively benchmarked.
    An early-access model appears only for a caller whose allowlist names it.
    """
    response = _describe(MODEL_REGISTRY, hidden)
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
