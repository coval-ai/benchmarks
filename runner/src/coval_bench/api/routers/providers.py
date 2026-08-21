# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/providers — catalogue of benchmarked providers and models.

The catalogue is sourced from the model registry
(``coval_bench.registries.MODEL_REGISTRY``) joined with ``model_state`` — the
same sources the orchestrator runs from, so the website can never drift from
the runner's reality. Stopped models (neither running nor shown) come back
``disabled=true``; Hidden models (running under embargo) are omitted for
public callers (a disabled entry would still leak existence) and included
enabled for callers whose grant names them.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends
from posthog import Posthog
from starlette.requests import Request

from coval_bench.api.deps import capture_api_event, get_model_states, get_posthog
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import (
    ModelInfo,
    ModelTagOut,
    ProviderInfo,
    ProvidersResponse,
    TagCategoryOut,
)
from coval_bench.db.model_state import DEFAULT_STATE, ModelKey, ModelState
from coval_bench.registries import (
    CATEGORY_LABELS,
    MODEL_REGISTRY,
    PROVIDER_VALUED_CATEGORIES,
    TAG_CATEGORIES,
    Benchmark,
    RegisteredModel,
    TagCategory,
    tag_value_label,
)

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["providers"])


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
    benchmark: Benchmark,
    hidden: frozenset[tuple[str, str]],
    states: dict[ModelKey, ModelState],
) -> dict[str, list[ModelInfo]]:
    """Build an ordered {provider: [ModelInfo, ...]} map from the model registry.

    Registry order throughout; a Hidden model appears only for a caller whose
    allowlist names it.
    """
    result: dict[str, list[ModelInfo]] = {}
    for m in MODEL_REGISTRY:
        if m.benchmark is not benchmark:
            continue
        if (m.provider, m.model) in hidden:
            continue
        state = states.get((m.benchmark, m.provider, m.model), DEFAULT_STATE)
        result.setdefault(m.provider, []).append(
            ModelInfo(
                model=m.model,
                disabled=not state.running and not state.shown,
                early_access=state.running and not state.shown,
                tags=_model_tags(m),
            )
        )
    return result


def _describe(
    hidden: frozenset[tuple[str, str]], states: dict[ModelKey, ModelState]
) -> ProvidersResponse:
    stt_map = _build_provider_map(Benchmark.STT, hidden, states)
    tts_map = _build_provider_map(Benchmark.TTS, hidden, states)
    s2s_map = _build_provider_map(Benchmark.S2S, hidden, states)

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
    states: dict[ModelKey, ModelState] = Depends(get_model_states),
) -> ProvidersResponse:
    """Return the catalogue of benchmarked providers and models.

    Sourced from the model registry (all entries, not just actively run ones).
    Each model includes a ``disabled`` flag that the frontend can use to
    hide or grey out models that are known but not actively benchmarked.
    A hidden model appears only for a caller whose allowlist names it.
    """
    response = _describe(hidden, states)
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
