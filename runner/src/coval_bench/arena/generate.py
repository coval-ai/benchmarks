# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""On-demand battle generation: synthesize a model pair on a prompt, persist a battle.

Both sides speak the same prompt (fairness). Which model lands as A vs B is
randomized to avoid position bias in voting. Synthesis runs concurrently to keep
the request fast. A battle is persisted only if both sides succeed — the
``audio_*_url`` columns are NOT NULL, so a half-synthesized battle is never stored.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

import structlog

from coval_bench.arena.audio_store import store_clip
from coval_bench.config import Settings
from coval_bench.db.arena_store import ArenaStore
from coval_bench.db.models import Battle
from coval_bench.providers.base import TTSResult
from coval_bench.providers.tts import TTS_PROVIDERS
from coval_bench.registries.models import RegisteredModel

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_SYNTH_TIMEOUT_S = 60.0


async def generate_battle(
    settings: Settings,
    store: ArenaStore,
    *,
    prompt: str,
    domain: str | None,
    pair: tuple[RegisteredModel, RegisteredModel],
    rng: random.Random | None = None,
) -> Battle | None:
    """Synthesize both sides of *pair* on *prompt*, store the clips, persist a battle.

    Returns the inserted ``Battle``, or ``None`` if either synthesis fails (no row
    is written). ``rng`` is injectable so the A/B assignment is deterministic in tests.
    """
    picker = rng if rng is not None else random.Random()  # noqa: S311
    first, second = pair
    model_a, model_b = (first, second) if picker.random() < 0.5 else (second, first)

    path_a, path_b = await asyncio.gather(
        _synthesize(settings, model_a, prompt),
        _synthesize(settings, model_b, prompt),
    )
    if path_a is None or path_b is None:
        logger.warning(
            "arena_battle_generation_failed",
            domain=domain,
            model_a=f"{model_a.provider}:{model_a.model}",
            model_b=f"{model_b.provider}:{model_b.model}",
            failed_a=path_a is None,
            failed_b=path_b is None,
        )
        return None

    battle = Battle(
        provider_a=model_a.provider,
        model_a=model_a.model,
        provider_b=model_b.provider,
        model_b=model_b.model,
        domain=domain,
        prompt_text=prompt,
        audio_a_url=store_clip(settings, path_a),
        audio_b_url=store_clip(settings, path_b),
    )
    inserted = await store.insert_battle(battle)
    logger.info(
        "arena_battle_created",
        battle_id=str(inserted.id),
        domain=domain,
        model_a=f"{model_a.provider}:{model_a.model}",
        model_b=f"{model_b.provider}:{model_b.model}",
    )
    return inserted


async def _synthesize(settings: Settings, model: RegisteredModel, prompt: str) -> Path | None:
    """Synthesize one side; return its WAV path, or ``None`` on any failure."""
    provider_cls: Any = TTS_PROVIDERS.get(model.provider)
    if provider_cls is None:
        logger.warning("arena_unknown_provider", provider=model.provider, model=model.model)
        return None
    if model.voice is None:
        logger.warning("arena_missing_voice", provider=model.provider, model=model.model)
        return None

    try:
        provider = provider_cls(settings=settings, model=model.model, voice=model.voice)
        result: TTSResult = await asyncio.wait_for(
            provider.synthesize(prompt), timeout=_SYNTH_TIMEOUT_S
        )
    except TimeoutError:
        logger.warning(
            "arena_synthesis_timeout",
            provider=model.provider,
            model=model.model,
            timeout_s=_SYNTH_TIMEOUT_S,
        )
        return None
    except Exception as exc:
        logger.warning(
            "arena_synthesis_exception",
            provider=model.provider,
            model=model.model,
            error=str(exc),
        )
        return None
    if result.error is not None or result.audio_path is None:
        logger.warning(
            "arena_synthesis_failed",
            provider=model.provider,
            model=model.model,
            reason="provider_error" if result.error is not None else "no_audio",
            error=result.error,
        )
        return None
    return result.audio_path
