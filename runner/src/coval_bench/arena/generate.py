# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""On-demand battle generation: synthesize a model pair on a prompt, persist a battle.

Both sides speak the same prompt (fairness). Which model lands as A vs B is
randomized to avoid position bias in voting. Synthesis runs concurrently to keep
the request fast. A battle is persisted only if both sides succeed — the
``audio_*_url`` columns are NOT NULL, so a half-synthesized battle is never stored.

A side whose key is out of credits does not sink the battle: it is swapped for an
alternate, keeping the clip that did synthesize, and the failure is reported so the key
is alerted on and drops out of pairing (see ``arena.provider_health``).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from coval_bench.arena import provider_health
from coval_bench.arena.audio_store import store_clip
from coval_bench.arena.pairing import voice_for
from coval_bench.arena.provider_health import KeyFailure
from coval_bench.config import Settings
from coval_bench.db.arena_store import ArenaStore
from coval_bench.db.models import Battle
from coval_bench.providers.base import TTSResult
from coval_bench.providers.tts import TTS_PROVIDERS
from coval_bench.registries.models import Gender, RegisteredModel, Voice

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_SYNTH_TIMEOUT_S = 60.0

# Whole-battle wall clock, replacements included. The browser gives up at 30s, so every
# synthesis is capped at whatever is left of this rather than its own fresh 60s.
_BATTLE_BUDGET_S = 25.0

# Too little time left to be worth a paid call.
_MIN_ATTEMPT_S = 4.0

# Replacements per battle, not per side: two dead providers must not cost four extra
# syntheses.
_MAX_SWAPS = 2


@dataclass
class _Budget:
    """What is left of this battle: wall clock, and replacements."""

    started: float
    swaps_left: int

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        return _BATTLE_BUDGET_S - self.elapsed()


@dataclass(frozen=True)
class _Synthesized:
    """One side's attempt: its clip, or why there is none."""

    path: Path | None
    status_code: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.path is not None


async def generate_battle(
    settings: Settings,
    store: ArenaStore,
    *,
    prompt: str,
    domain: str | None,
    pair: tuple[RegisteredModel, RegisteredModel],
    gender: Gender,
    alternates: Sequence[RegisteredModel] = (),
    benched: frozenset[str] = frozenset(),
    rng: random.Random | None = None,
) -> Battle | None:
    """Synthesize both sides of *pair* on *prompt*, store the clips, persist a battle.

    Both sides speak in *gender*, so a listener compares delivery rather than
    register. Returns the inserted ``Battle``, or ``None`` if a battle could not be
    assembled (no row is written). ``alternates`` are stand-ins for a side whose key
    fails, minus the already-``benched`` providers the caller resolved. ``rng`` is
    injectable so the A/B assignment is deterministic in tests.
    """
    budget = _Budget(started=time.monotonic(), swaps_left=_MAX_SWAPS)
    picker = rng if rng is not None else random.Random()  # noqa: S311
    first, second = pair
    model_a, model_b = (first, second) if picker.random() < 0.5 else (second, first)

    # Resolved before synthesis: a roster that cannot field this gender is a
    # caller error, and it should surface without paying for any audio.
    voice_a = voice_for(model_a, gender)
    voice_b = voice_for(model_b, gender)

    result_a, result_b = await asyncio.gather(
        _synthesize(settings, model_a, prompt, voice_a, budget.remaining()),
        _synthesize(settings, model_b, prompt, voice_b, budget.remaining()),
    )
    unusable = set(benched)
    for model, result in ((model_a, result_a), (model_b, result_b)):
        if _report(model, result):
            unusable.add(model.provider)

    spent = {model_a.provider, model_b.provider}
    # Gender is filtered here, not at ``voice_for``: an alternate that cannot sing this
    # gender would raise mid-battle, after both sides were already paid for, and take the
    # surviving clip's cleanup down with it.
    pool = [
        m
        for m in alternates
        if m.provider not in spent
        and m.provider not in unusable
        and any(v.gender is gender for v in m.voices)
    ]
    picker.shuffle(pool)

    if not result_a.ok:
        model_a, voice_a, result_a = await _swap_in(
            settings, prompt, gender, pool, budget, spent, model_a, voice_a, result_a
        )
    if not result_b.ok:
        model_b, voice_b, result_b = await _swap_in(
            settings, prompt, gender, pool, budget, spent, model_b, voice_b, result_b
        )

    if not result_a.ok or not result_b.ok:
        for result in (result_a, result_b):
            if result.path is not None:
                result.path.unlink(missing_ok=True)
        logger.warning(
            "arena_battle_generation_failed",
            domain=domain,
            model_a=f"{model_a.provider}:{model_a.model}",
            model_b=f"{model_b.provider}:{model_b.model}",
            failed_a=not result_a.ok,
            failed_b=not result_b.ok,
            elapsed_s=round(budget.elapsed(), 1),
        )
        return None

    path_a, path_b = result_a.path, result_b.path
    assert path_a is not None and path_b is not None  # noqa: S101 - narrowed by .ok above
    try:
        audio_a_url, audio_b_url = await asyncio.gather(
            asyncio.to_thread(store_clip, settings, path_a),
            asyncio.to_thread(store_clip, settings, path_b),
        )
    except Exception:
        path_a.unlink(missing_ok=True)
        path_b.unlink(missing_ok=True)
        raise
    battle = Battle(
        provider_a=model_a.provider,
        model_a=model_a.model,
        provider_b=model_b.provider,
        model_b=model_b.model,
        domain=domain,
        prompt_text=prompt,
        audio_a_url=audio_a_url,
        audio_b_url=audio_b_url,
        voice_a=voice_a.id,
        voice_b=voice_b.id,
        gender=gender,
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


async def _swap_in(
    settings: Settings,
    prompt: str,
    gender: Gender,
    pool: list[RegisteredModel],
    budget: _Budget,
    spent: set[str],
    dead_model: RegisteredModel,
    dead_voice: Voice,
    dead_result: _Synthesized,
) -> tuple[RegisteredModel, Voice, _Synthesized]:
    """The first stand-in from *pool* that synthesizes, else the failed side unchanged.

    Consumes *pool* and *budget*, both shared with the other side, so two dead providers
    cannot between them outspend one battle's allowance.
    """
    last = dead_model, dead_voice, dead_result
    while pool and budget.swaps_left > 0:
        remaining = budget.remaining()
        if remaining < _MIN_ATTEMPT_S:
            logger.warning(
                "arena_swap_budget_exhausted",
                elapsed_s=round(budget.elapsed(), 1),
                budget_s=_BATTLE_BUDGET_S,
            )
            break
        model = pool.pop(0)
        budget.swaps_left -= 1
        spent.add(model.provider)
        voice = voice_for(model, gender)
        logger.info(
            "arena_side_swapped",
            replaced=f"{dead_model.provider}:{dead_model.model}",
            provider=model.provider,
            model=model.model,
        )
        attempt = await _synthesize(settings, model, prompt, voice, remaining)
        if _report(model, attempt):
            pool[:] = [m for m in pool if m.provider != model.provider]
        last = model, voice, attempt
        if attempt.ok:
            break
    return last


def _report(model: RegisteredModel, result: _Synthesized) -> bool:
    """Alert on a dead key; True if this provider must not stand in for the other side.

    Nothing is written: pairing reads the TTS benchmark, and this battle only needs to
    remember the provider long enough not to reach for it again.
    """
    if result.ok:
        return False
    reason = provider_health.classify_failure(result.status_code, result.error)
    if reason not in provider_health.BENCHING_REASONS:
        if reason is KeyFailure.RATE_LIMIT:
            logger.warning(
                "arena_provider_rate_limited",
                provider=model.provider,
                model=model.model,
                status_code=result.status_code,
            )
        # Not the key's fault: the provider may still stand in for the other side.
        return False
    provider_health.report_key_failure(
        provider=model.provider,
        model=model.model,
        reason=reason,
        status_code=result.status_code,
    )
    return True


async def _synthesize(
    settings: Settings, model: RegisteredModel, prompt: str, voice: Voice, remaining_s: float
) -> _Synthesized:
    """Synthesize one side in *voice* within *remaining_s*; the path, or why there is none."""
    provider_cls: Any = TTS_PROVIDERS.get(model.provider)
    if provider_cls is None:
        logger.warning("arena_unknown_provider", provider=model.provider, model=model.model)
        return _Synthesized(path=None, error="unknown provider")

    timeout = min(_SYNTH_TIMEOUT_S, max(remaining_s, 0.0))
    try:
        provider = provider_cls(settings=settings, model=model.model, voice=voice.id)
        result: TTSResult = await asyncio.wait_for(provider.synthesize(prompt), timeout=timeout)
    except TimeoutError:
        logger.warning(
            "arena_synthesis_timeout",
            provider=model.provider,
            model=model.model,
            timeout_s=round(timeout, 1),
        )
        return _Synthesized(path=None, error="timeout")
    except Exception as exc:
        logger.warning(
            "arena_synthesis_exception",
            provider=model.provider,
            model=model.model,
            error=str(exc),
        )
        return _Synthesized(path=None, error=str(exc))
    if result.error is not None or result.audio_path is None or not result.audio_path.is_file():
        logger.warning(
            "arena_synthesis_failed",
            provider=model.provider,
            model=model.model,
            reason="provider_error" if result.error is not None else "no_audio",
            error=result.error,
            status_code=result.status_code,
        )
        return _Synthesized(path=None, status_code=result.status_code, error=result.error)
    return _Synthesized(path=result.audio_path)
