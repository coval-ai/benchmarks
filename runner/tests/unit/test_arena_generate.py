# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for on-demand battle generation (stubbed TTS — no real upstream calls)."""

from __future__ import annotations

import asyncio
import random
import time
import wave
from pathlib import Path
from uuid import uuid4

import pytest

from coval_bench.arena import generate as generate_module
from coval_bench.arena import provider_health
from coval_bench.arena.generate import generate_battle
from coval_bench.config import Settings
from coval_bench.db.models import Battle
from coval_bench.providers.base import TTSResult
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import Gender, RegisteredModel, Voice


def _model(provider: str, model: str) -> RegisteredModel:
    return RegisteredModel(
        benchmark=Benchmark.TTS,
        provider=provider,
        model=model,
        voice="v",
        voices=(
            Voice(id=f"{model}-f", gender=Gender.FEMALE),
            Voice(id=f"{model}-m", gender=Gender.MALE),
        ),
        collected=True,
        published=True,
    )


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * 100)


def _fake_provider_cls(tmp_path: Path, fail_models: set[str]) -> type:
    """A stub TTS provider: writes a tiny WAV, or returns an error for *fail_models*."""

    class _FakeProvider:
        def __init__(self, settings: Settings, model: str, voice: str) -> None:
            self.model = model

        async def synthesize(self, text: str) -> TTSResult:
            if self.model in fail_models:
                return TTSResult(
                    provider="fake",
                    model=self.model,
                    voice="v",
                    ttfa_ms=None,
                    audio_path=None,
                    error="synth boom",
                )
            path = tmp_path / f"{self.model}-{uuid4().hex}.wav"
            _write_wav(path)
            return TTSResult(
                provider="fake",
                model=self.model,
                voice="v",
                ttfa_ms=12.0,
                audio_path=path,
                error=None,
            )

    return _FakeProvider


class _FakeStore:
    """Minimal ArenaStore stand-in: records inserts and assigns an id."""

    def __init__(self) -> None:
        self.inserted: list[Battle] = []

    async def insert_battle(self, battle: Battle) -> Battle:
        stored = battle.model_copy(update={"id": uuid4()})
        self.inserted.append(stored)
        return stored


async def test_generate_battle_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models = (_model("provider-a", "model-a"), _model("provider-b", "model-b"))
    provider = _fake_provider_cls(tmp_path, fail_models=set())
    monkeypatch.setattr(generate_module, "TTS_PROVIDERS", {m.provider: provider for m in models})
    settings = Settings(arena_audio_dir=tmp_path / "store")
    store = _FakeStore()

    prompt = "Your test results came back normal, so no follow-up is needed."
    battle = await generate_battle(
        settings,
        store,  # type: ignore[arg-type]
        prompt=prompt,
        domain="healthcare",
        pair=models,
        gender=Gender.FEMALE,
        rng=random.Random(0),
    )

    assert battle is not None
    assert battle.id is not None
    assert battle.domain == "healthcare"
    assert battle.prompt_text == prompt
    assert {battle.model_a, battle.model_b} == {"model-a", "model-b"}
    for url in (battle.audio_a_url, battle.audio_b_url):
        assert (settings.arena_audio_dir / url.lstrip("/")).is_file()

    # The pooled female voice sang, not the scalar ``voice="v"``, and the row
    # records which one so the rating can be traced back to a speaker.
    assert battle.gender is Gender.FEMALE
    assert battle.voice_a is not None and battle.voice_a.endswith("-f")
    assert battle.voice_b is not None and battle.voice_b.endswith("-f")
    assert {battle.voice_a, battle.voice_b} == {"model-a-f", "model-b-f"}


async def test_generate_battle_skips_when_a_side_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = (_model("provider-a", "model-a"), _model("provider-b", "model-b"))
    provider = _fake_provider_cls(tmp_path, fail_models={"model-b"})
    monkeypatch.setattr(generate_module, "TTS_PROVIDERS", {m.provider: provider for m in models})
    settings = Settings(arena_audio_dir=tmp_path / "store")
    store = _FakeStore()

    battle = await generate_battle(
        settings,
        store,  # type: ignore[arg-type]
        prompt="Your claim has been approved and payment will arrive within five business days.",
        domain="insurance",
        pair=models,
        gender=Gender.MALE,
        rng=random.Random(0),
    )

    assert battle is None
    assert store.inserted == []


@pytest.mark.asyncio
async def test_generate_battle_uses_the_requested_gender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The male half of each pool sings when a male battle is asked for."""
    models = (_model("provider-a", "model-a"), _model("provider-b", "model-b"))
    provider = _fake_provider_cls(tmp_path, fail_models=set())
    monkeypatch.setattr(generate_module, "TTS_PROVIDERS", {m.provider: provider for m in models})
    settings = Settings(arena_audio_dir=tmp_path / "store")
    store = _FakeStore()

    battle = await generate_battle(
        settings,
        store,  # type: ignore[arg-type]
        prompt="Your appointment has been moved to Thursday afternoon.",
        domain="healthcare",
        pair=models,
        gender=Gender.MALE,
        rng=random.Random(0),
    )

    assert battle is not None
    assert battle.gender is Gender.MALE
    assert {battle.voice_a, battle.voice_b} == {"model-a-m", "model-b-m"}


@pytest.mark.asyncio
async def test_generate_battle_refuses_a_model_without_that_gender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unfiltered roster is a caller bug, and it must not reach a provider.

    The raise has to land before synthesis, otherwise the healthy side has
    already been paid for by the time the broken one is noticed.
    """
    female_only = RegisteredModel(
        benchmark=Benchmark.TTS,
        provider="provider-b",
        model="model-b",
        voice="v",
        voices=(Voice(id="model-b-f", gender=Gender.FEMALE),),
        collected=True,
        published=True,
    )
    models = (_model("provider-a", "model-a"), female_only)
    provider = _fake_provider_cls(tmp_path, fail_models=set())
    monkeypatch.setattr(generate_module, "TTS_PROVIDERS", {m.provider: provider for m in models})
    settings = Settings(arena_audio_dir=tmp_path / "store")
    store = _FakeStore()

    with pytest.raises(ValueError, match="no male voice"):
        await generate_battle(
            settings,
            store,  # type: ignore[arg-type]
            prompt="This should never reach a provider.",
            domain="healthcare",
            pair=models,
            gender=Gender.MALE,
            rng=random.Random(0),
        )

    assert store.inserted == []
    assert not (settings.arena_audio_dir).exists() or not any(
        settings.arena_audio_dir.rglob("*.wav")
    )


# ---------------------------------------------------------------------------
# A dead provider key: swap the side, alert, keep the good clip
# ---------------------------------------------------------------------------

# One stub for every fallback test. ``script`` maps a model to what it does: a
# (status, error) pair to fail with, or a float to sleep for.
_Behaviour = tuple[int | None, str] | float


def _stub_provider(tmp_path: Path, script: dict[str, _Behaviour], attempted: list[str]) -> type:
    class _Stub:
        def __init__(self, settings: Settings, model: str, voice: str) -> None:
            self.model = model

        async def synthesize(self, text: str) -> TTSResult:
            attempted.append(self.model)
            behaviour = script.get(self.model)
            if isinstance(behaviour, tuple):
                status, error = behaviour
                return TTSResult(
                    provider="fake",
                    model=self.model,
                    voice="v",
                    ttfa_ms=None,
                    audio_path=None,
                    error=error,
                    status_code=status,
                )
            if isinstance(behaviour, float):
                await asyncio.sleep(behaviour)
            path = tmp_path / f"{self.model}-{uuid4().hex}.wav"
            _write_wav(path)
            return TTSResult(
                provider="fake",
                model=self.model,
                voice="v",
                ttfa_ms=1.0,
                audio_path=path,
                error=None,
            )

    return _Stub


def _register(monkeypatch: pytest.MonkeyPatch, *models: RegisteredModel, provider: type) -> None:
    monkeypatch.setattr(generate_module, "TTS_PROVIDERS", {m.provider: provider for m in models})


DEAD_KEY = (402, "HTTP 402: out of credits")


async def test_a_dead_key_is_swapped_out_and_alerted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The battle still lands, on the stand-in, and the key is announced."""
    dead = _model("provider-dead", "model-dead")
    alive = _model("provider-alive", "model-alive")
    standin = _model("provider-standin", "model-standin")
    reported: list[dict[str, object]] = []
    monkeypatch.setattr(provider_health, "report_key_failure", lambda **kw: reported.append(kw))
    _register(
        monkeypatch,
        dead,
        alive,
        standin,
        provider=_stub_provider(tmp_path, {"model-dead": DEAD_KEY}, []),
    )
    settings = Settings(arena_audio_dir=tmp_path / "store")

    battle = await generate_battle(
        settings,
        _FakeStore(),  # type: ignore[arg-type]
        prompt="Your prescription is ready for pickup at the front counter.",
        domain="healthcare",
        pair=(dead, alive),
        gender=Gender.FEMALE,
        alternates=[dead, alive, standin],
        rng=random.Random(0),
    )

    assert battle is not None
    assert {battle.model_a, battle.model_b} == {"model-alive", "model-standin"}
    assert [r["provider"] for r in reported] == ["provider-dead"]
    assert reported[0]["reason"] is provider_health.KeyFailure.CREDIT


async def test_a_rate_limited_side_is_swapped_but_not_alerted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """429 is transient, so it costs this battle a swap and nothing more."""
    limited = _model("provider-limited", "model-limited")
    alive = _model("provider-alive", "model-alive")
    standin = _model("provider-standin", "model-standin")
    reported: list[dict[str, object]] = []
    monkeypatch.setattr(provider_health, "report_key_failure", lambda **kw: reported.append(kw))
    _register(
        monkeypatch,
        limited,
        alive,
        standin,
        provider=_stub_provider(tmp_path, {"model-limited": (429, "HTTP 429: slow down")}, []),
    )

    battle = await generate_battle(
        Settings(arena_audio_dir=tmp_path / "store"),
        _FakeStore(),  # type: ignore[arg-type]
        prompt="Your table for two is ready whenever you are.",
        domain="hospitality",
        pair=(limited, alive),
        gender=Gender.FEMALE,
        alternates=[limited, alive, standin],
        rng=random.Random(0),
    )

    assert battle is not None
    assert reported == []


async def test_both_sides_dead_share_one_swap_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two dead providers must not cost four extra syntheses between them."""
    dead_a = _model("provider-dead-a", "model-dead-a")
    dead_b = _model("provider-dead-b", "model-dead-b")
    spares = [_model(f"provider-spare-{i}", f"model-spare-{i}") for i in range(4)]
    attempted: list[str] = []
    # Every alternate is dead too, so the run cannot stop early: a shared budget spends
    # _MAX_SWAPS in total, a per-side one would spend that many twice.
    script: dict[str, _Behaviour] = {m.model: DEAD_KEY for m in (dead_a, dead_b, *spares)}
    _register(
        monkeypatch,
        dead_a,
        dead_b,
        *spares,
        provider=_stub_provider(tmp_path, script, attempted),
    )

    battle = await generate_battle(
        Settings(arena_audio_dir=tmp_path / "store"),
        _FakeStore(),  # type: ignore[arg-type]
        prompt="Both sides are out of credits.",
        domain="healthcare",
        pair=(dead_a, dead_b),
        gender=Gender.FEMALE,
        alternates=[dead_a, dead_b, *spares],
        rng=random.Random(0),
    )

    assert len(attempted) == 2 + generate_module._MAX_SWAPS, attempted
    assert battle is None


async def test_a_replacement_is_cancelled_by_the_remaining_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget caps how long a swap may run, not merely whether it may start."""
    dead = _model("provider-dead", "model-dead")
    alive = _model("provider-alive", "model-alive")
    standin = _model("provider-standin", "model-standin")
    attempted: list[str] = []
    script: dict[str, _Behaviour] = {"model-dead": DEAD_KEY, "model-standin": 30.0}
    _register(
        monkeypatch,
        dead,
        alive,
        standin,
        provider=_stub_provider(tmp_path, script, attempted),
    )
    # The stand-in must actually be started, then cut off — so the floor has to be low.
    monkeypatch.setattr(generate_module, "_BATTLE_BUDGET_S", 1.0)
    monkeypatch.setattr(generate_module, "_MIN_ATTEMPT_S", 0.1)

    started = time.monotonic()
    battle = await generate_battle(
        Settings(arena_audio_dir=tmp_path / "store"),
        _FakeStore(),  # type: ignore[arg-type]
        prompt="The stand-in hangs, and must not hold the request open.",
        domain="healthcare",
        pair=(dead, alive),
        gender=Gender.FEMALE,
        alternates=[dead, alive, standin],
        rng=random.Random(0),
    )
    elapsed = time.monotonic() - started

    assert "model-standin" in attempted, "the hanging stand-in was never started"
    assert battle is None
    assert elapsed < 5, f"the deadline did not cancel synthesis: {elapsed:.1f}s"


async def test_a_benched_provider_is_not_used_as_a_stand_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dead = _model("provider-dead", "model-dead")
    alive = _model("provider-alive", "model-alive")
    benched = _model("provider-benched", "model-benched")
    standin = _model("provider-standin", "model-standin")
    _register(
        monkeypatch,
        dead,
        alive,
        benched,
        standin,
        provider=_stub_provider(tmp_path, {"model-dead": DEAD_KEY}, []),
    )

    battle = await generate_battle(
        Settings(arena_audio_dir=tmp_path / "store"),
        _FakeStore(),  # type: ignore[arg-type]
        prompt="The stand-in must not be a provider the benchmark says is dead.",
        domain="healthcare",
        pair=(dead, alive),
        gender=Gender.FEMALE,
        alternates=[benched, standin],
        benched=frozenset({"provider-benched"}),
        rng=random.Random(0),
    )

    assert battle is not None
    assert {battle.model_a, battle.model_b} == {"model-alive", "model-standin"}


async def test_a_stand_in_without_that_gender_is_never_drawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filtered where the pool is built, not discovered by ``voice_for`` raising after
    both sides have already been paid for."""
    dead = _model("provider-dead", "model-dead")
    alive = _model("provider-alive", "model-alive")
    female_only = RegisteredModel(
        benchmark=Benchmark.TTS,
        provider="provider-female-only",
        model="model-female-only",
        voice="v",
        voices=(Voice(id="model-female-only-f", gender=Gender.FEMALE),),
        collected=True,
        published=True,
    )
    standin = _model("provider-standin", "model-standin")
    _register(
        monkeypatch,
        dead,
        alive,
        female_only,
        standin,
        provider=_stub_provider(tmp_path, {"model-dead": DEAD_KEY}, []),
    )

    battle = await generate_battle(
        Settings(arena_audio_dir=tmp_path / "store"),
        _FakeStore(),  # type: ignore[arg-type]
        prompt="A male battle cannot be rescued by a female-only voice.",
        domain="healthcare",
        pair=(dead, alive),
        gender=Gender.MALE,
        alternates=[female_only, standin],
        rng=random.Random(0),
    )

    assert battle is not None
    assert {battle.model_a, battle.model_b} == {"model-alive", "model-standin"}
