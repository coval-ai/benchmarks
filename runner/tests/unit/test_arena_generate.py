# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for on-demand battle generation (stubbed TTS — no real upstream calls)."""

from __future__ import annotations

import random
import wave
from pathlib import Path
from uuid import uuid4

import pytest

from coval_bench.arena import generate as generate_module
from coval_bench.arena.generate import generate_battle
from coval_bench.config import Settings
from coval_bench.db.models import Battle
from coval_bench.providers.base import TTSResult
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import Gender, ModelStatus, RegisteredModel, Voice


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
        status=ModelStatus.ACTIVE,
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
        status=ModelStatus.ACTIVE,
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
