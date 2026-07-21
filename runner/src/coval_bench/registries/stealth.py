# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Env-defined stealth models — embargoed identities never enter this repo.

Real provider, model, and voice IDs live only in the ``STEALTH_MODELS`` env
var (JSON, mounted from Secret Manager); registry entries, result rows, and
API responses carry only the alias, with voices persisted as positional
labels (``voice-0`` pinned, ``voice-1``… pool). Unset, placeholder, or
malformed input logs a warning and yields no stealth models — misconfiguration
skips the models, it never leaks a name.

JSON shape, keyed by alias::

    {
      "stealth-01": {
        "benchmark": "TTS",
        "provider": "elevenlabs",
        "model": "real-model-id",
        "voice": "real-voice-id",
        "voices": ["real-voice-f", "real-voice-m"],
        "api_key": "optional; defaults to the provider's configured key"
      }
    }
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, SecretStr, TypeAdapter, ValidationError, field_validator

from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import ModelStatus, RegisteredModel

if TYPE_CHECKING:
    from coval_bench.config import Settings

logger = structlog.get_logger("coval_bench.registries.stealth")

# Pseudo-provider under which every stealth model's results are keyed.
STEALTH_PROVIDER = "stealth"


def _voice_label(index: int) -> str:
    return f"voice-{index}"


class StealthUpstream(BaseModel, frozen=True, extra="forbid"):
    """The real identity behind one stealth alias."""

    benchmark: Benchmark
    provider: str
    model: str
    voice: str | None = None
    voices: tuple[str, ...] = ()
    api_key: SecretStr | None = None

    @field_validator("benchmark")
    @classmethod
    def _runnable_benchmark(cls, value: Benchmark) -> Benchmark:
        if value is Benchmark.S2S:
            raise ValueError("stealth models support STT and TTS only (S2S is fetch-only)")
        return value

    def resolve_voice(self, label: str | None) -> str | None:
        """Map a persisted voice label back to the real voice ID."""
        if label is None:
            return None
        index = int(label.removeprefix("voice-"))
        return self.voice if index == 0 else self.voices[index - 1]


_UPSTREAMS_ADAPTER: TypeAdapter[dict[str, StealthUpstream]] = TypeAdapter(
    dict[str, StealthUpstream]
)


@functools.lru_cache(maxsize=4)
def _parse(raw: str) -> dict[str, StealthUpstream]:
    try:
        return _UPSTREAMS_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        # Locations and types only — a rendered ValidationError includes input
        # values, which for this payload are the real names and keys.
        errors = [
            f"{'.'.join(map(str, err['loc'])) or '<root>'}: {err['type']}"
            for err in exc.errors(include_input=False)
        ]
        logger.warning("stealth_models_invalid", errors=errors)
        return {}


def stealth_upstreams(settings: Settings) -> dict[str, StealthUpstream]:
    """Alias → real upstream. Empty when ``STEALTH_MODELS`` is unset or invalid."""
    if settings.stealth_models is None:
        return {}
    return _parse(settings.stealth_models.get_secret_value())


def stealth_entries(settings: Settings) -> list[RegisteredModel]:
    """Alias-only registry entries for the env-defined stealth models."""
    return [
        RegisteredModel(
            benchmark=upstream.benchmark,
            provider=STEALTH_PROVIDER,
            model=alias,
            voice=_voice_label(0) if upstream.voice is not None else None,
            voices=tuple(_voice_label(i + 1) for i in range(len(upstream.voices))),
            status=ModelStatus.EARLY_ACCESS,
            arena_enabled=False,
        )
        for alias, upstream in stealth_upstreams(settings).items()
    ]
