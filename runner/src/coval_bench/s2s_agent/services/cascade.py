# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The pinned cascade components as Pipecat's stock services: Deepgram, OpenAI, ElevenLabs."""

from __future__ import annotations

from dataclasses import dataclass

from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pydantic import SecretStr

from coval_bench.config import Settings
from coval_bench.s2s_agent.stack import LoadedStack

LANGUAGE = "en"


@dataclass(frozen=True)
class Keys:
    deepgram: str
    openai: str
    elevenlabs: str

    @classmethod
    def from_settings(cls, settings: Settings) -> Keys:
        values = {
            "DEEPGRAM_API_KEY": _secret(settings.deepgram_api_key),
            "OPENAI_API_KEY": _secret(settings.openai_api_key),
            "ELEVENLABS_API_KEY": _secret(settings.elevenlabs_api_key),
        }
        missing = sorted(name for name, value in values.items() if not value)
        if missing:
            raise RuntimeError(f"unset: {', '.join(missing)}")
        return cls(
            deepgram=values["DEEPGRAM_API_KEY"],
            openai=values["OPENAI_API_KEY"],
            elevenlabs=values["ELEVENLABS_API_KEY"],
        )


def _secret(value: SecretStr | None) -> str:
    return value.get_secret_value() if value is not None else ""


def build_stt(loaded: LoadedStack, keys: Keys) -> DeepgramSTTService:
    return DeepgramSTTService(
        api_key=keys.deepgram,
        sample_rate=loaded.stack.audio.in_sample_rate_hz,
        live_options=LiveOptions(model=loaded.components.stt.model, language=LANGUAGE),
    )


def build_llm(loaded: LoadedStack, keys: Keys) -> OpenAILLMService:
    return OpenAILLMService(
        api_key=keys.openai,
        model=loaded.components.llm.model,
        params=OpenAILLMService.InputParams(temperature=loaded.components.llm.temperature),
    )


def build_tts(loaded: LoadedStack, keys: Keys) -> ElevenLabsTTSService:
    return ElevenLabsTTSService(
        api_key=keys.elevenlabs,
        voice_id=loaded.components.tts.voice_id,
        model=loaded.components.tts.model,
        sample_rate=loaded.stack.audio.out_sample_rate_hz,
    )
