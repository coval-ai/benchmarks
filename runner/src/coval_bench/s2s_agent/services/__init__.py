# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pipecat's stock services, picked by what the stack file names.

Pipecat gives every STT, LLM and TTS service the same shape: ``api_key``,
``sample_rate`` and a ``Settings`` whose base fields are ``model``, ``voice``,
``language`` and ``temperature``. So a vendor here is just a class per role
and the Settings key it bills to. A stack is resolved before anything connects,
so an unknown provider or a missing key fails at startup with the supported
list, not mid-call.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal

from pipecat.processors.frame_processor import FrameProcessor

from coval_bench.config import Settings
from coval_bench.s2s_agent.stack import LoadedStack

Role = Literal["stt", "llm", "tts"]


@dataclass(frozen=True)
class Vendor:
    key_attr: str
    stt: str | None = None  # "module:Class" for each role the vendor fills
    llm: str | None = None
    tts: str | None = None


VENDORS: dict[str, Vendor] = {
    "deepgram": Vendor("deepgram_api_key", stt="pipecat.services.deepgram.stt:DeepgramSTTService"),
    "openai": Vendor("openai_api_key", llm="pipecat.services.openai.llm:OpenAILLMService"),
    "elevenlabs": Vendor(
        "elevenlabs_api_key", tts="pipecat.services.elevenlabs.tts:ElevenLabsTTSService"
    ),
    "soniox": Vendor(
        "soniox_api_key",
        stt="pipecat.services.soniox.stt:SonioxSTTService",
        tts="pipecat.services.soniox.tts:SonioxTTSService",
    ),
    "google": Vendor("gemini_api_key", llm="pipecat.services.google.llm:GoogleLLMService"),
}


@dataclass(frozen=True)
class Services:
    stt: FrameProcessor
    llm: FrameProcessor
    tts: FrameProcessor


def supported(role: Role) -> list[str]:
    return sorted(name for name, vendor in VENDORS.items() if getattr(vendor, role) is not None)


def _resolve_vendor(role: Role, provider: str, settings: Settings) -> tuple[str, str]:
    """The class path for this role and the key that pays for it, or why not."""
    vendor = VENDORS.get(provider)
    path: str | None = getattr(vendor, role) if vendor else None
    if vendor is None or path is None:
        raise ValueError(f"no {role} for {provider!r}; have {', '.join(supported(role))}")
    value = getattr(settings, vendor.key_attr)
    if value is None or not value.get_secret_value():
        raise RuntimeError(f"{vendor.key_attr.upper()} is unset")
    key: str = value.get_secret_value()
    return path, key


def _build(
    path: str, api_key: str, fields: dict[str, Any], sample_rate: int | None
) -> FrameProcessor:
    module_name, class_name = path.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    kwargs: dict[str, Any] = {"api_key": api_key, "settings": cls.Settings(**fields)}
    if sample_rate is not None:
        kwargs["sample_rate"] = sample_rate
    service: FrameProcessor = cls(**kwargs)
    return service


def resolve(loaded: LoadedStack, settings: Settings) -> Services:
    """Every role's vendor and key checked first, then the three services built."""
    stack = loaded.stack
    stt_path, stt_key = _resolve_vendor("stt", stack.stt.provider, settings)
    llm_path, llm_key = _resolve_vendor("llm", stack.llm.provider, settings)
    tts_path, tts_key = _resolve_vendor("tts", stack.tts.provider, settings)
    return Services(
        stt=_build(
            stt_path,
            stt_key,
            {"model": stack.stt.model, **stack.stt.options},
            stack.audio.in_sample_rate_hz,
        ),
        llm=_build(
            llm_path,
            llm_key,
            {
                "model": stack.llm.model,
                "temperature": stack.llm.temperature,
                "system_instruction": loaded.system_prompt,
                **stack.llm.options,
            },
            None,
        ),
        tts=_build(
            tts_path,
            tts_key,
            {"model": stack.tts.model, "voice": stack.tts.voice, **stack.tts.options},
            stack.audio.out_sample_rate_hz,
        ),
    )
