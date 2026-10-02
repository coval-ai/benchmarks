# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

pytest.importorskip("pipecat")

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.soniox.stt import SonioxSTTService
from pipecat.services.soniox.tts import SonioxTTSService

from coval_bench.config import Settings
from coval_bench.s2s_agent import services
from coval_bench.s2s_agent.stack import load_stack

KEYS: dict[str, Any] = {
    "deepgram_api_key": "dg",
    "openai_api_key": "oa",
    "elevenlabs_api_key": "el",
    "soniox_api_key": "sx",
    "gemini_api_key": "gm",
}


def test_each_role_lists_the_vendors_that_fill_it() -> None:
    assert services.supported("stt") == ["deepgram", "soniox"]
    assert services.supported("llm") == ["google", "openai"]
    assert services.supported("tts") == ["elevenlabs", "soniox"]


def test_the_reference_stack_resolves_to_deepgram_openai_elevenlabs() -> None:
    built = services.resolve(
        load_stack("bank", "cascade-nova3-gpt41-flash"), Settings(_env_file=None, **KEYS)
    )
    assert isinstance(built.stt, DeepgramSTTService)
    assert isinstance(built.llm, OpenAILLMService)
    assert isinstance(built.tts, ElevenLabsTTSService)


def test_the_soniox_gemini_stack_resolves_to_soniox_google_soniox() -> None:
    built = services.resolve(
        load_stack("bank", "cascade-soniox-gemini"), Settings(_env_file=None, **KEYS)
    )
    assert isinstance(built.stt, SonioxSTTService)
    assert isinstance(built.llm, GoogleLLMService)
    assert isinstance(built.tts, SonioxTTSService)


def test_an_unknown_provider_fails_before_anything_connects() -> None:
    loaded = load_stack("bank", "cascade-nova3-gpt41-flash")
    bad = loaded.stack.model_copy(
        update={"tts": loaded.stack.tts.model_copy(update={"provider": "nope"})}
    )
    with pytest.raises(ValueError, match="no tts for 'nope'; have elevenlabs, soniox"):
        services.resolve(
            loaded.__class__(**{**loaded.__dict__, "stack": bad}), Settings(_env_file=None, **KEYS)
        )


def test_a_missing_key_names_the_env_var() -> None:
    keys = {**KEYS, "soniox_api_key": None}
    with pytest.raises(RuntimeError, match="SONIOX_API_KEY is unset"):
        services.resolve(
            load_stack("bank", "cascade-soniox-gemini"), Settings(_env_file=None, **keys)
        )
