import dataclasses
import json
from typing import Any

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup

from coval_bench.s2s_agent import stack as stacks
from coval_bench.s2s_agent.coval import (
    SESSION_READY,
    CovalFrameSerializer,
    authorized,
    simulation_id,
)

SILENCE_16K_100MS = bytes(2 * 1600)


async def _serializer(pipeline_in_hz: int) -> CovalFrameSerializer:
    serializer = CovalFrameSerializer()
    plumbing: dict[str, Any] = {
        field.name: None
        for field in dataclasses.fields(FrameProcessorSetup)
        if not field.name.startswith("audio_")
    }
    await serializer.setup(
        FrameProcessorSetup(
            **plumbing, audio_in_sample_rate=pipeline_in_hz, audio_out_sample_rate=24000
        )
    )
    return serializer


@pytest.mark.asyncio
async def test_inbound_pcm_is_resampled_once_to_the_pipeline_rate() -> None:
    serializer = await _serializer(24000)
    total = 0
    for _ in range(6):
        frame = await serializer.deserialize(SILENCE_16K_100MS)
        assert isinstance(frame, InputAudioRawFrame)
        assert frame.sample_rate == 24000 and frame.num_channels == 1
        total += len(frame.audio)
    # The stream resampler emits in blocks and holds a little back, so pin the ratio over six.
    expected = len(SILENCE_16K_100MS) * 1.5 * 6
    assert expected * 0.85 <= total <= expected


@pytest.mark.asyncio
async def test_inbound_pcm_passes_through_at_a_matching_rate() -> None:
    serializer = await _serializer(16000)
    frame = await serializer.deserialize(SILENCE_16K_100MS)
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.audio == SILENCE_16K_100MS and frame.sample_rate == 16000


@pytest.mark.asyncio
async def test_speech_markers_become_transport_messages_and_other_text_is_dropped() -> None:
    serializer = await _serializer(24000)
    marker = {"type": "utterance_end", "utterance_id": "utterance-3"}
    frame = await serializer.deserialize(json.dumps(marker))
    assert isinstance(frame, InputTransportMessageFrame)
    assert frame.message == marker
    assert await serializer.deserialize(json.dumps({"type": "mark"})) is None
    assert await serializer.deserialize("not json") is None


@pytest.mark.asyncio
async def test_outbound_audio_and_control_go_out_as_coval_expects() -> None:
    serializer = await _serializer(24000)
    audio = bytes(2 * 2400)
    frame = TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1)
    out = await serializer.serialize(frame)
    assert out == audio
    ready = await serializer.serialize(OutputTransportMessageUrgentFrame(message=SESSION_READY))
    assert isinstance(ready, str) and json.loads(ready) == {"type": "session_ready"}


def test_simulation_id_comes_from_the_upgrade_header_case_insensitively() -> None:
    assert simulation_id({"X-Coval-Simulation-Id": "sim-1"}) == "sim-1"
    assert simulation_id({"x-coval-simulation-id": ""}) is None
    assert simulation_id({}) is None


def test_bearer_token_is_required_only_when_configured() -> None:
    assert authorized({}, None)
    assert authorized({"Authorization": "Bearer s3cret"}, "s3cret")
    assert not authorized({"Authorization": "Bearer other"}, "s3cret")
    assert not authorized({}, "s3cret")


REFERENCE = "cascade-nova3-gpt41-flash"


def test_stack_names_its_components_and_loads_the_default_prompt_and_hash() -> None:
    loaded = stacks.load_stack("bank", REFERENCE)
    assert loaded.stack.architecture == "cascade"
    assert (loaded.stack.stt.provider, loaded.stack.stt.model) == ("deepgram", "nova-3")
    assert (loaded.stack.llm.provider, loaded.stack.llm.model) == ("openai", "gpt-4.1")
    assert loaded.stack.tts.provider == "elevenlabs"
    assert loaded.stack.turn_taking.vad_stop_secs == 0.2
    assert loaded.prompt_file == "system-prompt.txt"
    assert loaded.system_prompt.startswith("# Role & Objective")
    assert len(loaded.digest) == 64
    assert loaded.digest == stacks.stack_sha256("bank", REFERENCE)


def test_the_reference_stack_mirrors_the_orchestration_benchmarks_pinned_layer() -> None:
    """Same STT, LLM and TTS as every platform variant, so the boards can be compared."""
    from coval_bench import scenarios

    pinned = scenarios.load_stack()
    stack = stacks.load_stack("bank", REFERENCE).stack
    assert (stack.stt.provider, stack.stt.model) == (pinned.stt.provider, pinned.stt.model)
    assert (stack.llm.provider, stack.llm.model, stack.llm.temperature) == (
        pinned.llm.provider,
        pinned.llm.model,
        pinned.llm.temperature,
    )
    assert (stack.tts.provider, stack.tts.model, stack.tts.voice) == (
        pinned.tts.provider,
        pinned.tts.model,
        pinned.tts.voice_id,
    )


def test_a_stack_without_an_override_falls_back_to_the_scenario_default() -> None:
    assert stacks.prompt_file("bank", REFERENCE) == "system-prompt.txt"
    assert stacks.prompt_file("bank", "some-other-stack") == "system-prompt.txt"


def test_the_hash_covers_the_stack_file_and_the_prompt() -> None:
    import hashlib

    expected = hashlib.sha256()
    expected.update(stacks._stack_bytes(REFERENCE))
    expected.update(stacks._prompt_bytes("bank", "system-prompt.txt"))
    assert stacks.stack_sha256("bank", REFERENCE) == expected.hexdigest()


def test_an_unknown_stack_names_the_ones_that_exist() -> None:
    assert stacks.stack_slugs() == [REFERENCE, "cascade-soniox-gemini"]
    with pytest.raises(ValueError, match=REFERENCE):
        stacks.load_stack("bank", "nope")


def test_stack_rejects_unknown_keys_but_keeps_rationale() -> None:
    data = json.loads(stacks._stack_bytes(REFERENCE))
    assert stacks.S2SStack.model_validate({**data, "_note": "fine"})
    with pytest.raises(ValueError, match="unknown key"):
        stacks.S2SStack.model_validate({**data, "voice": "alloy"})
    with pytest.raises(ValueError, match="unknown key"):
        stacks.S2SStack.model_validate({**data, "tts": {**data["tts"], "voise": "x"}})
