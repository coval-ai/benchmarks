# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("pipecat")

from coval_bench.s2s_agent.app import choose_stack


def test_the_menu_lists_each_stacks_components_and_takes_a_number(
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(["zero", "9", "2"])
    chosen = choose_stack("bank", read=lambda _prompt: next(answers))
    out = capsys.readouterr().out
    assert "1. cascade-nova3-gpt41-flash" in out
    assert "stt deepgram/nova-3  llm openai/gpt-4.1  tts elevenlabs/eleven_flash_v2" in out
    assert "2. cascade-soniox-gemini" in out
    assert "stt soniox/stt-rt-v5  llm google/gemini-2.5-flash  tts soniox/tts-rt-v2" in out
    assert out.count("pick a number from 1 to 2") == 2
    assert chosen.slug == "cascade-soniox-gemini"
