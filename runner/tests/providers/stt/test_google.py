# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.google (GoogleSTTProvider).

The google-cloud-speech package is optional (extra: ``google-stt``).
Tests are skipped when it is not installed.

To run these tests locally:
    uv sync --extra google-stt
    uv run pytest tests/providers/stt/test_google.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

google_speech = pytest.importorskip(
    "google.cloud.speech_v2",
    reason="google-cloud-speech not installed; install with: uv sync --extra google-stt",
)

from coval_bench.providers.stt.google import GoogleSTTProvider  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "google"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _load_fixture_responses() -> list[Any]:
    """Load google/events-success.json and convert to SimpleNamespace objects."""
    raw: list[dict[str, Any]] = json.loads((FIXTURES_DIR / "events-success.json").read_text())
    responses = []
    for item in raw:
        results = []
        for r in item.get("results", []):
            alts = [SimpleNamespace(transcript=a["transcript"]) for a in r.get("alternatives", [])]
            results.append(
                SimpleNamespace(
                    alternatives=alts,
                    is_final=r.get("is_final", False),
                )
            )
        responses.append(SimpleNamespace(results=results))
    return responses


def _make_provider(model: str = "default") -> GoogleSTTProvider:
    with patch(
        "coval_bench.providers.stt.google.SpeechClient",
        return_value=MagicMock(),
    ):
        return GoogleSTTProvider(
            api_key=SecretStr("unused"), model=model, project_id="test-project"
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_google_success(audio_pcm_bytes: bytes) -> None:
    responses = _load_fixture_responses()

    mock_client = MagicMock()

    def _streaming_recognize_side_effect(*, requests: Any, **kwargs: Any) -> Any:
        # Real gRPC consumes the request iterator (config + audio chunks). That
        # drive sets ``audio_start_time`` on the first chunk; without draining
        # ``requests``, TTFT stays None while transcripts can still finalize.
        for _ in requests:
            pass
        return iter(responses)

    mock_client.streaming_recognize.side_effect = _streaming_recognize_side_effect

    with patch(
        "coval_bench.providers.stt.google.SpeechClient",
        return_value=mock_client,
    ):
        provider = GoogleSTTProvider(
            api_key=SecretStr("unused"), model="default", project_id="test-project"
        )

    provider._client = mock_client

    result = await provider.measure_ttft(
        audio_data=audio_pcm_bytes,
        channels=1,
        sample_width=2,
        sample_rate=16000,
        realtime_resolution=0.5,
        audio_duration=3.0,
    )

    assert result.error is None
    assert result.ttft_seconds is not None
    assert result.complete_transcript is not None
    assert "hello world" in result.complete_transcript


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


def test_provider_name_default() -> None:
    p = _make_provider("default")
    assert p.name == "google"


def test_provider_name_chirp2() -> None:
    p = _make_provider("chirp_2")
    assert p.name == "google-chirp_2"


# ---------------------------------------------------------------------------
# get_model_name mapping
# ---------------------------------------------------------------------------


def test_get_model_name_default() -> None:
    p = _make_provider("default")
    assert p._get_model_name() == "chirp_2"


def test_get_model_name_long() -> None:
    p = _make_provider("long")
    assert p._get_model_name() == "long"


# ---------------------------------------------------------------------------
# Invalid model
# ---------------------------------------------------------------------------


def test_invalid_model_raises() -> None:
    with (
        patch("coval_bench.providers.stt.google.SpeechClient", return_value=MagicMock()),
        pytest.raises(ValueError, match="Invalid Google STT model"),
    ):
        GoogleSTTProvider(api_key=SecretStr("k"), model="bad-model", project_id="test-project")


# ---------------------------------------------------------------------------
# Failure path — streaming_recognize raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_google_streaming_error(audio_pcm_bytes: bytes) -> None:
    mock_client = MagicMock()
    mock_client.streaming_recognize.side_effect = RuntimeError("quota exceeded")

    with patch(
        "coval_bench.providers.stt.google.SpeechClient",
        return_value=mock_client,
    ):
        provider = GoogleSTTProvider(
            api_key=SecretStr("unused"), model="default", project_id="test-project"
        )

    provider._client = mock_client

    result = await provider.measure_ttft(
        audio_data=audio_pcm_bytes,
        channels=1,
        sample_width=2,
        sample_rate=16000,
        realtime_resolution=0.5,
    )

    assert result.error is not None
    assert "quota exceeded" in result.error
    assert result.complete_transcript is None
