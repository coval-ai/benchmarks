# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud Speech-to-Text v2 provider (gRPC streaming).

This provider is gated on the optional extra ``google-stt``:

    uv sync --extra google-stt

Auth: Application Default Credentials (the runner-rt service account in Cloud Run).
Models: chirp_2 (default, us-central1), chirp_3 (us).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time  # monotonic clock — wall-clock can step on NTP sync
from collections.abc import Generator
from typing import Any

from coval_bench.config import Settings
from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._pacing import paced_chunks_sync
from coval_bench.providers.stt._transcript_utils import (
    add_partial_transcript,
    finalize_transcript,
    set_first_token,
)

# Optional gRPC dependency
try:
    from google.api_core.client_options import ClientOptions
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

_RECOGNIZER_PATTERN = "projects/{project}/locations/{location}/recognizers/_"

# chirp_3 isn't served in us-central1; it lives in the us multi-region.
_MODEL_LOCATIONS = {"chirp_3": "us"}
_DEFAULT_LOCATION = "us-central1"


class GoogleSTTProvider(STTProvider):
    """Google Cloud Speech-to-Text v2 streaming provider.

    Install with:  uv sync --extra google-stt
    Requires:      Application Default Credentials (the runner service account in Cloud Run).
    """

    def __init__(self, settings: Settings, model: str = "default") -> None:
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "google-cloud-speech is not installed. Install it with: uv sync --extra google-stt"
            )
        super().__init__(settings, model)
        self._project_id = self._require_setting("google_project_id")
        self._location = _MODEL_LOCATIONS.get(self._get_model_name(), _DEFAULT_LOCATION)
        self._client: Any = SpeechClient(
            client_options=ClientOptions(api_endpoint=f"{self._location}-speech.googleapis.com")
        )

    @property
    def name(self) -> str:
        if self._model == "default":
            return "google"
        return f"google-{self._model}"

    def _get_model_name(self) -> str:
        """Return the v2 model name; 'default' maps to chirp_2."""
        return "chirp_2" if self._model == "default" else self._model

    def _get_recognizer_name(self) -> str:
        return _RECOGNIZER_PATTERN.format(project=self._project_id, location=self._location)

    async def _transcribe(
        self,
        audio_data: bytes,
        sample_rate: int,
        realtime_resolution: float,
        result: TranscriptionResult,
    ) -> None:
        config: Any = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                audio_channel_count=1,
            ),
            language_codes=["en-US"],
            model=self._get_model_name(),
            features=cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True),
        )
        streaming_config: Any = cloud_speech.StreamingRecognitionConfig(
            config=config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(interim_results=True),
        )

        def _request_iterator() -> Generator[Any, None, None]:
            # First request carries the config
            yield cloud_speech.StreamingRecognizeRequest(
                recognizer=self._get_recognizer_name(), streaming_config=streaming_config
            )
            byte_rate = sample_rate * 2
            chunk_size = int(byte_rate * realtime_resolution)
            for chunk, start in paced_chunks_sync(audio_data, chunk_size, byte_rate):
                if result.audio_start_time is None:
                    result.audio_start_time = start
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

        def _run_sync() -> None:
            finals: list[str] = []
            last_final_time: float | None = None
            try:
                responses = self._client.streaming_recognize(requests=_request_iterator())
                for response in responses:
                    now = time.monotonic()
                    transcript = self._extract_transcript(response)
                    if transcript:
                        set_first_token(result, transcript, now=now)
                        add_partial_transcript(result, transcript)
                    for item in getattr(response, "results", None) or []:
                        if not getattr(item, "is_final", False):
                            continue
                        last_final_time = now
                        if item.alternatives:
                            finals.append(str(item.alternatives[0].transcript))
            finally:
                # Commits received finals even when the stream raises mid-clip.
                if last_final_time is not None and result.audio_start_time is not None:
                    result.audio_to_final_seconds = last_final_time - result.audio_start_time
                finalize_transcript(result, final_segments=finals, partial_fallback="none")

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, _run_sync)

    def _extract_transcript(self, response: Any) -> str:
        try:
            if not response.results:
                return ""
            first = response.results[0]
            if not first.alternatives:
                return ""
            return str(first.alternatives[0].transcript)
        except (AttributeError, IndexError):
            return ""
