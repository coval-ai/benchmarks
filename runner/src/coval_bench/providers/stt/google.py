# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud Speech-to-Text v2 provider (gRPC streaming).

This provider is gated on the optional extra ``google-stt``:

    uv sync --extra google-stt

Auth: GOOGLE_APPLICATION_CREDENTIALS env var (path to service-account JSON
      mounted as a Secret-as-volume in Cloud Run).
Models: short, long, telephony, chirp_2 (default).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time  # monotonic clock — wall-clock can step on NTP sync
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult

logger = structlog.get_logger(__name__)

# Optional gRPC dependency
try:
    from google.api_core.client_options import ClientOptions
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

if TYPE_CHECKING:
    # These type stubs may not be present; used only for static analysis.
    pass

_VALID_MODELS = ("default", "short", "long", "telephony", "chirp_2")
_RECOGNIZER_PATTERN = "projects/{project}/locations/us-central1/recognizers/_"
_API_ENDPOINT = "us-central1-speech.googleapis.com"


class GoogleSTTProvider(STTProvider):
    """Google Cloud Speech-to-Text v2 streaming provider.

    Install with:  uv sync --extra google-stt
    Requires:      GOOGLE_APPLICATION_CREDENTIALS pointing to a service-account JSON.
    """

    def __init__(
        self,
        api_key: SecretStr,
        model: str = "default",
        project_id: str | None = None,
    ) -> None:
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "google-cloud-speech is not installed. Install it with: uv sync --extra google-stt"
            )
        if model not in _VALID_MODELS:
            raise ValueError(f"Invalid Google STT model {model!r}. Valid: {_VALID_MODELS}")
        if project_id is None:
            raise ValueError(
                "GoogleSTTProvider requires project_id (set Settings.google_project_id)"
            )
        # api_key is unused for Google; auth is via GOOGLE_APPLICATION_CREDENTIALS
        _ = api_key
        self._model = model
        self._project_id = project_id
        self._client: Any = SpeechClient(client_options=ClientOptions(api_endpoint=_API_ENDPOINT))

    @property
    def name(self) -> str:
        if self._model == "default":
            return "google"
        return f"google-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _get_model_name(self) -> str:
        """Return the v2 model name; 'default' maps to chirp_2."""
        return "chirp_2" if self._model == "default" else self._model

    def _get_recognizer_name(self) -> str:
        return _RECOGNIZER_PATTERN.format(project=self._project_id)

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
        audio_duration: float | None = None,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name)

        config: Any = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                audio_channel_count=channels,
            ),
            language_codes=["en-US"],
            model=self._get_model_name(),
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
            ),
        )
        streaming_config: Any = cloud_speech.StreamingRecognitionConfig(
            config=config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=True,
            ),
        )

        def _run_sync() -> None:
            final_transcripts: list[str] = []
            last_final_time: float | None = None

            def _request_iterator() -> Generator[Any, None, None]:
                # First request carries the config
                yield cloud_speech.StreamingRecognizeRequest(
                    recognizer=self._get_recognizer_name(),
                    streaming_config=streaming_config,
                )
                data = audio_data
                byte_rate = sample_width * sample_rate * channels
                first_chunk = True
                while data:
                    chunk_size = int(byte_rate * realtime_resolution)
                    chunk, data = data[:chunk_size], data[chunk_size:]
                    if first_chunk and chunk:
                        result.audio_start_time = time.monotonic()
                        first_chunk = False
                    yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
                    if chunk:
                        time.sleep(realtime_resolution)

            try:
                responses = self._client.streaming_recognize(requests=_request_iterator())
                for response in responses:
                    now = time.monotonic()
                    transcript = self._extract_transcript(response)
                    if transcript:
                        if result.ttft_seconds is None and result.audio_start_time is not None:
                            result.ttft_seconds = now - result.audio_start_time
                            result.first_token_content = transcript
                        result.partial_transcripts.append(transcript)

                    if hasattr(response, "results") and response.results:
                        for result_item in response.results:
                            if getattr(result_item, "is_final", False):
                                final_transcripts.append(transcript)
                                last_final_time = now
                                break
            except Exception as exc:
                logger.exception("google streaming_recognize failed", error=str(exc))
                raise

            if last_final_time is not None and result.audio_start_time is not None:
                result.audio_to_final_seconds = last_final_time - result.audio_start_time

            if final_transcripts:
                result.complete_transcript = " ".join(final_transcripts).strip() or None
                if result.complete_transcript:
                    result.transcript_length = len(result.complete_transcript)
                    result.word_count = len(result.complete_transcript.split())

        total_start = time.monotonic()
        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, _run_sync)
        except Exception as exc:
            logger.exception("google measure_ttft failed", error=str(exc))
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

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
