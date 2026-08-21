# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud Speech-to-Text v2 word timestamps for short prerecorded clips."""

from __future__ import annotations

import hashlib
import time
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import soundfile as sf
from google.api_core.retry import Retry, if_transient_error

from coval_bench.preprocessing.benchmarking.candidates import (
    CANDIDATES,
    GOOGLE_RETRY_DEADLINE_SECONDS,
    GOOGLE_RETRY_INITIAL_SECONDS,
    GOOGLE_RETRY_MAXIMUM_SECONDS,
    GOOGLE_RETRY_MULTIPLIER,
    candidate_processor_revision,
)
from coval_bench.preprocessing.benchmarking.contracts import (
    BenchmarkCandidateKind,
    CandidateSpecV1,
)
from coval_bench.preprocessing.benchmarking.metrics import normalize_word
from coval_bench.preprocessing.contracts import (
    TimestampWarningCode,
    TimestampWarningV1,
    WordProcessorProvenanceV1,
    WordTimestampsV1,
    WordTimestampV1,
)

GOOGLE_CHIRP_3_CANDIDATE_ID = "word-google-chirp-3-hosted-v1"
GOOGLE_CHIRP_3_LOCATION = "us"
GOOGLE_CHIRP_3_MODEL = "chirp_3"


def _candidate() -> CandidateSpecV1:
    return next(
        candidate
        for candidate in CANDIDATES
        if candidate.candidate_id == GOOGLE_CHIRP_3_CANDIDATE_ID
    )


class _DurationLike(Protocol):
    seconds: int
    nanos: int


class _WordLike(Protocol):
    word: str
    start_offset: _DurationLike
    end_offset: _DurationLike
    confidence: float


class _AlternativeLike(Protocol):
    words: tuple[_WordLike, ...]


class _ResultLike(Protocol):
    alternatives: tuple[_AlternativeLike, ...]


class _ResponseLike(Protocol):
    results: tuple[_ResultLike, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _duration_ms(path: Path) -> int:
    info = sf.info(path)
    return int(round(info.frames * 1_000 / info.samplerate))


def google_recognize_retry() -> Retry:
    """Return the bounded Google retry policy for transient service failures."""
    return Retry(
        predicate=if_transient_error,
        initial=GOOGLE_RETRY_INITIAL_SECONDS,
        maximum=GOOGLE_RETRY_MAXIMUM_SECONDS,
        multiplier=GOOGLE_RETRY_MULTIPLIER,
        deadline=GOOGLE_RETRY_DEADLINE_SECONDS,
    )


def _offset_ms(offset: _DurationLike | timedelta) -> int:
    if isinstance(offset, timedelta):
        return round(offset.total_seconds() * 1_000)
    seconds = int(offset.seconds)
    nanos = int(offset.nanos)
    return round(seconds * 1_000 + nanos / 1_000_000)


def parse_google_chirp_3_response(
    response: _ResponseLike,
    *,
    analysis_id: str,
    audio_sha256: str,
    transcript: str,
    duration_ms: int,
) -> WordTimestampsV1:
    """Convert the top alternative's word offsets without using the known transcript."""
    candidate = _candidate()
    words: list[WordTimestampV1] = []
    dropped = 0
    for result in response.results:
        alternatives = result.alternatives
        if not alternatives:
            continue
        for raw_word in alternatives[0].words:
            text = normalize_word(str(raw_word.word))
            start_ms = _offset_ms(raw_word.start_offset)
            end_ms = min(_offset_ms(raw_word.end_offset), duration_ms)
            if not text or end_ms <= start_ms:
                dropped += 1
                continue
            words.append(
                WordTimestampV1(
                    text=text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    # Chirp 3 does not support word-level confidence. The strict v1
                    # timestamp schema uses zero as the unavailable-value sentinel.
                    confidence=0.0,
                )
            )
    warnings: list[TimestampWarningV1] = []
    if dropped:
        warnings.append(
            TimestampWarningV1(
                code=TimestampWarningCode.PARTIAL_ALIGNMENT,
                message=f"Google returned {dropped} unusable word-offset entries",
            )
        )
    if not words:
        warnings.append(
            TimestampWarningV1(
                code=TimestampWarningCode.EMPTY_SPANS,
                message="Google Chirp 3 returned no timestamped words",
            )
        )
    return WordTimestampsV1(
        schema_version="WordTimestampsV1",
        analysis_id=analysis_id,
        audio_sha256=audio_sha256,
        transcript_sha256=_sha256_text(transcript),
        duration_ms=duration_ms,
        processor=WordProcessorProvenanceV1(
            aligner_name=candidate.model_name,
            aligner_revision=candidate_processor_revision(candidate),
            normalization_version=candidate.normalization_version,
        ),
        words=tuple(words),
        warnings=tuple(warnings),
    )


class GoogleChirp3WordRecognizer:
    """Freely predict words and offsets with Google Speech-to-Text v2 Recognize."""

    def __init__(self, *, project_id: str) -> None:
        if not project_id.strip():
            raise ValueError("project_id is required for the Google word-timestamp candidate")
        self.candidate = _candidate()
        if self.candidate.kind is not BenchmarkCandidateKind.WORD_ALIGNER:
            raise RuntimeError("Google candidate registry entry has the wrong kind")
        request_asset = self.candidate.assets[0]
        request_path = Path(__file__).parents[1] / request_asset.path
        if _sha256_file(request_path) != request_asset.sha256:
            raise ValueError("Google request configuration does not match its candidate SHA-256")
        self._project_id = project_id
        self._client: Any = None
        self.model_load_seconds = 0.0
        self.last_inference_seconds = 0.0

    @property
    def runtime_software(self) -> str:
        try:
            package_version = version("google-cloud-speech")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "Google candidate requires the optional google-stt dependency"
            ) from exc
        return f"google-cloud-speech=={package_version}"

    def _load(self) -> None:
        if self._client is not None:
            return
        from google.api_core.client_options import ClientOptions
        from google.cloud.speech_v2 import SpeechClient

        started = time.perf_counter()
        self._client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{GOOGLE_CHIRP_3_LOCATION}-speech.googleapis.com"
            )
        )
        self.model_load_seconds = time.perf_counter() - started

    def align(self, *, audio_path: Path, transcript: str) -> WordTimestampsV1:
        """Use audio only for recognition; ``transcript`` is hashed for evaluation identity."""
        from google.cloud.speech_v2.types import cloud_speech

        self._load()
        config: Any = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model=GOOGLE_CHIRP_3_MODEL,
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=False,
                enable_word_time_offsets=True,
            ),
        )
        request: Any = cloud_speech.RecognizeRequest(
            recognizer=(
                f"projects/{self._project_id}/locations/{GOOGLE_CHIRP_3_LOCATION}/recognizers/_"
            ),
            config=config,
            content=audio_path.read_bytes(),
        )
        started = time.perf_counter()
        response = self._client.recognize(
            request=request,
            retry=google_recognize_retry(),
            timeout=60.0,
        )
        self.last_inference_seconds = time.perf_counter() - started
        return parse_google_chirp_3_response(
            response,
            analysis_id=audio_path.stem,
            audio_sha256=_sha256_file(audio_path),
            transcript=transcript,
            duration_ms=_duration_ms(audio_path),
        )
