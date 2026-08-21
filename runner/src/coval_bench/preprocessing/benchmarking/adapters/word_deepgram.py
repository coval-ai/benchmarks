# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deepgram Nova-3 prerecorded word timestamps with response-version verification."""

from __future__ import annotations

import hashlib
import secrets
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf
from pydantic import SecretStr

from coval_bench.preprocessing.benchmarking.adapters.base import HostedProviderError
from coval_bench.preprocessing.benchmarking.candidates import (
    DEEPGRAM_MAX_REQUEST_ATTEMPTS,
    DEEPGRAM_RETRY_INITIAL_SECONDS,
    DEEPGRAM_RETRY_JITTER_MAX_MILLISECONDS,
    DEEPGRAM_RETRY_MULTIPLIER,
    DEEPGRAM_TRANSIENT_STATUS_CODES,
    candidate_processor_revision,
    candidate_spec_is_registered,
    deepgram_nova_3_candidate,
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

DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_NOVA_3_MODEL = "nova-3"


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


def deepgram_model_version(value: dict[str, Any]) -> str:
    """Return the exact service model version recorded in Deepgram response metadata."""
    try:
        metadata = value["metadata"]
        model_ids = metadata["models"]
        if not isinstance(model_ids, list) or len(model_ids) != 1:
            raise ValueError("Deepgram response must identify exactly one model")
        model_info = metadata["model_info"][model_ids[0]]
        model_version = model_info["version"]
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("Deepgram response model version is blank")
        return model_version
    except (KeyError, TypeError) as exc:
        raise ValueError("Deepgram response is missing exact model-version metadata") from exc


def parse_deepgram_nova_3_response(
    value: dict[str, Any],
    *,
    expected_model_version: str,
    analysis_id: str,
    audio_sha256: str,
    transcript: str,
    duration_ms: int,
) -> WordTimestampsV1:
    """Convert Nova-3 word timing after verifying the provider's model version."""
    observed_version = deepgram_model_version(value)
    if observed_version != expected_model_version:
        raise ValueError(
            "Deepgram response model version does not match the pinned candidate revision"
        )
    try:
        alternatives = value["results"]["channels"][0]["alternatives"]
        raw_words = alternatives[0]["words"] if alternatives else []
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Deepgram response is missing the top alternative's words") from exc

    words: list[WordTimestampV1] = []
    dropped = 0
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            dropped += 1
            continue
        text = normalize_word(str(raw_word.get("word", "")))
        try:
            start_ms = round(float(raw_word["start"]) * 1_000)
            end_ms = min(round(float(raw_word["end"]) * 1_000), duration_ms)
            confidence = float(raw_word.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        if not text or start_ms < 0 or end_ms <= start_ms:
            dropped += 1
            continue
        words.append(
            WordTimestampV1(
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=max(0.0, min(confidence, 1.0)),
            )
        )
    warnings: list[TimestampWarningV1] = []
    if dropped:
        warnings.append(
            TimestampWarningV1(
                code=TimestampWarningCode.PARTIAL_ALIGNMENT,
                message=f"Deepgram returned {dropped} unusable word-offset entries",
            )
        )
    if not words:
        warnings.append(
            TimestampWarningV1(
                code=TimestampWarningCode.EMPTY_SPANS,
                message="Deepgram Nova-3 returned no timestamped words",
            )
        )
    candidate = deepgram_nova_3_candidate(expected_model_version)
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


def _request(
    *,
    audio_path: Path,
    api_key: SecretStr,
    model_version: str,
) -> dict[str, Any]:
    response: httpx.Response | None = None
    audio = audio_path.read_bytes()
    for attempt in range(DEEPGRAM_MAX_REQUEST_ATTEMPTS):
        try:
            response = httpx.post(
                DEEPGRAM_LISTEN_URL,
                params={
                    "language": "en-US",
                    "model": DEEPGRAM_NOVA_3_MODEL,
                    "punctuate": "false",
                    "smart_format": "false",
                    "version": model_version,
                },
                headers={
                    "Authorization": f"Token {api_key.get_secret_value()}",
                    "Content-Type": "audio/wav",
                },
                content=audio,
                timeout=60.0,
            )
        except httpx.TimeoutException:
            if attempt + 1 == DEEPGRAM_MAX_REQUEST_ATTEMPTS:
                raise
        except httpx.TransportError as exc:
            if attempt + 1 == DEEPGRAM_MAX_REQUEST_ATTEMPTS:
                raise HostedProviderError(
                    code="provider_transient_exhausted",
                    message="Deepgram transport failed after bounded retries",
                ) from exc
        else:
            if response.status_code not in DEEPGRAM_TRANSIENT_STATUS_CODES:
                break
            if attempt + 1 == DEEPGRAM_MAX_REQUEST_ATTEMPTS:
                raise HostedProviderError(
                    code="provider_transient_exhausted",
                    message=(
                        "Deepgram transient request failed after bounded retries "
                        f"(HTTP {response.status_code})"
                    ),
                )
        delay_seconds = (
            DEEPGRAM_RETRY_INITIAL_SECONDS * (DEEPGRAM_RETRY_MULTIPLIER**attempt)
            + secrets.randbelow(DEEPGRAM_RETRY_JITTER_MAX_MILLISECONDS + 1) / 1_000
        )
        time.sleep(delay_seconds)

    if response is None:
        raise RuntimeError("Deepgram request retry loop completed without a response")
    if response.is_error:
        if response.status_code in {401, 403}:
            code = "provider_auth"
        elif response.status_code in {402, 429}:
            code = "provider_quota"
        else:
            code = "provider_permanent"
        raise HostedProviderError(
            code=code,
            message=f"Deepgram request failed (HTTP {response.status_code})",
        )
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("Deepgram response must be a JSON object")
    return value


def discover_deepgram_nova_3_version(*, audio_path: Path, api_key: SecretStr) -> str:
    """Probe ``latest`` once; callers must pin the returned version before the full run."""
    return deepgram_model_version(
        _request(audio_path=audio_path, api_key=api_key, model_version="latest")
    )


class DeepgramNova3WordRecognizer:
    """Freely predict words using one explicitly versioned Nova-3 service model."""

    def __init__(self, *, candidate: CandidateSpecV1, api_key: SecretStr) -> None:
        if candidate.kind is not BenchmarkCandidateKind.WORD_ALIGNER:
            raise ValueError("Deepgram candidate must be a word candidate")
        if candidate.model_name != "deepgram/nova-3" or not candidate.freely_predicts_words:
            raise ValueError("Deepgram candidate provenance is invalid")
        if not candidate_spec_is_registered(candidate):
            raise ValueError("Deepgram candidate request configuration is not canonical")
        self.candidate = candidate
        self._api_key = api_key
        self.model_load_seconds = 0.0
        self.last_inference_seconds = 0.0

    @property
    def runtime_software(self) -> str:
        return f"httpx=={version('httpx')}"

    def align(self, *, audio_path: Path, transcript: str) -> WordTimestampsV1:
        """Use audio only for recognition; ``transcript`` is hashed for evaluation identity."""
        started = time.perf_counter()
        response = _request(
            audio_path=audio_path,
            api_key=self._api_key,
            model_version=self.candidate.model_revision,
        )
        self.last_inference_seconds = time.perf_counter() - started
        return parse_deepgram_nova_3_response(
            response,
            expected_model_version=self.candidate.model_revision,
            analysis_id=audio_path.stem,
            audio_sha256=_sha256_file(audio_path),
            transcript=transcript,
            duration_ms=_duration_ms(audio_path),
        )
