# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Adversarial CPU-only tests for timestamp preprocessing contracts."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from coval_bench.preprocessing import (
    DEFAULT_MAX_BATCH_CLIPS,
    DEFAULT_MAX_BATCH_DURATION_MS,
    PhonemeProcessorProvenanceV1,
    PhonemeTimestampsV1,
    PhonemeTimestampV1,
    TimestampArtifactV1,
    TimestampWarningCode,
    TimestampWarningV1,
    WordProcessorProvenanceV1,
    WordTimestampsV1,
    WordTimestampV1,
    artifact_content_fingerprint,
    batch_by_duration,
    canonical_json_bytes,
    generator_fingerprint,
    immutable_artifact_key,
)

SOURCE_SHA = "a" * 64
TRANSCRIPT_SHA = "b" * 64
WEIGHTS_SHA = "c" * 64


def _word_processor(**overrides: Any) -> WordProcessorProvenanceV1:
    values: dict[str, Any] = {
        "aligner_name": "word-aligner",
        "aligner_revision": "revision-123",
        "normalization_version": "normalization-v1",
    }
    values.update(overrides)
    return WordProcessorProvenanceV1(**values)


def _phoneme_processor(**overrides: Any) -> PhonemeProcessorProvenanceV1:
    values: dict[str, Any] = {
        "model_name": "phoneme-model",
        "model_revision": "revision-456",
        "weights_sha256": WEIGHTS_SHA,
        "phone_inventory": ("HH", "AH"),
        "resampler": "soxr-hq-v1",
        "decoder": "ctc-greedy-v1",
    }
    values.update(overrides)
    return PhonemeProcessorProvenanceV1(**values)


def _word(text: str = "hello", start_ms: int = 0, end_ms: int = 250) -> WordTimestampV1:
    return WordTimestampV1(text=text, start_ms=start_ms, end_ms=end_ms, confidence=0.95)


def _phoneme(symbol: str = "HH", start_ms: int = 0, end_ms: int = 100) -> PhonemeTimestampV1:
    return PhonemeTimestampV1(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=0.9,
    )


def _word_artifact(**overrides: Any) -> WordTimestampsV1:
    values: dict[str, Any] = {
        "schema_version": "WordTimestampsV1",
        "analysis_id": "analysis-001",
        "audio_sha256": SOURCE_SHA,
        "transcript_sha256": TRANSCRIPT_SHA,
        "processor": _word_processor(),
        "duration_ms": 1_000,
        "words": (_word(),),
        "warnings": (),
    }
    values.update(overrides)
    return WordTimestampsV1(**values)


def _phoneme_artifact(**overrides: Any) -> PhonemeTimestampsV1:
    values: dict[str, Any] = {
        "schema_version": "PhonemeTimestampsV1",
        "analysis_id": "analysis-001",
        "audio_sha256": SOURCE_SHA,
        "processor": _phoneme_processor(),
        "duration_ms": 1_000,
        "phones": (_phoneme(),),
        "warnings": (),
    }
    values.update(overrides)
    return PhonemeTimestampsV1(**values)


def test_schema_versions_and_shapes_are_exact_and_independent() -> None:
    word = _word_artifact()
    phoneme = _phoneme_artifact()

    assert word.schema_version == "WordTimestampsV1"
    assert phoneme.schema_version == "PhonemeTimestampsV1"
    assert set(word.words[0].model_dump()) == {"text", "start_ms", "end_ms", "confidence"}
    assert set(phoneme.phones[0].model_dump()) == {
        "symbol",
        "start_ms",
        "end_ms",
        "confidence",
    }
    assert type(word.processor) is WordProcessorProvenanceV1
    assert type(phoneme.processor) is PhonemeProcessorProvenanceV1
    assert set(word.model_dump()) == {
        "schema_version",
        "analysis_id",
        "audio_sha256",
        "transcript_sha256",
        "duration_ms",
        "processor",
        "words",
        "warnings",
    }
    assert set(phoneme.model_dump()) == {
        "schema_version",
        "analysis_id",
        "audio_sha256",
        "duration_ms",
        "processor",
        "phones",
        "warnings",
    }
    assert set(word.processor.model_dump()) == {
        "aligner_name",
        "aligner_revision",
        "normalization_version",
    }
    assert set(phoneme.processor.model_dump()) == {
        "model_name",
        "model_revision",
        "weights_sha256",
        "phone_inventory",
        "resampler",
        "decoder",
    }


def test_discriminated_union_rejects_cross_schema_payloads() -> None:
    class Envelope(BaseModel):
        artifact: TimestampArtifactV1

    payload = {"artifact": _word_artifact().model_dump(mode="json")}
    parsed = Envelope.model_validate_json(json.dumps(payload))
    assert type(parsed.artifact) is WordTimestampsV1

    crossed = _word_artifact().model_dump(mode="json")
    crossed["schema_version"] = "PhonemeTimestampsV1"
    with pytest.raises(ValidationError):
        Envelope.model_validate_json(json.dumps({"artifact": crossed}))

    lowercase = _word_artifact().model_dump(mode="json")
    lowercase["schema_version"] = "wordtimestampsv1"
    with pytest.raises(ValidationError):
        Envelope.model_validate_json(json.dumps({"artifact": lowercase}))


def test_contracts_are_frozen_strict_and_forbid_extras() -> None:
    artifact = _word_artifact()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WordTimestampV1.model_validate({**artifact.words[0].model_dump(), "word_index": 0})
    with pytest.raises(ValidationError):
        WordTimestampV1(text="hello", start_ms=0.0, end_ms=1, confidence=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        artifact.duration_ms = 2_000  # type: ignore[misc]


@pytest.mark.parametrize("field", ["schema_version", "warnings"])
def test_required_top_level_fields_cannot_be_omitted(field: str) -> None:
    payload = _word_artifact().model_dump(mode="json")
    del payload[field]
    with pytest.raises(ValidationError, match="Field required"):
        WordTimestampsV1.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("value", ["", "   ", 123])
def test_analysis_id_is_strict_nonblank(value: object) -> None:
    with pytest.raises(ValidationError):
        _word_artifact(analysis_id=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_ms": -1},
        {"end_ms": 0},
        {"start_ms": 2, "end_ms": 2},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"confidence": float("nan")},
    ],
)
def test_word_span_rejects_invalid_intervals_and_confidence(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {"text": "hi", "start_ms": 0, "end_ms": 1, "confidence": 1.0}
    values.update(kwargs)
    with pytest.raises(ValidationError):
        WordTimestampV1(**values)


@pytest.mark.parametrize(
    "words",
    [
        (_word("a", 100, 200), _word("b", 50, 90)),
        (_word("a", 0, 200), _word("b", 199, 300)),
        (_word("a", 0, 1_001),),
    ],
)
def test_artifact_rejects_out_of_order_overlapping_or_unbounded_spans(
    words: tuple[WordTimestampV1, ...],
) -> None:
    with pytest.raises(ValidationError):
        _word_artifact(words=words)


def test_adjacent_spans_are_valid() -> None:
    artifact = _word_artifact(words=(_word("a", 0, 100), _word("b", 100, 200)))
    assert len(artifact.words) == 2


def test_empty_spans_require_registered_empty_warning() -> None:
    with pytest.raises(ValidationError, match="EMPTY_SPANS"):
        _word_artifact(words=())

    warning = TimestampWarningV1(code=TimestampWarningCode.EMPTY_SPANS, message="No speech")
    assert _word_artifact(words=(), warnings=(warning,)).words == ()
    with pytest.raises(ValidationError, match="EMPTY_SPANS"):
        _phoneme_artifact(phones=())
    assert _phoneme_artifact(phones=(), warnings=(warning,)).phones == ()


def test_warning_codes_and_intervals_are_strict_paired_and_bounded() -> None:
    with pytest.raises(ValidationError):
        TimestampWarningV1(code="UNKNOWN", message="Unknown")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="provided together"):
        TimestampWarningV1(
            code=TimestampWarningCode.PARTIAL_ALIGNMENT,
            message="Partial",
            start_ms=100,
        )
    warning = TimestampWarningV1(
        code=TimestampWarningCode.PARTIAL_ALIGNMENT,
        message="Partial",
        start_ms=900,
        end_ms=1_001,
    )
    with pytest.raises(ValidationError, match="duration_ms"):
        _word_artifact(warnings=(warning,))


def test_processor_phone_inventory_is_unique_nonempty_and_contains_every_symbol() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        _phoneme_processor(phone_inventory=())
    with pytest.raises(ValidationError, match="duplicate"):
        _phoneme_processor(phone_inventory=("HH", "HH"))
    with pytest.raises(ValidationError, match="phone_inventory"):
        _phoneme_artifact(phones=(_phoneme("ZZ"),))


def test_canonical_bytes_and_key_use_the_same_content_sha() -> None:
    artifact = _word_artifact()
    encoded = canonical_json_bytes(artifact)
    digest = hashlib.sha256(encoded).hexdigest()
    key = immutable_artifact_key(artifact)

    assert encoded == canonical_json_bytes(artifact.model_dump(mode="json"))
    assert digest == artifact_content_fingerprint(artifact)
    assert key.endswith(f"/{digest}.json")
    assert "/WordTimestampsV1/" in key


def test_generator_fingerprint_is_key_order_independent_and_config_sensitive() -> None:
    first = generator_fingerprint(_word_processor())
    same = generator_fingerprint(_word_processor())
    changed = generator_fingerprint(_word_processor(aligner_revision="revision-789"))
    assert first == same
    assert first != changed


def test_import_has_no_gpu_stack_side_effect() -> None:
    assert not {"torch", "torchaudio", "whisperx"}.intersection(sys.modules)


@dataclass(frozen=True)
class _Clip:
    identifier: str
    duration_ms: int


def _batch_ids(batches: tuple[tuple[_Clip, ...], ...]) -> list[list[str]]:
    return [[clip.identifier for clip in batch] for batch in batches]


def test_integer_batching_defaults_order_limits_and_exact_boundary() -> None:
    assert DEFAULT_MAX_BATCH_CLIPS == 32
    assert DEFAULT_MAX_BATCH_DURATION_MS == 300_000
    clips = [_Clip("a", 200_000), _Clip("b", 100_000), _Clip("c", 1), _Clip("d", 99)]
    batches = batch_by_duration(clips, duration_ms_of=lambda clip: clip.duration_ms)
    assert _batch_ids(batches) == [["a", "b"], ["c", "d"]]


def test_integer_batching_respects_clip_limit_and_oversized_singleton() -> None:
    clips = [_Clip("a", 1), _Clip("large", 300_001), _Clip("b", 1), _Clip("c", 1)]
    batches = batch_by_duration(
        clips,
        duration_ms_of=lambda clip: clip.duration_ms,
        max_clips=1,
    )
    assert _batch_ids(batches) == [["a"], ["large"], ["b"], ["c"]]


@pytest.mark.parametrize("duration", [0, -1, 1.0, True, "1"])
def test_integer_batching_rejects_nonpositive_or_noninteger_duration(duration: object) -> None:
    with pytest.raises(ValueError):
        batch_by_duration(
            [_Clip("bad", duration)],  # type: ignore[arg-type]
            duration_ms_of=lambda clip: clip.duration_ms,
        )
