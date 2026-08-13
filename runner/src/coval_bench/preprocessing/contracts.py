# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict, versioned contracts for timestamp preprocessing artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^[a-f0-9]{64}$"
NONBLANK_PATTERN = r"^.*\S.*$"

NonBlankStr = Annotated[str, Field(min_length=1, pattern=NONBLANK_PATTERN, strict=True)]
NonNegativeMilliseconds = Annotated[int, Field(ge=0, strict=True)]
PositiveMilliseconds = Annotated[int, Field(gt=0, strict=True)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]


class _FrozenContract(BaseModel):
    """Reject coercion and unknown fields, and prevent mutation after validation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class WordProcessorProvenanceV1(_FrozenContract):
    """Exact word-alignment processor identity used to produce an artifact."""

    aligner_name: NonBlankStr
    aligner_revision: NonBlankStr
    normalization_version: NonBlankStr


class PhonemeProcessorProvenanceV1(_FrozenContract):
    """Exact phoneme-alignment processor identity used to produce an artifact."""

    model_name: NonBlankStr
    model_revision: NonBlankStr
    weights_sha256: str = Field(pattern=SHA256_PATTERN, strict=True)
    phone_inventory: tuple[NonBlankStr, ...]
    resampler: NonBlankStr
    decoder: NonBlankStr

    @model_validator(mode="after")
    def validate_inventory(self) -> PhonemeProcessorProvenanceV1:
        """Require a nonempty, unique inventory as part of the processor identity."""
        if not self.phone_inventory:
            raise ValueError("phone_inventory must not be empty")
        if len(set(self.phone_inventory)) != len(self.phone_inventory):
            raise ValueError("phone_inventory must not contain duplicate symbols")
        return self


class TimestampWarningCode(StrEnum):
    """Closed initial registry of warnings accepted in v1 artifacts."""

    EMPTY_SPANS = "EMPTY_SPANS"
    PARTIAL_ALIGNMENT = "PARTIAL_ALIGNMENT"


class TimestampWarningV1(_FrozenContract):
    """A registered, machine-readable preprocessing warning."""

    code: TimestampWarningCode
    message: NonBlankStr
    start_ms: NonNegativeMilliseconds | None = None
    end_ms: PositiveMilliseconds | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> TimestampWarningV1:
        """Require warning bounds to be absent together or form a positive interval."""
        if (self.start_ms is None) != (self.end_ms is None):
            raise ValueError("warning start_ms and end_ms must be provided together")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms <= self.start_ms:
            raise ValueError("warning end_ms must be greater than start_ms")
        return self


class WordTimestampV1(_FrozenContract):
    """One word span. The wire shape is intentionally limited to these four fields."""

    text: NonBlankStr
    start_ms: NonNegativeMilliseconds
    end_ms: PositiveMilliseconds
    confidence: Confidence

    @model_validator(mode="after")
    def validate_interval(self) -> WordTimestampV1:
        """Require a positive-duration span."""
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class PhonemeTimestampV1(_FrozenContract):
    """One phone span. Word-index coupling is intentionally absent from this schema."""

    symbol: NonBlankStr
    start_ms: NonNegativeMilliseconds
    end_ms: PositiveMilliseconds
    confidence: Confidence

    @model_validator(mode="after")
    def validate_interval(self) -> PhonemeTimestampV1:
        """Require a positive-duration span."""
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class _TimestampSpan(Protocol):
    """Structural type shared by word and phoneme timestamp spans."""

    start_ms: int
    end_ms: int


def _validate_timeline(
    *,
    spans: tuple[_TimestampSpan, ...],
    warnings: tuple[TimestampWarningV1, ...],
    duration_ms: int,
) -> None:
    """Validate top-level span/warning invariants common to both artifact schemas."""
    previous_end_ms = 0
    for index, span in enumerate(spans):
        if span.end_ms > duration_ms:
            raise ValueError(f"span {index} ends after duration_ms")
        if index and span.start_ms < previous_end_ms:
            raise ValueError(f"span {index} overlaps or is out of order")
        previous_end_ms = span.end_ms

    has_empty_warning = any(
        warning.code is TimestampWarningCode.EMPTY_SPANS for warning in warnings
    )
    if not spans and not has_empty_warning:
        raise ValueError("empty spans require an EMPTY_SPANS warning")

    for index, warning in enumerate(warnings):
        if warning.end_ms is not None and warning.end_ms > duration_ms:
            raise ValueError(f"warning {index} ends after duration_ms")


class WordTimestampsV1(_FrozenContract):
    """Validated word timestamps for one source-audio sample."""

    schema_version: Literal["WordTimestampsV1"]
    analysis_id: NonBlankStr
    audio_sha256: str = Field(pattern=SHA256_PATTERN, strict=True)
    transcript_sha256: str = Field(pattern=SHA256_PATTERN, strict=True)
    duration_ms: PositiveMilliseconds
    processor: WordProcessorProvenanceV1
    words: tuple[WordTimestampV1, ...]
    warnings: tuple[TimestampWarningV1, ...]

    @model_validator(mode="after")
    def validate_timeline(self) -> WordTimestampsV1:
        """Enforce ordering, non-overlap, bounds, and explicit empty output."""
        _validate_timeline(spans=self.words, warnings=self.warnings, duration_ms=self.duration_ms)
        return self


class PhonemeTimestampsV1(_FrozenContract):
    """Validated phoneme timestamps for one source-audio sample."""

    schema_version: Literal["PhonemeTimestampsV1"]
    analysis_id: NonBlankStr
    audio_sha256: str = Field(pattern=SHA256_PATTERN, strict=True)
    duration_ms: PositiveMilliseconds
    processor: PhonemeProcessorProvenanceV1
    phones: tuple[PhonemeTimestampV1, ...]
    warnings: tuple[TimestampWarningV1, ...]

    @model_validator(mode="after")
    def validate_timeline_and_inventory(self) -> PhonemeTimestampsV1:
        """Enforce timeline invariants and membership in a unique phone inventory."""
        inventory = set(self.processor.phone_inventory)
        for index, phone in enumerate(self.phones):
            if phone.symbol not in inventory:
                raise ValueError(f"phone {index} symbol is not in processor.phone_inventory")
        _validate_timeline(
            spans=self.phones,
            warnings=self.warnings,
            duration_ms=self.duration_ms,
        )
        return self


type TimestampArtifactV1 = Annotated[
    WordTimestampsV1 | PhonemeTimestampsV1,
    Field(discriminator="schema_version"),
]
