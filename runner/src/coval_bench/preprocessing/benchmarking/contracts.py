# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict public contracts for timestamp-model evaluation inputs and provenance."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from coval_bench.preprocessing.contracts import (
    SHA256_PATTERN,
    NonBlankStr,
    NonNegativeMilliseconds,
    PositiveMilliseconds,
)

FLOATING_REVISIONS = frozenset({"head", "latest", "main", "master"})

NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
Ratio = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
type HostedBillingPolicy = Literal[
    "exact-audio-duration-v1",
    "per-request-ceil-1000ms-v1",
]


class _BenchmarkContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BenchmarkCandidateKind(StrEnum):
    WORD_ALIGNER = "word_aligner"
    PHONEME_RECOGNIZER = "phoneme_recognizer"


class BenchmarkMode(StrEnum):
    HUMAN_GROUND_TRUTH = "human_ground_truth"
    AGREEMENT = "agreement"


class ModelAssetV1(_BenchmarkContract):
    """One immutable model asset included in candidate provenance."""

    path: NonBlankStr
    sha256: str = Field(pattern=SHA256_PATTERN, strict=True)


class CandidateSpecV1(_BenchmarkContract):
    """A benchmark candidate with explicit license and runtime identity."""

    schema_version: Literal["CandidateSpecV1"] = "CandidateSpecV1"
    candidate_id: NonBlankStr
    kind: BenchmarkCandidateKind
    implementation: NonBlankStr
    implementation_revision: NonBlankStr
    model_name: NonBlankStr
    model_revision: NonBlankStr
    assets: tuple[ModelAssetV1, ...]
    decoder: NonBlankStr
    resampler: NonBlankStr
    normalization_version: NonBlankStr
    phone_inventory_version: NonBlankStr | None
    freely_predicts_words: bool | None
    license_id: NonBlankStr | None
    commercial_use_allowed: bool | None
    redistribution_allowed: bool | None
    benchmark_eligible: bool
    license_eligible_for_production: bool
    eligibility_notes: tuple[NonBlankStr, ...]

    @model_validator(mode="after")
    def validate_eligibility(self) -> CandidateSpecV1:
        if self.implementation_revision.casefold() in FLOATING_REVISIONS:
            raise ValueError("implementation_revision must be immutable")
        if self.model_revision.casefold() in FLOATING_REVISIONS:
            raise ValueError("model_revision must be immutable")
        if not self.assets:
            raise ValueError("candidate assets must not be empty")
        if self.kind is BenchmarkCandidateKind.PHONEME_RECOGNIZER:
            if self.phone_inventory_version is None:
                raise ValueError("phoneme candidates require phone_inventory_version")
            if self.freely_predicts_words is not None:
                raise ValueError("phoneme candidates must not declare freely_predicts_words")
        elif self.phone_inventory_version is not None:
            raise ValueError("word candidates must not declare phone_inventory_version")
        elif self.freely_predicts_words is None:
            raise ValueError("word candidates require freely_predicts_words")
        if self.license_eligible_for_production and (
            self.commercial_use_allowed is not True or self.license_id is None
        ):
            raise ValueError(
                "license eligibility for production requires a known commercial-use license"
            )
        return self


class ReferenceWordV1(_BenchmarkContract):
    text: NonBlankStr
    start_ms: NonNegativeMilliseconds
    end_ms: PositiveMilliseconds

    @model_validator(mode="after")
    def validate_interval(self) -> ReferenceWordV1:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class ReferencePhonemeV1(_BenchmarkContract):
    symbol: NonBlankStr
    start_ms: NonNegativeMilliseconds
    end_ms: PositiveMilliseconds

    @model_validator(mode="after")
    def validate_interval(self) -> ReferencePhonemeV1:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class _ReferenceSpan(Protocol):
    @property
    def start_ms(self) -> int: ...

    @property
    def end_ms(self) -> int: ...


def _validate_reference_timeline(*, spans: tuple[_ReferenceSpan, ...], duration_ms: int) -> None:
    previous_end_ms = 0
    for index, span in enumerate(spans):
        if span.end_ms > duration_ms:
            raise ValueError(f"reference span {index} ends after duration_ms")
        if index and span.start_ms < previous_end_ms:
            raise ValueError(f"reference span {index} overlaps or is out of order")
        previous_end_ms = span.end_ms


class WordGroundTruthV1(_BenchmarkContract):
    """Private human word-boundary annotations; values never belong in git."""

    schema_version: Literal["WordGroundTruthV1"] = "WordGroundTruthV1"
    analysis_id: NonBlankStr
    audio_sha256: str = Field(pattern=SHA256_PATTERN, strict=True)
    duration_ms: PositiveMilliseconds
    annotation_revision: NonBlankStr
    words: tuple[ReferenceWordV1, ...]

    @model_validator(mode="after")
    def validate_timeline(self) -> WordGroundTruthV1:
        _validate_reference_timeline(spans=self.words, duration_ms=self.duration_ms)
        return self


class PhonemeGroundTruthV1(_BenchmarkContract):
    """Private observed-phone annotations normalized to the public inventory."""

    schema_version: Literal["PhonemeGroundTruthV1"] = "PhonemeGroundTruthV1"
    analysis_id: NonBlankStr
    audio_sha256: str = Field(pattern=SHA256_PATTERN, strict=True)
    duration_ms: PositiveMilliseconds
    annotation_revision: NonBlankStr
    phone_inventory_version: NonBlankStr
    phones: tuple[ReferencePhonemeV1, ...]

    @model_validator(mode="after")
    def validate_timeline(self) -> PhonemeGroundTruthV1:
        from coval_bench.preprocessing.benchmarking.inventory import (
            COVAL_ENGLISH_PHONES_V1,
            PHONE_INVENTORY_VERSION,
        )

        if self.phone_inventory_version != PHONE_INVENTORY_VERSION:
            raise ValueError(f"phone_inventory_version must be {PHONE_INVENTORY_VERSION!r}")
        inventory = frozenset(COVAL_ENGLISH_PHONES_V1)
        unknown_symbols = sorted({phone.symbol for phone in self.phones} - inventory)
        if unknown_symbols:
            raise ValueError(
                "reference phones are outside the declared inventory: " + ", ".join(unknown_symbols)
            )
        _validate_reference_timeline(spans=self.phones, duration_ms=self.duration_ms)
        return self


class OperationalMeasurementV1(_BenchmarkContract):
    """Raw operational measurements for one isolated candidate invocation."""

    schema_version: Literal["OperationalMeasurementV1"] = "OperationalMeasurementV1"
    candidate_id: NonBlankStr
    hardware: NonBlankStr
    software: NonBlankStr
    batch_size: Annotated[int, Field(gt=0, strict=True)]
    attempted_clips: Annotated[int, Field(gt=0, strict=True)]
    successful_clips: Annotated[int, Field(ge=0, strict=True)]
    audio_duration_ms: PositiveMilliseconds
    model_load_seconds: NonNegativeFiniteFloat
    inference_seconds: PositiveFiniteFloat
    peak_gpu_allocated_bytes: Annotated[int, Field(ge=0, strict=True)] | None
    peak_gpu_reserved_bytes: Annotated[int, Field(ge=0, strict=True)] | None
    peak_host_rss_bytes: Annotated[int, Field(gt=0, strict=True)]
    gpu_hour_cost_usd: NonNegativeFiniteFloat | None = None
    audio_minute_cost_usd: NonNegativeFiniteFloat | None = None
    hosted_provider: NonBlankStr | None = None
    hosted_price_reference: NonBlankStr | None = None
    hosted_billing_policy: HostedBillingPolicy | None = None
    billable_audio_duration_ms: PositiveMilliseconds | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> OperationalMeasurementV1:
        if self.successful_clips > self.attempted_clips:
            raise ValueError("successful_clips cannot exceed attempted_clips")
        if self.gpu_hour_cost_usd is not None and self.audio_minute_cost_usd is not None:
            raise ValueError("operational cost must use either GPU-hour or audio-minute pricing")
        hosted_provenance = (
            self.hosted_provider,
            self.hosted_price_reference,
            self.hosted_billing_policy,
            self.billable_audio_duration_ms,
        )
        if self.audio_minute_cost_usd is not None and any(
            value is None for value in hosted_provenance
        ):
            raise ValueError(
                "hosted per-minute cost requires provider, price, and billing provenance"
            )
        if self.audio_minute_cost_usd is None and any(
            value is not None for value in hosted_provenance
        ):
            raise ValueError("hosted billing provenance requires audio_minute_cost_usd")
        if (
            self.billable_audio_duration_ms is not None
            and self.billable_audio_duration_ms < self.audio_duration_ms
        ):
            raise ValueError("billable_audio_duration_ms cannot be shorter than source audio")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def real_time_factor(self) -> float:
        return self.inference_seconds / (self.audio_duration_ms / 1_000)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def clips_per_minute(self) -> float:
        return self.successful_clips * 60 / self.inference_seconds

    @computed_field  # type: ignore[prop-decorator]
    @property
    def audio_minutes_per_minute(self) -> float:
        return (self.audio_duration_ms / 60_000) / (self.inference_seconds / 60)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failure_rate(self) -> float:
        return (self.attempted_clips - self.successful_clips) / self.attempted_clips

    def estimated_daily_cost_usd(self, *, daily_audio_duration_ms: int) -> float | None:
        if daily_audio_duration_ms < 0:
            raise ValueError("daily_audio_duration_ms must not be negative")
        if self.audio_minute_cost_usd is not None:
            if self.billable_audio_duration_ms is None:
                raise RuntimeError("hosted measurement is missing billable audio duration")
            rounding_ratio = self.billable_audio_duration_ms / self.audio_duration_ms
            return daily_audio_duration_ms * rounding_ratio / 60_000 * self.audio_minute_cost_usd
        if self.gpu_hour_cost_usd is None:
            return None
        gpu_hours = (daily_audio_duration_ms / 1_000) * self.real_time_factor / 3_600
        return gpu_hours * self.gpu_hour_cost_usd
