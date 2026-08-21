# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Local research runners for private timestamp agreement bundles."""

from __future__ import annotations

import hashlib
import platform
import resource
import time
from pathlib import Path
from typing import Annotated, Literal

import httpx
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

from coval_bench.preprocessing.benchmarking.adapters import (
    create_phoneme_recognizer,
    create_word_aligner,
)
from coval_bench.preprocessing.benchmarking.adapters.base import HostedProviderError
from coval_bench.preprocessing.benchmarking.candidates import (
    CANDIDATES,
    DEEPGRAM_NOVA_3_CANDIDATE_ID,
    candidate_spec_is_registered,
)
from coval_bench.preprocessing.benchmarking.contracts import (
    BenchmarkCandidateKind,
    BenchmarkMode,
    CandidateSpecV1,
    HostedBillingPolicy,
    OperationalMeasurementV1,
)
from coval_bench.preprocessing.benchmarking.privacy import validate_private_evidence_path
from coval_bench.preprocessing.benchmarking.reporting import (
    CandidateFailureV1,
    FailureKind,
    PhonemeAgreementCaseV1,
    PhonemeCandidateOutputV1,
    TimestampBenchmarkBundleV1,
    WordAgreementCaseV1,
    WordCandidateOutputV1,
)
from coval_bench.preprocessing.contracts import NonBlankStr

PositiveInteger = Annotated[int, Field(gt=0, strict=True)]


class _RuntimeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PublicSTTManifestItemV1(BaseModel):
    """The public fields needed to join a frozen benchmark selection."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    path: NonBlankStr
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$", strict=True)
    transcript: NonBlankStr
    duration_sec: Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]


class PublicSTTManifestV1(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    id: NonBlankStr
    items: tuple[PublicSTTManifestItemV1, ...]


class PrivateAudioClipV1(_RuntimeContract):
    analysis_id: NonBlankStr
    audio_path: Path
    audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$", strict=True)
    duration_ms: PositiveInteger

    @model_validator(mode="after")
    def validate_identity(self) -> PrivateAudioClipV1:
        if self.audio_path.stem != self.analysis_id:
            raise ValueError("analysis_id must equal the private audio filename stem")
        return self


class PrivateAudioSelectionV1(BaseModel):
    """A frozen clip list reused across isolated candidate manifests."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    dataset_id: NonBlankStr
    clips: tuple[PrivateAudioClipV1, ...]


class PhonemeAgreementRunManifestV1(_RuntimeContract):
    schema_version: Literal["PhonemeAgreementRunManifestV1"] = "PhonemeAgreementRunManifestV1"
    benchmark_id: NonBlankStr
    dataset_id: NonBlankStr
    candidate_ids: tuple[NonBlankStr, ...]
    clips: tuple[PrivateAudioClipV1, ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> PhonemeAgreementRunManifestV1:
        if len(self.candidate_ids) != 1:
            raise ValueError(
                "candidate_ids must contain exactly one candidate so operational measurements "
                "are process-isolated"
            )
        candidates = {candidate.candidate_id: candidate for candidate in CANDIDATES}
        for candidate_id in self.candidate_ids:
            candidate = candidates.get(candidate_id)
            if candidate is None or candidate.kind is not BenchmarkCandidateKind.PHONEME_RECOGNIZER:
                raise ValueError(f"{candidate_id!r} is not a registered phoneme candidate")
        if not self.clips:
            raise ValueError("clips must not be empty")
        if len({clip.analysis_id for clip in self.clips}) != len(self.clips):
            raise ValueError("clip analysis_id values must be unique")
        return self


class PrivateWordClipV1(PrivateAudioClipV1):
    transcript: NonBlankStr


class WordAgreementRunManifestV1(_RuntimeContract):
    schema_version: Literal["WordAgreementRunManifestV1"] = "WordAgreementRunManifestV1"
    benchmark_id: NonBlankStr
    dataset_id: NonBlankStr
    candidate_ids: tuple[NonBlankStr, ...]
    candidate_spec: CandidateSpecV1 | None = None
    clips: tuple[PrivateWordClipV1, ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> WordAgreementRunManifestV1:
        if len(self.candidate_ids) != 1:
            raise ValueError(
                "candidate_ids must contain exactly one candidate so operational measurements "
                "are process-isolated"
            )
        candidates = {candidate.candidate_id: candidate for candidate in CANDIDATES}
        candidate = self.candidate_spec or candidates.get(self.candidate_ids[0])
        if self.candidate_ids[0] == DEEPGRAM_NOVA_3_CANDIDATE_ID and self.candidate_spec is None:
            raise ValueError("Deepgram manifests require an exact candidate_spec from the probe")
        if (
            candidate is None
            or candidate.candidate_id != self.candidate_ids[0]
            or candidate.kind is not BenchmarkCandidateKind.WORD_ALIGNER
            or not candidate_spec_is_registered(candidate)
        ):
            raise ValueError(f"{self.candidate_ids[0]!r} is not a registered word candidate")
        if not self.clips:
            raise ValueError("clips must not be empty")
        if len({clip.analysis_id for clip in self.clips}) != len(self.clips):
            raise ValueError("clip analysis_id values must be unique")
        return self


class PhonemeOperationalRunV1(_RuntimeContract):
    schema_version: Literal["PhonemeOperationalRunV1"] = "PhonemeOperationalRunV1"
    benchmark_id: NonBlankStr
    dataset_id: NonBlankStr
    measurements: tuple[OperationalMeasurementV1, ...]


class WordOperationalRunV1(_RuntimeContract):
    schema_version: Literal["WordOperationalRunV1"] = "WordOperationalRunV1"
    benchmark_id: NonBlankStr
    dataset_id: NonBlankStr
    measurements: tuple[OperationalMeasurementV1, ...]


def build_word_agreement_manifest(
    *,
    dataset_manifest: PublicSTTManifestV1,
    selection_manifest: PrivateAudioSelectionV1,
    candidate_id: str,
    benchmark_id: str,
    candidate_spec: CandidateSpecV1 | None = None,
) -> WordAgreementRunManifestV1:
    """Join a frozen private clip selection to its public dataset transcripts."""
    if selection_manifest.dataset_id != dataset_manifest.id:
        raise ValueError("selection dataset_id does not match the public dataset manifest")
    items = {Path(item.path).stem: item for item in dataset_manifest.items}
    clips: list[PrivateWordClipV1] = []
    for selected in selection_manifest.clips:
        item = items.get(selected.analysis_id)
        if item is None:
            raise ValueError(
                f"selected clip {selected.analysis_id!r} is absent from the dataset manifest"
            )
        expected_duration_ms = round(item.duration_sec * 1_000)
        if (
            item.sha256 != selected.audio_sha256
            or abs(expected_duration_ms - selected.duration_ms) > 1
        ):
            raise ValueError(
                f"selected clip {selected.analysis_id!r} does not match the dataset manifest"
            )
        clips.append(
            PrivateWordClipV1(
                **selected.model_dump(),
                transcript=item.transcript,
            )
        )
    return WordAgreementRunManifestV1(
        benchmark_id=benchmark_id,
        dataset_id=dataset_manifest.id,
        candidate_ids=(candidate_id,),
        candidate_spec=candidate_spec,
        clips=tuple(clips),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_clip(clip: PrivateAudioClipV1) -> None:
    audio_path = validate_private_evidence_path(clip.audio_path)
    if not audio_path.is_file():
        raise ValueError(f"private audio does not exist for {clip.analysis_id!r}")
    if _sha256(audio_path) != clip.audio_sha256:
        raise ValueError(f"private audio SHA-256 mismatch for {clip.analysis_id!r}")
    info = sf.info(audio_path)
    actual_duration_ms = round(info.frames * 1_000 / info.samplerate)
    if abs(actual_duration_ms - clip.duration_ms) > 1:
        raise ValueError(f"private audio duration mismatch for {clip.analysis_id!r}")


def _peak_host_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1_024)


def _gpu_memory(device: str) -> tuple[int | None, int | None]:
    if not device.startswith("cuda"):
        return None, None
    import torch

    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def _failure_kind(error: Exception) -> FailureKind:
    """Keep timeout and invalid-output rates distinct from provider/runtime failures."""
    if isinstance(error, (TimeoutError, httpx.TimeoutException)) or type(error).__name__ in {
        "DeadlineExceeded",
        "ReadTimeout",
        "ConnectTimeout",
    }:
        return FailureKind.TIMEOUT
    if isinstance(error, (ValidationError, ValueError)):
        return FailureKind.INVALID_ARTIFACT
    return FailureKind.FAILURE


def _failure_code(error: Exception) -> str:
    """Return a stable sanitized code without provider payloads or credentials."""
    if isinstance(error, HostedProviderError):
        return error.code
    if _failure_kind(error) is FailureKind.TIMEOUT:
        return "provider_timeout"
    google_codes = {
        "Unauthenticated": "provider_auth",
        "PermissionDenied": "provider_auth",
        "ResourceExhausted": "provider_quota",
        "InvalidArgument": "provider_permanent",
        "FailedPrecondition": "provider_permanent",
        "ServiceUnavailable": "provider_transient_exhausted",
        "InternalServerError": "provider_transient_exhausted",
        "TooManyRequests": "provider_quota",
    }
    return google_codes.get(type(error).__name__, type(error).__name__)


def _billable_audio_duration_ms(
    durations_ms: tuple[int, ...],
    policy: HostedBillingPolicy,
) -> int:
    if policy == "exact-audio-duration-v1":
        return sum(durations_ms)
    return sum(((duration_ms + 999) // 1_000) * 1_000 for duration_ms in durations_ms)


def run_word_agreement(
    manifest: WordAgreementRunManifestV1,
    *,
    google_project_id: str | None = None,
    deepgram_api_key: SecretStr | None = None,
    device: str = "cpu",
    local_files_only: bool = False,
    audio_minute_cost_usd: float | None = None,
    hosted_price_reference: str | None = None,
    hosted_billing_policy: HostedBillingPolicy | None = None,
) -> tuple[TimestampBenchmarkBundleV1, WordOperationalRunV1]:
    """Run one freely predicting or forced word candidate over private local audio."""
    for clip in manifest.clips:
        _validate_clip(clip)
    candidate_id = manifest.candidate_ids[0]
    candidate = manifest.candidate_spec or next(
        value for value in CANDIDATES if value.candidate_id == candidate_id
    )
    hosted_provider = {
        "word-google-chirp-3-hosted-v1": "google-cloud",
        DEEPGRAM_NOVA_3_CANDIDATE_ID: "deepgram",
    }.get(candidate_id)
    if audio_minute_cost_usd is not None and hosted_provider is None:
        raise ValueError("audio-minute API pricing is only valid for hosted candidates")
    price_provenance = (hosted_price_reference, hosted_billing_policy)
    if audio_minute_cost_usd is not None and any(value is None for value in price_provenance):
        raise ValueError("hosted pricing requires a price reference and billing policy")
    if audio_minute_cost_usd is None and any(value is not None for value in price_provenance):
        raise ValueError("hosted price provenance requires audio-minute pricing")
    adapter = create_word_aligner(
        candidate_id=candidate_id,
        candidate=candidate,
        google_project_id=google_project_id,
        deepgram_api_key=deepgram_api_key,
        device=device,
        local_files_only=local_files_only,
    )
    runtime_software = adapter.runtime_software
    outputs_by_clip: dict[str, list[WordCandidateOutputV1]] = {
        clip.analysis_id: [] for clip in manifest.clips
    }
    failures_by_clip: dict[str, list[CandidateFailureV1]] = {
        clip.analysis_id: [] for clip in manifest.clips
    }
    started = time.perf_counter()
    successful = 0
    for clip in manifest.clips:
        try:
            artifact = adapter.align(audio_path=clip.audio_path, transcript=clip.transcript)
            if artifact.analysis_id != clip.analysis_id:
                raise ValueError("adapter artifact analysis_id does not match the manifest")
            outputs_by_clip[clip.analysis_id].append(
                WordCandidateOutputV1(
                    candidate_id=candidate_id,
                    freely_predicted_words=bool(candidate.freely_predicts_words),
                    artifact=artifact,
                )
            )
            successful += 1
        except Exception as error:  # noqa: BLE001 - failures are benchmark outcomes
            failures_by_clip[clip.analysis_id].append(
                CandidateFailureV1(
                    candidate_id=candidate_id,
                    kind=_failure_kind(error),
                    code=_failure_code(error),
                )
            )
    processing_seconds = max(time.perf_counter() - started - adapter.model_load_seconds, 1e-9)
    total_audio_duration_ms = sum(clip.duration_ms for clip in manifest.clips)
    billable_audio_duration_ms = (
        _billable_audio_duration_ms(
            tuple(clip.duration_ms for clip in manifest.clips),
            hosted_billing_policy,
        )
        if audio_minute_cost_usd is not None and hosted_billing_policy is not None
        else None
    )
    measurement = OperationalMeasurementV1(
        candidate_id=candidate_id,
        hardware=(
            f"hosted-api-client-{platform.machine()}"
            if hosted_provider is not None
            else f"{platform.machine()}-{device}"
        ),
        software=runtime_software,
        batch_size=1,
        attempted_clips=len(manifest.clips),
        successful_clips=successful,
        audio_duration_ms=total_audio_duration_ms,
        model_load_seconds=adapter.model_load_seconds,
        inference_seconds=processing_seconds,
        peak_gpu_allocated_bytes=None,
        peak_gpu_reserved_bytes=None,
        peak_host_rss_bytes=_peak_host_rss_bytes(),
        gpu_hour_cost_usd=None,
        audio_minute_cost_usd=audio_minute_cost_usd,
        hosted_provider=hosted_provider if audio_minute_cost_usd is not None else None,
        hosted_price_reference=(
            hosted_price_reference if audio_minute_cost_usd is not None else None
        ),
        hosted_billing_policy=(
            hosted_billing_policy if audio_minute_cost_usd is not None else None
        ),
        billable_audio_duration_ms=billable_audio_duration_ms,
    )
    cases = tuple(
        WordAgreementCaseV1(
            analysis_id=clip.analysis_id,
            audio_sha256=clip.audio_sha256,
            duration_ms=clip.duration_ms,
            reference_transcript=clip.transcript,
            transcript_sha256=hashlib.sha256(clip.transcript.encode("utf-8")).hexdigest(),
            outputs=tuple(outputs_by_clip[clip.analysis_id]),
            failures=tuple(failures_by_clip[clip.analysis_id]),
        )
        for clip in manifest.clips
    )
    return (
        TimestampBenchmarkBundleV1(
            benchmark_id=manifest.benchmark_id,
            dataset_id=manifest.dataset_id,
            mode=BenchmarkMode.AGREEMENT,
            kind=BenchmarkCandidateKind.WORD_ALIGNER,
            candidate_ids=manifest.candidate_ids,
            candidate_specs=(candidate,),
            cases=cases,
        ),
        WordOperationalRunV1(
            benchmark_id=manifest.benchmark_id,
            dataset_id=manifest.dataset_id,
            measurements=(measurement,),
        ),
    )


def run_phoneme_agreement(
    manifest: PhonemeAgreementRunManifestV1,
    *,
    device: str,
    local_files_only: bool,
    gpu_hour_cost_usd: float | None = None,
) -> tuple[TimestampBenchmarkBundleV1, PhonemeOperationalRunV1]:
    """Run candidates from audio only and retain every success or failure outcome."""
    for clip in manifest.clips:
        _validate_clip(clip)
    outputs_by_clip: dict[str, list[PhonemeCandidateOutputV1]] = {
        clip.analysis_id: [] for clip in manifest.clips
    }
    failures_by_clip: dict[str, list[CandidateFailureV1]] = {
        clip.analysis_id: [] for clip in manifest.clips
    }
    measurements: list[OperationalMeasurementV1] = []
    total_audio_duration_ms = sum(clip.duration_ms for clip in manifest.clips)
    for candidate_id in manifest.candidate_ids:
        adapter = create_phoneme_recognizer(
            candidate_id=candidate_id,
            device=device,
            local_files_only=local_files_only,
        )
        candidate_started = time.perf_counter()
        successful = 0
        for clip in manifest.clips:
            try:
                artifact = adapter.recognize(audio_path=clip.audio_path)
                if artifact.analysis_id != clip.analysis_id:
                    raise ValueError("adapter artifact analysis_id does not match the manifest")
                outputs_by_clip[clip.analysis_id].append(
                    PhonemeCandidateOutputV1(
                        candidate_id=candidate_id,
                        normalization_loss_rate=adapter.last_normalization_loss_rate,
                        artifact=artifact,
                    )
                )
                successful += 1
            except Exception as error:  # noqa: BLE001 - failures are benchmark outcomes
                failures_by_clip[clip.analysis_id].append(
                    CandidateFailureV1(
                        candidate_id=candidate_id,
                        kind=_failure_kind(error),
                        code=_failure_code(error),
                    )
                )
        elapsed_seconds = time.perf_counter() - candidate_started
        processing_seconds = max(elapsed_seconds - adapter.model_load_seconds, 1e-9)
        allocated, reserved = _gpu_memory(device)
        measurements.append(
            OperationalMeasurementV1(
                candidate_id=candidate_id,
                hardware=f"{platform.machine()}-{device}",
                software=adapter.runtime_software,
                batch_size=1,
                attempted_clips=len(manifest.clips),
                successful_clips=successful,
                audio_duration_ms=total_audio_duration_ms,
                model_load_seconds=adapter.model_load_seconds,
                inference_seconds=processing_seconds,
                peak_gpu_allocated_bytes=allocated,
                peak_gpu_reserved_bytes=reserved,
                peak_host_rss_bytes=_peak_host_rss_bytes(),
                gpu_hour_cost_usd=gpu_hour_cost_usd,
                audio_minute_cost_usd=None,
            )
        )
    cases = tuple(
        PhonemeAgreementCaseV1(
            analysis_id=clip.analysis_id,
            audio_sha256=clip.audio_sha256,
            duration_ms=clip.duration_ms,
            outputs=tuple(outputs_by_clip[clip.analysis_id]),
            failures=tuple(failures_by_clip[clip.analysis_id]),
        )
        for clip in manifest.clips
    )
    return (
        TimestampBenchmarkBundleV1(
            benchmark_id=manifest.benchmark_id,
            dataset_id=manifest.dataset_id,
            mode=BenchmarkMode.AGREEMENT,
            kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER,
            candidate_ids=manifest.candidate_ids,
            candidate_specs=tuple(
                candidate
                for candidate in CANDIDATES
                if candidate.candidate_id in manifest.candidate_ids
            ),
            cases=cases,
        ),
        PhonemeOperationalRunV1(
            benchmark_id=manifest.benchmark_id,
            dataset_id=manifest.dataset_id,
            measurements=tuple(measurements),
        ),
    )
