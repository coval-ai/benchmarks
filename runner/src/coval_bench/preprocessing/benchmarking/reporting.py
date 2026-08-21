# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict file-based inputs and deterministic reports for private benchmark runs."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from enum import StrEnum
from itertools import combinations
from statistics import fmean
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from coval_bench.preprocessing.benchmarking.candidates import (
    candidate_processor_revision,
    candidate_spec_is_registered,
)
from coval_bench.preprocessing.benchmarking.contracts import (
    BenchmarkCandidateKind,
    BenchmarkMode,
    CandidateSpecV1,
    NonNegativeFiniteFloat,
    PhonemeGroundTruthV1,
    Ratio,
    WordGroundTruthV1,
)
from coval_bench.preprocessing.benchmarking.inventory import (
    COVAL_ENGLISH_PHONES_V1,
    PHONE_INVENTORY_VERSION,
)
from coval_bench.preprocessing.benchmarking.metrics import (
    PhonemeAccuracyMetricsV1,
    PhonemeAgreementMetricsV1,
    WordAccuracyMetricsV1,
    WordAgreementMetricsV1,
    WordRecognitionMetricsV1,
    evaluate_phoneme_accuracy,
    evaluate_phoneme_agreement,
    evaluate_word_accuracy,
    evaluate_word_agreement,
    evaluate_word_recognition,
)
from coval_bench.preprocessing.contracts import (
    SHA256_PATTERN,
    NonBlankStr,
    PhonemeTimestampsV1,
    PositiveMilliseconds,
    TimestampWarningCode,
    WordTimestampsV1,
)

NonNegativeInteger = Annotated[int, Field(ge=0, strict=True)]


class _ReportContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FailureKind(StrEnum):
    FAILURE = "failure"
    INVALID_ARTIFACT = "invalid_artifact"
    TIMEOUT = "timeout"


class CandidateFailureV1(_ReportContract):
    candidate_id: NonBlankStr
    kind: FailureKind
    code: NonBlankStr


class WordCandidateOutputV1(_ReportContract):
    candidate_id: NonBlankStr
    freely_predicted_words: bool
    artifact: WordTimestampsV1


class PhonemeCandidateOutputV1(_ReportContract):
    candidate_id: NonBlankStr
    normalization_loss_rate: Ratio
    artifact: PhonemeTimestampsV1


class WordGroundTruthCaseV1(_ReportContract):
    schema_version: Literal["WordGroundTruthCaseV1"] = "WordGroundTruthCaseV1"
    reference: WordGroundTruthV1
    outputs: tuple[WordCandidateOutputV1, ...]
    failures: tuple[CandidateFailureV1, ...] = ()

    @model_validator(mode="after")
    def validate_artifact_identity(self) -> WordGroundTruthCaseV1:
        for output in self.outputs:
            artifact = output.artifact
            if artifact.analysis_id != self.reference.analysis_id:
                raise ValueError("word artifact analysis_id does not match its reference")
            if artifact.audio_sha256 != self.reference.audio_sha256:
                raise ValueError("word artifact audio_sha256 does not match its reference")
            if artifact.duration_ms != self.reference.duration_ms:
                raise ValueError("word artifact duration_ms does not match its reference")
        return self


class PhonemeGroundTruthCaseV1(_ReportContract):
    schema_version: Literal["PhonemeGroundTruthCaseV1"] = "PhonemeGroundTruthCaseV1"
    reference: PhonemeGroundTruthV1
    outputs: tuple[PhonemeCandidateOutputV1, ...]
    failures: tuple[CandidateFailureV1, ...] = ()

    @model_validator(mode="after")
    def validate_artifact_identity(self) -> PhonemeGroundTruthCaseV1:
        for output in self.outputs:
            artifact = output.artifact
            if artifact.analysis_id != self.reference.analysis_id:
                raise ValueError("phoneme artifact analysis_id does not match its reference")
            if artifact.audio_sha256 != self.reference.audio_sha256:
                raise ValueError("phoneme artifact audio_sha256 does not match its reference")
            if artifact.duration_ms != self.reference.duration_ms:
                raise ValueError("phoneme artifact duration_ms does not match its reference")
        return self


class WordAgreementCaseV1(_ReportContract):
    schema_version: Literal["WordAgreementCaseV1"] = "WordAgreementCaseV1"
    analysis_id: NonBlankStr
    audio_sha256: str = Field(pattern=SHA256_PATTERN, strict=True)
    duration_ms: PositiveMilliseconds
    reference_transcript: NonBlankStr | None = None
    transcript_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN, strict=True)
    outputs: tuple[WordCandidateOutputV1, ...]
    failures: tuple[CandidateFailureV1, ...] = ()

    @model_validator(mode="after")
    def validate_artifact_identity(self) -> WordAgreementCaseV1:
        if (self.reference_transcript is None) != (self.transcript_sha256 is None):
            raise ValueError("reference_transcript and transcript_sha256 must be set together")
        if self.reference_transcript is not None:
            expected_hash = hashlib.sha256(self.reference_transcript.encode("utf-8")).hexdigest()
            if expected_hash != self.transcript_sha256:
                raise ValueError("reference transcript does not match transcript_sha256")
        artifact_transcript_hashes = {output.artifact.transcript_sha256 for output in self.outputs}
        if len(artifact_transcript_hashes) > 1:
            raise ValueError("word outputs use different transcript revisions")
        for output in self.outputs:
            artifact = output.artifact
            if artifact.analysis_id != self.analysis_id:
                raise ValueError("word artifact analysis_id does not match its agreement case")
            if artifact.audio_sha256 != self.audio_sha256:
                raise ValueError("word artifact audio_sha256 does not match its agreement case")
            if artifact.duration_ms != self.duration_ms:
                raise ValueError("word artifact duration_ms does not match its agreement case")
            if self.transcript_sha256 is not None and (
                artifact.transcript_sha256 != self.transcript_sha256
            ):
                raise ValueError("word artifact transcript_sha256 does not match its reference")
            if output.freely_predicted_words and self.reference_transcript is None:
                raise ValueError("freely predicted word outputs require a reference transcript")
        return self


class PhonemeAgreementCaseV1(_ReportContract):
    schema_version: Literal["PhonemeAgreementCaseV1"] = "PhonemeAgreementCaseV1"
    analysis_id: NonBlankStr
    audio_sha256: str = Field(pattern=SHA256_PATTERN, strict=True)
    duration_ms: PositiveMilliseconds
    outputs: tuple[PhonemeCandidateOutputV1, ...]
    failures: tuple[CandidateFailureV1, ...] = ()

    @model_validator(mode="after")
    def validate_artifact_identity(self) -> PhonemeAgreementCaseV1:
        for output in self.outputs:
            artifact = output.artifact
            if artifact.analysis_id != self.analysis_id:
                raise ValueError("phoneme artifact analysis_id does not match its agreement case")
            if artifact.audio_sha256 != self.audio_sha256:
                raise ValueError("phoneme artifact audio_sha256 does not match its agreement case")
            if artifact.duration_ms != self.duration_ms:
                raise ValueError("phoneme artifact duration_ms does not match its agreement case")
        return self


type BenchmarkCaseV1 = Annotated[
    WordGroundTruthCaseV1 | PhonemeGroundTruthCaseV1 | WordAgreementCaseV1 | PhonemeAgreementCaseV1,
    Field(discriminator="schema_version"),
]


def _validate_candidate_output_provenance(
    output: WordCandidateOutputV1 | PhonemeCandidateOutputV1,
    candidate: CandidateSpecV1,
) -> None:
    if isinstance(output, WordCandidateOutputV1):
        word_processor = output.artifact.processor
        expected_revision = (
            candidate_processor_revision(candidate)
            if candidate_spec_is_registered(candidate)
            else candidate.model_revision
        )
        if (
            output.freely_predicted_words != candidate.freely_predicts_words
            or word_processor.aligner_name != candidate.model_name
            or word_processor.aligner_revision != expected_revision
            or word_processor.normalization_version != candidate.normalization_version
        ):
            raise ValueError(
                f"word artifact processor does not match candidate {candidate.candidate_id!r}"
            )
        return

    phoneme_processor = output.artifact.processor
    weight_hashes = {
        asset.sha256
        for asset in candidate.assets
        if asset.path.endswith(("pytorch_model.bin", "allophant.pt"))
    }
    if (
        phoneme_processor.model_name != candidate.model_name
        or phoneme_processor.model_revision
        != (
            candidate_processor_revision(candidate)
            if candidate_spec_is_registered(candidate)
            else candidate.model_revision
        )
        or phoneme_processor.weights_sha256 not in weight_hashes
        or phoneme_processor.phone_inventory != COVAL_ENGLISH_PHONES_V1
        or candidate.phone_inventory_version != PHONE_INVENTORY_VERSION
        or phoneme_processor.resampler != candidate.resampler
        or phoneme_processor.decoder != candidate.decoder
    ):
        raise ValueError(
            f"phoneme artifact processor does not match candidate {candidate.candidate_id!r}"
        )


class TimestampBenchmarkBundleV1(_ReportContract):
    """Private run input containing references and real per-clip artifacts."""

    schema_version: Literal["TimestampBenchmarkBundleV1"] = "TimestampBenchmarkBundleV1"
    benchmark_id: NonBlankStr
    dataset_id: NonBlankStr
    mode: BenchmarkMode
    kind: BenchmarkCandidateKind
    fixture_only: bool = False
    candidate_ids: tuple[NonBlankStr, ...]
    candidate_specs: tuple[CandidateSpecV1, ...]
    cases: tuple[BenchmarkCaseV1, ...]

    @model_validator(mode="after")
    def validate_case_matrix(self) -> TimestampBenchmarkBundleV1:
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be nonempty and unique")
        specs = {spec.candidate_id: spec for spec in self.candidate_specs}
        if len(specs) != len(self.candidate_specs) or set(specs) != set(self.candidate_ids):
            raise ValueError("candidate_specs must contain one spec for every candidate_id")
        if any(spec.kind is not self.kind for spec in self.candidate_specs):
            raise ValueError("candidate spec kind does not match bundle kind")
        if self.fixture_only:
            if (
                not self.benchmark_id.startswith("invented-")
                or not self.dataset_id.startswith("invented-")
                or any(
                    not candidate_id.startswith("invented-") for candidate_id in self.candidate_ids
                )
                or any(
                    spec.benchmark_eligible or spec.license_eligible_for_production
                    for spec in self.candidate_specs
                )
            ):
                raise ValueError("fixture_only bundles must contain ineligible invented identities")
        else:
            if any(not candidate_spec_is_registered(spec) for spec in self.candidate_specs):
                raise ValueError("candidate_specs must exactly match the public candidate registry")
        if not self.cases:
            raise ValueError("benchmark cases must not be empty")
        analysis_ids = tuple(
            case.reference.analysis_id
            if isinstance(case, WordGroundTruthCaseV1 | PhonemeGroundTruthCaseV1)
            else case.analysis_id
            for case in self.cases
        )
        if len(analysis_ids) != len(set(analysis_ids)):
            raise ValueError("benchmark case analysis_id values must be unique")
        expected_case_type = {
            (BenchmarkMode.HUMAN_GROUND_TRUTH, BenchmarkCandidateKind.WORD_ALIGNER): (
                WordGroundTruthCaseV1
            ),
            (BenchmarkMode.HUMAN_GROUND_TRUTH, BenchmarkCandidateKind.PHONEME_RECOGNIZER): (
                PhonemeGroundTruthCaseV1
            ),
            (BenchmarkMode.AGREEMENT, BenchmarkCandidateKind.WORD_ALIGNER): WordAgreementCaseV1,
            (BenchmarkMode.AGREEMENT, BenchmarkCandidateKind.PHONEME_RECOGNIZER): (
                PhonemeAgreementCaseV1
            ),
        }[(self.mode, self.kind)]
        declared = set(self.candidate_ids)
        for case in self.cases:
            if not isinstance(case, expected_case_type):
                raise ValueError("case type does not match benchmark mode and candidate kind")
            observed_ids = [output.candidate_id for output in case.outputs]
            observed_ids.extend(failure.candidate_id for failure in case.failures)
            if len(observed_ids) != len(set(observed_ids)):
                raise ValueError("each candidate must have exactly one outcome per case")
            if set(observed_ids) != declared:
                raise ValueError("each case must contain one outcome for every candidate")
            for output in case.outputs:
                _validate_candidate_output_provenance(output, specs[output.candidate_id])
        return self


class WordAccuracyObservationV1(_ReportContract):
    schema_version: Literal["WordAccuracyObservationV1"] = "WordAccuracyObservationV1"
    analysis_id: NonBlankStr
    candidate_id: NonBlankStr
    metrics: WordAccuracyMetricsV1


class PhonemeAccuracyObservationV1(_ReportContract):
    schema_version: Literal["PhonemeAccuracyObservationV1"] = "PhonemeAccuracyObservationV1"
    analysis_id: NonBlankStr
    candidate_id: NonBlankStr
    metrics: PhonemeAccuracyMetricsV1


class WordAgreementObservationV1(_ReportContract):
    schema_version: Literal["WordAgreementObservationV1"] = "WordAgreementObservationV1"
    analysis_id: NonBlankStr
    left_candidate_id: NonBlankStr
    right_candidate_id: NonBlankStr
    metrics: WordAgreementMetricsV1


class WordRecognitionObservationV1(_ReportContract):
    schema_version: Literal["WordRecognitionObservationV1"] = "WordRecognitionObservationV1"
    analysis_id: NonBlankStr
    candidate_id: NonBlankStr
    metrics: WordRecognitionMetricsV1


class PhonemeAgreementObservationV1(_ReportContract):
    schema_version: Literal["PhonemeAgreementObservationV1"] = "PhonemeAgreementObservationV1"
    analysis_id: NonBlankStr
    left_candidate_id: NonBlankStr
    right_candidate_id: NonBlankStr
    metrics: PhonemeAgreementMetricsV1


type BenchmarkObservationV1 = Annotated[
    WordAccuracyObservationV1
    | WordRecognitionObservationV1
    | PhonemeAccuracyObservationV1
    | WordAgreementObservationV1
    | PhonemeAgreementObservationV1,
    Field(discriminator="schema_version"),
]


class CandidateOutcomeSummaryV1(_ReportContract):
    candidate_id: NonBlankStr
    attempted_clips: NonNegativeInteger
    successful_clips: NonNegativeInteger
    empty_outputs: NonNegativeInteger
    partial_alignment_outputs: NonNegativeInteger
    failures: NonNegativeInteger
    invalid_artifacts: NonNegativeInteger
    timeouts: NonNegativeInteger
    artifact_success_rate: Ratio
    failure_rate: Ratio
    mean_timeline_coverage: Ratio | None
    mean_normalization_loss_rate: Ratio | None


class AggregateMetricV1(_ReportContract):
    name: NonBlankStr
    unit: NonBlankStr
    aggregation: Literal["macro_mean"] = "macro_mean"
    value: NonNegativeFiniteFloat
    contributing_clips: NonNegativeInteger


class CandidateMetricSummaryV1(_ReportContract):
    candidate_or_pair_id: NonBlankStr
    metrics: tuple[AggregateMetricV1, ...]


class PairOutcomeSummaryV1(_ReportContract):
    """Pairwise success/failure agreement for an agreement-only dataset."""

    left_candidate_id: NonBlankStr
    right_candidate_id: NonBlankStr
    attempted_clips: NonNegativeInteger
    both_succeeded: NonNegativeInteger
    only_left_succeeded: NonNegativeInteger
    only_right_succeeded: NonNegativeInteger
    both_failed: NonNegativeInteger
    failure_disagreement_rate: Ratio

    @model_validator(mode="after")
    def validate_counts(self) -> PairOutcomeSummaryV1:
        total = (
            self.both_succeeded
            + self.only_left_succeeded
            + self.only_right_succeeded
            + self.both_failed
        )
        if total != self.attempted_clips:
            raise ValueError("pair outcome counts must sum to attempted_clips")
        expected_disagreement = (
            (self.only_left_succeeded + self.only_right_succeeded) / self.attempted_clips
            if self.attempted_clips
            else 0.0
        )
        if self.failure_disagreement_rate != expected_disagreement:
            raise ValueError("failure_disagreement_rate does not match pair outcome counts")
        return self


class TimestampBenchmarkReportV1(_ReportContract):
    schema_version: Literal["TimestampBenchmarkReportV1"] = "TimestampBenchmarkReportV1"
    benchmark_id: NonBlankStr
    dataset_id: NonBlankStr
    mode: BenchmarkMode
    kind: BenchmarkCandidateKind
    candidate_specs: tuple[CandidateSpecV1, ...]
    outcomes: tuple[CandidateOutcomeSummaryV1, ...]
    pair_outcomes: tuple[PairOutcomeSummaryV1, ...]
    metric_summaries: tuple[CandidateMetricSummaryV1, ...]
    observations: tuple[BenchmarkObservationV1, ...]


def _analysis_id(case: BenchmarkCaseV1) -> str:
    if isinstance(case, WordGroundTruthCaseV1 | PhonemeGroundTruthCaseV1):
        return case.reference.analysis_id
    return case.analysis_id


def merge_benchmark_bundles(
    bundles: tuple[TimestampBenchmarkBundleV1, ...],
    *,
    benchmark_id: str,
) -> TimestampBenchmarkBundleV1:
    """Merge isolated candidate runs over an identical private case matrix."""
    if len(bundles) < 2:
        raise ValueError("at least two bundles are required for a merge")
    baseline = bundles[0]
    candidate_ids = tuple(
        candidate_id for bundle in bundles for candidate_id in bundle.candidate_ids
    )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ids must be disjoint across merged bundles")
    for bundle in bundles[1:]:
        if (bundle.dataset_id, bundle.mode, bundle.kind, bundle.fixture_only) != (
            baseline.dataset_id,
            baseline.mode,
            baseline.kind,
            baseline.fixture_only,
        ):
            raise ValueError(
                "merged bundles must have identical dataset, mode, kind, and fixture scope"
            )
        if len(bundle.cases) != len(baseline.cases):
            raise ValueError("merged bundles must have identical case counts")

    merged_cases: list[BenchmarkCaseV1] = []
    for case_index in range(len(baseline.cases)):
        case_group = tuple(bundle.cases[case_index] for bundle in bundles)
        baseline_case = case_group[0]
        if any(type(case) is not type(baseline_case) for case in case_group[1:]):
            raise ValueError("merged bundles must have identical case types and ordering")
        if any(_analysis_id(case) != _analysis_id(baseline_case) for case in case_group[1:]):
            raise ValueError("merged bundles must have identical case identities and ordering")
        if isinstance(baseline_case, WordAgreementCaseV1 | PhonemeAgreementCaseV1) and any(
            not isinstance(case, WordAgreementCaseV1 | PhonemeAgreementCaseV1)
            or case.audio_sha256 != baseline_case.audio_sha256
            or case.duration_ms != baseline_case.duration_ms
            for case in case_group[1:]
        ):
            raise ValueError("merged bundles must refer to identical agreement audio")
        if isinstance(baseline_case, WordAgreementCaseV1) and any(
            not isinstance(case, WordAgreementCaseV1)
            or case.reference_transcript != baseline_case.reference_transcript
            or case.transcript_sha256 != baseline_case.transcript_sha256
            or {output.artifact.transcript_sha256 for output in case.outputs}
            != {output.artifact.transcript_sha256 for output in baseline_case.outputs}
            for case in case_group[1:]
        ):
            raise ValueError("merged word bundles must use the identical transcript revision")

        if isinstance(baseline_case, WordGroundTruthCaseV1):
            word_cases = tuple(
                case for case in case_group if isinstance(case, WordGroundTruthCaseV1)
            )
            if any(case.reference != baseline_case.reference for case in word_cases[1:]):
                raise ValueError("merged bundles must have identical word references")
            merged_cases.append(
                WordGroundTruthCaseV1(
                    reference=baseline_case.reference,
                    outputs=tuple(output for case in word_cases for output in case.outputs),
                    failures=tuple(failure for case in word_cases for failure in case.failures),
                )
            )
        elif isinstance(baseline_case, PhonemeGroundTruthCaseV1):
            phone_cases = tuple(
                case for case in case_group if isinstance(case, PhonemeGroundTruthCaseV1)
            )
            if any(case.reference != baseline_case.reference for case in phone_cases[1:]):
                raise ValueError("merged bundles must have identical phoneme references")
            merged_cases.append(
                PhonemeGroundTruthCaseV1(
                    reference=baseline_case.reference,
                    outputs=tuple(output for case in phone_cases for output in case.outputs),
                    failures=tuple(failure for case in phone_cases for failure in case.failures),
                )
            )
        elif isinstance(baseline_case, WordAgreementCaseV1):
            word_agreement_cases = tuple(
                case for case in case_group if isinstance(case, WordAgreementCaseV1)
            )
            merged_cases.append(
                WordAgreementCaseV1(
                    analysis_id=baseline_case.analysis_id,
                    audio_sha256=baseline_case.audio_sha256,
                    duration_ms=baseline_case.duration_ms,
                    reference_transcript=baseline_case.reference_transcript,
                    transcript_sha256=baseline_case.transcript_sha256,
                    outputs=tuple(
                        output for case in word_agreement_cases for output in case.outputs
                    ),
                    failures=tuple(
                        failure for case in word_agreement_cases for failure in case.failures
                    ),
                )
            )
        else:
            phone_agreement_cases = tuple(
                case for case in case_group if isinstance(case, PhonemeAgreementCaseV1)
            )
            merged_cases.append(
                PhonemeAgreementCaseV1(
                    analysis_id=baseline_case.analysis_id,
                    audio_sha256=baseline_case.audio_sha256,
                    duration_ms=baseline_case.duration_ms,
                    outputs=tuple(
                        output for case in phone_agreement_cases for output in case.outputs
                    ),
                    failures=tuple(
                        failure for case in phone_agreement_cases for failure in case.failures
                    ),
                )
            )

    return TimestampBenchmarkBundleV1(
        benchmark_id=benchmark_id,
        dataset_id=baseline.dataset_id,
        mode=baseline.mode,
        kind=baseline.kind,
        fixture_only=baseline.fixture_only,
        candidate_ids=candidate_ids,
        candidate_specs=tuple(spec for bundle in bundles for spec in bundle.candidate_specs),
        cases=tuple(merged_cases),
    )


def select_candidate_subset(
    bundle: TimestampBenchmarkBundleV1,
    *,
    candidate_ids: tuple[str, ...],
    benchmark_id: str,
) -> TimestampBenchmarkBundleV1:
    """Create a validated comparison view without altering the source evidence."""
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ids must be nonempty and unique")
    if not set(candidate_ids).issubset(bundle.candidate_ids):
        raise ValueError("candidate subset contains an identity absent from the source bundle")
    selected = set(candidate_ids)
    cases = tuple(
        case.model_copy(
            update={
                "outputs": tuple(
                    output for output in case.outputs if output.candidate_id in selected
                ),
                "failures": tuple(
                    failure for failure in case.failures if failure.candidate_id in selected
                ),
            }
        )
        for case in bundle.cases
    )
    return TimestampBenchmarkBundleV1(
        benchmark_id=benchmark_id,
        dataset_id=bundle.dataset_id,
        mode=bundle.mode,
        kind=bundle.kind,
        fixture_only=bundle.fixture_only,
        candidate_ids=candidate_ids,
        candidate_specs=tuple(
            spec
            for candidate_id in candidate_ids
            for spec in bundle.candidate_specs
            if spec.candidate_id == candidate_id
        ),
        cases=cases,
    )


def _metric_values(
    metrics: WordAccuracyMetricsV1
    | WordRecognitionMetricsV1
    | PhonemeAccuracyMetricsV1
    | WordAgreementMetricsV1
    | PhonemeAgreementMetricsV1,
) -> tuple[tuple[str, str, float | None], ...]:
    if isinstance(metrics, WordAccuracyMetricsV1):
        return (
            ("matched_word_coverage", "ratio", metrics.matched_word_coverage),
            ("temporal_coverage", "ratio", metrics.temporal_coverage),
            ("mean_span_iou", "ratio", metrics.mean_span_iou),
            (
                "start_boundary_mean_absolute",
                "ms",
                metrics.start_boundary_error.mean_absolute_ms,
            ),
            ("end_boundary_mean_absolute", "ms", metrics.end_boundary_error.mean_absolute_ms),
            ("word_error_rate", "ratio", metrics.word_error_rate),
        )
    if isinstance(metrics, WordRecognitionMetricsV1):
        return (("word_error_rate", "ratio", metrics.word_error_rate),)
    if isinstance(metrics, PhonemeAccuracyMetricsV1):
        return (
            ("phone_error_rate", "ratio", metrics.phone_error_rate),
            ("exact_phone_coverage", "ratio", metrics.exact_phone_coverage),
            ("exact_temporal_coverage", "ratio", metrics.exact_temporal_coverage),
            ("normalization_loss_rate", "ratio", metrics.normalization_loss_rate),
            (
                "matched_start_boundary_mean_absolute",
                "ms",
                metrics.matched_start_boundary_error.mean_absolute_ms,
            ),
            (
                "matched_end_boundary_mean_absolute",
                "ms",
                metrics.matched_end_boundary_error.mean_absolute_ms,
            ),
        )
    if isinstance(metrics, WordAgreementMetricsV1):
        return (
            ("word_disagreement_rate", "ratio", metrics.word_disagreement_rate),
            ("paired_coverage", "ratio", metrics.paired_coverage),
            ("timeline_coverage_difference", "ratio", metrics.timeline_coverage_difference),
            (
                "matched_start_boundary_mean_absolute",
                "ms",
                metrics.matched_start_boundary_difference.mean_absolute_ms,
            ),
            (
                "matched_end_boundary_mean_absolute",
                "ms",
                metrics.matched_end_boundary_difference.mean_absolute_ms,
            ),
        )
    return (
        ("phone_disagreement_rate", "ratio", metrics.phone_disagreement_rate),
        ("exact_phone_agreement", "ratio", metrics.exact_phone_agreement),
        ("paired_coverage", "ratio", metrics.paired_coverage),
        ("timeline_coverage_difference", "ratio", metrics.timeline_coverage_difference),
        (
            "matched_start_boundary_mean_absolute",
            "ms",
            metrics.matched_start_boundary_difference.mean_absolute_ms,
        ),
        (
            "matched_end_boundary_mean_absolute",
            "ms",
            metrics.matched_end_boundary_difference.mean_absolute_ms,
        ),
    )


def _observation_key(observation: BenchmarkObservationV1) -> str:
    if isinstance(
        observation,
        WordAccuracyObservationV1 | WordRecognitionObservationV1 | PhonemeAccuracyObservationV1,
    ):
        return observation.candidate_id
    return f"{observation.left_candidate_id}__vs__{observation.right_candidate_id}"


def _summarize_metrics(
    observations: tuple[BenchmarkObservationV1, ...],
) -> tuple[CandidateMetricSummaryV1, ...]:
    values: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        key = _observation_key(observation)
        for name, unit, value in _metric_values(observation.metrics):
            if value is not None:
                values[key][(name, unit)].append(value)
    summaries: list[CandidateMetricSummaryV1] = []
    for key in sorted(values):
        metrics = tuple(
            AggregateMetricV1(
                name=name,
                unit=unit,
                value=float(fmean(metric_values)),
                contributing_clips=len(metric_values),
            )
            for (name, unit), metric_values in sorted(values[key].items())
        )
        summaries.append(CandidateMetricSummaryV1(candidate_or_pair_id=key, metrics=metrics))
    return tuple(summaries)


def _summarize_outcomes(
    bundle: TimestampBenchmarkBundleV1,
) -> tuple[CandidateOutcomeSummaryV1, ...]:
    summaries: list[CandidateOutcomeSummaryV1] = []
    for candidate_id in bundle.candidate_ids:
        successful = 0
        empty = 0
        partial = 0
        coverages: list[float] = []
        normalization_loss_rates: list[float] = []
        counts = {kind: 0 for kind in FailureKind}
        for case in bundle.cases:
            output = next(
                (item for item in case.outputs if item.candidate_id == candidate_id),
                None,
            )
            if output is not None:
                successful += 1
                spans = (
                    output.artifact.words
                    if isinstance(output, WordCandidateOutputV1)
                    else output.artifact.phones
                )
                empty += not spans
                partial += any(
                    warning.code is TimestampWarningCode.PARTIAL_ALIGNMENT
                    for warning in output.artifact.warnings
                )
                coverages.append(
                    sum(span.end_ms - span.start_ms for span in spans) / output.artifact.duration_ms
                )
                if isinstance(output, PhonemeCandidateOutputV1):
                    normalization_loss_rates.append(output.normalization_loss_rate)
                continue
            failure = next(item for item in case.failures if item.candidate_id == candidate_id)
            counts[failure.kind] += 1
        attempted = len(bundle.cases)
        summaries.append(
            CandidateOutcomeSummaryV1(
                candidate_id=candidate_id,
                attempted_clips=attempted,
                successful_clips=successful,
                empty_outputs=empty,
                partial_alignment_outputs=partial,
                failures=counts[FailureKind.FAILURE],
                invalid_artifacts=counts[FailureKind.INVALID_ARTIFACT],
                timeouts=counts[FailureKind.TIMEOUT],
                artifact_success_rate=successful / attempted,
                failure_rate=(attempted - successful) / attempted,
                mean_timeline_coverage=(float(fmean(coverages)) if coverages else None),
                mean_normalization_loss_rate=(
                    float(fmean(normalization_loss_rates)) if normalization_loss_rates else None
                ),
            )
        )
    return tuple(summaries)


def _summarize_pair_outcomes(
    bundle: TimestampBenchmarkBundleV1,
) -> tuple[PairOutcomeSummaryV1, ...]:
    if bundle.mode is not BenchmarkMode.AGREEMENT:
        return ()
    summaries: list[PairOutcomeSummaryV1] = []
    attempted = len(bundle.cases)
    for left_candidate_id, right_candidate_id in combinations(bundle.candidate_ids, 2):
        both_succeeded = 0
        only_left_succeeded = 0
        only_right_succeeded = 0
        both_failed = 0
        for case in bundle.cases:
            successful_ids = {output.candidate_id for output in case.outputs}
            left_succeeded = left_candidate_id in successful_ids
            right_succeeded = right_candidate_id in successful_ids
            if left_succeeded and right_succeeded:
                both_succeeded += 1
            elif left_succeeded:
                only_left_succeeded += 1
            elif right_succeeded:
                only_right_succeeded += 1
            else:
                both_failed += 1
        summaries.append(
            PairOutcomeSummaryV1(
                left_candidate_id=left_candidate_id,
                right_candidate_id=right_candidate_id,
                attempted_clips=attempted,
                both_succeeded=both_succeeded,
                only_left_succeeded=only_left_succeeded,
                only_right_succeeded=only_right_succeeded,
                both_failed=both_failed,
                failure_disagreement_rate=(only_left_succeeded + only_right_succeeded) / attempted,
            )
        )
    return tuple(summaries)


def build_report(bundle: TimestampBenchmarkBundleV1) -> TimestampBenchmarkReportV1:
    """Compute accuracy only for human references and agreement otherwise."""
    observations: list[BenchmarkObservationV1] = []
    for case in bundle.cases:
        analysis_id = _analysis_id(case)
        if isinstance(case, WordGroundTruthCaseV1):
            observations.extend(
                WordAccuracyObservationV1(
                    analysis_id=analysis_id,
                    candidate_id=output.candidate_id,
                    metrics=evaluate_word_accuracy(
                        case.reference,
                        output.artifact,
                        freely_predicted_words=output.freely_predicted_words,
                    ),
                )
                for output in case.outputs
            )
        elif isinstance(case, PhonemeGroundTruthCaseV1):
            observations.extend(
                PhonemeAccuracyObservationV1(
                    analysis_id=analysis_id,
                    candidate_id=output.candidate_id,
                    metrics=evaluate_phoneme_accuracy(
                        case.reference,
                        output.artifact,
                        normalization_loss_rate=output.normalization_loss_rate,
                    ),
                )
                for output in case.outputs
            )
        elif isinstance(case, WordAgreementCaseV1):
            word_outputs = {output.candidate_id: output for output in case.outputs}
            if case.reference_transcript is not None:
                observations.extend(
                    WordRecognitionObservationV1(
                        analysis_id=analysis_id,
                        candidate_id=output.candidate_id,
                        metrics=evaluate_word_recognition(
                            case.reference_transcript,
                            output.artifact,
                        ),
                    )
                    for output in case.outputs
                    if output.freely_predicted_words
                )
            for left_id, right_id in combinations(bundle.candidate_ids, 2):
                word_left = word_outputs.get(left_id)
                word_right = word_outputs.get(right_id)
                if word_left is None or word_right is None:
                    continue
                observations.append(
                    WordAgreementObservationV1(
                        analysis_id=analysis_id,
                        left_candidate_id=word_left.candidate_id,
                        right_candidate_id=word_right.candidate_id,
                        metrics=evaluate_word_agreement(
                            word_left.artifact,
                            word_right.artifact,
                        ),
                    )
                )
        else:
            phone_outputs = {output.candidate_id: output for output in case.outputs}
            for left_id, right_id in combinations(bundle.candidate_ids, 2):
                phone_left = phone_outputs.get(left_id)
                phone_right = phone_outputs.get(right_id)
                if phone_left is None or phone_right is None:
                    continue
                observations.append(
                    PhonemeAgreementObservationV1(
                        analysis_id=analysis_id,
                        left_candidate_id=phone_left.candidate_id,
                        right_candidate_id=phone_right.candidate_id,
                        metrics=evaluate_phoneme_agreement(
                            phone_left.artifact,
                            phone_right.artifact,
                        ),
                    )
                )
    observation_tuple = tuple(observations)
    return TimestampBenchmarkReportV1(
        benchmark_id=bundle.benchmark_id,
        dataset_id=bundle.dataset_id,
        mode=bundle.mode,
        kind=bundle.kind,
        candidate_specs=bundle.candidate_specs,
        outcomes=_summarize_outcomes(bundle),
        pair_outcomes=_summarize_pair_outcomes(bundle),
        metric_summaries=_summarize_metrics(observation_tuple),
        observations=observation_tuple,
    )
