# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-independent timestamp accuracy and agreement metrics."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from statistics import fmean, median
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field

from coval_bench.preprocessing.benchmarking.alignment import (
    AlignmentOperation,
    AlignmentStep,
    align_sequences,
)
from coval_bench.preprocessing.benchmarking.contracts import (
    PhonemeGroundTruthV1,
    WordGroundTruthV1,
)
from coval_bench.preprocessing.contracts import (
    PhonemeTimestampsV1,
    WordTimestampsV1,
)

WORD_NORMALIZATION_VERSION = "word-nfkc-casefold-alnum-v1"

NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False, strict=True)]
Ratio = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
NonNegativeInteger = Annotated[int, Field(ge=0, strict=True)]


class _MetricsContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BoundaryErrorSummaryV1(_MetricsContract):
    count: NonNegativeInteger
    mean_signed_ms: FiniteFloat | None
    mean_absolute_ms: NonNegativeFiniteFloat | None
    median_absolute_ms: NonNegativeFiniteFloat | None
    p95_absolute_ms: NonNegativeFiniteFloat | None
    within_20_ms: Ratio | None
    within_50_ms: Ratio | None
    within_100_ms: Ratio | None


class WordAccuracyMetricsV1(_MetricsContract):
    schema_version: str = "WordAccuracyMetricsV1"
    normalization_version: str = WORD_NORMALIZATION_VERSION
    reference_words: NonNegativeInteger
    predicted_words: NonNegativeInteger
    matches: NonNegativeInteger
    substitutions: NonNegativeInteger
    insertions: NonNegativeInteger
    deletions: NonNegativeInteger
    matched_word_coverage: Ratio
    temporal_coverage: Ratio
    mean_span_iou: Ratio | None
    mean_reference_overlap: Ratio | None
    start_boundary_error: BoundaryErrorSummaryV1
    end_boundary_error: BoundaryErrorSummaryV1
    wer_applicable: bool
    word_error_rate: NonNegativeFiniteFloat | None
    empty_output: bool


class PhonemeAccuracyMetricsV1(_MetricsContract):
    schema_version: str = "PhonemeAccuracyMetricsV1"
    reference_phones: NonNegativeInteger
    predicted_phones: NonNegativeInteger
    matches: NonNegativeInteger
    substitutions: NonNegativeInteger
    insertions: NonNegativeInteger
    deletions: NonNegativeInteger
    phone_error_rate: NonNegativeFiniteFloat
    exact_phone_coverage: Ratio
    paired_reference_coverage: Ratio
    exact_temporal_coverage: Ratio
    reference_timeline_coverage: Ratio
    predicted_timeline_coverage: Ratio
    normalization_loss_rate: Ratio
    matched_start_boundary_error: BoundaryErrorSummaryV1
    matched_end_boundary_error: BoundaryErrorSummaryV1
    substitution_start_boundary_error: BoundaryErrorSummaryV1
    substitution_end_boundary_error: BoundaryErrorSummaryV1
    empty_output: bool


class WordAgreementMetricsV1(_MetricsContract):
    schema_version: str = "WordAgreementMetricsV1"
    left_words: NonNegativeInteger
    right_words: NonNegativeInteger
    matches: NonNegativeInteger
    substitutions: NonNegativeInteger
    insertions: NonNegativeInteger
    deletions: NonNegativeInteger
    word_disagreement_rate: NonNegativeFiniteFloat
    paired_coverage: Ratio
    left_timeline_coverage: Ratio
    right_timeline_coverage: Ratio
    timeline_coverage_difference: Ratio
    matched_start_boundary_difference: BoundaryErrorSummaryV1
    matched_end_boundary_difference: BoundaryErrorSummaryV1


class WordRecognitionMetricsV1(_MetricsContract):
    """Transcript accuracy for a freely predicting model on an agreement dataset."""

    schema_version: str = "WordRecognitionMetricsV1"
    normalization_version: str = WORD_NORMALIZATION_VERSION
    reference_words: NonNegativeInteger
    predicted_words: NonNegativeInteger
    matches: NonNegativeInteger
    substitutions: NonNegativeInteger
    insertions: NonNegativeInteger
    deletions: NonNegativeInteger
    word_error_rate: NonNegativeFiniteFloat
    empty_output: bool


class PhonemeAgreementMetricsV1(_MetricsContract):
    schema_version: str = "PhonemeAgreementMetricsV1"
    left_phones: NonNegativeInteger
    right_phones: NonNegativeInteger
    matches: NonNegativeInteger
    substitutions: NonNegativeInteger
    insertions: NonNegativeInteger
    deletions: NonNegativeInteger
    phone_disagreement_rate: NonNegativeFiniteFloat
    exact_phone_agreement: Ratio
    paired_coverage: Ratio
    left_timeline_coverage: Ratio
    right_timeline_coverage: Ratio
    timeline_coverage_difference: Ratio
    matched_start_boundary_difference: BoundaryErrorSummaryV1
    matched_end_boundary_difference: BoundaryErrorSummaryV1


class _Span(Protocol):
    @property
    def start_ms(self) -> int: ...

    @property
    def end_ms(self) -> int: ...


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _boundary_summary(errors_ms: Sequence[int]) -> BoundaryErrorSummaryV1:
    if not errors_ms:
        return BoundaryErrorSummaryV1(
            count=0,
            mean_signed_ms=None,
            mean_absolute_ms=None,
            median_absolute_ms=None,
            p95_absolute_ms=None,
            within_20_ms=None,
            within_50_ms=None,
            within_100_ms=None,
        )
    absolute = [abs(error) for error in errors_ms]
    count = len(errors_ms)
    return BoundaryErrorSummaryV1(
        count=count,
        mean_signed_ms=float(fmean(errors_ms)),
        mean_absolute_ms=float(fmean(absolute)),
        median_absolute_ms=float(median(absolute)),
        p95_absolute_ms=_percentile(absolute, 0.95),
        within_20_ms=sum(error <= 20 for error in absolute) / count,
        within_50_ms=sum(error <= 50 for error in absolute) / count,
        within_100_ms=sum(error <= 100 for error in absolute) / count,
    )


def normalize_word(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return "".join(character for character in normalized if character.isalnum() or character == "'")


def _operation_counts[Item](
    steps: tuple[AlignmentStep[Item], ...],
) -> tuple[int, int, int, int]:
    return (
        sum(step.operation is AlignmentOperation.MATCH for step in steps),
        sum(step.operation is AlignmentOperation.SUBSTITUTION for step in steps),
        sum(step.operation is AlignmentOperation.INSERTION for step in steps),
        sum(step.operation is AlignmentOperation.DELETION for step in steps),
    )


def _paired_spans[ReferenceSpan: _Span, HypothesisSpan: _Span](
    steps: tuple[AlignmentStep[str], ...],
    reference_spans: tuple[ReferenceSpan, ...],
    hypothesis_spans: tuple[HypothesisSpan, ...],
    *,
    operation: AlignmentOperation,
) -> tuple[tuple[ReferenceSpan, HypothesisSpan], ...]:
    pairs: list[tuple[ReferenceSpan, HypothesisSpan]] = []
    for step in steps:
        if step.operation is not operation:
            continue
        if step.reference_index is None or step.hypothesis_index is None:
            raise RuntimeError("paired alignment operation is missing an index")
        pairs.append(
            (reference_spans[step.reference_index], hypothesis_spans[step.hypothesis_index])
        )
    return tuple(pairs)


def _intersection_ms(left: _Span, right: _Span) -> int:
    return max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))


def _span_iou(left: _Span, right: _Span) -> float:
    intersection = _intersection_ms(left, right)
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    return intersection / union if union else 0.0


def _reference_overlap(reference: _Span, hypothesis: _Span) -> float:
    return _intersection_ms(reference, hypothesis) / (reference.end_ms - reference.start_ms)


def _timeline_coverage(spans: Sequence[_Span], duration_ms: int) -> float:
    return sum(span.end_ms - span.start_ms for span in spans) / duration_ms


def _validate_matching_artifact(
    *, reference_analysis_id: str, reference_audio_sha256: str, artifact: object
) -> None:
    artifact_analysis_id = getattr(artifact, "analysis_id", None)
    artifact_audio_sha256 = getattr(artifact, "audio_sha256", None)
    if artifact_analysis_id != reference_analysis_id:
        raise ValueError("artifact analysis_id does not match ground truth")
    if artifact_audio_sha256 != reference_audio_sha256:
        raise ValueError("artifact audio_sha256 does not match ground truth")


def evaluate_word_accuracy(
    reference: WordGroundTruthV1,
    prediction: WordTimestampsV1,
    *,
    freely_predicted_words: bool,
) -> WordAccuracyMetricsV1:
    """Compare word spans to human boundaries without misreporting forced-aligner WER."""
    _validate_matching_artifact(
        reference_analysis_id=reference.analysis_id,
        reference_audio_sha256=reference.audio_sha256,
        artifact=prediction,
    )
    reference_tokens = tuple(normalize_word(word.text) for word in reference.words)
    predicted_tokens = tuple(normalize_word(word.text) for word in prediction.words)
    steps = align_sequences(reference_tokens, predicted_tokens)
    matches, substitutions, insertions, deletions = _operation_counts(steps)
    pairs = _paired_spans(
        steps,
        reference.words,
        prediction.words,
        operation=AlignmentOperation.MATCH,
    )
    start_errors = [predicted.start_ms - expected.start_ms for expected, predicted in pairs]
    end_errors = [predicted.end_ms - expected.end_ms for expected, predicted in pairs]
    reference_duration = sum(word.end_ms - word.start_ms for word in reference.words)
    matched_intersection = sum(
        _intersection_ms(expected, predicted) for expected, predicted in pairs
    )
    denominator = len(reference.words)
    edit_count = substitutions + insertions + deletions
    return WordAccuracyMetricsV1(
        reference_words=denominator,
        predicted_words=len(prediction.words),
        matches=matches,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        matched_word_coverage=matches / denominator if denominator else 1.0,
        temporal_coverage=matched_intersection / reference_duration if reference_duration else 1.0,
        mean_span_iou=float(fmean(_span_iou(*pair) for pair in pairs)) if pairs else None,
        mean_reference_overlap=(
            float(fmean(_reference_overlap(*pair) for pair in pairs)) if pairs else None
        ),
        start_boundary_error=_boundary_summary(start_errors),
        end_boundary_error=_boundary_summary(end_errors),
        wer_applicable=freely_predicted_words,
        word_error_rate=(
            edit_count / denominator if freely_predicted_words and denominator else None
        ),
        empty_output=not prediction.words,
    )


def evaluate_word_recognition(
    reference_transcript: str,
    prediction: WordTimestampsV1,
) -> WordRecognitionMetricsV1:
    """Measure WER when a hosted model predicted words without transcript conditioning."""
    reference_tokens = tuple(
        token for raw_token in reference_transcript.split() if (token := normalize_word(raw_token))
    )
    predicted_tokens = tuple(
        token for word in prediction.words if (token := normalize_word(word.text))
    )
    steps = align_sequences(reference_tokens, predicted_tokens)
    matches, substitutions, insertions, deletions = _operation_counts(steps)
    edit_count = substitutions + insertions + deletions
    denominator = len(reference_tokens)
    return WordRecognitionMetricsV1(
        reference_words=denominator,
        predicted_words=len(predicted_tokens),
        matches=matches,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        word_error_rate=edit_count / denominator if denominator else float(bool(edit_count)),
        empty_output=not predicted_tokens,
    )


def evaluate_phoneme_accuracy(
    reference: PhonemeGroundTruthV1,
    prediction: PhonemeTimestampsV1,
    *,
    normalization_loss_rate: float,
) -> PhonemeAccuracyMetricsV1:
    """Score independently recognized phones against observed human phones."""
    if not 0 <= normalization_loss_rate <= 1:
        raise ValueError("normalization_loss_rate must be between zero and one")
    _validate_matching_artifact(
        reference_analysis_id=reference.analysis_id,
        reference_audio_sha256=reference.audio_sha256,
        artifact=prediction,
    )
    reference_symbols = tuple(phone.symbol for phone in reference.phones)
    predicted_symbols = tuple(phone.symbol for phone in prediction.phones)
    steps = align_sequences(reference_symbols, predicted_symbols)
    matches, substitutions, insertions, deletions = _operation_counts(steps)
    matched_pairs = _paired_spans(
        steps,
        reference.phones,
        prediction.phones,
        operation=AlignmentOperation.MATCH,
    )
    substitution_pairs = _paired_spans(
        steps,
        reference.phones,
        prediction.phones,
        operation=AlignmentOperation.SUBSTITUTION,
    )
    matched_start_errors = [
        actual.start_ms - expected.start_ms for expected, actual in matched_pairs
    ]
    matched_end_errors = [actual.end_ms - expected.end_ms for expected, actual in matched_pairs]
    substitution_start_errors = [
        actual.start_ms - expected.start_ms for expected, actual in substitution_pairs
    ]
    substitution_end_errors = [
        actual.end_ms - expected.end_ms for expected, actual in substitution_pairs
    ]
    reference_count = len(reference.phones)
    reference_duration = sum(phone.end_ms - phone.start_ms for phone in reference.phones)
    exact_intersection = sum(
        _intersection_ms(expected, actual) for expected, actual in matched_pairs
    )
    edit_count = substitutions + insertions + deletions
    return PhonemeAccuracyMetricsV1(
        reference_phones=reference_count,
        predicted_phones=len(prediction.phones),
        matches=matches,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        phone_error_rate=(
            edit_count / reference_count if reference_count else float(bool(edit_count))
        ),
        exact_phone_coverage=matches / reference_count if reference_count else 1.0,
        paired_reference_coverage=(
            (matches + substitutions) / reference_count if reference_count else 1.0
        ),
        exact_temporal_coverage=(
            exact_intersection / reference_duration if reference_duration else 1.0
        ),
        reference_timeline_coverage=_timeline_coverage(reference.phones, reference.duration_ms),
        predicted_timeline_coverage=_timeline_coverage(prediction.phones, prediction.duration_ms),
        normalization_loss_rate=normalization_loss_rate,
        matched_start_boundary_error=_boundary_summary(matched_start_errors),
        matched_end_boundary_error=_boundary_summary(matched_end_errors),
        substitution_start_boundary_error=_boundary_summary(substitution_start_errors),
        substitution_end_boundary_error=_boundary_summary(substitution_end_errors),
        empty_output=not prediction.phones,
    )


def evaluate_word_agreement(
    left: WordTimestampsV1, right: WordTimestampsV1
) -> WordAgreementMetricsV1:
    """Compare two word processors symmetrically without calling agreement accuracy."""
    if left.analysis_id != right.analysis_id or left.audio_sha256 != right.audio_sha256:
        raise ValueError("word artifacts must describe the same audio analysis")
    left_tokens = tuple(normalize_word(word.text) for word in left.words)
    right_tokens = tuple(normalize_word(word.text) for word in right.words)
    steps = align_sequences(left_tokens, right_tokens)
    matches, substitutions, insertions, deletions = _operation_counts(steps)
    pairs = _paired_spans(steps, left.words, right.words, operation=AlignmentOperation.MATCH)
    denominator = max(len(left.words), len(right.words))
    left_coverage = _timeline_coverage(left.words, left.duration_ms)
    right_coverage = _timeline_coverage(right.words, right.duration_ms)
    return WordAgreementMetricsV1(
        left_words=len(left.words),
        right_words=len(right.words),
        matches=matches,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        word_disagreement_rate=(
            (substitutions + insertions + deletions) / denominator if denominator else 0.0
        ),
        paired_coverage=matches / denominator if denominator else 1.0,
        left_timeline_coverage=left_coverage,
        right_timeline_coverage=right_coverage,
        timeline_coverage_difference=abs(left_coverage - right_coverage),
        matched_start_boundary_difference=_boundary_summary(
            [right_span.start_ms - left_span.start_ms for left_span, right_span in pairs]
        ),
        matched_end_boundary_difference=_boundary_summary(
            [right_span.end_ms - left_span.end_ms for left_span, right_span in pairs]
        ),
    )


def evaluate_phoneme_agreement(
    left: PhonemeTimestampsV1, right: PhonemeTimestampsV1
) -> PhonemeAgreementMetricsV1:
    """Compare two independent phone recognizers without implying ground truth."""
    if left.analysis_id != right.analysis_id or left.audio_sha256 != right.audio_sha256:
        raise ValueError("phoneme artifacts must describe the same audio analysis")
    left_symbols = tuple(phone.symbol for phone in left.phones)
    right_symbols = tuple(phone.symbol for phone in right.phones)
    steps = align_sequences(left_symbols, right_symbols)
    matches, substitutions, insertions, deletions = _operation_counts(steps)
    pairs = _paired_spans(steps, left.phones, right.phones, operation=AlignmentOperation.MATCH)
    denominator = max(len(left.phones), len(right.phones))
    paired = matches + substitutions
    left_coverage = _timeline_coverage(left.phones, left.duration_ms)
    right_coverage = _timeline_coverage(right.phones, right.duration_ms)
    return PhonemeAgreementMetricsV1(
        left_phones=len(left.phones),
        right_phones=len(right.phones),
        matches=matches,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        phone_disagreement_rate=(
            (substitutions + insertions + deletions) / denominator if denominator else 0.0
        ),
        exact_phone_agreement=matches / denominator if denominator else 1.0,
        paired_coverage=paired / denominator if denominator else 1.0,
        left_timeline_coverage=left_coverage,
        right_timeline_coverage=right_coverage,
        timeline_coverage_difference=abs(left_coverage - right_coverage),
        matched_start_boundary_difference=_boundary_summary(
            [right_span.start_ms - left_span.start_ms for left_span, right_span in pairs]
        ),
        matched_end_boundary_difference=_boundary_summary(
            [right_span.end_ms - left_span.end_ms for left_span, right_span in pairs]
        ),
    )
