# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-independent timestamp benchmark contracts and metrics."""

from coval_bench.preprocessing.benchmarking.alignment import (
    AlignmentOperation,
    AlignmentStep,
    align_sequences,
)
from coval_bench.preprocessing.benchmarking.candidates import (
    CANDIDATES,
    DEEPGRAM_NOVA_3_CANDIDATE_ID,
    PRIMARY_PHONEME_CANDIDATE_IDS,
    PRIMARY_WORD_CANDIDATE_IDS,
    deepgram_nova_3_candidate,
)
from coval_bench.preprocessing.benchmarking.contracts import (
    BenchmarkCandidateKind,
    BenchmarkMode,
    CandidateSpecV1,
    ModelAssetV1,
    OperationalMeasurementV1,
    PhonemeGroundTruthV1,
    ReferencePhonemeV1,
    ReferenceWordV1,
    WordGroundTruthV1,
)
from coval_bench.preprocessing.benchmarking.inventory import (
    COVAL_ENGLISH_PHONES_V1,
    PHONE_INVENTORY_VERSION,
    PhoneNormalizationResult,
    PhoneSource,
    normalize_phone_sequence,
)
from coval_bench.preprocessing.benchmarking.metrics import (
    BoundaryErrorSummaryV1,
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

__all__ = [
    "CANDIDATES",
    "COVAL_ENGLISH_PHONES_V1",
    "DEEPGRAM_NOVA_3_CANDIDATE_ID",
    "PHONE_INVENTORY_VERSION",
    "PRIMARY_PHONEME_CANDIDATE_IDS",
    "PRIMARY_WORD_CANDIDATE_IDS",
    "AlignmentOperation",
    "AlignmentStep",
    "BenchmarkCandidateKind",
    "BenchmarkMode",
    "BoundaryErrorSummaryV1",
    "CandidateSpecV1",
    "ModelAssetV1",
    "OperationalMeasurementV1",
    "PhoneNormalizationResult",
    "PhoneSource",
    "PhonemeAccuracyMetricsV1",
    "PhonemeAgreementMetricsV1",
    "PhonemeGroundTruthV1",
    "ReferencePhonemeV1",
    "ReferenceWordV1",
    "WordAccuracyMetricsV1",
    "WordAgreementMetricsV1",
    "WordRecognitionMetricsV1",
    "WordGroundTruthV1",
    "align_sequences",
    "evaluate_phoneme_accuracy",
    "evaluate_phoneme_agreement",
    "evaluate_word_accuracy",
    "evaluate_word_agreement",
    "evaluate_word_recognition",
    "deepgram_nova_3_candidate",
    "normalize_phone_sequence",
]
