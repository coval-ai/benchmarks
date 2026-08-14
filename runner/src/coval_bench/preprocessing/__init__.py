# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only timestamp preprocessing contracts and deterministic utilities."""

from coval_bench.preprocessing.artifacts import (
    artifact_content_fingerprint,
    canonical_json_bytes,
    canonical_json_sha256,
    generator_fingerprint,
    immutable_artifact_key,
)
from coval_bench.preprocessing.batching import (
    DEFAULT_MAX_BATCH_CLIPS,
    DEFAULT_MAX_BATCH_DURATION_MS,
    batch_by_duration,
)
from coval_bench.preprocessing.contracts import (
    PhonemeProcessorProvenanceV1,
    PhonemeTimestampsV1,
    PhonemeTimestampV1,
    TimestampArtifactV1,
    TimestampWarningCode,
    TimestampWarningV1,
    WordProcessorProvenanceV1,
    WordTimestampsV1,
    WordTimestampV1,
)

__all__ = [
    "DEFAULT_MAX_BATCH_CLIPS",
    "DEFAULT_MAX_BATCH_DURATION_MS",
    "PhonemeProcessorProvenanceV1",
    "PhonemeTimestampV1",
    "PhonemeTimestampsV1",
    "TimestampArtifactV1",
    "TimestampWarningCode",
    "TimestampWarningV1",
    "WordProcessorProvenanceV1",
    "WordTimestampV1",
    "WordTimestampsV1",
    "artifact_content_fingerprint",
    "batch_by_duration",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "generator_fingerprint",
    "immutable_artifact_key",
]
