# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical bytes and immutable identities for timestamp artifacts."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel

from coval_bench.preprocessing.contracts import (
    PhonemeProcessorProvenanceV1,
    PhonemeTimestampsV1,
    WordProcessorProvenanceV1,
    WordTimestampsV1,
)

TimestampArtifact = WordTimestampsV1 | PhonemeTimestampsV1
ProcessorProvenance = WordProcessorProvenanceV1 | PhonemeProcessorProvenanceV1


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value to deterministic UTF-8 bytes."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be JSON-compatible and contain no NaN values") from exc


def canonical_json_sha256(value: object) -> str:
    """Hash the exact bytes returned by :func:`canonical_json_bytes`."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def generator_fingerprint(processor: ProcessorProvenance) -> str:
    """Hash one validated, fully typed processor identity."""
    return canonical_json_sha256(processor)


def artifact_content_fingerprint(artifact: TimestampArtifact) -> str:
    """Return the SHA-256 of the artifact's canonical serialized bytes."""
    return canonical_json_sha256(artifact)


def immutable_artifact_key(artifact: TimestampArtifact) -> str:
    """Return a content-addressed key whose final digest hashes canonical bytes."""
    processor_fingerprint = generator_fingerprint(artifact.processor)
    return (
        f"timestamp-artifacts/{artifact.schema_version}/"
        f"{artifact.audio_sha256}/{processor_fingerprint}/"
        f"{artifact_content_fingerprint(artifact)}.json"
    )
