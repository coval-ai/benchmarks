# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry of preprocessing artifact contracts emitted by the runner."""

from __future__ import annotations

SUPPORTED_PREPROCESSING_ARTIFACT_CONTRACTS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("word_timestamps", "WordTimestampsV1", "v1"),
        ("phoneme_timestamps", "PhonemeTimestampsV1", "v1"),
    }
)


def validate_preprocessing_artifact_contract(
    artifact_name: str, schema_name: str, schema_version: str
) -> None:
    """Reject preprocessing contracts that this runner does not know how to emit."""
    contract = (artifact_name, schema_name, schema_version)
    if contract not in SUPPORTED_PREPROCESSING_ARTIFACT_CONTRACTS:
        raise ValueError(
            "unknown preprocessing artifact contract "
            f"{artifact_name!r}/{schema_name!r}/{schema_version!r}"
        )
