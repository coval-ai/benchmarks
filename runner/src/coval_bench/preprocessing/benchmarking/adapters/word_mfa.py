# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Import pinned Montreal Forced Aligner JSON into ``WordTimestampsV1``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from coval_bench.preprocessing.benchmarking.candidates import (
    CANDIDATES,
    candidate_processor_revision,
)
from coval_bench.preprocessing.benchmarking.metrics import (
    normalize_word,
)
from coval_bench.preprocessing.contracts import (
    TimestampWarningCode,
    TimestampWarningV1,
    WordProcessorProvenanceV1,
    WordTimestampsV1,
    WordTimestampV1,
)

_MFA_CANDIDATE = next(
    candidate for candidate in CANDIDATES if candidate.candidate_id == "word-mfa-english-us-arpa-v1"
)
MFA_ALIGNER_REVISION = candidate_processor_revision(_MFA_CANDIDATE)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _entries(value: dict[str, Any], tier: str) -> tuple[tuple[float, float, str], ...]:
    try:
        raw_entries = value["tiers"][tier]["entries"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"MFA JSON is missing the {tier!r} interval tier") from exc
    entries: list[tuple[float, float, str]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, list) or len(entry) != 3:
            raise ValueError(f"MFA {tier} entry {index} is not a three-item interval")
        start, end, label = entry
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise ValueError(f"MFA {tier} entry {index} has nonnumeric boundaries")
        if not isinstance(label, str):
            raise ValueError(f"MFA {tier} entry {index} has a nonstring label")
        entries.append((float(start), float(end), label))
    return tuple(entries)


def parse_mfa_json_artifact(
    output_path: Path,
    *,
    analysis_id: str,
    audio_sha256: str,
    transcript: str,
) -> WordTimestampsV1:
    """Convert MFA 3.3 JSON while surfacing unknown-word spans as partial alignment."""
    try:
        value = json.loads(output_path.read_text(encoding="utf-8"))
        duration_ms = round(float(value["end"]) * 1_000)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read MFA JSON artifact: {exc}") from exc

    words = tuple(
        WordTimestampV1(
            text=normalized,
            start_ms=round(start * 1_000),
            end_ms=round(end * 1_000),
            # MFA JSON has no calibrated word confidence. Zero is an explicit sentinel.
            confidence=0.0,
        )
        for start, end, label in _entries(value, "words")
        if (normalized := normalize_word(label))
    )
    unknown_intervals = tuple(
        (round(start * 1_000), round(end * 1_000))
        for start, end, label in _entries(value, "phones")
        if label.casefold() == "spn"
    )
    warnings = tuple(
        TimestampWarningV1(
            code=TimestampWarningCode.PARTIAL_ALIGNMENT,
            message="MFA emitted the unknown-word phone 'spn' for this interval",
            start_ms=start_ms,
            end_ms=end_ms,
        )
        for start_ms, end_ms in unknown_intervals
        if end_ms > start_ms
    )
    return WordTimestampsV1(
        schema_version="WordTimestampsV1",
        analysis_id=analysis_id,
        audio_sha256=audio_sha256,
        transcript_sha256=_sha256_text(transcript),
        duration_ms=duration_ms,
        processor=WordProcessorProvenanceV1(
            aligner_name=_MFA_CANDIDATE.model_name,
            aligner_revision=MFA_ALIGNER_REVISION,
            normalization_version=_MFA_CANDIDATE.normalization_version,
        ),
        words=words,
        warnings=warnings,
    )
