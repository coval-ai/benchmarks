# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate interfaces enforce the transcript boundary between benchmark tasks."""

from pathlib import Path
from typing import Protocol

from coval_bench.preprocessing.benchmarking.contracts import CandidateSpecV1
from coval_bench.preprocessing.contracts import PhonemeTimestampsV1, WordTimestampsV1


class HostedProviderError(RuntimeError):
    """Sanitized hosted failure with a stable, non-secret classification code."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def source_duration_ms(*, sample_count: int, sample_rate: int) -> int:
    """Round the original decoded source duration for strict artifact identity."""
    if sample_count <= 0 or sample_rate <= 0:
        raise ValueError("source audio must have a positive sample count and sample rate")
    return round(sample_count * 1_000 / sample_rate)


class WordAligner(Protocol):
    """Known-transcript word alignment is permitted."""

    @property
    def candidate(self) -> CandidateSpecV1: ...

    @property
    def model_load_seconds(self) -> float: ...

    @property
    def last_inference_seconds(self) -> float: ...

    @property
    def runtime_software(self) -> str: ...

    def align(self, *, audio_path: Path, transcript: str) -> WordTimestampsV1: ...


class PhonemeRecognizer(Protocol):
    """Observed phones must be inferred from audio without reference text."""

    @property
    def candidate(self) -> CandidateSpecV1: ...

    @property
    def model_load_seconds(self) -> float: ...

    @property
    def last_normalization_loss_rate(self) -> float: ...

    @property
    def runtime_software(self) -> str: ...

    def recognize(self, *, audio_path: Path) -> PhonemeTimestampsV1: ...
