# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hard evidence gates that must pass before a timestamp processor is selected."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from coval_bench.preprocessing.benchmarking.candidates import candidate_uses_hosted_model
from coval_bench.preprocessing.benchmarking.contracts import (
    BenchmarkCandidateKind,
    CandidateSpecV1,
    NonNegativeFiniteFloat,
    Ratio,
)
from coval_bench.preprocessing.contracts import NonBlankStr

NonNegativeInteger = Annotated[int, Field(ge=0, strict=True)]


class SelectionEvidenceV1(BaseModel):
    """Private aggregate evidence; it contains no audio, transcript, or clip detail."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["SelectionEvidenceV1"] = "SelectionEvidenceV1"
    candidate_id: NonBlankStr
    kind: BenchmarkCandidateKind
    human_ground_truth_clips: NonNegativeInteger
    manually_checked_in_domain_clips: NonNegativeInteger
    artifact_success_rate: Ratio
    projected_daily_runtime_minutes: NonNegativeFiniteFloat | None
    target_gpu: NonBlankStr | None
    target_gpu_benchmark_completed: bool | None
    weights_hashes_verified: bool | None
    hosted_reproducibility_policy_approved: bool | None = None
    provider_revision_drift_handling_verified: bool | None = None
    raw_response_hashes_recorded: bool | None = None
    deterministic_artifacts_verified: bool
    confidence_thresholds_calibrated: bool
    normalization_loss_rate: Ratio | None

    @model_validator(mode="after")
    def validate_kind_specific_fields(self) -> SelectionEvidenceV1:
        if self.kind is BenchmarkCandidateKind.PHONEME_RECOGNIZER:
            if self.normalization_loss_rate is None:
                raise ValueError("phoneme selection evidence requires normalization_loss_rate")
        elif self.normalization_loss_rate is not None:
            raise ValueError("word selection evidence must not include normalization_loss_rate")
        return self


def selection_blockers(
    candidate: CandidateSpecV1,
    evidence: SelectionEvidenceV1,
    *,
    minimum_human_ground_truth_clips: int = 1,
    minimum_in_domain_clips: int = 50,
    minimum_success_rate: float = 0.95,
    daily_deadline_minutes: float = 30,
) -> tuple[str, ...]:
    """Return concrete blockers without converting agreement into accuracy."""
    if candidate.candidate_id != evidence.candidate_id or candidate.kind is not evidence.kind:
        raise ValueError("candidate and selection evidence identities do not match")
    blockers: list[str] = []
    if not candidate.benchmark_eligible:
        blockers.append("candidate is not eligible for benchmark evaluation")
    if not candidate.license_eligible_for_production:
        blockers.append("candidate license/provenance is not eligible for production use")
    if evidence.human_ground_truth_clips < minimum_human_ground_truth_clips:
        blockers.append(f"human-ground-truth clips are below {minimum_human_ground_truth_clips}")
    if evidence.manually_checked_in_domain_clips < minimum_in_domain_clips:
        blockers.append(f"manually checked in-domain clips are below {minimum_in_domain_clips}")
    if candidate_uses_hosted_model(candidate):
        if not evidence.hosted_reproducibility_policy_approved:
            blockers.append("hosted-model reproducibility policy was not approved")
        if not evidence.provider_revision_drift_handling_verified:
            blockers.append("provider revision-drift handling was not verified")
        if not evidence.raw_response_hashes_recorded:
            blockers.append("private raw-response hashes were not recorded")
    else:
        if not evidence.target_gpu or not evidence.target_gpu_benchmark_completed:
            blockers.append("target-GPU benchmark was not completed")
        if not evidence.weights_hashes_verified:
            blockers.append("model asset hashes were not verified at runtime")
    if not evidence.deterministic_artifacts_verified:
        blockers.append("repeat runs did not verify deterministic artifact hashes")
    if not evidence.confidence_thresholds_calibrated:
        blockers.append("agreement/confidence thresholds were not calibrated against human labels")
    if evidence.artifact_success_rate < minimum_success_rate:
        blockers.append(f"artifact success rate is below {minimum_success_rate:.0%}")
    if evidence.projected_daily_runtime_minutes is None:
        blockers.append("target-GPU daily runtime projection is unavailable")
    elif evidence.projected_daily_runtime_minutes > daily_deadline_minutes:
        blockers.append(f"projected daily runtime exceeds {daily_deadline_minutes:g} minutes")
    return tuple(blockers)
