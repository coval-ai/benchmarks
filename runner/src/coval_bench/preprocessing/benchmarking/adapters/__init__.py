# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Interfaces for timestamp candidates; heavy implementations stay optional."""

from pydantic import SecretStr

from coval_bench.preprocessing.benchmarking.adapters.base import (
    PhonemeRecognizer,
    WordAligner,
)
from coval_bench.preprocessing.benchmarking.adapters.phoneme_allophant import (
    ALLOPHANT_ALLOPHOIBLE_ENGLISH_INVENTORY,
    ALLOPHANT_ENGLISH_INVENTORY,
    AllophantPhonemeRecognizer,
)
from coval_bench.preprocessing.benchmarking.adapters.phoneme_hf import (
    DecodedPhoneSequence,
    HuggingFacePhonemeRecognizer,
    RawPhoneSpan,
    build_phoneme_artifact,
    decode_phone_frames,
    normalize_phone_spans,
)
from coval_bench.preprocessing.benchmarking.adapters.word_ctc import (
    CTCAlignmentError,
    CTCTokenSpan,
    HuggingFaceCTCWordAligner,
    ctc_transcript_token_ids,
    force_align_ctc,
    normalize_ctc_transcript,
    token_spans_to_words,
)
from coval_bench.preprocessing.benchmarking.adapters.word_deepgram import (
    DeepgramNova3WordRecognizer,
    deepgram_model_version,
    discover_deepgram_nova_3_version,
    parse_deepgram_nova_3_response,
)
from coval_bench.preprocessing.benchmarking.adapters.word_google import (
    GOOGLE_CHIRP_3_CANDIDATE_ID,
    GoogleChirp3WordRecognizer,
    google_recognize_retry,
    parse_google_chirp_3_response,
)
from coval_bench.preprocessing.benchmarking.adapters.word_mfa import (
    MFA_ALIGNER_REVISION,
    parse_mfa_json_artifact,
)
from coval_bench.preprocessing.benchmarking.candidates import DEEPGRAM_NOVA_3_CANDIDATE_ID
from coval_bench.preprocessing.benchmarking.contracts import CandidateSpecV1


def create_phoneme_recognizer(
    *,
    candidate_id: str,
    device: str = "cpu",
    local_files_only: bool = False,
) -> PhonemeRecognizer:
    """Construct a lazy candidate adapter without importing optional runtimes."""
    if candidate_id == "phone-allophant-en-multitask-v1":
        return AllophantPhonemeRecognizer(
            candidate_id=candidate_id,
            device=device,
            local_files_only=local_files_only,
        )
    return HuggingFacePhonemeRecognizer(
        candidate_id=candidate_id,
        device=device,
        local_files_only=local_files_only,
    )


def create_word_aligner(
    *,
    candidate_id: str,
    candidate: CandidateSpecV1 | None = None,
    google_project_id: str | None = None,
    deepgram_api_key: SecretStr | None = None,
    device: str = "cpu",
    local_files_only: bool = False,
) -> WordAligner:
    """Construct a registered word candidate without importing optional SDKs eagerly."""
    if candidate_id == GOOGLE_CHIRP_3_CANDIDATE_ID:
        if google_project_id is None:
            raise ValueError("google_project_id is required for the Google candidate")
        return GoogleChirp3WordRecognizer(project_id=google_project_id)
    if candidate_id == DEEPGRAM_NOVA_3_CANDIDATE_ID:
        if candidate is None or deepgram_api_key is None:
            raise ValueError("Deepgram requires an exact candidate spec and DEEPGRAM_API_KEY")
        return DeepgramNova3WordRecognizer(candidate=candidate, api_key=deepgram_api_key)
    if candidate_id == "word-wav2vec2-base-960h-ctc-v1":
        return HuggingFaceCTCWordAligner(
            device=device,
            local_files_only=local_files_only,
        )
    raise ValueError(
        f"word candidate {candidate_id!r} has no direct in-process adapter; "
        "MFA artifacts are imported from its isolated CLI runtime"
    )


__all__ = [
    "ALLOPHANT_ALLOPHOIBLE_ENGLISH_INVENTORY",
    "DecodedPhoneSequence",
    "DeepgramNova3WordRecognizer",
    "ALLOPHANT_ENGLISH_INVENTORY",
    "AllophantPhonemeRecognizer",
    "CTCAlignmentError",
    "CTCTokenSpan",
    "HuggingFacePhonemeRecognizer",
    "HuggingFaceCTCWordAligner",
    "GoogleChirp3WordRecognizer",
    "MFA_ALIGNER_REVISION",
    "PhonemeRecognizer",
    "RawPhoneSpan",
    "WordAligner",
    "build_phoneme_artifact",
    "create_phoneme_recognizer",
    "create_word_aligner",
    "ctc_transcript_token_ids",
    "decode_phone_frames",
    "deepgram_model_version",
    "discover_deepgram_nova_3_version",
    "force_align_ctc",
    "google_recognize_retry",
    "normalize_ctc_transcript",
    "normalize_phone_spans",
    "parse_deepgram_nova_3_response",
    "parse_google_chirp_3_response",
    "parse_mfa_json_artifact",
    "token_spans_to_words",
]
