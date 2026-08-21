# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned initial candidates and hosted candidate builders for BENCH-668."""

import hashlib
import json

from coval_bench.preprocessing.benchmarking.contracts import (
    BenchmarkCandidateKind,
    CandidateSpecV1,
    ModelAssetV1,
)
from coval_bench.preprocessing.benchmarking.inventory import PHONE_INVENTORY_VERSION

_HARNESS_REVISION = "timestamp-benchmark-v1"
_RESAMPLER = "scipy-resample-poly-16000-v1"
DEEPGRAM_NOVA_3_CANDIDATE_ID = "word-deepgram-nova-3-hosted-v1"
GOOGLE_RETRY_POLICY = "google-api-core-if-transient-error-v1"
GOOGLE_RETRY_INITIAL_SECONDS = 0.5
GOOGLE_RETRY_MAXIMUM_SECONDS = 4.0
GOOGLE_RETRY_MULTIPLIER = 2.0
GOOGLE_RETRY_DEADLINE_SECONDS = 30.0
DEEPGRAM_RETRY_POLICY = "bounded-transient-http-v1"
DEEPGRAM_MAX_REQUEST_ATTEMPTS = 3
DEEPGRAM_TRANSIENT_STATUS_CODES = (408, 500, 502, 503, 504)
DEEPGRAM_RETRY_INITIAL_SECONDS = 0.5
DEEPGRAM_RETRY_MULTIPLIER = 2.0
DEEPGRAM_RETRY_JITTER_MAX_MILLISECONDS = 250
HOSTED_CANDIDATE_IDS = frozenset(
    {
        "word-google-chirp-3-hosted-v1",
        DEEPGRAM_NOVA_3_CANDIDATE_ID,
    }
)

PRIMARY_WORD_CANDIDATE_IDS = (
    "word-mfa-english-us-arpa-v1",
    "word-google-chirp-3-hosted-v1",
    DEEPGRAM_NOVA_3_CANDIDATE_ID,
)
PRIMARY_PHONEME_CANDIDATE_IDS = (
    "phone-meta-espeak-ctc-midpoint-v1",
    "phone-allophant-en-multitask-v1",
)

CANDIDATES = (
    CandidateSpecV1(
        candidate_id="word-wav2vec2-base-960h-ctc-v1",
        kind=BenchmarkCandidateKind.WORD_ALIGNER,
        implementation="coval-ctc-forced-alignment",
        implementation_revision=_HARNESS_REVISION,
        model_name="facebook/wav2vec2-base-960h",
        model_revision="fd6acee659c35d4a148bf50caac693b056efd8d1",
        assets=(
            ModelAssetV1(
                path="pytorch_model.bin",
                sha256="c34f9827b034a1b9141dbf6f652f8a60eda61cdf5771c9e05bfa99033c92cd96",
            ),
            ModelAssetV1(
                path="vocab.json",
                sha256="19727f8944fe6459fc3f240ae2c198395b740f6a029bd23e06656266b83bcf64",
            ),
        ),
        decoder="known-transcript-ctc-trellis-v1",
        resampler=_RESAMPLER,
        normalization_version="word-nfkc-casefold-alnum-v1",
        phone_inventory_version=None,
        freely_predicts_words=False,
        license_id="Apache-2.0",
        commercial_use_allowed=True,
        redistribution_allowed=True,
        benchmark_eligible=True,
        license_eligible_for_production=True,
        eligibility_notes=("Selection still requires human-boundary and GPU results.",),
    ),
    CandidateSpecV1(
        candidate_id="word-mfa-english-us-arpa-v1",
        kind=BenchmarkCandidateKind.WORD_ALIGNER,
        implementation="montreal-forced-aligner-3.3.0",
        implementation_revision="3.3.0",
        model_name="MontrealCorpusTools/mfa-models/english_us_arpa",
        model_revision="v3.0.0",
        assets=(
            ModelAssetV1(
                path="acoustic/english_us_arpa.zip",
                sha256="d35ce271ded357d833d2f4b8d1041dc3748b9538567ba13f2c697f4e4126711b",
            ),
            ModelAssetV1(
                path="dictionary/english_us_arpa.dict",
                sha256="e8c6c7b036ae2b7c78d2768b8dc6b1f9359175b842956d00b48c53c9c332e6b0",
            ),
        ),
        decoder="mfa-gmm-hmm-single-speaker-no-adaptation-v1",
        resampler="mfa-16000hz-mfcc-v3.3.0",
        normalization_version="word-nfkc-casefold-alnum-v1",
        phone_inventory_version=None,
        freely_predicts_words=False,
        license_id="CC-BY-4.0",
        commercial_use_allowed=True,
        redistribution_allowed=True,
        benchmark_eligible=True,
        license_eligible_for_production=False,
        eligibility_notes=(
            "Model assets are CC-BY-4.0, but the production runtime dependency license "
            "scan is unresolved.",
            "The current disposable conda solve includes a GPL ffmpeg build and must not "
            "be copied into production.",
            "Selection still requires human-boundary and target-hardware results.",
        ),
    ),
    CandidateSpecV1(
        candidate_id="word-google-chirp-3-hosted-v1",
        kind=BenchmarkCandidateKind.WORD_ALIGNER,
        implementation="google-cloud-speech-v2-recognize",
        implementation_revision=_HARNESS_REVISION,
        model_name="google-cloud-speech-v2/chirp_3",
        model_revision="provider-managed-chirp_3",
        assets=(
            ModelAssetV1(
                path="request_configs/google-chirp-3-en-us-v1.json",
                sha256="bc9f564805b90c25df4246c2793b795635411ffb7bc9957d5bf97043e522f9c1",
            ),
        ),
        decoder="google-v2-top-alternative-word-offsets-confidence-unavailable-v1",
        resampler="google-v2-auto-decoding-v1",
        normalization_version="word-nfkc-casefold-alnum-v1",
        phone_inventory_version=None,
        freely_predicts_words=True,
        license_id="Google Cloud Platform Terms",
        commercial_use_allowed=True,
        redistribution_allowed=False,
        benchmark_eligible=True,
        license_eligible_for_production=False,
        eligibility_notes=(
            "Google exposes the chirp_3 service model ID but no immutable model revision or "
            "weights hash.",
            "Google Chirp 3 does not support requested word-level confidence; artifacts use "
            "zero as an unavailable-value sentinel that must not be calibrated.",
            "The checked request configuration is hashed; production selection remains blocked "
            "on an accepted hosted-model reproducibility policy.",
            "Selection still requires authorized human-boundary evaluation.",
        ),
    ),
    CandidateSpecV1(
        candidate_id="phone-meta-espeak-ctc-sparse-v1",
        kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER,
        implementation="transformers-wav2vec2-ctc",
        implementation_revision=_HARNESS_REVISION,
        model_name="facebook/wav2vec2-lv-60-espeak-cv-ft",
        model_revision="ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4",
        assets=(
            ModelAssetV1(
                path="pytorch_model.bin",
                sha256="3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e",
            ),
            ModelAssetV1(
                path="vocab.json",
                sha256="d732ab2456c0c017930001dc9af0b41b3b93d25b2eb9740bf9d925508d7d87d0",
            ),
        ),
        decoder="ctc-greedy-sparse-v1",
        resampler=_RESAMPLER,
        normalization_version=PHONE_INVENTORY_VERSION,
        phone_inventory_version=PHONE_INVENTORY_VERSION,
        freely_predicts_words=None,
        license_id="Apache-2.0",
        commercial_use_allowed=True,
        redistribution_allowed=True,
        benchmark_eligible=True,
        license_eligible_for_production=True,
        eligibility_notes=(
            "Selection still requires observed-phone ground truth and GPU results.",
        ),
    ),
    CandidateSpecV1(
        candidate_id="phone-meta-espeak-ctc-midpoint-v1",
        kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER,
        implementation="transformers-wav2vec2-ctc",
        implementation_revision=_HARNESS_REVISION,
        model_name="facebook/wav2vec2-lv-60-espeak-cv-ft",
        model_revision="ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4",
        assets=(
            ModelAssetV1(
                path="pytorch_model.bin",
                sha256="3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e",
            ),
            ModelAssetV1(
                path="vocab.json",
                sha256="d732ab2456c0c017930001dc9af0b41b3b93d25b2eb9740bf9d925508d7d87d0",
            ),
        ),
        decoder="ctc-greedy-midpoint-fill-v1",
        resampler=_RESAMPLER,
        normalization_version=PHONE_INVENTORY_VERSION,
        phone_inventory_version=PHONE_INVENTORY_VERSION,
        freely_predicts_words=None,
        license_id="Apache-2.0",
        commercial_use_allowed=True,
        redistribution_allowed=True,
        benchmark_eligible=True,
        license_eligible_for_production=True,
        eligibility_notes=(
            "Dense blank-gap timing must beat sparse decoding on human boundaries.",
        ),
    ),
    CandidateSpecV1(
        candidate_id="phone-allophant-en-multitask-v1",
        kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER,
        implementation="allophant-1.0.0-py312-compat",
        implementation_revision=_HARNESS_REVISION,
        model_name="kgnlp/allophant",
        model_revision="ad69d315e4c42991cb3faecd294476515195237d",
        assets=(
            ModelAssetV1(
                path="allophant.pt",
                sha256="0a1a28183544199e82c0d3574968d5518fc4fbaa10efe9f6ab467110de474dcb",
            ),
            ModelAssetV1(
                path=(
                    "facebook/wav2vec2-xls-r-300m@"
                    "1a640f32ac3e39899438a2931f9924c02f080a54/config.json"
                ),
                sha256="0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac",
            ),
            ModelAssetV1(
                path=(
                    "facebook/wav2vec2-xls-r-300m@"
                    "1a640f32ac3e39899438a2931f9924c02f080a54/preprocessor_config.json"
                ),
                sha256="a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335",
            ),
        ),
        decoder="ctc-greedy-midpoint-fill-en39-v1",
        resampler=_RESAMPLER,
        normalization_version=PHONE_INVENTORY_VERSION,
        phone_inventory_version=PHONE_INVENTORY_VERSION,
        freely_predicts_words=None,
        license_id="Apache-2.0",
        commercial_use_allowed=True,
        redistribution_allowed=True,
        benchmark_eligible=True,
        license_eligible_for_production=False,
        eligibility_notes=(
            "Checkpoint is Apache-2.0 and the Allophant source package is MIT licensed.",
            "The checkpoint's acoustic base is pinned to wav2vec2-xls-r-300m revision "
            "1a640f32ac3e39899438a2931f9924c02f080a54.",
            "The explicit 39-phone English inventory restores /p t k/, which are absent "
            "from Allophant's built-in Allophoible English inventory.",
            "Allophant 1.0.0 has a stale Python 3.12 dependency solve; production packaging "
            "and target-GPU validation remain required.",
        ),
    ),
    CandidateSpecV1(
        candidate_id="phone-charsiu-frame-10ms-v1",
        kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER,
        implementation="transformers-wav2vec2-ctc-as-frame-classifier",
        implementation_revision=_HARNESS_REVISION,
        model_name="charsiu/en_w2v2_fc_10ms",
        model_revision="e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f",
        assets=(
            ModelAssetV1(
                path="pytorch_model.bin",
                sha256="6dc8a18422db7c22e951d5f72dc2afc267b942eb0b8459ac6dcc0cf412536de1",
            ),
            ModelAssetV1(
                path=(
                    "charsiu/tokenizer_en_cmu@10507401aedf5e0aba164128535b49225ff95260/vocab.json"
                ),
                sha256="781514e251e9615ea4a05bbf3760ffcf8ec2ebe67355f411177ea64b0602d897",
            ),
        ),
        decoder="adjacent-frame-collapse-v1",
        resampler=_RESAMPLER,
        normalization_version=PHONE_INVENTORY_VERSION,
        phone_inventory_version=PHONE_INVENTORY_VERSION,
        freely_predicts_words=None,
        license_id=None,
        commercial_use_allowed=None,
        redistribution_allowed=None,
        benchmark_eligible=True,
        license_eligible_for_production=False,
        eligibility_notes=(
            "Checkpoint has no published license; source repository MIT license is insufficient.",
        ),
    ),
)


def deepgram_nova_3_request_config(model_version: str) -> dict[str, str]:
    """Return the canonical request parameters hashed into a Deepgram candidate."""
    if not model_version.strip() or model_version.casefold() in {
        "latest",
        "main",
        "master",
        "head",
    }:
        raise ValueError("Deepgram model_version must be an exact provider version")
    return {
        "endpoint": "https://api.deepgram.com/v1/listen",
        "language": "en-US",
        "model": "nova-3",
        "punctuate": "false",
        "retry_initial_seconds": str(DEEPGRAM_RETRY_INITIAL_SECONDS),
        "retry_jitter_max_milliseconds": str(DEEPGRAM_RETRY_JITTER_MAX_MILLISECONDS),
        "retry_max_attempts": str(DEEPGRAM_MAX_REQUEST_ATTEMPTS),
        "retry_multiplier": str(DEEPGRAM_RETRY_MULTIPLIER),
        "retry_policy": DEEPGRAM_RETRY_POLICY,
        "retry_transient_status_codes": ",".join(
            str(status_code) for status_code in DEEPGRAM_TRANSIENT_STATUS_CODES
        ),
        "smart_format": "false",
        "timeout_seconds": "60",
        "version": model_version,
    }


def deepgram_nova_3_candidate(model_version: str) -> CandidateSpecV1:
    """Build the exact public candidate identity after a one-clip version probe."""
    request_config = deepgram_nova_3_request_config(model_version)
    request_sha256 = hashlib.sha256(
        json.dumps(
            request_config,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return CandidateSpecV1(
        candidate_id=DEEPGRAM_NOVA_3_CANDIDATE_ID,
        kind=BenchmarkCandidateKind.WORD_ALIGNER,
        implementation="deepgram-prerecorded-rest-v1",
        implementation_revision=_HARNESS_REVISION,
        model_name="deepgram/nova-3",
        model_revision=model_version,
        assets=(
            ModelAssetV1(
                path="request_configs/deepgram-nova-3-en-us-v1.json",
                sha256=request_sha256,
            ),
        ),
        decoder="deepgram-top-alternative-word-offsets-v1",
        resampler="deepgram-provider-decoding-v1",
        normalization_version="word-nfkc-casefold-alnum-v1",
        phone_inventory_version=None,
        freely_predicts_words=True,
        license_id="Deepgram Terms of Service",
        commercial_use_allowed=True,
        redistribution_allowed=False,
        benchmark_eligible=True,
        license_eligible_for_production=False,
        eligibility_notes=(
            "The exact response model version and canonical request configuration are pinned.",
            "Hosted weights are not redistributable and their hash is not exposed.",
            "Production selection remains blocked on an accepted hosted-model "
            "reproducibility policy.",
        ),
    )


def candidate_spec_is_registered(candidate: CandidateSpecV1) -> bool:
    """Validate a static registry entry or an exactly versioned Deepgram candidate."""
    if candidate.candidate_id == DEEPGRAM_NOVA_3_CANDIDATE_ID:
        return candidate == deepgram_nova_3_candidate(candidate.model_revision)
    return candidate in CANDIDATES


def candidate_fingerprint(candidate: CandidateSpecV1) -> str:
    """Hash every candidate pin into one stable processor identity component."""
    payload = json.dumps(
        candidate.model_dump(mode="json"),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidate_processor_revision(candidate: CandidateSpecV1) -> str:
    """Retain the readable model revision while binding all candidate pins."""
    return f"{candidate.model_revision}@{candidate_fingerprint(candidate)}"


def candidate_uses_hosted_model(candidate: CandidateSpecV1) -> bool:
    """Return whether selection must use hosted reproducibility evidence."""
    return candidate.candidate_id in HOSTED_CANDIDATE_IDS
