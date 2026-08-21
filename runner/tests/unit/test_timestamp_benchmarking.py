# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for timestamp model evaluation behavior."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from datetime import timedelta
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from click.testing import CliRunner
from pydantic import SecretStr, ValidationError

from coval_bench.preprocessing.artifacts import generator_fingerprint, immutable_artifact_key
from coval_bench.preprocessing.benchmarking import (
    CANDIDATES,
    COVAL_ENGLISH_PHONES_V1,
    DEEPGRAM_NOVA_3_CANDIDATE_ID,
    PRIMARY_PHONEME_CANDIDATE_IDS,
    PRIMARY_WORD_CANDIDATE_IDS,
    AlignmentOperation,
    BenchmarkCandidateKind,
    BenchmarkMode,
    CandidateSpecV1,
    ModelAssetV1,
    OperationalMeasurementV1,
    PhonemeGroundTruthV1,
    PhoneSource,
    ReferencePhonemeV1,
    ReferenceWordV1,
    WordGroundTruthV1,
    align_sequences,
    deepgram_nova_3_candidate,
    evaluate_phoneme_accuracy,
    evaluate_phoneme_agreement,
    evaluate_word_accuracy,
    evaluate_word_agreement,
    evaluate_word_recognition,
    normalize_phone_sequence,
)
from coval_bench.preprocessing.benchmarking.adapters import (
    ALLOPHANT_ALLOPHOIBLE_ENGLISH_INVENTORY,
    ALLOPHANT_ENGLISH_INVENTORY,
    AllophantPhonemeRecognizer,
    CTCAlignmentError,
    PhonemeRecognizer,
    WordAligner,
    create_phoneme_recognizer,
    create_word_aligner,
    ctc_transcript_token_ids,
    decode_phone_frames,
    deepgram_model_version,
    force_align_ctc,
    google_recognize_retry,
    normalize_phone_spans,
    parse_deepgram_nova_3_response,
    parse_google_chirp_3_response,
    parse_mfa_json_artifact,
    token_spans_to_words,
)
from coval_bench.preprocessing.benchmarking.adapters.base import (
    HostedProviderError,
    source_duration_ms,
)
from coval_bench.preprocessing.benchmarking.candidates import (
    DEEPGRAM_MAX_REQUEST_ATTEMPTS,
    DEEPGRAM_RETRY_INITIAL_SECONDS,
    DEEPGRAM_RETRY_JITTER_MAX_MILLISECONDS,
    DEEPGRAM_RETRY_MULTIPLIER,
    DEEPGRAM_RETRY_POLICY,
    DEEPGRAM_TRANSIENT_STATUS_CODES,
    GOOGLE_RETRY_DEADLINE_SECONDS,
    GOOGLE_RETRY_INITIAL_SECONDS,
    GOOGLE_RETRY_MAXIMUM_SECONDS,
    GOOGLE_RETRY_MULTIPLIER,
    GOOGLE_RETRY_POLICY,
    candidate_processor_revision,
    deepgram_nova_3_request_config,
)
from coval_bench.preprocessing.benchmarking.cli import timestamp_benchmark
from coval_bench.preprocessing.benchmarking.privacy import validate_private_evidence_path
from coval_bench.preprocessing.benchmarking.reporting import (
    CandidateFailureV1,
    FailureKind,
    PhonemeAgreementCaseV1,
    PhonemeCandidateOutputV1,
    TimestampBenchmarkBundleV1,
    WordAgreementCaseV1,
    WordCandidateOutputV1,
    WordGroundTruthCaseV1,
    build_report,
    merge_benchmark_bundles,
    select_candidate_subset,
)
from coval_bench.preprocessing.benchmarking.runtime import (
    PhonemeAgreementRunManifestV1,
    PrivateAudioClipV1,
    PrivateAudioSelectionV1,
    PrivateWordClipV1,
    PublicSTTManifestItemV1,
    PublicSTTManifestV1,
    WordAgreementRunManifestV1,
    _billable_audio_duration_ms,
    build_word_agreement_manifest,
)
from coval_bench.preprocessing.benchmarking.sampling import (
    clip_stratum,
    select_stratified_stt_items,
)
from coval_bench.preprocessing.benchmarking.selection import (
    SelectionEvidenceV1,
    selection_blockers,
)
from coval_bench.preprocessing.contracts import (
    PhonemeProcessorProvenanceV1,
    PhonemeTimestampsV1,
    PhonemeTimestampV1,
    TimestampWarningCode,
    TimestampWarningV1,
    WordProcessorProvenanceV1,
    WordTimestampsV1,
    WordTimestampV1,
)

SOURCE_SHA = "a" * 64
TRANSCRIPT_SHA = "b" * 64
WEIGHTS_SHA = "c" * 64


@dataclass(frozen=True)
class _SelectionItem:
    path: str
    sha256: str
    transcript: str
    duration_sec: float


def _word(text: str, start_ms: int, end_ms: int) -> WordTimestampV1:
    return WordTimestampV1(text=text, start_ms=start_ms, end_ms=end_ms, confidence=0.9)


def _phone(symbol: str, start_ms: int, end_ms: int) -> PhonemeTimestampV1:
    return PhonemeTimestampV1(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=0.8,
    )


def _word_artifact(
    words: tuple[WordTimestampV1, ...], *, analysis_id: str = "invented-001"
) -> WordTimestampsV1:
    warnings = (
        (
            TimestampWarningV1(
                code=TimestampWarningCode.EMPTY_SPANS,
                message="invented empty output",
            ),
        )
        if not words
        else ()
    )
    return WordTimestampsV1(
        schema_version="WordTimestampsV1",
        analysis_id=analysis_id,
        audio_sha256=SOURCE_SHA,
        transcript_sha256=TRANSCRIPT_SHA,
        duration_ms=1_000,
        processor=WordProcessorProvenanceV1(
            aligner_name="invented-word-aligner",
            aligner_revision="revision-001",
            normalization_version="word-nfkc-casefold-alnum-v1",
        ),
        words=words,
        warnings=warnings,
    )


def _phone_artifact(
    phones: tuple[PhonemeTimestampV1, ...], *, analysis_id: str = "invented-001"
) -> PhonemeTimestampsV1:
    warnings = (
        (
            TimestampWarningV1(
                code=TimestampWarningCode.EMPTY_SPANS,
                message="invented empty output",
            ),
        )
        if not phones
        else ()
    )
    return PhonemeTimestampsV1(
        schema_version="PhonemeTimestampsV1",
        analysis_id=analysis_id,
        audio_sha256=SOURCE_SHA,
        duration_ms=1_000,
        processor=PhonemeProcessorProvenanceV1(
            model_name="invented-phone-recognizer",
            model_revision="revision-001",
            weights_sha256=WEIGHTS_SHA,
            phone_inventory=COVAL_ENGLISH_PHONES_V1,
            resampler="invented-resampler-v1",
            decoder="audio-only-decoder-v1",
        ),
        phones=phones,
        warnings=warnings,
    )


def _word_ground_truth() -> WordGroundTruthV1:
    return WordGroundTruthV1(
        analysis_id="invented-001",
        audio_sha256=SOURCE_SHA,
        duration_ms=1_000,
        annotation_revision="invented-human-v1",
        words=(
            ReferenceWordV1(text="Hello,", start_ms=100, end_ms=300),
            ReferenceWordV1(text="world", start_ms=400, end_ms=700),
        ),
    )


def _phone_ground_truth() -> PhonemeGroundTruthV1:
    return PhonemeGroundTruthV1(
        analysis_id="invented-001",
        audio_sha256=SOURCE_SHA,
        duration_ms=1_000,
        annotation_revision="invented-human-v1",
        phone_inventory_version="coval-english-arpabet-v1",
        phones=(
            ReferencePhonemeV1(symbol="HH", start_ms=100, end_ms=200),
            ReferencePhonemeV1(symbol="AH", start_ms=200, end_ms=350),
            ReferencePhonemeV1(symbol="L", start_ms=350, end_ms=500),
        ),
    )


def _candidate(**overrides: Any) -> CandidateSpecV1:
    values: dict[str, Any] = {
        "candidate_id": "invented-word-candidate",
        "kind": BenchmarkCandidateKind.WORD_ALIGNER,
        "implementation": "invented",
        "implementation_revision": "revision-1",
        "model_name": "invented/model",
        "model_revision": "revision-2",
        "assets": (ModelAssetV1(path="weights.bin", sha256=WEIGHTS_SHA),),
        "decoder": "invented-decoder-v1",
        "resampler": "invented-resampler-v1",
        "normalization_version": "invented-normalization-v1",
        "phone_inventory_version": None,
        "freely_predicts_words": False,
        "license_id": "Apache-2.0",
        "commercial_use_allowed": True,
        "redistribution_allowed": True,
        "benchmark_eligible": True,
        "license_eligible_for_production": True,
        "eligibility_notes": (),
    }
    values.update(overrides)
    return CandidateSpecV1(**values)


def _invented_word_candidate(candidate_id: str) -> CandidateSpecV1:
    return _candidate(
        candidate_id=candidate_id,
        model_name="invented-word-aligner",
        model_revision="revision-001",
        normalization_version="word-nfkc-casefold-alnum-v1",
        benchmark_eligible=False,
        license_eligible_for_production=False,
    )


def _invented_phone_candidate(candidate_id: str) -> CandidateSpecV1:
    return _candidate(
        candidate_id=candidate_id,
        kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER,
        model_name="invented-phone-recognizer",
        model_revision="revision-001",
        assets=(ModelAssetV1(path="pytorch_model.bin", sha256=WEIGHTS_SHA),),
        decoder="audio-only-decoder-v1",
        resampler="invented-resampler-v1",
        normalization_version="coval-english-arpabet-v1",
        phone_inventory_version="coval-english-arpabet-v1",
        freely_predicts_words=None,
        benchmark_eligible=False,
        license_eligible_for_production=False,
    )


def test_candidate_revisions_are_immutable_and_licenses_gate_production() -> None:
    with pytest.raises(ValidationError):
        _candidate(model_revision="main")
    with pytest.raises(ValidationError, match="commercial-use license"):
        _candidate(
            license_id=None,
            commercial_use_allowed=None,
            redistribution_allowed=None,
        )
    with pytest.raises(ValidationError, match="phone_inventory_version"):
        _candidate(kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER)

    charsiu = next(candidate for candidate in CANDIDATES if "charsiu" in candidate.candidate_id)
    assert charsiu.benchmark_eligible is True
    assert charsiu.license_eligible_for_production is False
    assert charsiu.license_id is None

    allophant = next(candidate for candidate in CANDIDATES if "allophant" in candidate.candidate_id)
    assert allophant.model_revision == "ad69d315e4c42991cb3faecd294476515195237d"
    assert allophant.assets[0].sha256 == (
        "0a1a28183544199e82c0d3574968d5518fc4fbaa10efe9f6ab467110de474dcb"
    )
    assert allophant.license_id == "Apache-2.0"
    assert allophant.license_eligible_for_production is False

    google = next(
        candidate for candidate in CANDIDATES if "google-chirp-3" in candidate.candidate_id
    )
    assert google.freely_predicts_words is True
    assert google.model_revision == "provider-managed-chirp_3"
    assert "confidence-unavailable" in google.decoder
    request_config_path = (
        Path(__file__).parents[2]
        / "src/coval_bench/preprocessing/benchmarking/request_configs"
        / "google-chirp-3-en-us-v1.json"
    )
    request_config = json.loads(request_config_path.read_text())
    assert request_config["features"] == {
        "enable_automatic_punctuation": False,
        "enable_word_time_offsets": True,
    }
    assert request_config["retry"] == {
        "backoff_initial_seconds": GOOGLE_RETRY_INITIAL_SECONDS,
        "backoff_maximum_seconds": GOOGLE_RETRY_MAXIMUM_SECONDS,
        "backoff_multiplier": GOOGLE_RETRY_MULTIPLIER,
        "deadline_seconds": GOOGLE_RETRY_DEADLINE_SECONDS,
        "policy": GOOGLE_RETRY_POLICY,
    }
    assert google.assets[0].sha256 == hashlib.sha256(request_config_path.read_bytes()).hexdigest()
    assert google.license_eligible_for_production is False


def test_every_candidate_pin_is_bound_into_the_strict_processor_fingerprint() -> None:
    candidate = next(
        candidate
        for candidate in CANDIDATES
        if candidate.candidate_id == "phone-meta-espeak-ctc-midpoint-v1"
    )
    changed_vocab = candidate.assets[1].model_copy(update={"sha256": "d" * 64})
    changed_candidate = candidate.model_copy(
        update={"assets": (candidate.assets[0], changed_vocab)}
    )
    assert candidate_processor_revision(candidate) != candidate_processor_revision(
        changed_candidate
    )

    artifact = _phone_artifact((_phone("HH", 100, 200),))
    original_processor = artifact.processor.model_copy(
        update={
            "model_name": candidate.model_name,
            "model_revision": candidate_processor_revision(candidate),
            "weights_sha256": candidate.assets[0].sha256,
            "resampler": candidate.resampler,
            "decoder": candidate.decoder,
        }
    )
    changed_processor = original_processor.model_copy(
        update={"model_revision": candidate_processor_revision(changed_candidate)}
    )
    original_artifact = artifact.model_copy(update={"processor": original_processor})
    changed_artifact = artifact.model_copy(update={"processor": changed_processor})
    assert generator_fingerprint(original_processor) != generator_fingerprint(changed_processor)
    assert immutable_artifact_key(original_artifact) != immutable_artifact_key(changed_artifact)


def test_primary_comparison_uses_one_meta_phone_model_and_one_independent_model() -> None:
    assert PRIMARY_WORD_CANDIDATE_IDS == (
        "word-mfa-english-us-arpa-v1",
        "word-google-chirp-3-hosted-v1",
        "word-deepgram-nova-3-hosted-v1",
    )
    assert PRIMARY_PHONEME_CANDIDATE_IDS == (
        "phone-meta-espeak-ctc-midpoint-v1",
        "phone-allophant-en-multitask-v1",
    )
    assert sum("phone-meta" in candidate_id for candidate_id in PRIMARY_PHONEME_CANDIDATE_IDS) == 1


def test_google_word_candidate_factory_requires_project_without_loading_sdk() -> None:
    with pytest.raises(ValueError, match="google_project_id"):
        create_word_aligner(candidate_id="word-google-chirp-3-hosted-v1")

    adapter = create_word_aligner(
        candidate_id="word-google-chirp-3-hosted-v1",
        google_project_id="invented-project",
    )
    assert adapter.candidate.model_name == "google-cloud-speech-v2/chirp_3"
    assert adapter.model_load_seconds == 0.0


def test_google_runtime_provenance_fails_before_requests_when_sdk_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = create_word_aligner(
        candidate_id="word-google-chirp-3-hosted-v1",
        google_project_id="invented-project",
    )

    def missing_distribution(_: str) -> str:
        raise PackageNotFoundError("google-cloud-speech")

    monkeypatch.setattr(
        "coval_bench.preprocessing.benchmarking.adapters.word_google.version",
        missing_distribution,
    )
    with pytest.raises(RuntimeError, match="optional google-stt dependency"):
        _ = adapter.runtime_software


def test_google_retry_policy_is_bounded_and_transient_only() -> None:
    from google.api_core.exceptions import InvalidArgument, ServiceUnavailable

    retry = google_recognize_retry()
    assert retry._initial == 0.5
    assert retry._maximum == 4.0
    assert retry._multiplier == 2.0
    assert retry._timeout == 30.0
    assert retry._predicate(ServiceUnavailable("invented transient")) is True
    assert retry._predicate(InvalidArgument("invented permanent")) is False


def test_deepgram_candidate_probe_result_is_exact_and_factory_ready() -> None:
    candidate = deepgram_nova_3_candidate("2026-08-01.12345")
    assert candidate.candidate_id == DEEPGRAM_NOVA_3_CANDIDATE_ID
    assert candidate.model_revision == "2026-08-01.12345"
    request_config = deepgram_nova_3_request_config(candidate.model_revision)
    assert request_config["retry_policy"] == DEEPGRAM_RETRY_POLICY
    assert request_config["retry_max_attempts"] == str(DEEPGRAM_MAX_REQUEST_ATTEMPTS)
    assert request_config["retry_transient_status_codes"] == ",".join(
        str(status_code) for status_code in DEEPGRAM_TRANSIENT_STATUS_CODES
    )
    assert request_config["retry_initial_seconds"] == str(DEEPGRAM_RETRY_INITIAL_SECONDS)
    assert request_config["retry_multiplier"] == str(DEEPGRAM_RETRY_MULTIPLIER)
    assert request_config["retry_jitter_max_milliseconds"] == str(
        DEEPGRAM_RETRY_JITTER_MAX_MILLISECONDS
    )
    assert (
        candidate.assets[0].sha256
        == hashlib.sha256(
            json.dumps(request_config, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
    )
    with pytest.raises(ValueError, match="exact provider version"):
        deepgram_nova_3_candidate("latest")
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        create_word_aligner(
            candidate_id=DEEPGRAM_NOVA_3_CANDIDATE_ID,
            candidate=candidate,
        )
    adapter = create_word_aligner(
        candidate_id=DEEPGRAM_NOVA_3_CANDIDATE_ID,
        candidate=candidate,
        deepgram_api_key=SecretStr("invented-secret"),
    )
    assert adapter.candidate == candidate


def test_deepgram_retries_transient_statuses_with_bounded_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "invented.wav"
    audio.write_bytes(b"invented audio")
    request = httpx.Request("POST", "https://api.deepgram.com/v1/listen")
    responses = iter(
        (
            httpx.Response(503, request=request),
            httpx.Response(503, request=request),
            httpx.Response(
                200,
                request=request,
                json={
                    "metadata": {
                        "models": ["invented-model"],
                        "model_info": {"invented-model": {"version": "2026-08-01.12345"}},
                    }
                },
            ),
        )
    )
    calls = 0
    sleeps: list[float] = []

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(
        "coval_bench.preprocessing.benchmarking.adapters.word_deepgram.httpx.post",
        fake_post,
    )
    monkeypatch.setattr(
        "coval_bench.preprocessing.benchmarking.adapters.word_deepgram.secrets.randbelow",
        lambda _: 0,
    )
    monkeypatch.setattr(
        "coval_bench.preprocessing.benchmarking.adapters.word_deepgram.time.sleep",
        sleeps.append,
    )
    from coval_bench.preprocessing.benchmarking.adapters import discover_deepgram_nova_3_version

    assert (
        discover_deepgram_nova_3_version(
            audio_path=audio,
            api_key=SecretStr("invented-secret"),
        )
        == "2026-08-01.12345"
    )
    assert calls == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    [(401, "provider_auth"), (402, "provider_quota"), (400, "provider_permanent")],
)
def test_deepgram_permanent_errors_are_sanitized_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    expected_code: str,
) -> None:
    audio = tmp_path / "invented.wav"
    audio.write_bytes(b"invented audio")
    request = httpx.Request("POST", "https://api.deepgram.com/v1/listen")
    calls = 0

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, request=request, text="sensitive provider payload")

    monkeypatch.setattr(
        "coval_bench.preprocessing.benchmarking.adapters.word_deepgram.httpx.post",
        fake_post,
    )
    from coval_bench.preprocessing.benchmarking.adapters import discover_deepgram_nova_3_version

    with pytest.raises(HostedProviderError) as captured:
        discover_deepgram_nova_3_version(
            audio_path=audio,
            api_key=SecretStr("invented-secret"),
        )
    assert captured.value.code == expected_code
    assert "sensitive provider payload" not in str(captured.value)
    assert "invented-secret" not in str(captured.value)
    assert calls == 1


def test_google_response_parser_emits_freely_predicted_word_timestamps() -> None:
    response = SimpleNamespace(
        results=[
            SimpleNamespace(
                alternatives=[
                    SimpleNamespace(
                        words=[
                            SimpleNamespace(
                                word="Hello,",
                                start_offset=timedelta(milliseconds=100),
                                end_offset=timedelta(milliseconds=400),
                                confidence=0.91,
                            ),
                            SimpleNamespace(
                                word="world",
                                start_offset=SimpleNamespace(seconds=0, nanos=500_000_000),
                                end_offset=SimpleNamespace(seconds=0, nanos=900_000_000),
                                confidence=0.82,
                            ),
                        ]
                    )
                ]
            )
        ]
    )
    artifact = parse_google_chirp_3_response(
        response,
        analysis_id="invented-001",
        audio_sha256=SOURCE_SHA,
        transcript="known reference text",
        duration_ms=1_000,
    )
    assert [(word.text, word.start_ms, word.end_ms) for word in artifact.words] == [
        ("hello", 100, 400),
        ("world", 500, 900),
    ]
    assert [word.confidence for word in artifact.words] == [0.0, 0.0]
    google = next(
        candidate for candidate in CANDIDATES if "google-chirp-3" in candidate.candidate_id
    )
    assert artifact.processor.aligner_revision == candidate_processor_revision(google)


def test_deepgram_response_parser_requires_and_records_exact_service_version() -> None:
    response = {
        "metadata": {
            "models": ["model-uuid"],
            "model_info": {
                "model-uuid": {
                    "name": "nova-3",
                    "version": "2026-08-01.12345",
                    "arch": "nova-3",
                }
            },
        },
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {
                                    "word": "Hello",
                                    "start": 0.1,
                                    "end": 0.4,
                                    "confidence": 0.93,
                                }
                            ]
                        }
                    ]
                }
            ]
        },
    }
    assert deepgram_model_version(response) == "2026-08-01.12345"
    artifact = parse_deepgram_nova_3_response(
        response,
        expected_model_version="2026-08-01.12345",
        analysis_id="invented-001",
        audio_sha256=SOURCE_SHA,
        transcript="known reference text",
        duration_ms=1_000,
    )
    assert artifact.words == (_word("hello", 100, 400).model_copy(update={"confidence": 0.93}),)
    assert artifact.processor.aligner_revision == candidate_processor_revision(
        deepgram_nova_3_candidate("2026-08-01.12345")
    )
    with pytest.raises(ValueError, match="pinned candidate revision"):
        parse_deepgram_nova_3_response(
            response,
            expected_model_version="different-version",
            analysis_id="invented-001",
            audio_sha256=SOURCE_SHA,
            transcript="known reference text",
            duration_ms=1_000,
        )


def test_phoneme_interface_cannot_receive_reference_text_or_phones() -> None:
    word_parameters = inspect.signature(WordAligner.align).parameters
    phone_parameters = inspect.signature(PhonemeRecognizer.recognize).parameters
    assert set(word_parameters) == {"self", "audio_path", "transcript"}
    assert set(phone_parameters) == {"self", "audio_path"}
    assert "transcript" not in phone_parameters
    assert "phones" not in phone_parameters


def test_source_duration_identity_is_not_shifted_by_resampler_rounding() -> None:
    sample_count = 64
    sample_rate = 44_100
    source_ms = source_duration_ms(sample_count=sample_count, sample_rate=sample_rate)
    resampled_count = (sample_count * 16_000 + sample_rate - 1) // sample_rate
    resampled_ms = round(resampled_count * 1_000 / 16_000)
    assert source_ms == 1
    assert resampled_ms == 2


def test_phoneme_factory_keeps_allophant_optional_and_audio_only() -> None:
    adapter = create_phoneme_recognizer(
        candidate_id="phone-allophant-en-multitask-v1",
        local_files_only=True,
    )
    assert isinstance(adapter, AllophantPhonemeRecognizer)
    assert adapter.candidate.model_name == "kgnlp/allophant"


def test_operational_manifest_requires_one_process_isolated_candidate() -> None:
    clip = PrivateAudioClipV1(
        analysis_id="invented-001",
        audio_path=Path("/private/invented-001.wav"),
        audio_sha256=SOURCE_SHA,
        duration_ms=1_000,
    )
    isolated = PhonemeAgreementRunManifestV1(
        benchmark_id="invented-allophant",
        dataset_id="invented",
        candidate_ids=("phone-allophant-en-multitask-v1",),
        clips=(clip,),
    )
    assert isolated.candidate_ids == ("phone-allophant-en-multitask-v1",)
    with pytest.raises(ValidationError, match="process-isolated"):
        PhonemeAgreementRunManifestV1(
            benchmark_id="invented-mixed",
            dataset_id="invented",
            candidate_ids=(
                "phone-allophant-en-multitask-v1",
                "phone-meta-espeak-ctc-midpoint-v1",
            ),
            clips=(clip,),
        )


def test_word_operational_manifest_keeps_transcripts_private_and_process_isolated() -> None:
    clip = PrivateWordClipV1(
        analysis_id="invented-001",
        audio_path=Path("/private/invented-001.wav"),
        audio_sha256=SOURCE_SHA,
        duration_ms=1_000,
        transcript="invented transcript",
    )
    manifest = WordAgreementRunManifestV1(
        benchmark_id="invented-google",
        dataset_id="invented",
        candidate_ids=("word-google-chirp-3-hosted-v1",),
        clips=(clip,),
    )
    assert manifest.clips[0].transcript == "invented transcript"
    with pytest.raises(ValidationError, match="process-isolated"):
        WordAgreementRunManifestV1(
            benchmark_id="invented-mixed-word",
            dataset_id="invented",
            candidate_ids=(
                "word-google-chirp-3-hosted-v1",
                "word-mfa-english-us-arpa-v1",
            ),
            clips=(clip,),
        )

    deepgram = WordAgreementRunManifestV1(
        benchmark_id="invented-deepgram",
        dataset_id="invented",
        candidate_ids=(DEEPGRAM_NOVA_3_CANDIDATE_ID,),
        candidate_spec=deepgram_nova_3_candidate("2026-08-01.12345"),
        clips=(clip,),
    )
    assert deepgram.candidate_spec is not None
    with pytest.raises(ValidationError, match="exact candidate_spec"):
        WordAgreementRunManifestV1(
            benchmark_id="invented-deepgram",
            dataset_id="invented",
            candidate_ids=(DEEPGRAM_NOVA_3_CANDIDATE_ID,),
            clips=(clip,),
        )


def test_word_manifest_builder_joins_frozen_clip_identity_to_transcript() -> None:
    selection = PrivateAudioSelectionV1(
        dataset_id="invented-data",
        clips=(
            PrivateAudioClipV1(
                analysis_id="invented-001",
                audio_path=Path("/private/invented-001.wav"),
                audio_sha256=SOURCE_SHA,
                duration_ms=1_000,
            ),
        ),
    )
    dataset = PublicSTTManifestV1(
        id="invented-data",
        items=(
            PublicSTTManifestItemV1(
                path="audio/invented-001.wav",
                sha256=SOURCE_SHA,
                transcript="invented transcript",
                duration_sec=1.0,
            ),
        ),
    )
    manifest = build_word_agreement_manifest(
        dataset_manifest=dataset,
        selection_manifest=selection,
        candidate_id="word-google-chirp-3-hosted-v1",
        benchmark_id="invented-google",
    )
    assert manifest.clips[0].transcript == "invented transcript"
    assert manifest.clips[0].audio_path == Path("/private/invented-001.wav")

    with pytest.raises(ValueError, match="selection dataset_id"):
        build_word_agreement_manifest(
            dataset_manifest=dataset.model_copy(update={"id": "other-data"}),
            selection_manifest=selection,
            candidate_id="word-google-chirp-3-hosted-v1",
            benchmark_id="invented-google",
        )


def test_known_transcript_ctc_alignment_uses_blank_expanded_viterbi_path() -> None:
    vocabulary = {"<pad>": 0, "A": 1, "|": 2, "B": 3}
    words, symbols, token_ids = ctc_transcript_token_ids(
        "A b",
        vocabulary=vocabulary,
    )
    emissions = np.full((7, 4), -10.0, dtype=np.float64)
    for frame, token_id in enumerate((0, 1, 0, 2, 0, 3, 0)):
        emissions[frame, token_id] = 0.0
    spans = force_align_ctc(
        emissions,
        token_symbols=symbols,
        token_ids=token_ids,
        blank_id=0,
    )
    assert [(span.symbol, span.start_frame, span.end_frame) for span in spans] == [
        ("A", 1, 2),
        ("|", 3, 4),
        ("B", 5, 6),
    ]
    aligned_words = token_spans_to_words(
        words,
        spans,
        frame_count=7,
        duration_ms=700,
    )
    assert [(word.text, word.start_ms, word.end_ms) for word in aligned_words] == [
        ("a", 100, 200),
        ("b", 500, 600),
    ]


def test_known_transcript_ctc_alignment_rejects_unverbalized_oov_symbols() -> None:
    with pytest.raises(CTCAlignmentError, match="outside the candidate vocabulary"):
        ctc_transcript_token_ids("route 66", vocabulary={"R": 1, "O": 2, "U": 3, "T": 4, "E": 5})


def test_mfa_json_import_surfaces_unknown_word_phone(tmp_path: Path) -> None:
    output = tmp_path / "invented.json"
    output.write_text(
        json.dumps(
            {
                "start": 0.0,
                "end": 1.0,
                "tiers": {
                    "words": {
                        "type": "interval",
                        "entries": [[0.1, 0.4, "Hello,"], [0.5, 0.9, "123"]],
                    },
                    "phones": {
                        "type": "interval",
                        "entries": [[0.1, 0.4, "HH"], [0.5, 0.9, "spn"]],
                    },
                },
            }
        )
    )
    artifact = parse_mfa_json_artifact(
        output,
        analysis_id="invented-001",
        audio_sha256=SOURCE_SHA,
        transcript="Hello, 123",
    )
    assert [
        (word.text, word.start_ms, word.end_ms, word.confidence) for word in artifact.words
    ] == [
        ("hello", 100, 400, 0.0),
        ("123", 500, 900, 0.0),
    ]
    assert len(artifact.warnings) == 1
    assert artifact.warnings[0].code is TimestampWarningCode.PARTIAL_ALIGNMENT
    assert artifact.warnings[0].start_ms == 500
    assert artifact.warnings[0].end_ms == 900


def test_audio_only_ctc_decoders_make_blank_gap_policy_explicit() -> None:
    frame_ids = (0, 1, 1, 0, 0, 2, 2, 0)
    frame_confidences = (0.9,) * len(frame_ids)
    vocabulary = {0: "<pad>", 1: "h", 2: "əl"}
    sparse = decode_phone_frames(
        frame_ids,
        frame_confidences,
        id_to_token=vocabulary,
        decoder="sparse",
        blank_id=0,
    )
    midpoint = decode_phone_frames(
        frame_ids,
        frame_confidences,
        id_to_token=vocabulary,
        decoder="midpoint_fill",
        blank_id=0,
    )
    assert [(span.start_frame, span.end_frame) for span in sparse] == [(1.0, 3.0), (5.0, 7.0)]
    assert [(span.start_frame, span.end_frame) for span in midpoint] == [(1.0, 4.0), (4.0, 7.0)]

    normalized = normalize_phone_spans(midpoint, source=PhoneSource.META_ESPEAK_IPA)
    assert tuple(span.symbol for span in normalized.spans) == ("HH", "AH", "L")
    assert normalized.spans[1].start_frame == 4.0
    assert normalized.spans[1].end_frame == 5.5
    assert normalized.normalization_loss_rate == 0.0


def test_frame_classifier_collapse_keeps_silence_explicit_until_normalization() -> None:
    raw = decode_phone_frames(
        (0, 0, 1, 1, 2),
        (0.8, 0.9, 0.7, 0.8, 0.6),
        id_to_token={0: "[SIL]", 1: "AH0", 2: "L"},
        decoder="frame_collapse",
        blank_id=None,
    )
    assert tuple(span.symbol for span in raw) == ("[SIL]", "AH0", "L")
    normalized = normalize_phone_spans(raw, source=PhoneSource.CHARSIU_CMU)
    assert tuple(span.symbol for span in normalized.spans) == ("AH", "L")


def test_phone_normalization_is_versioned_explicit_and_loss_preserving() -> None:
    timit = normalize_phone_sequence(
        ("h#", "sh", "ix", "q", "tcl", "ux"),
        source=PhoneSource.TIMIT_61,
    )
    assert timit.symbols == ("SH", "IH", "UNK", "UW")
    assert timit.source_count == 6
    assert timit.ignored_count == 2
    assert timit.unknown_count == 1
    assert timit.loss_rate == 0.25

    meta = normalize_phone_sequence(
        ("h", "əl", "ɑːɹ", "not-a-phone"),
        source=PhoneSource.META_ESPEAK_IPA,
    )
    assert meta.symbols == ("HH", "AH", "L", "AA", "R", "UNK")
    assert meta.unknown_count == 1
    assert meta.loss_rate == 0.25

    charsiu = normalize_phone_sequence(("AH0", "ZH", "[SIL]"), source=PhoneSource.CHARSIU_CMU)
    assert charsiu.symbols == ("AH", "ZH")
    assert charsiu.ignored_count == 1

    allophant = normalize_phone_sequence(
        ALLOPHANT_ENGLISH_INVENTORY,
        source=PhoneSource.ALLOPHANT_IPA,
    )
    assert allophant.source_count == 39
    assert allophant.unknown_count == 0
    assert allophant.loss_rate == 0.0
    assert {"K", "P", "T"}.issubset(allophant.symbols)
    assert len(ALLOPHANT_ALLOPHOIBLE_ENGLISH_INVENTORY) == 36
    assert len(ALLOPHANT_ENGLISH_INVENTORY) == 39
    assert tuple(
        symbol
        for symbol in ALLOPHANT_ENGLISH_INVENTORY
        if symbol not in ALLOPHANT_ALLOPHOIBLE_ENGLISH_INVENTORY
    ) == ("k", "p", "t")
    allophant_candidate = next(
        candidate for candidate in CANDIDATES if "allophant" in candidate.candidate_id
    )
    assert allophant_candidate.decoder == "ctc-greedy-midpoint-fill-en39-v1"


def test_phone_ground_truth_rejects_wrong_inventory_or_symbol() -> None:
    with pytest.raises(ValidationError, match="phone_inventory_version"):
        PhonemeGroundTruthV1(
            analysis_id="invented-001",
            audio_sha256=SOURCE_SHA,
            duration_ms=1_000,
            annotation_revision="invented-human-v1",
            phone_inventory_version="invented-inventory-v2",
            phones=(ReferencePhonemeV1(symbol="HH", start_ms=100, end_ms=200),),
        )
    with pytest.raises(ValidationError, match="outside the declared inventory"):
        PhonemeGroundTruthV1(
            analysis_id="invented-001",
            audio_sha256=SOURCE_SHA,
            duration_ms=1_000,
            annotation_revision="invented-human-v1",
            phone_inventory_version="coval-english-arpabet-v1",
            phones=(ReferencePhonemeV1(symbol="NOT_A_PHONE", start_ms=100, end_ms=200),),
        )


def test_sequence_alignment_has_stable_substitution_insertion_and_deletion_counts() -> None:
    steps = align_sequences(("A", "B", "C"), ("A", "X", "C", "D"))
    assert tuple(step.operation for step in steps) == (
        AlignmentOperation.MATCH,
        AlignmentOperation.SUBSTITUTION,
        AlignmentOperation.MATCH,
        AlignmentOperation.INSERTION,
    )
    assert align_sequences(("A", "B"), ("A",))[1].operation is AlignmentOperation.DELETION


def test_word_accuracy_reports_boundaries_overlap_and_forced_wer_as_not_applicable() -> None:
    prediction = _word_artifact(
        (
            _word("hello", 110, 280),
            _word("world", 390, 730),
        )
    )
    metrics = evaluate_word_accuracy(_word_ground_truth(), prediction, freely_predicted_words=False)
    assert metrics.matches == 2
    assert metrics.matched_word_coverage == 1.0
    assert metrics.temporal_coverage == pytest.approx(0.94)
    assert metrics.start_boundary_error.mean_signed_ms == 0.0
    assert metrics.start_boundary_error.mean_absolute_ms == 10.0
    assert metrics.end_boundary_error.mean_absolute_ms == 25.0
    assert metrics.start_boundary_error.within_20_ms == 1.0
    assert metrics.end_boundary_error.within_20_ms == 0.5
    assert metrics.wer_applicable is False
    assert metrics.word_error_rate is None


def test_word_accuracy_computes_wer_only_for_freely_predicted_words() -> None:
    prediction = _word_artifact((_word("hello", 100, 300), _word("there", 400, 700)))
    metrics = evaluate_word_accuracy(_word_ground_truth(), prediction, freely_predicted_words=True)
    assert metrics.matches == 1
    assert metrics.substitutions == 1
    assert metrics.word_error_rate == 0.5


def test_agreement_report_adds_wer_only_for_freely_predicted_candidate() -> None:
    transcript = "hello world"
    transcript_sha = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    prediction = _word_artifact((_word("hello", 100, 300), _word("there", 400, 700))).model_copy(
        update={"transcript_sha256": transcript_sha}
    )
    candidate = _invented_word_candidate("invented-hosted").model_copy(
        update={"freely_predicts_words": True}
    )
    bundle = TimestampBenchmarkBundleV1(
        benchmark_id="invented-word-agreement",
        dataset_id="invented-dataset",
        mode=BenchmarkMode.AGREEMENT,
        kind=BenchmarkCandidateKind.WORD_ALIGNER,
        fixture_only=True,
        candidate_ids=(candidate.candidate_id,),
        candidate_specs=(candidate,),
        cases=(
            WordAgreementCaseV1(
                analysis_id="invented-001",
                audio_sha256=SOURCE_SHA,
                duration_ms=1_000,
                reference_transcript=transcript,
                transcript_sha256=transcript_sha,
                outputs=(
                    WordCandidateOutputV1(
                        candidate_id=candidate.candidate_id,
                        freely_predicted_words=True,
                        artifact=prediction,
                    ),
                ),
            ),
        ),
    )
    report = build_report(bundle)
    assert len(report.observations) == 1
    assert report.metric_summaries[0].metrics[0].name == "word_error_rate"
    assert report.metric_summaries[0].metrics[0].value == 0.5
    assert evaluate_word_recognition(transcript, prediction).word_error_rate == 0.5


def test_word_bundle_merge_rejects_different_transcript_revisions() -> None:
    bundles = []
    for candidate_id, transcript_sha in (
        ("invented-candidate-a", TRANSCRIPT_SHA),
        ("invented-candidate-b", "d" * 64),
    ):
        candidate = _invented_word_candidate(candidate_id)
        artifact = _word_artifact((_word("hello", 100, 300),)).model_copy(
            update={"transcript_sha256": transcript_sha}
        )
        bundles.append(
            TimestampBenchmarkBundleV1(
                benchmark_id=f"invented-{candidate_id}",
                dataset_id="invented-dataset",
                mode=BenchmarkMode.AGREEMENT,
                kind=BenchmarkCandidateKind.WORD_ALIGNER,
                fixture_only=True,
                candidate_ids=(candidate_id,),
                candidate_specs=(candidate,),
                cases=(
                    WordAgreementCaseV1(
                        analysis_id="invented-001",
                        audio_sha256=SOURCE_SHA,
                        duration_ms=1_000,
                        outputs=(
                            WordCandidateOutputV1(
                                candidate_id=candidate_id,
                                freely_predicted_words=False,
                                artifact=artifact,
                            ),
                        ),
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="identical transcript revision"):
        merge_benchmark_bundles(tuple(bundles), benchmark_id="invented-merged")


def test_phoneme_accuracy_keeps_identity_and_boundary_metrics_separate() -> None:
    prediction = _phone_artifact(
        (
            _phone("HH", 90, 210),
            _phone("EH", 210, 340),
            _phone("L", 360, 510),
            _phone("Z", 600, 650),
        )
    )
    metrics = evaluate_phoneme_accuracy(
        _phone_ground_truth(), prediction, normalization_loss_rate=0.1
    )
    assert metrics.matches == 2
    assert metrics.substitutions == 1
    assert metrics.insertions == 1
    assert metrics.deletions == 0
    assert metrics.phone_error_rate == pytest.approx(2 / 3)
    assert metrics.exact_phone_coverage == pytest.approx(2 / 3)
    assert metrics.paired_reference_coverage == 1.0
    assert metrics.matched_start_boundary_error.mean_absolute_ms == 10.0
    assert metrics.substitution_start_boundary_error.mean_signed_ms == 10.0
    assert metrics.normalization_loss_rate == 0.1


def test_agreement_metrics_never_expose_accuracy_fields() -> None:
    left_words = _word_artifact((_word("hello", 100, 300), _word("world", 400, 700)))
    right_words = _word_artifact((_word("hello", 120, 320), _word("there", 410, 680)))
    word_agreement = evaluate_word_agreement(left_words, right_words)
    assert word_agreement.word_disagreement_rate == 0.5
    assert "accuracy" not in word_agreement.model_dump()

    left_phones = _phone_artifact((_phone("HH", 100, 200), _phone("AH", 200, 350)))
    right_phones = _phone_artifact((_phone("HH", 110, 210), _phone("EH", 210, 340)))
    phone_agreement = evaluate_phoneme_agreement(left_phones, right_phones)
    assert phone_agreement.phone_disagreement_rate == 0.5
    assert phone_agreement.exact_phone_agreement == 0.5
    assert phone_agreement.matched_start_boundary_difference.mean_signed_ms == 10.0
    assert "accuracy" not in phone_agreement.model_dump()


def test_ground_truth_comparison_rejects_different_audio() -> None:
    with pytest.raises(ValueError, match="analysis_id"):
        evaluate_word_accuracy(
            _word_ground_truth(),
            _word_artifact((_word("hello", 100, 300),), analysis_id="different"),
            freely_predicted_words=False,
        )


def test_operational_measurement_derives_throughput_failure_and_private_cost_input() -> None:
    measurement = OperationalMeasurementV1(
        candidate_id="invented",
        hardware="invented-gpu",
        software="container-sha-001",
        batch_size=4,
        attempted_clips=10,
        successful_clips=9,
        audio_duration_ms=60_000,
        model_load_seconds=2.0,
        inference_seconds=30.0,
        peak_gpu_allocated_bytes=1_000,
        peak_gpu_reserved_bytes=2_000,
        peak_host_rss_bytes=3_000,
        gpu_hour_cost_usd=2.0,
    )
    assert measurement.real_time_factor == 0.5
    assert measurement.clips_per_minute == 18.0
    assert measurement.audio_minutes_per_minute == 2.0
    assert measurement.failure_rate == pytest.approx(0.1)
    assert measurement.estimated_daily_cost_usd(daily_audio_duration_ms=3_600_000) == 1.0

    hosted = OperationalMeasurementV1(
        **(
            measurement.model_dump(exclude_computed_fields=True)
            | {
                "gpu_hour_cost_usd": None,
                "audio_minute_cost_usd": 0.016,
                "hosted_provider": "google-cloud",
                "hosted_price_reference": "google-stt-v2-list-2026-08-20",
                "hosted_billing_policy": "per-request-ceil-1000ms-v1",
                "billable_audio_duration_ms": 61_000,
            }
        )
    )
    assert hosted.estimated_daily_cost_usd(daily_audio_duration_ms=3_600_000) == 0.976
    assert _billable_audio_duration_ms((100, 1_100), "exact-audio-duration-v1") == 1_200
    assert _billable_audio_duration_ms((100, 1_100), "per-request-ceil-1000ms-v1") == 3_000
    with pytest.raises(ValidationError, match="either GPU-hour or audio-minute"):
        OperationalMeasurementV1(
            **(
                measurement.model_dump(exclude_computed_fields=True)
                | {"audio_minute_cost_usd": 0.016}
            )
        )


def test_private_path_guard_rejects_sibling_or_installed_source_checkout(tmp_path: Path) -> None:
    sibling = tmp_path / "sibling-benchmarks"
    (sibling / "runner" / "src" / "coval_bench").mkdir(parents=True)
    (sibling / "runner" / "pyproject.toml").write_text("[project]\nname='invented'\n")
    with pytest.raises(ValueError, match="outside the repository"):
        validate_private_evidence_path(sibling / "private-results.json")


def test_candidate_blind_stt_selection_is_stratified_and_deterministic() -> None:
    items = tuple(
        _SelectionItem(
            path=f"audio/{index:04d}.wav",
            sha256=f"{index:064x}",
            transcript=(
                "Add milk to my list"
                if index % 2 == 0
                else "I was thinking about the same thing yesterday"
            ),
            duration_sec=3.0 if index < 6 else 9.0,
        )
        for index in range(12)
    )
    first = select_stratified_stt_items(items, sample_size=8, seed="invented-seed")
    second = select_stratified_stt_items(items, sample_size=8, seed="invented-seed")
    assert first == second
    strata = {clip_stratum(item) for item in first}
    assert {stratum.style for stratum in strata} == {
        "command_like",
        "conversational_or_question",
    }
    assert {stratum.duration for stratum in strata} == {"short", "long"}


def test_selection_gate_requires_provenance_success_runtime_and_determinism() -> None:
    candidate = _candidate()
    evidence = SelectionEvidenceV1(
        candidate_id=candidate.candidate_id,
        kind=candidate.kind,
        human_ground_truth_clips=50,
        manually_checked_in_domain_clips=50,
        artifact_success_rate=0.94,
        projected_daily_runtime_minutes=31.0,
        target_gpu="invented-gpu",
        target_gpu_benchmark_completed=False,
        weights_hashes_verified=False,
        deterministic_artifacts_verified=False,
        confidence_thresholds_calibrated=False,
        normalization_loss_rate=None,
    )
    assert selection_blockers(candidate, evidence) == (
        "target-GPU benchmark was not completed",
        "model asset hashes were not verified at runtime",
        "repeat runs did not verify deterministic artifact hashes",
        "agreement/confidence thresholds were not calibrated against human labels",
        "artifact success rate is below 95%",
        "projected daily runtime exceeds 30 minutes",
    )


def test_selection_gate_blocks_unlicensed_checkpoint_even_with_good_results() -> None:
    charsiu = next(candidate for candidate in CANDIDATES if "charsiu" in candidate.candidate_id)
    evidence = SelectionEvidenceV1(
        candidate_id=charsiu.candidate_id,
        kind=charsiu.kind,
        human_ground_truth_clips=50,
        manually_checked_in_domain_clips=50,
        artifact_success_rate=1.0,
        projected_daily_runtime_minutes=5.0,
        target_gpu="invented-gpu",
        target_gpu_benchmark_completed=True,
        weights_hashes_verified=True,
        deterministic_artifacts_verified=True,
        confidence_thresholds_calibrated=True,
        normalization_loss_rate=0.0,
    )
    assert selection_blockers(charsiu, evidence) == (
        "candidate license/provenance is not eligible for production use",
    )


def test_selection_gate_requires_real_human_and_in_domain_evidence() -> None:
    candidate = _candidate()
    evidence = SelectionEvidenceV1(
        candidate_id=candidate.candidate_id,
        kind=candidate.kind,
        human_ground_truth_clips=0,
        manually_checked_in_domain_clips=0,
        artifact_success_rate=1.0,
        projected_daily_runtime_minutes=None,
        target_gpu="not-run",
        target_gpu_benchmark_completed=False,
        weights_hashes_verified=True,
        deterministic_artifacts_verified=True,
        confidence_thresholds_calibrated=False,
        normalization_loss_rate=None,
    )
    assert selection_blockers(candidate, evidence) == (
        "human-ground-truth clips are below 1",
        "manually checked in-domain clips are below 50",
        "target-GPU benchmark was not completed",
        "agreement/confidence thresholds were not calibrated against human labels",
        "target-GPU daily runtime projection is unavailable",
    )


def test_selection_gate_rejects_candidate_that_is_not_benchmark_eligible() -> None:
    candidate = _candidate(benchmark_eligible=False)
    evidence = SelectionEvidenceV1(
        candidate_id=candidate.candidate_id,
        kind=candidate.kind,
        human_ground_truth_clips=50,
        manually_checked_in_domain_clips=50,
        artifact_success_rate=1.0,
        projected_daily_runtime_minutes=5.0,
        target_gpu="invented-gpu",
        target_gpu_benchmark_completed=True,
        weights_hashes_verified=True,
        deterministic_artifacts_verified=True,
        confidence_thresholds_calibrated=True,
        normalization_loss_rate=None,
    )
    assert selection_blockers(candidate, evidence) == (
        "candidate is not eligible for benchmark evaluation",
    )


def test_hosted_selection_uses_reproducibility_evidence_instead_of_false_weight_hashes() -> None:
    google = next(
        candidate
        for candidate in CANDIDATES
        if candidate.candidate_id == "word-google-chirp-3-hosted-v1"
    ).model_copy(update={"license_eligible_for_production": True})
    evidence = SelectionEvidenceV1(
        candidate_id=google.candidate_id,
        kind=google.kind,
        human_ground_truth_clips=50,
        manually_checked_in_domain_clips=50,
        artifact_success_rate=1.0,
        projected_daily_runtime_minutes=5.0,
        target_gpu=None,
        target_gpu_benchmark_completed=None,
        weights_hashes_verified=None,
        deterministic_artifacts_verified=True,
        confidence_thresholds_calibrated=True,
        normalization_loss_rate=None,
        hosted_reproducibility_policy_approved=True,
        provider_revision_drift_handling_verified=True,
        raw_response_hashes_recorded=True,
    )
    assert selection_blockers(google, evidence) == ()
    incomplete = evidence.model_copy(update={"raw_response_hashes_recorded": False})
    assert selection_blockers(google, incomplete) == (
        "private raw-response hashes were not recorded",
    )


@pytest.mark.parametrize("path", [Path("audio.wav"), Path("private/audio.wav")])
def test_protocol_examples_accept_paths_without_reading_audio(path: Path) -> None:
    assert isinstance(path, Path)


def test_bundle_requires_exactly_one_outcome_per_declared_candidate() -> None:
    with pytest.raises(ValidationError, match="one outcome"):
        TimestampBenchmarkBundleV1(
            benchmark_id="invented-bundle",
            dataset_id="invented-data",
            mode=BenchmarkMode.HUMAN_GROUND_TRUTH,
            kind=BenchmarkCandidateKind.WORD_ALIGNER,
            fixture_only=True,
            candidate_ids=("invented-candidate-a", "invented-candidate-b"),
            candidate_specs=(
                _invented_word_candidate("invented-candidate-a"),
                _invented_word_candidate("invented-candidate-b"),
            ),
            cases=(
                WordGroundTruthCaseV1(
                    reference=_word_ground_truth(),
                    outputs=(
                        WordCandidateOutputV1(
                            candidate_id="invented-candidate-a",
                            freely_predicted_words=False,
                            artifact=_word_artifact((_word("hello", 100, 300),)),
                        ),
                    ),
                ),
            ),
        )


def test_bundle_rejects_spoofed_registry_spec_duplicate_cases_and_forced_wer() -> None:
    phone_candidate = next(
        candidate
        for candidate in CANDIDATES
        if candidate.candidate_id == "phone-meta-espeak-ctc-midpoint-v1"
    )
    failure_case = PhonemeAgreementCaseV1(
        analysis_id="invented-001",
        audio_sha256=SOURCE_SHA,
        duration_ms=1_000,
        outputs=(),
        failures=(
            CandidateFailureV1(
                candidate_id=phone_candidate.candidate_id,
                kind=FailureKind.FAILURE,
                code="invented-failure",
            ),
        ),
    )
    with pytest.raises(ValidationError, match="public candidate registry"):
        TimestampBenchmarkBundleV1(
            benchmark_id="real-benchmark",
            dataset_id="private-dataset",
            mode=BenchmarkMode.AGREEMENT,
            kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER,
            candidate_ids=(phone_candidate.candidate_id,),
            candidate_specs=(
                phone_candidate.model_copy(update={"model_revision": "spoofed-revision"}),
            ),
            cases=(failure_case,),
        )

    invented_id = "invented-phone"
    with pytest.raises(ValidationError, match="analysis_id values must be unique"):
        TimestampBenchmarkBundleV1(
            benchmark_id="invented-duplicates",
            dataset_id="invented-audio",
            mode=BenchmarkMode.AGREEMENT,
            kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER,
            fixture_only=True,
            candidate_ids=(invented_id,),
            candidate_specs=(_invented_phone_candidate(invented_id),),
            cases=(
                failure_case.model_copy(
                    update={
                        "failures": (
                            CandidateFailureV1(
                                candidate_id=invented_id,
                                kind=FailureKind.FAILURE,
                                code="invented-failure",
                            ),
                        )
                    }
                ),
                failure_case.model_copy(
                    update={
                        "failures": (
                            CandidateFailureV1(
                                candidate_id=invented_id,
                                kind=FailureKind.FAILURE,
                                code="invented-failure",
                            ),
                        )
                    }
                ),
            ),
        )

    word_candidate = next(
        candidate
        for candidate in CANDIDATES
        if candidate.candidate_id == "word-wav2vec2-base-960h-ctc-v1"
    )
    artifact = _word_artifact((_word("hello", 100, 300),)).model_copy(
        update={
            "processor": WordProcessorProvenanceV1(
                aligner_name=word_candidate.model_name,
                aligner_revision=word_candidate.model_revision,
                normalization_version=word_candidate.normalization_version,
            )
        }
    )
    with pytest.raises(ValidationError, match="word artifact processor"):
        TimestampBenchmarkBundleV1(
            benchmark_id="real-forced-word",
            dataset_id="private-dataset",
            mode=BenchmarkMode.HUMAN_GROUND_TRUTH,
            kind=BenchmarkCandidateKind.WORD_ALIGNER,
            candidate_ids=(word_candidate.candidate_id,),
            candidate_specs=(word_candidate,),
            cases=(
                WordGroundTruthCaseV1(
                    reference=_word_ground_truth(),
                    outputs=(
                        WordCandidateOutputV1(
                            candidate_id=word_candidate.candidate_id,
                            freely_predicted_words=True,
                            artifact=artifact,
                        ),
                    ),
                ),
            ),
        )


def test_ground_truth_case_rejects_mismatched_artifact_identity() -> None:
    with pytest.raises(ValidationError, match="analysis_id does not match"):
        WordGroundTruthCaseV1(
            reference=_word_ground_truth(),
            outputs=(
                WordCandidateOutputV1(
                    candidate_id="invented-word",
                    freely_predicted_words=False,
                    artifact=_word_artifact(
                        (_word("hello", 100, 300),),
                        analysis_id="different-analysis",
                    ),
                ),
            ),
        )


def test_isolated_candidate_bundles_merge_without_mixing_model_runtimes() -> None:
    left = TimestampBenchmarkBundleV1(
        benchmark_id="invented-left",
        dataset_id="invented-audio",
        mode=BenchmarkMode.AGREEMENT,
        kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER,
        fixture_only=True,
        candidate_ids=("invented-candidate-a",),
        candidate_specs=(_invented_phone_candidate("invented-candidate-a"),),
        cases=(
            PhonemeAgreementCaseV1(
                analysis_id="invented-001",
                audio_sha256=SOURCE_SHA,
                duration_ms=1_000,
                outputs=(
                    PhonemeCandidateOutputV1(
                        candidate_id="invented-candidate-a",
                        normalization_loss_rate=0.0,
                        artifact=_phone_artifact((_phone("HH", 100, 200),)),
                    ),
                ),
            ),
        ),
    )
    right = TimestampBenchmarkBundleV1(
        benchmark_id="invented-right",
        dataset_id="invented-audio",
        mode=BenchmarkMode.AGREEMENT,
        kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER,
        fixture_only=True,
        candidate_ids=("invented-candidate-b",),
        candidate_specs=(_invented_phone_candidate("invented-candidate-b"),),
        cases=(
            PhonemeAgreementCaseV1(
                analysis_id="invented-001",
                audio_sha256=SOURCE_SHA,
                duration_ms=1_000,
                outputs=(
                    PhonemeCandidateOutputV1(
                        candidate_id="invented-candidate-b",
                        normalization_loss_rate=0.0,
                        artifact=_phone_artifact((_phone("HH", 110, 210),)),
                    ),
                ),
            ),
        ),
    )
    merged = merge_benchmark_bundles(
        (left, right),
        benchmark_id="invented-merged",
    )
    assert merged.candidate_ids == ("invented-candidate-a", "invented-candidate-b")
    assert len(merged.cases[0].outputs) == 2
    assert len(build_report(merged).observations) == 1
    subset = select_candidate_subset(
        merged,
        candidate_ids=("invented-candidate-b",),
        benchmark_id="invented-subset",
    )
    assert subset.candidate_ids == ("invented-candidate-b",)
    assert tuple(output.candidate_id for output in subset.cases[0].outputs) == (
        "invented-candidate-b",
    )

    different_sha = "d" * 64
    right_output = right.cases[0].outputs[0]
    different_right = right.model_copy(
        update={
            "cases": (
                PhonemeAgreementCaseV1(
                    analysis_id="invented-001",
                    audio_sha256=different_sha,
                    duration_ms=1_000,
                    outputs=(
                        right_output.model_copy(
                            update={
                                "artifact": right_output.artifact.model_copy(
                                    update={"audio_sha256": different_sha}
                                )
                            }
                        ),
                    ),
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="identical agreement audio"):
        merge_benchmark_bundles(
            (left, different_right),
            benchmark_id="invented-mismatched-audio",
        )


def test_report_canonicalizes_agreement_output_order_by_candidate_registry() -> None:
    candidate_ids = ("invented-phone-a", "invented-phone-b")

    def output(candidate_id: str, analysis_id: str, offset_ms: int) -> PhonemeCandidateOutputV1:
        return PhonemeCandidateOutputV1(
            candidate_id=candidate_id,
            normalization_loss_rate=0.0,
            artifact=_phone_artifact(
                (_phone("HH", 100 + offset_ms, 200 + offset_ms),),
                analysis_id=analysis_id,
            ),
        )

    first = (
        output(candidate_ids[0], "invented-001", 0),
        output(candidate_ids[1], "invented-001", 10),
    )
    second = (
        output(candidate_ids[1], "invented-002", 10),
        output(candidate_ids[0], "invented-002", 0),
    )
    bundle = TimestampBenchmarkBundleV1(
        benchmark_id="invented-output-order",
        dataset_id="invented-audio",
        mode=BenchmarkMode.AGREEMENT,
        kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER,
        fixture_only=True,
        candidate_ids=candidate_ids,
        candidate_specs=tuple(_invented_phone_candidate(value) for value in candidate_ids),
        cases=(
            PhonemeAgreementCaseV1(
                analysis_id="invented-001",
                audio_sha256=SOURCE_SHA,
                duration_ms=1_000,
                outputs=first,
            ),
            PhonemeAgreementCaseV1(
                analysis_id="invented-002",
                audio_sha256=SOURCE_SHA,
                duration_ms=1_000,
                outputs=second,
            ),
        ),
    )
    report = build_report(bundle)
    assert len(report.metric_summaries) == 1
    assert {
        (observation.left_candidate_id, observation.right_candidate_id)
        for observation in report.observations
    } == {candidate_ids}


def test_build_report_aggregates_outcomes_without_renaming_agreement_accuracy() -> None:
    left = PhonemeCandidateOutputV1(
        candidate_id="phone-a",
        normalization_loss_rate=0.0,
        artifact=_phone_artifact((_phone("HH", 100, 200), _phone("AH", 200, 350))),
    )
    right = PhonemeCandidateOutputV1(
        candidate_id="phone-b",
        normalization_loss_rate=0.0,
        artifact=_phone_artifact((_phone("HH", 110, 210), _phone("EH", 210, 340))),
    )
    bundle = TimestampBenchmarkBundleV1(
        benchmark_id="invented-agreement",
        dataset_id="invented-audio",
        mode=BenchmarkMode.AGREEMENT,
        kind=BenchmarkCandidateKind.PHONEME_RECOGNIZER,
        fixture_only=True,
        candidate_ids=("invented-phone-a", "invented-phone-b"),
        candidate_specs=(
            _invented_phone_candidate("invented-phone-a"),
            _invented_phone_candidate("invented-phone-b"),
        ),
        cases=(
            PhonemeAgreementCaseV1(
                analysis_id="invented-001",
                audio_sha256=SOURCE_SHA,
                duration_ms=1_000,
                outputs=(
                    left.model_copy(update={"candidate_id": "invented-phone-a"}),
                    right.model_copy(update={"candidate_id": "invented-phone-b"}),
                ),
            ),
            PhonemeAgreementCaseV1(
                analysis_id="invented-002",
                audio_sha256=SOURCE_SHA,
                duration_ms=1_000,
                outputs=(
                    PhonemeCandidateOutputV1(
                        candidate_id="invented-phone-a",
                        normalization_loss_rate=0.0,
                        artifact=_phone_artifact(
                            (_phone("HH", 100, 200),), analysis_id="invented-002"
                        ),
                    ),
                ),
                failures=(
                    CandidateFailureV1(
                        candidate_id="invented-phone-b",
                        kind=FailureKind.TIMEOUT,
                        code="invented-timeout",
                    ),
                ),
            ),
        ),
    )
    report = build_report(bundle)
    assert len(report.observations) == 1
    assert report.observations[0].schema_version == "PhonemeAgreementObservationV1"
    assert report.outcomes[0].artifact_success_rate == 1.0
    assert report.outcomes[0].mean_timeline_coverage == pytest.approx(0.175)
    assert report.outcomes[0].mean_normalization_loss_rate == 0.0
    assert report.outcomes[1].artifact_success_rate == 0.5
    assert report.outcomes[1].timeouts == 1
    assert report.pair_outcomes[0].both_succeeded == 1
    assert report.pair_outcomes[0].only_left_succeeded == 1
    assert report.pair_outcomes[0].only_right_succeeded == 0
    assert report.pair_outcomes[0].both_failed == 0
    assert report.pair_outcomes[0].failure_disagreement_rate == 0.5
    assert report.candidate_specs == bundle.candidate_specs
    serialized = report.model_dump_json()
    assert "phone_disagreement_rate" in serialized
    assert '"accuracy"' not in serialized


def test_cli_validates_invented_fixture_and_writes_private_report(tmp_path: Path) -> None:
    fixture = (
        Path(__file__).parents[1]
        / "fixtures"
        / "preprocessing"
        / "benchmarking"
        / "invented-word-ground-truth.json"
    )
    runner = CliRunner()
    validation = runner.invoke(timestamp_benchmark, ["validate", "--input", str(fixture)])
    assert validation.exit_code == 0, validation.output
    assert json.loads(validation.output)["cases"] == 1

    output = tmp_path / "report.json"
    summary = runner.invoke(
        timestamp_benchmark,
        ["summarize", "--input", str(fixture), "--output", str(output)],
    )
    assert summary.exit_code == 0, summary.output
    report = json.loads(output.read_text())
    assert report["mode"] == "human_ground_truth"
    assert report["observations"][0]["metrics"]["wer_applicable"] is False


def test_private_evidence_path_allows_only_the_exact_fixture_as_input() -> None:
    repository_root = Path(__file__).parents[3]
    fixture = (
        repository_root
        / "runner"
        / "tests"
        / "fixtures"
        / "preprocessing"
        / "benchmarking"
        / "invented-word-ground-truth.json"
    )
    assert (
        validate_private_evidence_path(
            fixture,
            allow_invented_fixture=True,
        )
        == fixture.resolve()
    )
    for repository_path in (
        Path(__file__),
        repository_root / "docs" / "timestamp-model-benchmark.md",
        fixture,
        fixture.parent / "real-output.json",
    ):
        with pytest.raises(ValueError, match="outside the repository"):
            validate_private_evidence_path(repository_path)
    with pytest.raises(ValueError, match="outside the repository"):
        validate_private_evidence_path(
            fixture.parent / "real-output.json",
            allow_invented_fixture=True,
        )
