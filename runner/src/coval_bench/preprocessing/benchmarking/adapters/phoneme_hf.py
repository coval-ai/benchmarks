# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Audio-only Hugging Face phoneme candidates and deterministic decoders."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from coval_bench.preprocessing.benchmarking.adapters.base import source_duration_ms
from coval_bench.preprocessing.benchmarking.candidates import (
    CANDIDATES,
    candidate_processor_revision,
)
from coval_bench.preprocessing.benchmarking.contracts import (
    BenchmarkCandidateKind,
    CandidateSpecV1,
)
from coval_bench.preprocessing.benchmarking.inventory import (
    COVAL_ENGLISH_PHONES_V1,
    PhoneSource,
    normalize_phone_sequence,
)
from coval_bench.preprocessing.contracts import (
    PhonemeProcessorProvenanceV1,
    PhonemeTimestampsV1,
    PhonemeTimestampV1,
    TimestampWarningCode,
    TimestampWarningV1,
)

TARGET_SAMPLE_RATE = 16_000
CHARSIU_TOKENIZER_NAME = "charsiu/tokenizer_en_cmu"
CHARSIU_TOKENIZER_REVISION = "10507401aedf5e0aba164128535b49225ff95260"

PhoneDecoder = Literal["sparse", "midpoint_fill", "frame_collapse"]


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "missing"


def _runtime_software() -> str:
    packages = (
        "torch",
        "transformers",
        "numpy",
        "scipy",
        "soundfile",
        "huggingface-hub",
    )
    return "_".join(f"{package}-{_installed_version(package)}" for package in packages)


@dataclass(frozen=True, slots=True)
class RawPhoneSpan:
    """One source-vocabulary span over model output frames."""

    symbol: str
    start_frame: float
    end_frame: float
    confidence: float


@dataclass(frozen=True, slots=True)
class DecodedPhoneSequence:
    """Normalized spans plus the loss introduced by inventory conversion."""

    spans: tuple[RawPhoneSpan, ...]
    normalization_loss_rate: float


def _frame_runs(
    frame_ids: tuple[int, ...],
    frame_confidences: tuple[float, ...],
    *,
    id_to_token: dict[int, str],
) -> tuple[tuple[int, int, int, str, float], ...]:
    if not frame_ids:
        raise ValueError("frame_ids must not be empty")
    if len(frame_ids) != len(frame_confidences):
        raise ValueError("frame_ids and frame_confidences must have equal lengths")
    runs: list[tuple[int, int, int, str, float]] = []
    start = 0
    for end in range(1, len(frame_ids) + 1):
        if end < len(frame_ids) and frame_ids[end] == frame_ids[start]:
            continue
        token_id = frame_ids[start]
        if token_id not in id_to_token:
            raise ValueError(f"model output token id {token_id} is absent from the vocabulary")
        runs.append(
            (
                token_id,
                start,
                end,
                id_to_token[token_id],
                float(fmean(frame_confidences[start:end])),
            )
        )
        start = end
    return tuple(runs)


def decode_phone_frames(
    frame_ids: tuple[int, ...],
    frame_confidences: tuple[float, ...],
    *,
    id_to_token: dict[int, str],
    decoder: PhoneDecoder,
    blank_id: int | None,
) -> tuple[RawPhoneSpan, ...]:
    """Decode model frames without using a transcript or expected phone sequence."""
    runs = _frame_runs(
        frame_ids,
        frame_confidences,
        id_to_token=id_to_token,
    )
    if decoder == "frame_collapse":
        if blank_id is not None:
            raise ValueError("frame_collapse must not declare a CTC blank id")
        return tuple(
            RawPhoneSpan(
                symbol=symbol,
                start_frame=float(start),
                end_frame=float(end),
                confidence=confidence,
            )
            for _, start, end, symbol, confidence in runs
        )
    if blank_id is None:
        raise ValueError("CTC decoders require blank_id")
    speech_runs = [run for run in runs if run[0] != blank_id]
    if decoder == "sparse" or len(speech_runs) < 2:
        return tuple(
            RawPhoneSpan(
                symbol=symbol,
                start_frame=float(start),
                end_frame=float(end),
                confidence=confidence,
            )
            for _, start, end, symbol, confidence in speech_runs
        )

    spans = [
        RawPhoneSpan(
            symbol=symbol,
            start_frame=float(start),
            end_frame=float(end),
            confidence=confidence,
        )
        for _, start, end, symbol, confidence in speech_runs
    ]
    for index in range(len(spans) - 1):
        left = spans[index]
        right = spans[index + 1]
        midpoint = (left.end_frame + right.start_frame) / 2
        spans[index] = RawPhoneSpan(
            symbol=left.symbol,
            start_frame=left.start_frame,
            end_frame=midpoint,
            confidence=left.confidence,
        )
        spans[index + 1] = RawPhoneSpan(
            symbol=right.symbol,
            start_frame=midpoint,
            end_frame=right.end_frame,
            confidence=right.confidence,
        )
    return tuple(spans)


def normalize_phone_spans(
    spans: tuple[RawPhoneSpan, ...],
    *,
    source: PhoneSource,
) -> DecodedPhoneSequence:
    """Normalize source labels and split multi-phone mappings over the same time span."""
    normalized_spans: list[RawPhoneSpan] = []
    source_symbols = tuple(span.symbol for span in spans)
    normalization = normalize_phone_sequence(source_symbols, source=source)
    for span in spans:
        mapped = normalize_phone_sequence((span.symbol,), source=source).symbols
        if not mapped:
            continue
        frame_width = (span.end_frame - span.start_frame) / len(mapped)
        for index, symbol in enumerate(mapped):
            normalized_spans.append(
                RawPhoneSpan(
                    symbol=symbol,
                    start_frame=span.start_frame + index * frame_width,
                    end_frame=span.start_frame + (index + 1) * frame_width,
                    confidence=span.confidence,
                )
            )
    return DecodedPhoneSequence(
        spans=tuple(normalized_spans),
        normalization_loss_rate=normalization.loss_rate,
    )


def build_phoneme_artifact(
    *,
    analysis_id: str,
    audio_sha256: str,
    duration_ms: int,
    frame_count: int,
    candidate: CandidateSpecV1,
    decoded: DecodedPhoneSequence,
) -> PhonemeTimestampsV1:
    """Convert decoded model frames into the merged strict artifact schema."""
    if candidate.kind is not BenchmarkCandidateKind.PHONEME_RECOGNIZER:
        raise ValueError("candidate must be a phoneme recognizer")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    weights = next(
        (
            asset
            for asset in candidate.assets
            if asset.path.endswith(("pytorch_model.bin", "allophant.pt"))
        ),
        None,
    )
    if weights is None:
        raise ValueError("candidate has no pinned model-weight asset")
    phones: list[PhonemeTimestampV1] = []
    dropped_zero_width = False
    for span in decoded.spans:
        start_ms = round(span.start_frame * duration_ms / frame_count)
        end_ms = round(span.end_frame * duration_ms / frame_count)
        if end_ms <= start_ms:
            dropped_zero_width = True
            continue
        phones.append(
            PhonemeTimestampV1(
                symbol=span.symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=span.confidence,
            )
        )
    warnings: list[TimestampWarningV1] = []
    if not phones:
        warnings.append(
            TimestampWarningV1(
                code=TimestampWarningCode.EMPTY_SPANS,
                message="phoneme decoder produced no normalized speech-phone spans",
            )
        )
    elif dropped_zero_width:
        warnings.append(
            TimestampWarningV1(
                code=TimestampWarningCode.PARTIAL_ALIGNMENT,
                message="sub-millisecond phone spans were omitted during quantization",
            )
        )
    return PhonemeTimestampsV1(
        schema_version="PhonemeTimestampsV1",
        analysis_id=analysis_id,
        audio_sha256=audio_sha256,
        duration_ms=duration_ms,
        processor=PhonemeProcessorProvenanceV1(
            model_name=candidate.model_name,
            model_revision=candidate_processor_revision(candidate),
            weights_sha256=weights.sha256,
            phone_inventory=COVAL_ENGLISH_PHONES_V1,
            resampler=candidate.resampler,
            decoder=candidate.decoder,
        ),
        phones=tuple(phones),
        warnings=tuple(warnings),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate(candidate_id: str) -> CandidateSpecV1:
    try:
        return next(candidate for candidate in CANDIDATES if candidate.candidate_id == candidate_id)
    except StopIteration as error:
        raise ValueError(f"unknown candidate_id {candidate_id!r}") from error


class HuggingFacePhonemeRecognizer:
    """Lazy heavy-runtime adapter; the public protocol exposes audio only."""

    def __init__(
        self,
        *,
        candidate_id: str,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        candidate = _candidate(candidate_id)
        if candidate.kind is not BenchmarkCandidateKind.PHONEME_RECOGNIZER:
            raise ValueError("candidate_id must identify a phoneme recognizer")
        self.candidate = candidate
        self.device = device
        self.local_files_only = local_files_only
        self.model_load_seconds = 0.0
        self.last_inference_seconds = 0.0
        self.last_normalization_loss_rate = 0.0
        self.runtime_software = _runtime_software()
        self._model: Any = None
        self._feature_extractor: Any = None
        self._id_to_token: dict[int, str] = {}
        self._blank_id: int | None = None
        self._decoder: PhoneDecoder
        self._source: PhoneSource

    def _load(self) -> None:
        if self._model is not None:
            return
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        started = time.perf_counter()
        weight_asset = next(
            asset for asset in self.candidate.assets if asset.path.endswith("pytorch_model.bin")
        )
        weight_path = Path(
            hf_hub_download(
                repo_id=self.candidate.model_name,
                filename="pytorch_model.bin",
                revision=self.candidate.model_revision,
                local_files_only=self.local_files_only,
            )
        )
        if _sha256(weight_path) != weight_asset.sha256:
            raise ValueError("downloaded model weights do not match the candidate SHA-256")
        self._feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=TARGET_SAMPLE_RATE,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )
        model: Any = Wav2Vec2ForCTC.from_pretrained(
            self.candidate.model_name,
            revision=self.candidate.model_revision,
            local_files_only=self.local_files_only,
        )
        self._model = model.to(self.device).eval()
        if self.candidate.candidate_id.startswith("phone-charsiu-"):
            tokenizer_vocab_path = Path(
                hf_hub_download(
                    repo_id=CHARSIU_TOKENIZER_NAME,
                    filename="vocab.json",
                    revision=CHARSIU_TOKENIZER_REVISION,
                    local_files_only=self.local_files_only,
                )
            )
            tokenizer_asset = next(
                asset
                for asset in self.candidate.assets
                if asset.path.startswith("charsiu/tokenizer_en_cmu@")
            )
            if _sha256(tokenizer_vocab_path) != tokenizer_asset.sha256:
                raise ValueError("downloaded Charsiu tokenizer does not match its SHA-256")
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                CHARSIU_TOKENIZER_NAME,
                revision=CHARSIU_TOKENIZER_REVISION,
                local_files_only=self.local_files_only,
            )
            self._decoder = "frame_collapse"
            self._source = PhoneSource.CHARSIU_CMU
            self._blank_id = None
        else:
            vocab_path = Path(
                hf_hub_download(
                    repo_id=self.candidate.model_name,
                    filename="vocab.json",
                    revision=self.candidate.model_revision,
                    local_files_only=self.local_files_only,
                )
            )
            vocab_asset = next(
                asset for asset in self.candidate.assets if asset.path == "vocab.json"
            )
            if _sha256(vocab_path) != vocab_asset.sha256:
                raise ValueError("downloaded model vocabulary does not match its SHA-256")
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            self._id_to_token = {int(token_id): str(token) for token, token_id in vocab.items()}
            self._decoder = (
                "midpoint_fill" if "midpoint" in self.candidate.candidate_id else "sparse"
            )
            self._source = PhoneSource.META_ESPEAK_IPA
            self._blank_id = int(self._model.config.pad_token_id)
            self.model_load_seconds = time.perf_counter() - started
            return
        self._id_to_token = {
            token_id: str(token)
            for token_id, token in enumerate(tokenizer.convert_ids_to_tokens(range(len(tokenizer))))
        }
        self.model_load_seconds = time.perf_counter() - started

    def recognize(self, *, audio_path: Path) -> PhonemeTimestampsV1:
        """Infer phone identities and boundaries from audio without reference text."""
        self._load()
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if samples.ndim != 1:
            samples = samples.mean(axis=1)
        duration_ms = source_duration_ms(sample_count=len(samples), sample_rate=sample_rate)
        if sample_rate != TARGET_SAMPLE_RATE:
            divisor = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
            samples = resample_poly(
                samples,
                TARGET_SAMPLE_RATE // divisor,
                sample_rate // divisor,
            ).astype(np.float32)
        inputs = self._feature_extractor(
            samples,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)
        import torch

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(input_values).logits[0]
            probabilities = logits.softmax(dim=-1)
            frame_confidences, frame_ids = probabilities.max(dim=-1)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.last_inference_seconds = time.perf_counter() - started
        raw_spans = decode_phone_frames(
            tuple(int(value) for value in frame_ids.cpu().tolist()),
            tuple(float(value) for value in frame_confidences.cpu().tolist()),
            id_to_token=self._id_to_token,
            decoder=self._decoder,
            blank_id=self._blank_id,
        )
        decoded = normalize_phone_spans(raw_spans, source=self._source)
        self.last_normalization_loss_rate = decoded.normalization_loss_rate
        return build_phoneme_artifact(
            analysis_id=audio_path.stem,
            audio_sha256=_sha256(audio_path),
            duration_ms=duration_ms,
            frame_count=len(frame_ids),
            candidate=self.candidate,
            decoded=decoded,
        )
