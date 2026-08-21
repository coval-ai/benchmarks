# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Known-transcript CTC word alignment with an immutable model candidate."""

from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any

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
)
from coval_bench.preprocessing.benchmarking.metrics import (
    normalize_word,
)
from coval_bench.preprocessing.contracts import (
    WordProcessorProvenanceV1,
    WordTimestampsV1,
    WordTimestampV1,
)

TARGET_SAMPLE_RATE = 16_000


class CTCAlignmentError(ValueError):
    """The pinned CTC vocabulary cannot produce a valid forced alignment."""


@dataclass(frozen=True, slots=True)
class CTCTokenSpan:
    symbol: str
    start_frame: int
    end_frame: int
    confidence: float


def normalize_ctc_transcript(transcript: str) -> tuple[str, ...]:
    """Apply the public word normalization independently to whitespace tokens."""
    normalized = unicodedata.normalize("NFKC", transcript).replace("’", "'")
    words = tuple(filter(None, (normalize_word(word) for word in normalized.split())))
    if not words:
        raise CTCAlignmentError("transcript contains no alignable words")
    return words


def ctc_transcript_token_ids(
    transcript: str,
    *,
    vocabulary: dict[str, int],
    delimiter: str = "|",
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    """Map normalized known text into the exact CTC vocabulary or fail explicitly."""
    words = normalize_ctc_transcript(transcript)
    symbols = tuple(delimiter.join(words).upper())
    unknown = sorted({symbol for symbol in symbols if symbol not in vocabulary})
    if unknown:
        raise CTCAlignmentError(
            "normalized transcript contains symbols outside the candidate vocabulary: "
            + ", ".join(unknown)
        )
    return words, symbols, tuple(vocabulary[symbol] for symbol in symbols)


def force_align_ctc(
    log_probabilities: np.ndarray,
    *,
    token_symbols: tuple[str, ...],
    token_ids: tuple[int, ...],
    blank_id: int,
) -> tuple[CTCTokenSpan, ...]:
    """Viterbi-align tokens with the standard blank-expanded CTC state graph."""
    if log_probabilities.ndim != 2:
        raise ValueError("log_probabilities must have shape [frames, vocabulary]")
    if len(token_symbols) != len(token_ids) or not token_ids:
        raise ValueError("token symbols and ids must be nonempty and have equal lengths")
    frame_count, vocabulary_size = log_probabilities.shape
    if frame_count <= 0:
        raise ValueError("log_probabilities must contain at least one frame")
    if blank_id < 0 or blank_id >= vocabulary_size:
        raise ValueError("blank_id is outside the emission vocabulary")
    if any(token_id < 0 or token_id >= vocabulary_size for token_id in token_ids):
        raise ValueError("token id is outside the emission vocabulary")

    state_count = 2 * len(token_ids) + 1
    state_tokens = np.full(state_count, blank_id, dtype=np.int64)
    state_tokens[1::2] = token_ids
    scores = np.full((frame_count, state_count), -np.inf, dtype=np.float64)
    predecessors = np.full((frame_count, state_count), -1, dtype=np.int32)
    scores[0, 0] = float(log_probabilities[0, blank_id])
    scores[0, 1] = float(log_probabilities[0, token_ids[0]])

    for frame in range(1, frame_count):
        for state in range(state_count):
            candidates = [(scores[frame - 1, state], state)]
            if state > 0:
                candidates.append((scores[frame - 1, state - 1], state - 1))
            if state > 1 and state % 2 == 1 and state_tokens[state] != state_tokens[state - 2]:
                candidates.append((scores[frame - 1, state - 2], state - 2))
            previous_score, previous_state = max(candidates, key=lambda item: (item[0], item[1]))
            scores[frame, state] = previous_score + float(
                log_probabilities[frame, state_tokens[state]]
            )
            predecessors[frame, state] = previous_state

    final_states = (state_count - 2, state_count - 1)
    final_state = max(final_states, key=lambda state: scores[-1, state])
    if not np.isfinite(scores[-1, final_state]):
        raise CTCAlignmentError("emissions are too short to align the normalized transcript")
    path = [final_state]
    for frame in range(frame_count - 1, 0, -1):
        final_state = int(predecessors[frame, final_state])
        if final_state < 0:
            raise CTCAlignmentError("CTC backtracking reached an invalid predecessor")
        path.append(final_state)
    path.reverse()

    spans: list[CTCTokenSpan] = []
    for token_index, (symbol, token_id) in enumerate(zip(token_symbols, token_ids, strict=True)):
        token_state = 2 * token_index + 1
        frames = [frame for frame, state in enumerate(path) if state == token_state]
        if not frames:
            raise CTCAlignmentError(f"token {token_index} has no aligned emission frame")
        probabilities = [float(np.exp(log_probabilities[frame, token_id])) for frame in frames]
        spans.append(
            CTCTokenSpan(
                symbol=symbol,
                start_frame=frames[0],
                end_frame=frames[-1] + 1,
                confidence=float(fmean(probabilities)),
            )
        )
    return tuple(spans)


def token_spans_to_words(
    words: tuple[str, ...],
    token_spans: tuple[CTCTokenSpan, ...],
    *,
    frame_count: int,
    duration_ms: int,
    delimiter: str = "|",
) -> tuple[WordTimestampV1, ...]:
    """Group aligned character emissions into normalized word spans."""
    grouped: list[list[CTCTokenSpan]] = [[]]
    for span in token_spans:
        if span.symbol == delimiter:
            grouped.append([])
        else:
            grouped[-1].append(span)
    if len(grouped) != len(words) or any(not group for group in grouped):
        raise CTCAlignmentError("aligned token delimiters do not match normalized words")
    output: list[WordTimestampV1] = []
    for word, characters in zip(words, grouped, strict=True):
        start_ms = round(characters[0].start_frame * duration_ms / frame_count)
        end_ms = round(characters[-1].end_frame * duration_ms / frame_count)
        if end_ms <= start_ms:
            raise CTCAlignmentError("word alignment collapsed during millisecond quantization")
        output.append(
            WordTimestampV1(
                text=word,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=float(fmean(character.confidence for character in characters)),
            )
        )
    return tuple(output)


def _sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HuggingFaceCTCWordAligner:
    """Pinned direct CTC aligner; accepts known transcript by design."""

    def __init__(self, *, device: str = "cpu", local_files_only: bool = False) -> None:
        self.candidate = next(
            candidate
            for candidate in CANDIDATES
            if candidate.candidate_id == "word-wav2vec2-base-960h-ctc-v1"
        )
        if self.candidate.kind is not BenchmarkCandidateKind.WORD_ALIGNER:
            raise RuntimeError("word CTC candidate registry entry has the wrong kind")
        self.device = device
        self.local_files_only = local_files_only
        self.model_load_seconds = 0.0
        self.last_inference_seconds = 0.0
        self._model: Any = None
        self._feature_extractor: Any = None
        self._vocabulary: dict[str, int] = {}

    @property
    def runtime_software(self) -> str:
        return (
            f"torch=={version('torch')};transformers=={version('transformers')};"
            f"huggingface-hub=={version('huggingface-hub')}"
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        started = time.perf_counter()
        weight_asset = next(
            asset for asset in self.candidate.assets if asset.path == "pytorch_model.bin"
        )
        weight_path = Path(
            hf_hub_download(
                repo_id=self.candidate.model_name,
                filename=weight_asset.path,
                revision=self.candidate.model_revision,
                local_files_only=self.local_files_only,
            )
        )
        if _sha256_file(weight_path) != weight_asset.sha256:
            raise ValueError("downloaded word model weights do not match the candidate SHA-256")
        vocab_path = Path(
            hf_hub_download(
                repo_id=self.candidate.model_name,
                filename="vocab.json",
                revision=self.candidate.model_revision,
                local_files_only=self.local_files_only,
            )
        )
        vocab_asset = next(asset for asset in self.candidate.assets if asset.path == "vocab.json")
        if _sha256_file(vocab_path) != vocab_asset.sha256:
            raise ValueError(
                "downloaded word model vocabulary does not match the candidate SHA-256"
            )
        vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
        self._vocabulary = {str(symbol): int(token_id) for symbol, token_id in vocabulary.items()}
        self._feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=TARGET_SAMPLE_RATE,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )
        self._model = (
            Wav2Vec2ForCTC.from_pretrained(
                self.candidate.model_name,
                revision=self.candidate.model_revision,
                local_files_only=self.local_files_only,
            )
            .to(self.device)
            .eval()
        )
        self.model_load_seconds = time.perf_counter() - started

    def align(self, *, audio_path: Path, transcript: str) -> WordTimestampsV1:
        self._load()
        words, symbols, token_ids = ctc_transcript_token_ids(
            transcript,
            vocabulary=self._vocabulary,
        )
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
            log_probabilities = self._model(input_values).logits[0].log_softmax(dim=-1)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.last_inference_seconds = time.perf_counter() - started
        emissions = log_probabilities.cpu().numpy()
        token_spans = force_align_ctc(
            emissions,
            token_symbols=symbols,
            token_ids=token_ids,
            blank_id=int(self._model.config.pad_token_id),
        )
        aligned_words = token_spans_to_words(
            words,
            token_spans,
            frame_count=emissions.shape[0],
            duration_ms=duration_ms,
        )
        return WordTimestampsV1(
            schema_version="WordTimestampsV1",
            analysis_id=audio_path.stem,
            audio_sha256=_sha256_file(audio_path),
            transcript_sha256=_sha256_bytes(transcript),
            duration_ms=duration_ms,
            processor=WordProcessorProvenanceV1(
                aligner_name=self.candidate.model_name,
                aligner_revision=candidate_processor_revision(self.candidate),
                normalization_version=self.candidate.normalization_version,
            ),
            words=aligned_words,
            warnings=(),
        )
