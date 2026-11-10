# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""SemDist: one minus the cosine similarity of sentence embeddings.

Scored on the same normalized text as WER and CER. The embedder is passed in, so
the metric doesn't care which model produced the vectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel

from coval_bench.metrics.wer import normalize_text

_MAX_TOKENS = 256


class SentenceEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Return one unit-length row per text."""
        ...


class SemDistResult(BaseModel):
    distance: float
    normalized_reference: str
    normalized_hypothesis: str


class OnnxSentenceEmbedder:
    """Mean-pooled, unit-length embeddings from a BERT-style ONNX export."""

    def __init__(self, model_dir: Path) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=_MAX_TOKENS)
        self._tokenizer.enable_padding()
        self._session = onnxruntime.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self._input_names = {node.name for node in self._session.get_inputs()}

    def embed(self, texts: Sequence[str]) -> np.ndarray[Any, np.dtype[np.float32]]:
        encoded = self._tokenizer.encode_batch(list(texts))
        mask = np.array([row.attention_mask for row in encoded], dtype=np.int64)
        feeds = {
            "input_ids": np.array([row.ids for row in encoded], dtype=np.int64),
            "attention_mask": mask,
            "token_type_ids": np.array([row.type_ids for row in encoded], dtype=np.int64),
        }
        hidden = self._session.run(
            None, {name: value for name, value in feeds.items() if name in self._input_names}
        )[0]
        weights = mask[:, :, None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / weights.sum(axis=1)
        unit: np.ndarray[Any, np.dtype[np.float32]] = pooled / np.linalg.norm(
            pooled, axis=1, keepdims=True
        )
        return unit


@lru_cache(maxsize=4)
def load_sentence_embedder(model_dir: Path) -> OnnxSentenceEmbedder:
    return OnnxSentenceEmbedder(model_dir)


def compute_semdist(reference: str, hypothesis: str, embedder: SentenceEmbedder) -> SemDistResult:
    norm_ref = normalize_text(reference).strip()
    norm_hyp = normalize_text(hypothesis).strip()
    if norm_ref == norm_hyp:
        distance = 0.0
    elif not norm_ref or not norm_hyp:
        distance = 1.0
    else:
        ref_vector, hyp_vector = embedder.embed([norm_ref, norm_hyp])
        distance = min(2.0, max(0.0, 1.0 - float(ref_vector @ hyp_vector)))
    return SemDistResult(
        distance=distance, normalized_reference=norm_ref, normalized_hypothesis=norm_hyp
    )
