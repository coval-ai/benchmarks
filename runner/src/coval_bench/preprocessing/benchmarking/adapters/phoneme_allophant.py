# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Audio-only Allophant candidate with pinned model and base-model assets."""

from __future__ import annotations

import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from coval_bench.preprocessing.benchmarking.adapters.base import source_duration_ms
from coval_bench.preprocessing.benchmarking.adapters.phoneme_hf import (
    TARGET_SAMPLE_RATE,
    _candidate,
    _sha256,
    build_phoneme_artifact,
    decode_phone_frames,
    normalize_phone_spans,
)
from coval_bench.preprocessing.benchmarking.contracts import BenchmarkCandidateKind
from coval_bench.preprocessing.benchmarking.inventory import PhoneSource
from coval_bench.preprocessing.contracts import PhonemeTimestampsV1

ALLOPHANT_BASE_MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
ALLOPHANT_BASE_MODEL_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
ALLOPHANT_ALLOPHOIBLE_ENGLISH_INVENTORY = (
    "æ",
    "aɪ",
    "aʊ",
    "ɑ",
    "b",
    "d",
    "ð",
    "d̠ʒ",
    "eɪ̯",
    "ə",
    "ɛ",
    "ɚː",
    "f",
    "ɡ",
    "h",
    "iɪ",
    "ɪ",
    "j",
    "l",
    "m",
    "n",
    "ŋ",
    "oʊ",
    "ɔɪ",
    "ɹ",
    "s",
    "ʃ",
    "t̠ʃ",
    "uː",
    "ʊ",
    "v",
    "ʌ",
    "w",
    "z",
    "ʒ",
    "θ",
)
ALLOPHANT_ENGLISH_INVENTORY = (
    *ALLOPHANT_ALLOPHOIBLE_ENGLISH_INVENTORY[:18],
    "k",
    *ALLOPHANT_ALLOPHOIBLE_ENGLISH_INVENTORY[18:23],
    "p",
    *ALLOPHANT_ALLOPHOIBLE_ENGLISH_INVENTORY[23:27],
    "t",
    *ALLOPHANT_ALLOPHOIBLE_ENGLISH_INVENTORY[27:],
)


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "missing"


def _runtime_software() -> str:
    packages = (
        "allophant",
        "torch",
        "torchaudio",
        "transformers",
        "numpy",
        "pandas",
        "scipy",
        "soundfile",
        "huggingface-hub",
    )
    versions = "_".join(f"{package}-{_installed_version(package)}" for package in packages)
    return f"{versions}_compat-readcsvbuffer-v1"


def _prepare_pandas_compatibility() -> None:
    """Backfill an import-only pandas alias removed after Allophant's pinned pandas."""
    from pandas.io.parsers import readers

    if not hasattr(readers, "ReadCsvBuffer"):
        readers.ReadCsvBuffer = Any


class AllophantPhonemeRecognizer:
    """Infer English phones without a transcript using Allophant 1.0.0."""

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
        if candidate.model_name != "kgnlp/allophant":
            raise ValueError("candidate_id must identify the Allophant candidate")
        self.candidate = candidate
        self.device = device
        self.local_files_only = local_files_only
        self.model_load_seconds = 0.0
        self.last_inference_seconds = 0.0
        self.last_normalization_loss_rate = 0.0
        self.runtime_software = _runtime_software()
        self._model: Any = None
        self._target_feature_indices: Any = None
        self._id_to_token: dict[int, str] = {}

    def _verified_download(
        self,
        *,
        repo_id: str,
        filename: str,
        revision: str,
        asset_suffix: str,
    ) -> Path:
        from huggingface_hub import hf_hub_download

        asset = next(
            (item for item in self.candidate.assets if item.path.endswith(asset_suffix)),
            None,
        )
        if asset is None:
            raise ValueError(f"candidate has no pinned {asset_suffix!r} asset")
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_files_only=self.local_files_only,
            )
        )
        if _sha256(path) != asset.sha256:
            raise ValueError(f"downloaded {asset_suffix} does not match its SHA-256")
        return path

    def _load(self) -> None:
        if self._model is not None:
            return
        _prepare_pandas_compatibility()
        from allophant.estimator import Checkpoint, Estimator

        started = time.perf_counter()
        checkpoint_path = self._verified_download(
            repo_id=self.candidate.model_name,
            filename="allophant.pt",
            revision=self.candidate.model_revision,
            asset_suffix="allophant.pt",
        )
        base_config_path = self._verified_download(
            repo_id=ALLOPHANT_BASE_MODEL_NAME,
            filename="config.json",
            revision=ALLOPHANT_BASE_MODEL_REVISION,
            asset_suffix=f"{ALLOPHANT_BASE_MODEL_REVISION}/config.json",
        )
        self._verified_download(
            repo_id=ALLOPHANT_BASE_MODEL_NAME,
            filename="preprocessor_config.json",
            revision=ALLOPHANT_BASE_MODEL_REVISION,
            asset_suffix=f"{ALLOPHANT_BASE_MODEL_REVISION}/preprocessor_config.json",
        )
        checkpoint = Checkpoint.restore(checkpoint_path, device=self.device)
        checkpoint.config.nn.acoustic_model.model_id = str(base_config_path.parent)
        model, indexer = Estimator.restore(checkpoint, device=self.device)
        if model.sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"unexpected Allophant sample rate {model.sample_rate}")
        allophoible_inventory = tuple(indexer.phoneme_inventory("en"))
        if allophoible_inventory != ALLOPHANT_ALLOPHOIBLE_ENGLISH_INVENTORY:
            raise ValueError(
                "Allophant English inventory differs from the pinned Allophoible inventory"
            )
        inventory = ALLOPHANT_ENGLISH_INVENTORY
        categories = tuple(indexer.attributes.subset(list(inventory)).feature_categories("phoneme"))
        if categories != inventory:
            raise ValueError(
                "Allophant phoneme output categories differ from its English inventory"
            )
        self._id_to_token = {index + 1: symbol for index, symbol in enumerate(inventory)}
        self._target_feature_indices = indexer.composition_feature_matrix(list(inventory)).to(
            self.device
        )
        model.model.eval()
        self._model = model
        self.model_load_seconds = time.perf_counter() - started

    def recognize(self, *, audio_path: Path) -> PhonemeTimestampsV1:
        """Infer phone identities and boundaries using audio as the only input."""
        self._load()
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if samples.ndim != 1:
            samples = samples.mean(axis=1)
        if not len(samples) or not np.isfinite(samples).all():
            raise ValueError("audio samples must be non-empty and finite")
        duration_ms = source_duration_ms(sample_count=len(samples), sample_rate=sample_rate)
        if sample_rate != TARGET_SAMPLE_RATE:
            divisor = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
            samples = resample_poly(
                samples,
                TARGET_SAMPLE_RATE // divisor,
                sample_rate // divisor,
            ).astype(np.float32)

        import torch
        from allophant.dataset_processing import Batch

        audio = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
        batch = Batch(
            audio.to(self.device),
            torch.tensor([audio.shape[1]], dtype=torch.long, device=self.device),
            torch.zeros(1, dtype=torch.long, device=self.device),
        )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = self._model.predict(batch, self._target_feature_indices)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.last_inference_seconds = time.perf_counter() - started

        log_probabilities = outputs.outputs["phoneme"]
        if log_probabilities.ndim != 3 or log_probabilities.shape[1] != 1:
            raise ValueError("unexpected Allophant phoneme output shape")
        frame_count = int(outputs.lengths[0].item())
        if frame_count <= 0 or frame_count > log_probabilities.shape[0]:
            raise ValueError("invalid Allophant output frame count")
        probabilities = log_probabilities[:frame_count, 0].exp()
        frame_confidences, frame_ids = probabilities.max(dim=-1)
        raw_spans = decode_phone_frames(
            tuple(int(value) for value in frame_ids.cpu().tolist()),
            tuple(float(value) for value in frame_confidences.cpu().tolist()),
            id_to_token={0: "<blank>", **self._id_to_token},
            decoder="midpoint_fill",
            blank_id=0,
        )
        decoded = normalize_phone_spans(raw_spans, source=PhoneSource.ALLOPHANT_IPA)
        self.last_normalization_loss_rate = decoded.normalization_loss_rate
        return build_phoneme_artifact(
            analysis_id=audio_path.stem,
            audio_sha256=_sha256(audio_path),
            duration_ms=duration_ms,
            frame_count=frame_count,
            candidate=self.candidate,
            decoded=decoded,
        )
