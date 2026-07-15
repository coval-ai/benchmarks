# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""coval_bench.datasets — manifest-driven, GCS-backed dataset loader.

Public surface::

    from coval_bench.datasets import (
        Dataset,
        DatasetIntegrityError,
        DatasetItem,
        TTSDataset,
        TTSDatasetItem,
        load_dataset,
        load_stt_dataset,
        load_tts_dataset,
        Manifest,
        STTManifestItem,
        TTSManifestItem,
    )
"""

from coval_bench.datasets.loader import (
    WILDASR_ENV_FAMILY,
    Dataset,
    DatasetIntegrityError,
    DatasetItem,
    ManifestAlignmentError,
    TTSDataset,
    TTSDatasetItem,
    family_rng,
    load_dataset,
    load_stt_dataset,
    load_tts_dataset,
)
from coval_bench.datasets.manifest import Manifest, STTManifestItem, TTSManifestItem

__all__ = [
    "WILDASR_ENV_FAMILY",
    "Dataset",
    "DatasetIntegrityError",
    "DatasetItem",
    "ManifestAlignmentError",
    "family_rng",
    "TTSDataset",
    "TTSDatasetItem",
    "load_dataset",
    "load_stt_dataset",
    "load_tts_dataset",
    "Manifest",
    "STTManifestItem",
    "TTSManifestItem",
]
