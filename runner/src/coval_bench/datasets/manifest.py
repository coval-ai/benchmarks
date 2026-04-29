# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for coval-bench dataset manifests.

Manifests are JSON files shipped inside the wheel under
``coval_bench/datasets/manifests/``.  They are the SHA-pinned
source of truth for a dataset version — the runner validates
every downloaded file against the hash before use.

Schema matches ARCHITECTURE.md § "GCS dataset bucket — manifest.json schema".
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class STTManifestItem(BaseModel):
    """A single STT audio file entry in the manifest."""

    model_config = ConfigDict(frozen=True)

    path: str  # relative path, e.g. "audio/0001.wav"
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    transcript: str
    duration_sec: float = Field(gt=0)
    speaker_id: str | None = None  # LibriSpeech provenance
    chapter_id: str | None = None
    utterance_id: str | None = None


class TTSManifestItem(BaseModel):
    """A single TTS prompt entry in the manifest."""

    model_config = ConfigDict(frozen=True)

    testcase_id: str
    transcript: str


class Manifest(BaseModel):
    """Top-level manifest object for a dataset version.

    The ``items`` field is either a homogeneous list of
    :class:`STTManifestItem` or :class:`TTSManifestItem`.
    Mixed lists are rejected by the ``items_consistent`` validator.
    """

    model_config = ConfigDict(frozen=True)

    id: str  # "stt-v1" or "tts-v1"
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    license: str  # "CC-BY-4.0" for STT, internal for TTS
    source: str  # e.g. "LibriSpeech test-clean"
    items: list[STTManifestItem | TTSManifestItem]

    @model_validator(mode="after")
    def items_consistent(self) -> Manifest:
        """Ensure all items are the same concrete type."""
        if not self.items:
            return self
        first_type = type(self.items[0])
        for item in self.items[1:]:
            if type(item) is not first_type:
                raise ValueError(
                    f"Manifest '{self.id}' contains mixed item types: "
                    f"expected all {first_type.__name__}, "
                    f"found {type(item).__name__}"
                )
        return self
