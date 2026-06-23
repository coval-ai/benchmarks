# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage for generated arena clips: move a synthesized WAV into durable storage.

Local-dir backend for dev; a GCS backend reimplements just ``store_clip`` later
(generate key -> upload -> return URL) without touching callers. Keys are random
and opaque so a served URL never reveals which model produced it (blind voting) —
identity is recovered from ``arena.battles``, never the filename.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from coval_bench.config import Settings


def store_clip(settings: Settings, src_path: Path) -> str:
    """Move a synthesized WAV into arena storage under a fresh opaque key; return its URL.

    The source is a provider's temp WAV (``TTSResult.audio_path``); we own it once
    synthesis returns, so it is moved, not copied. The URL prepends
    ``arena_audio_base_url`` when set, else is root-relative for the web layer.
    """
    key = f"clips/{uuid.uuid4().hex}.wav"
    dest = settings.arena_audio_dir / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_path), dest)

    base = settings.arena_audio_base_url.rstrip("/")
    return f"{base}/{key}" if base else f"/{key}"
