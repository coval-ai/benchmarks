# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Dataset loader: manifest-driven, GCS-backed, SHA256-verified.

Design choice: two public entry points — ``load_stt_dataset`` and
``load_tts_dataset`` — rather than a single overloaded ``load_dataset``.
This avoids a Union return type that the orchestrator would have to
discriminate on anyway.  ``load_dataset`` is kept as a thin dispatcher for
back-compat and convenience; its return type is the union of both.

GCS access
----------
* When ``google_application_credentials`` is None, the GCS client is
  initialised with ``anonymous_credentials`` so OSS contributors can read
  the public bucket without a GCP account.
* In Cloud Run, the runner-rt SA credentials are attached automatically via
  the metadata server.
* Tests inject a fake ``storage.Client`` via the ``storage_client`` kwarg —
  the real GCS is *never* called from tests.

Manifest loading
----------------
Manifests live *inside the wheel* at
``coval_bench/datasets/manifests/<dataset_id>.json`` and are loaded via
``importlib.resources``.  This gives the runner a SHA-pinned source of truth
before any network call.

Cache directory
---------------
Default: ``$XDG_CACHE_HOME/coval-bench`` or ``~/.cache/coval-bench``.
If that path is unwritable (e.g. a constrained Cloud Run sandbox), the
loader falls back to ``/tmp/coval-bench``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from importlib.resources import files
from pathlib import Path

from google.cloud import storage  # type: ignore[attr-defined]
from pydantic import BaseModel, Field

from coval_bench.config import Settings
from coval_bench.datasets.manifest import (
    Manifest,
    STTManifestItem,
    TTSManifestItem,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Dataset",
    "DatasetIntegrityError",
    "DatasetItem",
    "TTSDatasetItem",
    "TTSDataset",
    "load_dataset",
    "load_stt_dataset",
    "load_tts_dataset",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DatasetIntegrityError(Exception):
    """Raised when a downloaded file's SHA256 does not match the manifest.

    This is a hard failure — the run is aborted.  Never fall back to an
    alternative file; that masking behaviour was the root cause of the
    legacy 100%-WER bug.
    """

    def __init__(self, path: str, expected: str, actual: str) -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(f"SHA256 mismatch for '{path}': expected={expected} actual={actual}")


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class DatasetItem(BaseModel):
    """A single STT benchmark item, ready for a provider."""

    path: Path  # local on-disk path after fetch
    transcript: str  # ground-truth transcript
    duration_sec: float
    sha256: str
    metadata: dict[str, str] = Field(default_factory=dict)  # speaker_id etc.


class Dataset(BaseModel):
    """An STT dataset, fully fetched and verified."""

    id: str
    version: str
    items: list[DatasetItem]


class TTSDatasetItem(BaseModel):
    """A single TTS benchmark item (text-only, no audio to fetch)."""

    testcase_id: str
    transcript: str


class TTSDataset(BaseModel):
    """A TTS dataset (text prompts only)."""

    id: str
    version: str
    items: list[TTSDatasetItem]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_cache_dir() -> Path:
    """Return the default cache directory, falling back to /tmp if needed."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    candidate = Path(xdg) / "coval-bench" if xdg else Path.home() / ".cache" / "coval-bench"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        # Verify writability with a probe
        probe = candidate / ".write_probe"
        probe.touch()
        probe.unlink()
        return candidate
    except OSError:
        import tempfile  # noqa: PLC0415

        fallback = Path(tempfile.gettempdir()) / "coval-bench"  # noqa: S108
        fallback.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "Default cache dir %s is not writable; falling back to %s",
            candidate,
            fallback,
        )
        return fallback


def _load_manifest(dataset_id: str) -> Manifest:
    """Load and validate the packaged manifest for *dataset_id*."""
    manifest_text = (
        files("coval_bench.datasets.manifests")
        .joinpath(f"{dataset_id}.json")
        .read_text(encoding="utf-8")
    )
    return Manifest.model_validate_json(manifest_text)


def _sha256_file(path: Path) -> str:
    """Return the hex SHA256 digest of the file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_storage_client(settings: Settings) -> storage.Client:
    """Create a GCS client, using anonymous creds for public-bucket reads."""
    if settings.google_application_credentials is None:
        return storage.Client.create_anonymous_client()
    return storage.Client()


def _fetch_blob(
    client: storage.Client,
    bucket_name: str,
    object_path: str,
    dest: Path,
) -> None:
    """Download *object_path* from *bucket_name* to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    blob.download_to_filename(str(dest))


def _fetch_and_verify(
    *,
    client: storage.Client,
    bucket: str,
    dataset_id: str,
    item: STTManifestItem,
    cache_dir: Path,
) -> Path:
    """Return local path for *item*, downloading from GCS if needed.

    If the cached file already has the correct SHA256, the download is skipped.
    On SHA256 mismatch after download, raises :class:`DatasetIntegrityError`.
    """
    local_path = cache_dir / dataset_id / item.path

    # Cache hit check
    if local_path.exists():
        if _sha256_file(local_path) == item.sha256:
            logger.debug("Cache hit for %s", item.path)
            return local_path
        logger.warning("Cached file %s has wrong hash — re-downloading", local_path)

    # Download
    gcs_object = f"{dataset_id}/{item.path}"
    logger.info("Fetching gs://%s/%s", bucket, gcs_object)
    _fetch_blob(client, bucket, gcs_object, local_path)

    # Post-download integrity check
    actual = _sha256_file(local_path)
    if actual != item.sha256:
        raise DatasetIntegrityError(item.path, item.sha256, actual)

    return local_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_stt_dataset(
    dataset_id: str,
    *,
    settings: Settings,
    cache_dir: Path | None = None,
    storage_client: storage.Client | None = None,
) -> Dataset:
    """Load an STT dataset from the packaged manifest + GCS.

    Parameters
    ----------
    dataset_id:
        Manifest identifier, e.g. ``"stt-v1"``.
    settings:
        Runner settings; ``dataset_bucket`` names the GCS bucket.
    cache_dir:
        Override the local file cache directory.
    storage_client:
        Injectable GCS client (for tests; production passes ``None``).

    Returns
    -------
    Dataset
        Fully-verified dataset with local file paths.
    """
    manifest = _load_manifest(dataset_id)
    resolved_cache = cache_dir if cache_dir is not None else _default_cache_dir()
    client = storage_client if storage_client is not None else _make_storage_client(settings)

    items: list[DatasetItem] = []
    for raw_item in manifest.items:
        if not isinstance(raw_item, STTManifestItem):
            raise TypeError(
                f"Dataset '{dataset_id}' contains non-STT items; use load_tts_dataset instead."
            )
        local_path = _fetch_and_verify(
            client=client,
            bucket=settings.dataset_bucket,
            dataset_id=dataset_id,
            item=raw_item,
            cache_dir=resolved_cache,
        )
        meta: dict[str, str] = {}
        if raw_item.speaker_id is not None:
            meta["speaker_id"] = raw_item.speaker_id
        if raw_item.chapter_id is not None:
            meta["chapter_id"] = raw_item.chapter_id
        if raw_item.utterance_id is not None:
            meta["utterance_id"] = raw_item.utterance_id
        items.append(
            DatasetItem(
                path=local_path,
                transcript=raw_item.transcript,
                duration_sec=raw_item.duration_sec,
                sha256=raw_item.sha256,
                metadata=meta,
            )
        )

    return Dataset(id=manifest.id, version=manifest.version, items=items)


def load_tts_dataset(
    dataset_id: str,
    *,
    settings: Settings,  # noqa: ARG001 – kept for uniform call signature
    cache_dir: Path | None = None,  # noqa: ARG001 – no files to cache for TTS
    storage_client: storage.Client | None = None,  # noqa: ARG001 – no GCS needed
) -> TTSDataset:
    """Load a TTS dataset (text prompts only; no GCS fetch required).

    Parameters
    ----------
    dataset_id:
        Manifest identifier, e.g. ``"tts-v1"``.
    settings:
        Accepted for call-signature symmetry; not used for TTS.
    cache_dir:
        Accepted for call-signature symmetry; not used for TTS.
    storage_client:
        Accepted for call-signature symmetry; not used for TTS.

    Returns
    -------
    TTSDataset
        Dataset with text-only items.
    """
    manifest = _load_manifest(dataset_id)

    tts_items: list[TTSDatasetItem] = []
    for raw_item in manifest.items:
        if not isinstance(raw_item, TTSManifestItem):
            raise TypeError(
                f"Dataset '{dataset_id}' contains non-TTS items; use load_stt_dataset instead."
            )
        tts_items.append(
            TTSDatasetItem(
                testcase_id=raw_item.testcase_id,
                transcript=raw_item.transcript,
            )
        )

    return TTSDataset(id=manifest.id, version=manifest.version, items=tts_items)


def load_dataset(
    dataset_id: str,
    *,
    settings: Settings,
    cache_dir: Path | None = None,
    storage_client: storage.Client | None = None,
) -> Dataset | TTSDataset:
    """Dispatcher: load STT or TTS dataset based on manifest content.

    Prefer the explicit ``load_stt_dataset`` / ``load_tts_dataset`` functions
    when the caller knows the dataset type ahead of time.
    """
    manifest = _load_manifest(dataset_id)
    if manifest.items and isinstance(manifest.items[0], STTManifestItem):
        return load_stt_dataset(
            dataset_id,
            settings=settings,
            cache_dir=cache_dir,
            storage_client=storage_client,
        )
    if not manifest.items or isinstance(manifest.items[0], TTSManifestItem):
        return load_tts_dataset(
            dataset_id,
            settings=settings,
            cache_dir=cache_dir,
            storage_client=storage_client,
        )
    return load_tts_dataset(
        dataset_id,
        settings=settings,
        cache_dir=cache_dir,
        storage_client=storage_client,
    )
