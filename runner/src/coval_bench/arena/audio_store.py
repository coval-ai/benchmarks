# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Arena clip storage.

``store_clip`` stores a WAV under an opaque key and returns the key (kept on the battle
row); ``clip_url`` turns a key into a playable URL at serve time — a fresh signed URL for
GCS, so rows never serve stale links. Opaque keys keep voting blind.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coval_bench.config import Settings

if TYPE_CHECKING:
    from google.cloud import storage

_SIGNED_URL_TTL = timedelta(days=1)


def store_clip(
    settings: Settings,
    src_path: Path,
    *,
    storage_client: storage.Client | None = None,
) -> str:
    """Consume a WAV into storage (GCS upload or local move); return its opaque key."""
    key = f"clips/{uuid.uuid4().hex}.wav"
    if settings.arena_gcs_bucket:
        _upload_gcs(settings.arena_gcs_bucket, key, src_path, storage_client)
    else:
        dest = settings.arena_audio_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), dest)
    return key


def clip_url(
    settings: Settings,
    key: str,
    *,
    storage_client: storage.Client | None = None,
) -> str:
    """Build a playable URL for a key at serve time: a fresh signed URL for GCS, else local."""
    if settings.arena_gcs_bucket:
        return _signed_url(settings.arena_gcs_bucket, key, storage_client=storage_client)
    base = settings.arena_audio_base_url.rstrip("/")
    return f"{base}/{key}" if base else f"/{key}"


def _upload_gcs(
    bucket_name: str,
    key: str,
    src_path: Path,
    client: storage.Client | None,
) -> None:
    if client is None:
        from google.cloud import storage

        client = storage.Client()
    blob = client.bucket(bucket_name).blob(key)
    blob.upload_from_filename(str(src_path), content_type="audio/wav")
    src_path.unlink(missing_ok=True)


def _signed_url(
    bucket_name: str,
    key: str,
    *,
    storage_client: storage.Client | None = None,
) -> str:
    """Fresh V4 signed GET URL via the SA's IAM SignBlob (no key file; needs tokenCreator)."""
    import google.auth
    import google.auth.transport.requests

    if storage_client is None:
        from google.cloud import storage

        storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(key)

    credentials, _ = google.auth.default()
    signer: Any = credentials
    if not hasattr(signer, "service_account_email"):
        raise RuntimeError(
            "Signing arena clip URLs requires service-account credentials with a "
            "tokenCreator grant; got credentials without a service account."
        )
    signer.refresh(google.auth.transport.requests.Request())
    url: str = blob.generate_signed_url(
        version="v4",
        expiration=_SIGNED_URL_TTL,
        method="GET",
        service_account_email=signer.service_account_email,
        access_token=signer.token,
    )
    return url
