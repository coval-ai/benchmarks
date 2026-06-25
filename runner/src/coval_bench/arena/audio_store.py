# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage for generated arena clips: move a synthesized WAV into durable storage.

Local-dir backend for dev; GCS backend for prod, selected by
``settings.arena_gcs_bucket``. Keys are random and opaque so a served URL never
reveals which model produced it (blind voting) — identity is recovered from
``arena.battles``, never the filename.
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
    """Consume a synthesized WAV into arena storage under a fresh opaque key; return its URL.

    The source is a provider's temp WAV (``TTSResult.audio_path``); we own it once
    synthesis returns, so it is consumed — moved locally, or uploaded then deleted.
    With ``arena_gcs_bucket`` set the clip goes to GCS and a time-limited V4 signed
    URL is returned (the bucket is private); otherwise it is stored under
    ``arena_audio_dir`` and a root-relative (or ``arena_audio_base_url``-prefixed) URL
    is returned.
    """
    key = f"clips/{uuid.uuid4().hex}.wav"
    if settings.arena_gcs_bucket:
        return _store_clip_gcs(settings.arena_gcs_bucket, key, src_path, storage_client)

    dest = settings.arena_audio_dir / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_path), dest)

    base = settings.arena_audio_base_url.rstrip("/")
    return f"{base}/{key}" if base else f"/{key}"


def _store_clip_gcs(
    bucket_name: str,
    key: str,
    src_path: Path,
    client: storage.Client | None,
) -> str:
    if client is None:
        from google.cloud import storage

        client = storage.Client()
    blob = client.bucket(bucket_name).blob(key)
    blob.upload_from_filename(str(src_path), content_type="audio/wav")
    url = _signed_url(blob)
    src_path.unlink(missing_ok=True)
    return url


def _signed_url(blob: storage.Blob) -> str:
    """V4 signed GET URL signed via the runtime SA's IAM SignBlob (no key file).

    Cloud Run's default credentials carry no private key, so signing is delegated to
    the IAM API using the SA's own token — this needs ``roles/iam.serviceAccounts.tokenCreator``
    on the runtime SA.
    """
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default()
    signer: Any = credentials
    if not hasattr(signer, "service_account_email"):
        raise RuntimeError(
            "Signing arena clip URLs requires service-account credentials with a "
            "tokenCreator grant; got credentials without a service account. Set "
            "arena_gcs_bucket only where the runtime SA is configured."
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
