# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared access to private GCS buckets.

Signing goes through the runtime service account's IAM SignBlob, so no key file
is ever held: the service account needs tokenCreator on itself plus read access
to the bucket. Callers pass their own TTL — arena clips and S2S samples want
very different lifetimes.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from google.api_core.exceptions import NotFound

if TYPE_CHECKING:
    from google.cloud import storage


def client() -> storage.Client:
    from google.cloud import storage

    return storage.Client()


def read_json(
    bucket_name: str,
    key: str,
    *,
    storage_client: storage.Client | None = None,
) -> object | None:
    gcs = storage_client if storage_client is not None else client()
    try:
        decoded: object = json.loads(gcs.bucket(bucket_name).blob(key).download_as_bytes())
    except NotFound:
        return None
    return decoded


def signed_url(
    bucket_name: str,
    key: str,
    *,
    ttl: timedelta,
    storage_client: storage.Client | None = None,
) -> str:
    import google.auth
    import google.auth.transport.requests

    gcs = storage_client if storage_client is not None else client()
    blob = gcs.bucket(bucket_name).blob(key)

    credentials, _ = google.auth.default()
    signer: Any = credentials
    if not hasattr(signer, "service_account_email"):
        raise RuntimeError(
            "Signing GCS URLs requires service-account credentials with a "
            "tokenCreator grant; got credentials without a service account."
        )
    signer.refresh(google.auth.transport.requests.Request())
    url: str = blob.generate_signed_url(
        version="v4",
        expiration=ttl,
        method="GET",
        service_account_email=signer.service_account_email,
        access_token=signer.token,
    )
    return url
