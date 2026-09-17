# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Immutable private GCS artifacts for normalized benchmark observations."""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from typing import Any, cast

from google.api_core.exceptions import GoogleAPIError, PreconditionFailed
from google.cloud import storage

from coval_bench.db.models import ObservationArtifact, ObservationArtifactType


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _artifact_key(artifact_type: ObservationArtifactType, digest: str, extension: str) -> str:
    return f"observation-artifacts/v1/{artifact_type}/{digest[:2]}/{digest}.{extension}"


def immutable_object_key(prefix: str, digest: str, suffix: str = "json") -> str:
    """Return a deterministic private object key for a frozen payload."""
    if not prefix or prefix.startswith("/") or ".." in prefix.split("/"):
        raise ValueError("prefix must be a relative object prefix")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("digest must be a lowercase SHA-256 hex digest")
    return f"{prefix.rstrip('/')}/{digest[:2]}/{digest}.{suffix}"


def immutable_object_uri(bucket_name: str, key: str) -> str:
    return f"gs://{bucket_name}/{key}"


def upload_immutable_object(
    client: storage.Client,
    bucket_name: str,
    key: str,
    payload: bytes,
    *,
    content_type: str = "application/octet-stream",
    max_attempts: int = 3,
) -> str:
    """Create an object once, accepting only an exact immutable collision.

    The operation is retried only for transport/service failures.  A
    precondition conflict is verified and is never retried.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    digest = hashlib.sha256(payload).hexdigest()
    blob = client.bucket(bucket_name).blob(key)
    blob.metadata = {"sha256": digest}

    def verify() -> None:
        blob.reload()
        remote = cast(bytes, blob.download_as_bytes())
        metadata = blob.metadata or {}
        if (
            remote != payload
            or blob.size != len(payload)
            or blob.content_type != content_type
            or metadata.get("sha256") != digest
        ):
            raise ValueError("immutable object collision does not match expected content")

    for attempt in range(max_attempts):
        try:
            blob.upload_from_string(payload, content_type=content_type, if_generation_match=0)
            verify()
            return immutable_object_uri(bucket_name, key)
        except PreconditionFailed:
            verify()
            return immutable_object_uri(bucket_name, key)
        except (GoogleAPIError, OSError, TimeoutError):
            if attempt + 1 >= max_attempts:
                raise
    raise AssertionError("unreachable")


def upload_immutable_json(
    client: storage.Client,
    bucket_name: str,
    prefix: str,
    value: object,
    *,
    max_attempts: int = 3,
) -> tuple[str, str]:
    """Serialize and upload a canonical JSON object; return URI and digest."""
    payload = _canonical_json(value)
    digest = hashlib.sha256(payload).hexdigest()
    key = immutable_object_key(prefix, digest)
    return (
        upload_immutable_object(
            client,
            bucket_name,
            key,
            payload,
            content_type="application/json",
            max_attempts=max_attempts,
        ),
        digest,
    )


def prepare_provider_transcript(
    transcript: str,
) -> tuple[ObservationArtifactType, bytes, str, str, str]:
    """Return the canonical, upload-independent transcript descriptor.

    Keeping preparation separate lets migrations determine their exact dry-run
    payload without constructing a storage client.
    """
    return (
        ObservationArtifactType.PROVIDER_TRANSCRIPT,
        _canonical_json({"schema_version": "v1", "transcript": transcript}),
        "json",
        "application/json",
        "ProviderTranscript",
    )


def prepare_timing_events(
    events: dict[str, Any],
) -> tuple[ObservationArtifactType, bytes, str, str, str]:
    """Return the canonical, upload-independent timing-events descriptor."""
    return (
        ObservationArtifactType.TIMING_EVENTS,
        _canonical_json({"schema_version": "v1", "events": events}),
        "json",
        "application/json",
        "TimingEvents",
    )


def _upload(
    client: storage.Client,
    bucket_name: str,
    artifact_type: ObservationArtifactType,
    payload: bytes,
    *,
    extension: str,
    content_type: str,
    schema_name: str,
    duration_ms: float | None = None,
) -> ObservationArtifact:
    digest = hashlib.sha256(payload).hexdigest()
    key = _artifact_key(artifact_type, digest, extension)
    blob = client.bucket(bucket_name).blob(key)
    blob.metadata = {"sha256": digest}
    try:
        blob.upload_from_string(payload, content_type=content_type, if_generation_match=0)
    except PreconditionFailed:
        blob.reload()
        remote = blob.download_as_bytes()
        if (
            remote != payload
            or blob.size != len(payload)
            or blob.content_type != content_type
            or (blob.metadata or {}).get("sha256") != digest
        ):
            raise ValueError(
                "immutable artifact collision does not match expected content"
            ) from None
    return ObservationArtifact(
        artifact_type=artifact_type,
        schema_name=schema_name,
        schema_version="v1",
        gcs_uri=f"gs://{bucket_name}/{key}",
        content_sha256=digest,
        size_bytes=len(payload),
        duration_ms=duration_ms,
    )


def upload_prepared_observation_artifact(
    client: storage.Client,
    bucket_name: str,
    artifact_type: ObservationArtifactType,
    payload: bytes,
    *,
    extension: str,
    content_type: str,
    schema_name: str,
    schema_version: str = "v1",
    duration_ms: float | None = None,
) -> ObservationArtifact:
    """Upload frozen artifact bytes without reconstructing their payload."""
    if schema_version != "v1":
        raise ValueError(f"unsupported observation artifact schema version {schema_version!r}")
    return _upload(
        client,
        bucket_name,
        artifact_type,
        payload,
        extension=extension,
        content_type=content_type,
        schema_name=schema_name,
        duration_ms=duration_ms,
    )


def upload_provider_transcript(
    client: storage.Client, bucket_name: str, transcript: str
) -> ObservationArtifact:
    artifact_type, payload, extension, content_type, schema_name = prepare_provider_transcript(
        transcript
    )
    return _upload(
        client,
        bucket_name,
        artifact_type,
        payload,
        extension=extension,
        content_type=content_type,
        schema_name=schema_name,
    )


def upload_timing_events(
    client: storage.Client, bucket_name: str, events: dict[str, Any]
) -> ObservationArtifact:
    artifact_type, payload, extension, content_type, schema_name = prepare_timing_events(events)
    return _upload(
        client,
        bucket_name,
        artifact_type,
        payload,
        extension=extension,
        content_type=content_type,
        schema_name=schema_name,
    )


def snapshot_generated_audio(path: Path) -> tuple[bytes, float]:
    """Read a temporary WAV before yielding to asynchronous persistence work."""
    payload = path.read_bytes()
    with wave.open(str(path), "rb") as wav:
        duration_ms = wav.getnframes() / wav.getframerate() * 1000
    return payload, duration_ms


def upload_generated_audio(
    client: storage.Client, bucket_name: str, payload: bytes, duration_ms: float
) -> ObservationArtifact:
    return _upload(
        client,
        bucket_name,
        ObservationArtifactType.GENERATED_AUDIO,
        payload,
        extension="wav",
        content_type="audio/wav",
        schema_name="GeneratedAudio",
        duration_ms=duration_ms,
    )
