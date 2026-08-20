# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Immutable private GCS artifacts for normalized benchmark observations."""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from typing import Any

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage

from coval_bench.db.models import ObservationArtifact, ObservationArtifactType


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _artifact_key(artifact_type: ObservationArtifactType, digest: str, extension: str) -> str:
    return f"observation-artifacts/v1/{artifact_type}/{digest[:2]}/{digest}.{extension}"


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


def upload_provider_transcript(
    client: storage.Client, bucket_name: str, transcript: str
) -> ObservationArtifact:
    return _upload(
        client,
        bucket_name,
        ObservationArtifactType.PROVIDER_TRANSCRIPT,
        _canonical_json({"schema_version": "v1", "transcript": transcript}),
        extension="json",
        content_type="application/json",
        schema_name="ProviderTranscript",
    )


def upload_timing_events(
    client: storage.Client, bucket_name: str, events: dict[str, Any]
) -> ObservationArtifact:
    return _upload(
        client,
        bucket_name,
        ObservationArtifactType.TIMING_EVENTS,
        _canonical_json({"schema_version": "v1", "events": events}),
        extension="json",
        content_type="application/json",
        schema_name="TimingEvents",
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
