# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Durable, immutable capture envelopes for normalized benchmark writes.

This module deliberately contains no database or provider calls.  Producers
freeze their result and artifact bytes here; recovery can then replay that
frozen payload without rerunning a provider or consulting current mappings.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Self, cast

from google.api_core.exceptions import NotFound
from google.cloud import storage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from coval_bench.observation_artifacts import (
    immutable_object_key,
    immutable_object_uri,
    upload_immutable_json,
    upload_immutable_object,
)

CAPTURE_SCHEMA_VERSION = "v1"
CAPTURE_PREFIX = "normalized-captures/v1"


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_capture_ids(values: list[str]) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError("expected_capture_ids must be unique")
    if any(not _is_sha256(value) for value in values):
        raise ValueError("expected_capture_ids must contain SHA-256 digests")
    return values


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CAPTURE_SCHEMA_VERSION
    run_id: int = Field(gt=0)
    scheduled_at: datetime
    benchmark_kind: str = Field(min_length=1)
    source: str = Field(min_length=1)
    datasets: dict[str, str]
    expected_capture_ids: list[str]

    @field_validator("datasets")
    @classmethod
    def _valid_datasets(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or any(
            not dataset or not _is_sha256(digest) for dataset, digest in value.items()
        ):
            raise ValueError("datasets must map non-empty IDs to SHA-256 digests")
        return value

    _valid_capture_ids = field_validator("expected_capture_ids")(_validate_capture_ids)


class RunSeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CAPTURE_SCHEMA_VERSION
    run_id: int = Field(gt=0)
    intended_status: str = Field(min_length=1)
    stored_status: str = Field(min_length=1)
    finished_at: datetime
    error: str | None = None
    expected_capture_ids: list[str]

    _valid_capture_ids = field_validator("expected_capture_ids")(_validate_capture_ids)


class FinalizedReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CAPTURE_SCHEMA_VERSION
    run_id: int = Field(gt=0)
    seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ImportRunIdentity(BaseModel):
    """Stable source identity for one Coval import generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = "coval-api"
    external_run_id: str = Field(min_length=1)
    workspace_id: str = ""
    benchmark: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    persona_id: str = ""


class ImportRunClaim(BaseModel):
    """Immutable mapping from an external import generation to a DB run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CAPTURE_SCHEMA_VERSION
    identity: ImportRunIdentity
    generation: int = Field(ge=0)
    run_id: int = Field(gt=0)
    started_at: datetime
    scheduled_at: datetime
    captured_at: datetime
    metric_types: list[str] = Field(min_length=1)

    @field_validator("metric_types")
    @classmethod
    def _unique_metric_types(cls, value: list[str]) -> list[str]:
        if any(not metric for metric in value) or value != sorted(set(value)):
            raise ValueError("metric_types must be non-empty, sorted, and unique")
        return value


class LegacyResultAllocation(BaseModel):
    """Immutable allocation of legacy result primary keys to one envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CAPTURE_SCHEMA_VERSION
    envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_ids: list[int]

    @field_validator("result_ids")
    @classmethod
    def _unique_result_ids(cls, value: list[int]) -> list[int]:
        if any(result_id <= 0 for result_id in value) or len(value) != len(set(value)):
            raise ValueError("result_ids must be positive and unique")
        return value


def read_immutable_object(client: storage.Client, bucket_name: str, uri_or_key: str) -> bytes:
    """Read one immutable object and verify its stored content hash."""
    key = uri_or_key.removeprefix(f"gs://{bucket_name}/")
    blob = client.bucket(bucket_name).blob(key)
    blob.reload()
    payload = cast(bytes, blob.download_as_bytes())
    expected = (blob.metadata or {}).get("sha256")
    actual = hashlib.sha256(payload).hexdigest()
    if expected != actual or (blob.size is not None and blob.size != len(payload)):
        raise ValueError("immutable capture object failed content verification")
    return payload


def read_immutable_json(client: storage.Client, bucket_name: str, uri_or_key: str) -> object:
    return json.loads(read_immutable_object(client, bucket_name, uri_or_key))


def list_capture_objects(
    client: storage.Client, bucket_name: str, *, prefix: str = CAPTURE_PREFIX
) -> list[str]:
    """Return sanitized capture object URIs in deterministic order."""
    return sorted(
        immutable_object_uri(bucket_name, blob.name)
        for blob in client.list_blobs(bucket_name, prefix=prefix)
    )


def run_prefix(run_id: int) -> str:
    if run_id <= 0:
        raise ValueError("run_id must be positive")
    return f"{CAPTURE_PREFIX}/runs/{run_id}"


def _run_state_key(run_id: int, kind: str) -> str:
    if kind not in {"manifest", "seal", "finalized"}:
        raise ValueError("unsupported run capture state")
    return f"{run_prefix(run_id)}/{kind}.json"


def upload_run_state(
    client: storage.Client,
    bucket_name: str,
    run_id: int,
    kind: str,
    value: BaseModel | Mapping[str, Any],
    *,
    max_attempts: int = 3,
) -> tuple[str, str]:
    payload_value = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    payload = canonical_bytes(payload_value)
    key = _run_state_key(run_id, kind)
    return (
        upload_immutable_object(
            client,
            bucket_name,
            key,
            payload,
            content_type="application/json",
            max_attempts=max_attempts,
        ),
        hashlib.sha256(payload).hexdigest(),
    )


def read_run_state(
    client: storage.Client, bucket_name: str, run_id: int, kind: str
) -> dict[str, Any] | None:
    try:
        value = read_immutable_json(client, bucket_name, _run_state_key(run_id, kind))
    except NotFound:
        return None
    if not isinstance(value, dict):
        raise TypeError("capture run state must be a JSON object")
    return value


def preflight_capture_storage(client: storage.Client, bucket_name: str) -> None:
    """Verify the create/read/list capabilities required by recovery."""
    payload = canonical_bytes({"schema_version": CAPTURE_SCHEMA_VERSION, "kind": "preflight"})
    key = f"{CAPTURE_PREFIX}/preflight.json"
    upload_immutable_object(
        client, bucket_name, key, payload, content_type="application/json", max_attempts=3
    )
    if read_immutable_object(client, bucket_name, key) != payload:
        raise ValueError("capture preflight read did not match its write")
    names = {blob.name for blob in client.list_blobs(bucket_name, prefix=key, max_results=2)}
    if key not in names:
        raise ValueError("capture preflight object was not visible to list")


def list_envelope_uris(
    client: storage.Client,
    bucket_name: str,
    *,
    run_id: int | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> tuple[list[str], str | None]:
    if limit < 1:
        raise ValueError("limit must be positive")
    prefix = f"{run_prefix(run_id)}/" if run_id is not None else f"{CAPTURE_PREFIX}/runs/"
    start_offset = cursor.removeprefix(f"gs://{bucket_name}/") if cursor is not None else None
    uris: list[str] = []
    for blob in client.list_blobs(bucket_name, prefix=prefix, start_offset=start_offset):
        uri = immutable_object_uri(bucket_name, blob.name)
        if "/envelope/" not in blob.name or (cursor is not None and uri <= cursor):
            continue
        uris.append(uri)
        if len(uris) > limit:
            break
    page = uris[:limit]
    return page, page[-1] if len(uris) > limit and page else None


def missing_expected_envelope_ids(
    client: storage.Client,
    bucket_name: str,
    run_id: int,
    expected_capture_ids: list[str],
) -> list[str]:
    """Find manifest identities with no envelope using one bounded lookup each."""
    _validate_capture_ids(expected_capture_ids)
    missing: list[str] = []
    for capture_id in expected_capture_ids:
        prefix = f"{run_prefix(run_id)}/{capture_id}/envelope/"
        if not any(client.list_blobs(bucket_name, prefix=prefix, max_results=1)):
            missing.append(capture_id)
    return missing


def load_envelope(client: storage.Client, bucket_name: str, uri_or_key: str) -> CaptureEnvelope:
    value = read_immutable_json(client, bucket_name, uri_or_key)
    envelope = CaptureEnvelope.model_validate(value)
    key = uri_or_key.removeprefix(f"gs://{bucket_name}/")
    object_digest = key.rsplit("/", 1)[-1].removesuffix(".json")
    if object_digest != envelope.envelope_digest():
        raise ValueError("capture envelope key does not match its content digest")
    return envelope


def upload_receipt(
    client: storage.Client,
    bucket_name: str,
    envelope: CaptureEnvelope,
    *,
    status: str = "persisted",
    max_attempts: int = 3,
) -> tuple[str, str]:
    receipt = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "status": status,
        "identity": envelope.identity.model_dump(mode="json"),
        "envelope_sha256": envelope.envelope_digest(),
    }
    return upload_immutable_json(
        client,
        bucket_name,
        f"{capture_prefix(envelope.identity)}/receipt",
        receipt,
        max_attempts=max_attempts,
    )


def read_receipt(
    client: storage.Client, bucket_name: str, envelope: CaptureEnvelope
) -> dict[str, Any] | None:
    key = immutable_object_key(
        f"{capture_prefix(envelope.identity)}/receipt",
        payload_digest(
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "status": "persisted",
                "identity": envelope.identity.model_dump(mode="json"),
                "envelope_sha256": envelope.envelope_digest(),
            }
        ),
    )
    try:
        value = read_immutable_json(client, bucket_name, key)
    except NotFound:
        return None
    if not isinstance(value, dict):
        raise TypeError("capture receipt must be a JSON object")
    return value


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def payload_digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


class CaptureIdentity(BaseModel):
    """Stable logical identity used by claims and replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: int = Field(gt=0)
    benchmark: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    voice: str | None = None
    capture_id: str = Field(min_length=1)


def build_capture_identity(
    *,
    run_id: int,
    benchmark: str,
    dataset_id: str,
    sample_id: str,
    provider: str,
    model: str,
    voice: str | None = None,
    capture_id: str | None = None,
) -> CaptureIdentity:
    """Build the one canonical identity shared by manifests and envelopes."""
    benchmark = benchmark.upper()
    return CaptureIdentity(
        run_id=run_id,
        benchmark=benchmark,
        dataset_id=dataset_id,
        sample_id=sample_id,
        provider=provider,
        model=model,
        voice=voice,
        capture_id=capture_id or f"{benchmark}:{sample_id}:{provider}:{model}:{voice or ''}",
    )


class CaptureEnvelope(BaseModel):
    """Complete frozen input for one normalized observation replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CAPTURE_SCHEMA_VERSION
    identity: CaptureIdentity
    payload: dict[str, Any]
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_bytes: dict[str, str] = Field(default_factory=dict)
    artifact_sha256: dict[str, str] = Field(default_factory=dict)

    @field_validator("payload_sha256")
    @classmethod
    def _digest_matches(cls, value: str, info: ValidationInfo) -> str:
        payload = info.data.get("payload")
        if payload is not None and payload_digest(payload) != value:
            raise ValueError("payload_sha256 does not match frozen payload")
        return value

    @classmethod
    def _validate_artifacts(cls, value: dict[str, str], info: ValidationInfo) -> dict[str, str]:
        encoded = info.data.get("artifact_bytes", {})
        if set(value) != set(encoded):
            raise ValueError("artifact_sha256 must cover every frozen artifact")
        for name, digest in value.items():
            try:
                raw = base64.b64decode(encoded[name], validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError(f"artifact {name!r} is not valid base64") from exc
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError(f"artifact {name!r} digest does not match frozen bytes")
        return value

    _artifacts_match = field_validator("artifact_sha256")(_validate_artifacts)

    @model_validator(mode="after")
    def _identity_matches_payload(self) -> Self:
        compared = (
            "run_id",
            "benchmark",
            "dataset_id",
            "sample_id",
            "provider",
            "model",
            "voice",
        )
        mismatches = [
            field for field in compared if self.payload.get(field) != getattr(self.identity, field)
        ]
        if mismatches:
            raise ValueError(
                "capture identity conflicts with frozen payload: " + ", ".join(mismatches)
            )
        payload_artifacts = self.payload.get("artifacts")
        if not isinstance(payload_artifacts, list):
            raise TypeError("frozen payload artifacts must be a list")
        artifact_names = {item.get("name") for item in payload_artifacts if isinstance(item, dict)}
        if artifact_names != set(self.artifact_bytes):
            raise ValueError("frozen payload artifacts do not match embedded bytes")
        return self

    @classmethod
    def freeze(
        cls,
        identity: CaptureIdentity,
        payload: Mapping[str, Any],
        *,
        artifact_bytes: Mapping[str, bytes] | None = None,
    ) -> CaptureEnvelope:
        encoded = {
            name: base64.b64encode(data).decode("ascii")
            for name, data in (artifact_bytes or {}).items()
        }
        digests = {
            name: hashlib.sha256(data).hexdigest() for name, data in (artifact_bytes or {}).items()
        }
        frozen_payload = dict(payload)
        return cls(
            identity=identity,
            payload=frozen_payload,
            payload_sha256=payload_digest(frozen_payload),
            artifact_bytes=encoded,
            artifact_sha256=digests,
        )

    def envelope_digest(self) -> str:
        """Digest the complete frozen envelope, including embedded artifact bytes."""
        return payload_digest(self.model_dump(mode="json"))

    def as_claim(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.model_dump(mode="json"),
            "payload_sha256": self.payload_sha256,
            "envelope_sha256": self.envelope_digest(),
            "artifact_sha256": self.artifact_sha256,
        }


def capture_prefix(identity: CaptureIdentity) -> str:
    # Never place provider/model/sample strings in object paths.  The digest
    # is deterministic for one logical identity and safe as a single segment.
    return f"{CAPTURE_PREFIX}/runs/{identity.run_id}/{identity_digest(identity)}"


def identity_digest(identity: CaptureIdentity) -> str:
    return payload_digest(identity.model_dump(mode="json"))


def import_identity_digest(identity: ImportRunIdentity) -> str:
    return payload_digest(identity.model_dump(mode="json"))


def _import_claim_key(identity: ImportRunIdentity, generation: int) -> str:
    if generation < 0:
        raise ValueError("generation must be non-negative")
    digest = import_identity_digest(identity)
    return f"{CAPTURE_PREFIX}/imports/{digest[:2]}/{digest}/generation-{generation}.json"


def read_import_run_claim(
    client: storage.Client,
    bucket_name: str,
    identity: ImportRunIdentity,
    generation: int,
) -> ImportRunClaim | None:
    try:
        value = read_immutable_json(client, bucket_name, _import_claim_key(identity, generation))
    except NotFound:
        return None
    return ImportRunClaim.model_validate(value)


def upload_import_run_claim(
    client: storage.Client,
    bucket_name: str,
    claim: ImportRunClaim,
    *,
    max_attempts: int = 3,
) -> ImportRunClaim:
    """Elect one run allocation for a stable external import generation."""
    payload = canonical_bytes(claim.model_dump(mode="json"))
    key = _import_claim_key(claim.identity, claim.generation)
    try:
        upload_immutable_object(
            client,
            bucket_name,
            key,
            payload,
            content_type="application/json",
            max_attempts=max_attempts,
        )
        return claim
    except ValueError:
        stored = read_import_run_claim(client, bucket_name, claim.identity, claim.generation)
        if stored is None:
            raise
        if (
            stored.identity != claim.identity
            or stored.generation != claim.generation
            or stored.scheduled_at != claim.scheduled_at
            or stored.metric_types != claim.metric_types
        ):
            raise ValueError(
                "external import generation conflicts with its durable claim"
            ) from None
        return stored


def capture_key(identity: CaptureIdentity, kind: str, digest: str) -> str:
    if kind not in {"manifest", "envelope", "claim", "receipt", "seal", "finalized"}:
        raise ValueError("unsupported capture object kind")
    return immutable_object_key(f"{capture_prefix(identity)}/{kind}", digest)


def upload_envelope(
    client: storage.Client,
    bucket_name: str,
    envelope: CaptureEnvelope,
    *,
    max_attempts: int = 3,
) -> tuple[str, str]:
    """Write and verify the envelope, the first acknowledged durable payload."""
    return upload_immutable_json(
        client,
        bucket_name,
        f"{capture_prefix(envelope.identity)}/envelope",
        envelope.model_dump(mode="json"),
        max_attempts=max_attempts,
    )


def upload_claim(
    client: storage.Client,
    bucket_name: str,
    envelope: CaptureEnvelope,
    *,
    max_attempts: int = 3,
) -> tuple[str, str]:
    """Create the identity claim bound to the envelope payload digest."""
    payload = canonical_bytes(envelope.as_claim())
    digest = hashlib.sha256(payload).hexdigest()
    key = immutable_object_key(
        f"{capture_prefix(envelope.identity)}/claim", identity_digest(envelope.identity)
    )
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


def _legacy_allocation_key(envelope: CaptureEnvelope) -> str:
    return f"{capture_prefix(envelope.identity)}/legacy-allocation.json"


def read_legacy_result_allocation(
    client: storage.Client, bucket_name: str, envelope: CaptureEnvelope
) -> LegacyResultAllocation | None:
    try:
        value = read_immutable_json(client, bucket_name, _legacy_allocation_key(envelope))
    except NotFound:
        return None
    allocation = LegacyResultAllocation.model_validate(value)
    if allocation.envelope_sha256 != envelope.envelope_digest():
        raise ValueError("legacy result allocation belongs to a different envelope")
    return allocation


def upload_legacy_result_allocation(
    client: storage.Client,
    bucket_name: str,
    envelope: CaptureEnvelope,
    result_ids: list[int],
    *,
    max_attempts: int = 3,
) -> LegacyResultAllocation:
    """Elect one ordered result-ID allocation for concurrent replay workers."""
    allocation = LegacyResultAllocation(
        envelope_sha256=envelope.envelope_digest(), result_ids=result_ids
    )
    payload = canonical_bytes(allocation.model_dump(mode="json"))
    key = _legacy_allocation_key(envelope)
    try:
        upload_immutable_object(
            client,
            bucket_name,
            key,
            payload,
            content_type="application/json",
            max_attempts=max_attempts,
        )
        return allocation
    except ValueError:
        stored = read_legacy_result_allocation(client, bucket_name, envelope)
        if stored is None:
            raise
        if len(stored.result_ids) != len(result_ids):
            raise ValueError("legacy result allocation has a different row count") from None
        return stored
