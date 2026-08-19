from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import pytest
from google.api_core.exceptions import PreconditionFailed

from coval_bench.observation_artifacts import (
    snapshot_generated_audio,
    upload_generated_audio,
    upload_provider_transcript,
)


class Blob:
    def __init__(self) -> None:
        self.metadata: dict[str, str] | None = None
        self.payload = b""
        self.size = 0
        self.content_type: str | None = None
        self.kwargs: dict[str, object] = {}
        self.conflict = False

    def upload_from_string(self, payload: bytes, **kwargs: object) -> None:
        self.kwargs = kwargs
        if self.conflict:
            raise PreconditionFailed("exists")
        self.payload = payload
        self.size = len(payload)
        self.content_type = str(kwargs["content_type"])

    def reload(self) -> None:
        return None

    def download_as_bytes(self) -> bytes:
        return self.payload


class Bucket:
    def __init__(self, blob: Blob) -> None:
        self._blob = blob

    def blob(self, _key: str) -> Blob:
        return self._blob


class Client:
    def __init__(self, blob: Blob) -> None:
        self._bucket = Bucket(blob)

    def bucket(self, _name: str) -> Bucket:
        return self._bucket


def test_transcript_upload_is_create_only_private_and_opaque() -> None:
    blob = Blob()
    artifact = upload_provider_transcript(Client(blob), "private", "secret transcript")  # type: ignore[arg-type]
    assert blob.kwargs["if_generation_match"] == 0
    assert artifact.gcs_uri.startswith("gs://private/observation-artifacts/v1/")
    assert "secret" not in artifact.gcs_uri
    assert artifact.content_sha256 == hashlib.sha256(blob.payload).hexdigest()
    assert artifact.size_bytes == len(blob.payload)
    assert blob.metadata == {"sha256": artifact.content_sha256}
    assert blob.content_type == "application/json"


def test_exact_collision_accepts_but_corruption_rejects() -> None:
    blob = Blob()
    client = Client(blob)
    upload_provider_transcript(client, "private", "same")  # type: ignore[arg-type]
    blob.conflict = True
    upload_provider_transcript(client, "private", "same")  # type: ignore[arg-type]
    blob.content_type = "text/plain"
    with pytest.raises(ValueError, match="collision"):
        upload_provider_transcript(client, "private", "same")  # type: ignore[arg-type]
    blob.content_type = "application/json"
    blob.payload = b"wrong"
    blob.size = len(blob.payload)
    with pytest.raises(ValueError, match="collision"):
        upload_provider_transcript(client, "private", "same")  # type: ignore[arg-type]


def test_wav_snapshot_keeps_bytes_and_duration(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16_000, 160, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 160)
    payload, duration_ms = snapshot_generated_audio(path)
    assert payload == path.read_bytes()
    assert duration_ms == 10

    blob = Blob()
    artifact = upload_generated_audio(  # type: ignore[arg-type]
        Client(blob), "private", payload, duration_ms
    )
    assert blob.payload == payload
    assert blob.content_type == "audio/wav"
    assert artifact.duration_ms == duration_ms
