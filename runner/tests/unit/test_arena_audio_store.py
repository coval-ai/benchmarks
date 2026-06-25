# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for arena clip storage (local-dir and GCS backends)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from coval_bench.arena.audio_store import store_clip
from coval_bench.config import Settings


def _wav(tmp_path: Path) -> Path:
    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFFsynth")
    return src


def test_store_clip_local_root_relative(tmp_path: Path) -> None:
    settings = Settings(arena_audio_dir=tmp_path / "store")
    url = store_clip(settings, _wav(tmp_path))

    assert url.startswith("/clips/") and url.endswith(".wav")
    assert (settings.arena_audio_dir / url.lstrip("/")).is_file()


def test_store_clip_local_base_url_prefix(tmp_path: Path) -> None:
    settings = Settings(arena_audio_dir=tmp_path / "store", arena_audio_base_url="https://cdn.x/")
    url = store_clip(settings, _wav(tmp_path))

    assert url.startswith("https://cdn.x/clips/")


class _FakeBlob:
    def __init__(self) -> None:
        self.uploaded_from: str | None = None
        self.content_type: str | None = None
        self.signed_kwargs: dict[str, Any] = {}

    def upload_from_filename(self, filename: str, *, content_type: str) -> None:
        self.uploaded_from = filename
        self.content_type = content_type

    def generate_signed_url(self, **kwargs: Any) -> str:
        self.signed_kwargs = kwargs
        return f"https://signed.example/{kwargs['method']}"


class _FakeBucket:
    def __init__(self, blob: _FakeBlob) -> None:
        self._blob = blob
        self.key: str | None = None

    def blob(self, key: str) -> _FakeBlob:
        self.key = key
        return self._blob


class _FakeClient:
    def __init__(self, blob: _FakeBlob) -> None:
        self.bucket_obj = _FakeBucket(blob)
        self.bucket_name: str | None = None

    def bucket(self, name: str) -> _FakeBucket:
        self.bucket_name = name
        return self.bucket_obj


def test_store_clip_gcs_uploads_and_signs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    creds = SimpleNamespace(
        service_account_email="api-rt@proj.iam.gserviceaccount.com",
        token="tok",  # noqa: S106 — fake credential token for the stubbed signer
        refresh=lambda _request: None,
    )
    monkeypatch.setattr("google.auth.default", lambda: (creds, "proj"))
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())

    blob = _FakeBlob()
    client = _FakeClient(blob)
    src = _wav(tmp_path)
    settings = Settings(arena_gcs_bucket="coval-benchmarks-arena")

    url = store_clip(settings, src, storage_client=client)

    assert url == "https://signed.example/GET"
    assert client.bucket_name == "coval-benchmarks-arena"
    assert client.bucket_obj.key is not None and client.bucket_obj.key.startswith("clips/")
    assert blob.uploaded_from == str(src)
    assert blob.content_type == "audio/wav"
    assert blob.signed_kwargs["version"] == "v4"
    assert blob.signed_kwargs["expiration"] == timedelta(days=1)
    assert not src.exists()
