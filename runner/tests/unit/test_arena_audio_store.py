# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for arena clip storage: store_clip (key) and clip_url (serve-time URL)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from coval_bench.arena.audio_store import clip_url, store_clip
from coval_bench.config import Settings


def _wav(tmp_path: Path) -> Path:
    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFFsynth")
    return src


# --- store_clip: returns an opaque key, consumes the source -----------------


def test_store_clip_local_returns_key_and_moves_file(tmp_path: Path) -> None:
    settings = Settings(arena_audio_dir=tmp_path / "store")
    key = store_clip(settings, _wav(tmp_path))

    assert key.startswith("clips/") and key.endswith(".wav")
    assert (settings.arena_audio_dir / key).is_file()


def test_store_clip_gcs_uploads_and_returns_key(tmp_path: Path) -> None:
    blob = _FakeBlob()
    client = _FakeClient(blob)
    src = _wav(tmp_path)
    settings = Settings(arena_gcs_bucket="coval-benchmarks-arena")

    key = store_clip(settings, src, storage_client=client)

    assert key.startswith("clips/") and key.endswith(".wav")
    assert client.bucket_name == "coval-benchmarks-arena"
    assert client.bucket_obj.key == key
    assert blob.uploaded_from == str(src)
    assert blob.content_type == "audio/wav"
    assert not src.exists()


# --- clip_url: builds a playable URL at serve time --------------------------


def test_clip_url_local_root_relative() -> None:
    settings = Settings()
    assert clip_url(settings, "clips/abc.wav") == "/clips/abc.wav"


def test_clip_url_local_base_url_prefix() -> None:
    settings = Settings(arena_audio_base_url="https://cdn.x/")
    assert clip_url(settings, "clips/abc.wav") == "https://cdn.x/clips/abc.wav"


def test_clip_url_gcs_signs_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    creds = SimpleNamespace(
        service_account_email="api-rt@proj.iam.gserviceaccount.com",
        token="tok",  # noqa: S106 — fake credential token for the stubbed signer
        refresh=lambda _request: None,
    )
    monkeypatch.setattr("google.auth.default", lambda: (creds, "proj"))
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())

    blob = _FakeBlob()
    client = _FakeClient(blob)
    settings = Settings(arena_gcs_bucket="coval-benchmarks-arena")

    url = clip_url(settings, "clips/abc.wav", storage_client=client)

    assert url == "https://signed.example/GET"
    assert client.bucket_name == "coval-benchmarks-arena"
    assert client.bucket_obj.key == "clips/abc.wav"
    assert blob.signed_kwargs["version"] == "v4"
    assert blob.signed_kwargs["expiration"] == timedelta(days=1)


def test_clip_url_gcs_rejects_non_service_account_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    creds = SimpleNamespace(token="tok", refresh=lambda _request: None)  # noqa: S106 — fake user creds
    monkeypatch.setattr("google.auth.default", lambda: (creds, "proj"))

    settings = Settings(arena_gcs_bucket="coval-benchmarks-arena")

    with pytest.raises(RuntimeError, match="service-account credentials"):
        clip_url(settings, "clips/abc.wav", storage_client=_FakeClient(_FakeBlob()))


# --- fakes ------------------------------------------------------------------


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
