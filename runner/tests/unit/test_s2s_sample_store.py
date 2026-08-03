# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve-time reads of the S2S samples bucket: index, filtering, object lookup."""

from __future__ import annotations

import json
from typing import Any

import pytest
from google.api_core.exceptions import NotFound

from coval_bench.s2s.samples import audio_object_key, load_sample, load_sample_ids

_BUCKET = "coval-benchmarks-s2s-samples"
_SAMPLE = "2026-07-30T00:00:00Z"
_MANIFEST_KEY = f"s2s-samples/{_SAMPLE}/manifest.json"
_HIDDEN = frozenset({("xai", "grok-voice-think-fast-1.0")})


def _recording(provider: str, model: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "object": f"s2s-samples/{_SAMPLE}/{provider}/{model}.wav",
        "coval_run_id": "run-1",
        "sim_id": "sim-1",
        "agent_id": "agent-1",
        "turns": [{"index": 0, "role": "user", "content": "hi"}],
    }


def _manifest(*recordings: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "bucket_at": _SAMPLE,
        "test_case_id": "tc-1",
        "recordings": list(recordings),
    }


# --- load_sample_ids --------------------------------------------------------


def test_load_sample_ids_newest_first() -> None:
    client = _FakeClient({"s2s-samples/index.json": ["2026-07-28T00:00:00Z", _SAMPLE]})
    assert load_sample_ids(_BUCKET, storage_client=client) == [_SAMPLE, "2026-07-28T00:00:00Z"]


def test_load_sample_ids_missing_index_is_empty() -> None:
    assert load_sample_ids(_BUCKET, storage_client=_FakeClient({})) == []


def test_load_sample_ids_drops_non_strings() -> None:
    client = _FakeClient({"s2s-samples/index.json": [_SAMPLE, 7, None]})
    assert load_sample_ids(_BUCKET, storage_client=client) == [_SAMPLE]


# --- load_sample: the embargo filter ----------------------------------------


def test_load_sample_strips_hidden_recordings() -> None:
    client = _FakeClient(
        {
            _MANIFEST_KEY: _manifest(
                _recording("openai", "gpt-realtime"), _recording("xai", "grok-voice-think-fast-1.0")
            )
        }
    )

    sample = load_sample(_BUCKET, _SAMPLE, hidden=_HIDDEN, storage_client=client)

    assert sample is not None
    assert [(r["provider"], r["model"]) for r in sample["recordings"]] == [
        ("openai", "gpt-realtime")
    ]


def test_load_sample_hidden_recording_takes_its_transcript_with_it() -> None:
    client = _FakeClient({_MANIFEST_KEY: _manifest(_recording("xai", "grok-voice-think-fast-1.0"))})

    sample = load_sample(_BUCKET, _SAMPLE, hidden=_HIDDEN, storage_client=client)

    assert sample is not None
    assert sample["recordings"] == []
    assert "hi" not in json.dumps(sample)


def test_load_sample_keeps_everything_for_an_unrestricted_caller() -> None:
    client = _FakeClient(
        {
            _MANIFEST_KEY: _manifest(
                _recording("openai", "gpt-realtime"), _recording("xai", "grok-voice-think-fast-1.0")
            )
        }
    )

    sample = load_sample(_BUCKET, _SAMPLE, hidden=frozenset(), storage_client=client)

    assert sample is not None
    assert len(sample["recordings"]) == 2


def test_load_sample_tolerates_v1_manifests() -> None:
    v1 = {
        "bucket_at": _SAMPLE,
        "test_case_id": "tc-1",
        "transcript": None,
        "input_audio_url": None,
        "recordings": [
            {
                "provider": "openai",
                "model": "gpt-realtime",
                "object": f"s2s-samples/{_SAMPLE}/openai.wav",
                "coval_run_id": "run-1",
                "sim_id": "sim-1",
            }
        ],
    }
    client = _FakeClient({_MANIFEST_KEY: v1})

    sample = load_sample(_BUCKET, _SAMPLE, hidden=_HIDDEN, storage_client=client)

    assert sample is not None
    assert len(sample["recordings"]) == 1
    assert "turns" not in sample["recordings"][0]


def test_load_sample_drops_recordings_without_an_object() -> None:
    broken = _manifest(_recording("openai", "gpt-realtime"))
    del broken["recordings"][0]["object"]
    client = _FakeClient({_MANIFEST_KEY: broken})

    sample = load_sample(_BUCKET, _SAMPLE, hidden=frozenset(), storage_client=client)

    assert sample is not None
    assert sample["recordings"] == []


def test_load_sample_drops_an_object_from_another_sample() -> None:
    strayed = _manifest(_recording("openai", "gpt-realtime"))
    strayed["recordings"][0]["object"] = "s2s-samples/2026-01-01T00:00:00Z/openai.wav"
    client = _FakeClient({_MANIFEST_KEY: strayed})

    sample = load_sample(_BUCKET, _SAMPLE, hidden=frozenset(), storage_client=client)

    assert sample is not None
    assert sample["recordings"] == []


def test_load_sample_drops_an_object_that_is_not_audio() -> None:
    strayed = _manifest(_recording("openai", "gpt-realtime"))
    strayed["recordings"][0]["object"] = _MANIFEST_KEY
    client = _FakeClient({_MANIFEST_KEY: strayed})

    sample = load_sample(_BUCKET, _SAMPLE, hidden=frozenset(), storage_client=client)

    assert sample is not None
    assert sample["recordings"] == []


def test_load_sample_drops_a_traversing_object() -> None:
    strayed = _manifest(_recording("openai", "gpt-realtime"))
    strayed["recordings"][0]["object"] = f"s2s-samples/{_SAMPLE}/../../secrets/leak.wav"
    client = _FakeClient({_MANIFEST_KEY: strayed})

    sample = load_sample(_BUCKET, _SAMPLE, hidden=frozenset(), storage_client=client)

    assert sample is not None
    assert sample["recordings"] == []


def test_load_sample_missing_manifest_is_none() -> None:
    assert load_sample(_BUCKET, _SAMPLE, hidden=frozenset(), storage_client=_FakeClient({})) is None


# --- audio_object_key: the check the redirect route relies on ---------------


def test_audio_object_key_returns_the_stored_path() -> None:
    client = _FakeClient({_MANIFEST_KEY: _manifest(_recording("openai", "gpt-realtime"))})

    key = audio_object_key(
        _BUCKET, _SAMPLE, "openai", "gpt-realtime", hidden=frozenset(), storage_client=client
    )

    assert key == f"s2s-samples/{_SAMPLE}/openai/gpt-realtime.wav"


def test_audio_object_key_refuses_a_hidden_model() -> None:
    client = _FakeClient({_MANIFEST_KEY: _manifest(_recording("xai", "grok-voice-think-fast-1.0"))})

    key = audio_object_key(
        _BUCKET, _SAMPLE, "xai", "grok-voice-think-fast-1.0", hidden=_HIDDEN, storage_client=client
    )

    assert key is None


def test_audio_object_key_refuses_an_object_outside_the_sample() -> None:
    strayed = _manifest(_recording("openai", "gpt-realtime"))
    strayed["recordings"][0]["object"] = "s2s-samples/2026-01-01T00:00:00Z/openai.wav"
    client = _FakeClient({_MANIFEST_KEY: strayed})

    key = audio_object_key(
        _BUCKET,
        _SAMPLE,
        "openai",
        "gpt-realtime",
        hidden=frozenset(),
        storage_client=client,
    )

    assert key is None


def test_audio_object_key_unknown_model_is_none() -> None:
    client = _FakeClient({_MANIFEST_KEY: _manifest(_recording("openai", "gpt-realtime"))})

    key = audio_object_key(
        _BUCKET, _SAMPLE, "openai", "no-such-model", hidden=frozenset(), storage_client=client
    )

    assert key is None


def test_audio_object_key_missing_manifest_is_none() -> None:
    key = audio_object_key(
        _BUCKET,
        _SAMPLE,
        "openai",
        "gpt-realtime",
        hidden=frozenset(),
        storage_client=_FakeClient({}),
    )

    assert key is None


# --- fakes ------------------------------------------------------------------


class _FakeBlob:
    def __init__(self, payload: object | None) -> None:
        self._payload = payload

    def download_as_bytes(self) -> bytes:
        if self._payload is None:
            raise NotFound("no such object")  # type: ignore[no-untyped-call]
        return json.dumps(self._payload).encode()


class _FakeBucket:
    def __init__(self, objects: dict[str, object]) -> None:
        self._objects = objects

    def blob(self, key: str) -> _FakeBlob:
        return _FakeBlob(self._objects.get(key))


class _FakeClient:
    def __init__(self, objects: dict[str, object]) -> None:
        self._objects = objects
        self.bucket_name: str | None = None

    def bucket(self, name: str) -> _FakeBucket:
        self.bucket_name = name
        return _FakeBucket(self._objects)


@pytest.fixture(autouse=True)
def _no_real_gcs(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode() -> None:
        raise AssertionError("test reached for a real GCS client")

    monkeypatch.setattr("coval_bench.gcs.client", _explode)
