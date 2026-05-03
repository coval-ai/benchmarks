# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for coval_bench.datasets.

Audio fixture ``tests/unit/fixtures/datasets/audio/0001.wav`` is a
1-second 440 Hz sine wave at 16 kHz mono PCM_16, generated once during
agent execution by:

    python3 -c "
    import wave, struct, math
    sample_rate = 16000
    amplitude = 16384
    freq = 440
    samples = [int(amplitude * math.sin(2*math.pi*freq*i/sample_rate))
               for i in range(sample_rate)]
    with wave.open('0001.wav', 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        wf.writeframes(struct.pack('<' + 'h'*sample_rate, *samples))
    "

Its SHA256 is 61648b306c4b096538b83eafa50fea3a22ae49918d48430859d29725243fb1d4.

Tests NEVER hit GCS — a fake ``storage.Client`` is injected via the
``storage_client`` kwarg on every ``load_stt_dataset`` call.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from coval_bench.config import Settings
from coval_bench.datasets.loader import (
    DatasetIntegrityError,
    TTSDataset,
    load_stt_dataset,
    load_tts_dataset,
)
from coval_bench.datasets.manifest import Manifest

# ---------------------------------------------------------------------------
# Fixtures directory helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "datasets"
AUDIO_DIR = FIXTURES_DIR / "audio"
FIXTURE_WAV = AUDIO_DIR / "0001.wav"
FIXTURE_SHA256 = "61648b306c4b096538b83eafa50fea3a22ae49918d48430859d29725243fb1d4"

VALID_MANIFEST_PATH = FIXTURES_DIR / "manifest-valid.json"
BAD_HASH_MANIFEST_PATH = FIXTURES_DIR / "manifest-bad-hash.json"


# ---------------------------------------------------------------------------
# Shared test settings
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_settings() -> Settings:
    return Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",  # type: ignore[arg-type]
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
    )


# ---------------------------------------------------------------------------
# Fake GCS client factory
#
# The fake client copies files from the local AUDIO_DIR into the
# requested download destination — no real network call is made.
# ---------------------------------------------------------------------------


def _make_fake_storage_client(audio_dir: Path) -> MagicMock:
    """Return a mock ``google.cloud.storage.Client`` that copies local files.

    The fake client intercepts ``download_to_filename`` and copies the
    matching file from *audio_dir* to the requested destination path.
    The blob ``name`` attribute is set to whatever string was passed to
    ``bucket.blob(name)`` so the side-effect can derive the filename.
    """

    def make_blob(name: str) -> MagicMock:
        blob = MagicMock()
        blob.name = name  # real attribute on GCS Blob objects

        def fake_download(dest: str, **_kwargs: object) -> None:
            filename = name.split("/")[-1]
            src = audio_dir / filename
            shutil.copy(str(src), dest)

        blob.download_to_filename.side_effect = fake_download
        return blob

    bucket_mock = MagicMock()
    bucket_mock.blob.side_effect = make_blob

    client_mock = MagicMock()
    client_mock.bucket.return_value = bucket_mock

    return client_mock


# ---------------------------------------------------------------------------
# Test 1: happy path — 2-item manifest, both items cached correctly
# ---------------------------------------------------------------------------


def test_happy_path(test_settings: Settings, tmp_path: Path) -> None:
    """load_stt_dataset returns a Dataset with 2 items; both files exist."""
    fake_client = _make_fake_storage_client(AUDIO_DIR)

    with patch(
        "coval_bench.datasets.loader._load_manifest",
        return_value=Manifest.model_validate_json(VALID_MANIFEST_PATH.read_text()),
    ):
        dataset = load_stt_dataset(
            "stt-v1",
            settings=test_settings,
            cache_dir=tmp_path,
            storage_client=fake_client,
        )

    assert dataset.id == "stt-v1"
    assert dataset.version == "1.0.0"
    assert len(dataset.items) == 2
    for item in dataset.items:
        assert item.path.exists(), f"Expected {item.path} to exist on disk"
        assert item.sha256 == FIXTURE_SHA256


# ---------------------------------------------------------------------------
# Test 2: bad hash raises DatasetIntegrityError
# ---------------------------------------------------------------------------


def test_bad_hash_raises_integrity_error(test_settings: Settings, tmp_path: Path) -> None:
    """Manifest with wrong SHA256 must raise DatasetIntegrityError."""
    fake_client = _make_fake_storage_client(AUDIO_DIR)

    with (
        patch(
            "coval_bench.datasets.loader._load_manifest",
            return_value=Manifest.model_validate_json(BAD_HASH_MANIFEST_PATH.read_text()),
        ),
        pytest.raises(DatasetIntegrityError) as exc_info,
    ):
        load_stt_dataset(
            "stt-v1",
            settings=test_settings,
            cache_dir=tmp_path,
            storage_client=fake_client,
        )

    err = exc_info.value
    assert "audio/0001.wav" in err.path
    assert err.expected == "a" * 64  # manifest-bad-hash.json value
    assert err.actual == FIXTURE_SHA256
    # Message must include all three pieces of info
    msg = str(err)
    assert "audio/0001.wav" in msg
    assert err.expected in msg
    assert err.actual in msg


# ---------------------------------------------------------------------------
# Test 3: cache hit — GCS is called only once
# ---------------------------------------------------------------------------


def test_cache_hit_skips_download(test_settings: Settings, tmp_path: Path) -> None:
    """Second call to load_stt_dataset with same cache_dir skips download."""
    fake_client = _make_fake_storage_client(AUDIO_DIR)

    manifest = Manifest.model_validate_json(VALID_MANIFEST_PATH.read_text())

    with patch("coval_bench.datasets.loader._load_manifest", return_value=manifest):
        # First call — triggers download for each item
        load_stt_dataset(
            "stt-v1",
            settings=test_settings,
            cache_dir=tmp_path,
            storage_client=fake_client,
        )

        # blob() is called once per item during download; record the count
        blob_call_count_after_first = fake_client.bucket.return_value.blob.call_count

        # Second call — all items should be cache hits; blob() must NOT be called again
        load_stt_dataset(
            "stt-v1",
            settings=test_settings,
            cache_dir=tmp_path,
            storage_client=fake_client,
        )

        blob_call_count_after_second = fake_client.bucket.return_value.blob.call_count

    assert blob_call_count_after_second == blob_call_count_after_first, (
        "Expected no additional GCS calls on second load (cache hit); "
        f"got {blob_call_count_after_second - blob_call_count_after_first} extra call(s)"
    )


# ---------------------------------------------------------------------------
# Test 4: manifest validation failure propagates ValidationError
# ---------------------------------------------------------------------------


def test_manifest_validation_failure(test_settings: Settings, tmp_path: Path) -> None:
    """A manifest missing 'version' causes Pydantic ValidationError."""
    bad_manifest_json = json.dumps(
        {
            "id": "stt-v1",
            # version is missing
            "license": "CC-BY-4.0",
            "source": "test",
            "items": [],
        }
    )

    with (
        patch(
            "coval_bench.datasets.loader._load_manifest",
            side_effect=lambda _id: Manifest.model_validate_json(bad_manifest_json),
        ),
        pytest.raises(ValidationError),
    ):
        load_stt_dataset(
            "stt-v1",
            settings=test_settings,
            cache_dir=tmp_path,
            storage_client=MagicMock(),
        )


# ---------------------------------------------------------------------------
# Test 5: TTS-only path — items come back with text only
# ---------------------------------------------------------------------------


def test_tts_dataset_text_only(test_settings: Settings, tmp_path: Path) -> None:
    """load_tts_dataset returns TTSDataset with text-only items."""
    from coval_bench.datasets.manifest import TTSManifestItem

    tts_manifest = Manifest(
        id="tts-v1",
        version="1.0.0",
        license="proprietary",
        source="test",
        items=[
            TTSManifestItem(testcase_id="TC001", transcript="hello world"),
            TTSManifestItem(testcase_id="TC002", transcript="goodbye world"),
        ],
    )

    with patch("coval_bench.datasets.loader._load_manifest", return_value=tts_manifest):
        result = load_tts_dataset(
            "tts-v1",
            settings=test_settings,
            cache_dir=tmp_path,
            storage_client=MagicMock(),
        )

    assert isinstance(result, TTSDataset)
    assert len(result.items) == 2
    assert result.items[0].testcase_id == "TC001"
    assert result.items[0].transcript == "hello world"
    # TTS items have no path or sha256
    assert not hasattr(result.items[0], "path")
    assert not hasattr(result.items[0], "sha256")


# ---------------------------------------------------------------------------
# Test 6: GCS path resolution — URI matches expected pattern
# ---------------------------------------------------------------------------


def test_gcs_path_resolution(test_settings: Settings, tmp_path: Path) -> None:
    """GCS object path is exactly '<dataset_id>/<item.path>'."""
    fake_client = _make_fake_storage_client(AUDIO_DIR)

    manifest = Manifest.model_validate_json(
        json.dumps(
            {
                "_license": "Apache-2.0",
                "id": "stt-v1",
                "version": "1.0.0",
                "license": "CC-BY-4.0",
                "source": "test",
                "items": [
                    {
                        "path": "audio/0001.wav",
                        "sha256": FIXTURE_SHA256,
                        "transcript": "test",
                        "duration_sec": 1.0,
                    }
                ],
            }
        )
    )

    with patch("coval_bench.datasets.loader._load_manifest", return_value=manifest):
        load_stt_dataset(
            "stt-v1",
            settings=test_settings,
            cache_dir=tmp_path,
            storage_client=fake_client,
        )

    # bucket name must be settings.dataset_bucket
    fake_client.bucket.assert_called_once_with(test_settings.dataset_bucket)
    # blob name must be "<dataset_id>/<item.path>"
    fake_client.bucket.return_value.blob.assert_called_once_with("stt-v1/audio/0001.wav")


# ---------------------------------------------------------------------------
# Test 7: empty manifest — returns Dataset with empty items, no error
# ---------------------------------------------------------------------------


def test_empty_manifest_items(test_settings: Settings, tmp_path: Path) -> None:
    """Manifest with items: [] → Dataset with empty items list; no exception."""
    empty_manifest = Manifest(
        id="stt-v1",
        version="1.0.0",
        license="CC-BY-4.0",
        source="test",
        items=[],
    )

    with patch("coval_bench.datasets.loader._load_manifest", return_value=empty_manifest):
        result = load_stt_dataset(
            "stt-v1",
            settings=test_settings,
            cache_dir=tmp_path,
            storage_client=MagicMock(),
        )

    assert result.items == []
    assert result.id == "stt-v1"


# ---------------------------------------------------------------------------
# Sanity: fixture WAV exists and has the expected SHA256
# ---------------------------------------------------------------------------


def test_fixture_wav_exists_and_has_correct_sha256() -> None:
    """Regression guard: the fixture WAV must exist with the recorded SHA256."""
    assert FIXTURE_WAV.exists(), f"Fixture WAV not found at {FIXTURE_WAV}"
    with FIXTURE_WAV.open("rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    assert actual == FIXTURE_SHA256, (
        f"Fixture WAV SHA256 changed!\n  expected: {FIXTURE_SHA256}\n  actual:   {actual}"
    )


# ---------------------------------------------------------------------------
# Coverage helpers: _load_manifest, load_dataset dispatcher, stale cache
# ---------------------------------------------------------------------------


def test_load_manifest_reads_packaged_tts_manifest() -> None:
    """_load_manifest round-trips the packaged tts-v1.json manifest."""
    from coval_bench.datasets.loader import _load_manifest

    manifest = _load_manifest("tts-v1")
    assert manifest.id == "tts-v1"
    assert len(manifest.items) == 30  # 30 curated TTS prompts


def test_load_manifest_reads_packaged_stt_manifest() -> None:
    """_load_manifest round-trips the packaged stt-v1.json manifest."""
    from coval_bench.datasets.loader import _load_manifest

    manifest = _load_manifest("stt-v1")
    assert manifest.id == "stt-v1"
    assert len(manifest.items) == 50  # 50-utterance LibriSpeech sample (ADR-020)


def test_load_dataset_dispatcher_stt(test_settings: Settings, tmp_path: Path) -> None:
    """load_dataset routes to load_stt_dataset when manifest has STT items."""
    from coval_bench.datasets.loader import Dataset, load_dataset

    fake_client = _make_fake_storage_client(AUDIO_DIR)
    manifest = Manifest.model_validate_json(VALID_MANIFEST_PATH.read_text())

    with patch("coval_bench.datasets.loader._load_manifest", return_value=manifest):
        result = load_dataset(
            "stt-v1",
            settings=test_settings,
            cache_dir=tmp_path,
            storage_client=fake_client,
        )

    assert isinstance(result, Dataset)
    assert len(result.items) == 2


def test_load_dataset_dispatcher_tts(test_settings: Settings, tmp_path: Path) -> None:
    """load_dataset routes to load_tts_dataset when manifest has TTS items."""
    from coval_bench.datasets.loader import TTSDataset, load_dataset
    from coval_bench.datasets.manifest import TTSManifestItem

    tts_manifest = Manifest(
        id="tts-v1",
        version="1.0.0",
        license="proprietary",
        source="test",
        items=[TTSManifestItem(testcase_id="TC001", transcript="hello")],
    )

    with patch("coval_bench.datasets.loader._load_manifest", return_value=tts_manifest):
        result = load_dataset(
            "tts-v1",
            settings=test_settings,
            cache_dir=tmp_path,
            storage_client=MagicMock(),
        )

    assert isinstance(result, TTSDataset)


def test_load_dataset_dispatcher_empty(test_settings: Settings, tmp_path: Path) -> None:
    """load_dataset with empty items falls through to TTS path without error."""
    from coval_bench.datasets.loader import TTSDataset, load_dataset

    empty_manifest = Manifest(
        id="tts-v1",
        version="1.0.0",
        license="proprietary",
        source="test",
        items=[],
    )

    with patch("coval_bench.datasets.loader._load_manifest", return_value=empty_manifest):
        result = load_dataset(
            "tts-v1",
            settings=test_settings,
            cache_dir=tmp_path,
            storage_client=MagicMock(),
        )

    assert isinstance(result, TTSDataset)
    assert result.items == []


def test_stale_cache_redownloads(test_settings: Settings, tmp_path: Path) -> None:
    """A cached file with wrong SHA is re-downloaded (stale-cache branch)."""
    fake_client = _make_fake_storage_client(AUDIO_DIR)

    # Use a single-item manifest pointing at 0001.wav
    manifest = Manifest.model_validate_json(
        json.dumps(
            {
                "_license": "Apache-2.0",
                "id": "stt-v1",
                "version": "1.0.0",
                "license": "CC-BY-4.0",
                "source": "test",
                "items": [
                    {
                        "path": "audio/0001.wav",
                        "sha256": FIXTURE_SHA256,
                        "transcript": "test",
                        "duration_sec": 1.0,
                    }
                ],
            }
        )
    )

    # Pre-seed the cache with a corrupt file
    stale_path = tmp_path / "stt-v1" / "audio" / "0001.wav"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_bytes(b"corrupt")

    with patch("coval_bench.datasets.loader._load_manifest", return_value=manifest):
        dataset = load_stt_dataset(
            "stt-v1",
            settings=test_settings,
            cache_dir=tmp_path,
            storage_client=fake_client,
        )

    # Should have re-downloaded and now have the correct hash
    assert dataset.items[0].sha256 == FIXTURE_SHA256
    # download_to_filename was called (re-download happened)
    fake_client.bucket.return_value.blob.assert_called_once()


def test_make_storage_client_anonymous(test_settings: Settings) -> None:
    """_make_storage_client returns anonymous client when no credentials set."""
    from coval_bench.datasets.loader import _make_storage_client

    # test_settings has google_application_credentials=None
    assert test_settings.google_application_credentials is None
    client = _make_storage_client(test_settings)
    # Just check we get a Client object back without error
    assert client is not None


def test_default_cache_dir_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_default_cache_dir uses XDG_CACHE_HOME when set."""
    from coval_bench.datasets.loader import _default_cache_dir

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    result = _default_cache_dir()
    assert result == tmp_path / "coval-bench"
    assert result.is_dir()
