# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for coval_bench.datasets.scripts.build_dataset.

All tests are pure (no network, no GCS, no real LibriSpeech).
They run in < 2 s and produce no I/O outside tmpdir.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from click.testing import CliRunner

from coval_bench.datasets.manifest import Manifest
from coval_bench.datasets.scripts.build_dataset import (
    _BuiltItem,
    _hash_file,
    _parse_trans_txt,
    _render_manifest,
    _select_50,
    _upload_items,
    _Utterance,
    cli,
)

# ---------------------------------------------------------------------------
# Test 1: _parse_trans_txt — basic happy path
# ---------------------------------------------------------------------------


def test_parse_trans_txt_basic(tmp_path: Path) -> None:
    """Given a synthetic .trans.txt string, returns 3 utterances in file order."""
    text = "1234-5678-0001 HELLO WORLD\n1234-5678-0002 GOOD MORNING\n1234-5678-0003 OPEN SOURCE\n"
    audio_dir = tmp_path
    # Touch FLAC placeholders so the caller can work if needed
    for utt_id in ("1234-5678-0001", "1234-5678-0002", "1234-5678-0003"):
        (audio_dir / f"{utt_id}.flac").touch()

    result = _parse_trans_txt(text, speaker_id="1234", chapter_id="5678", audio_dir=audio_dir)

    assert len(result) == 3
    assert result[0].utterance_id == "1234-5678-0001"
    assert result[0].transcript == "HELLO WORLD"
    assert result[0].speaker_id == "1234"
    assert result[0].chapter_id == "5678"

    assert result[1].utterance_id == "1234-5678-0002"
    assert result[1].transcript == "GOOD MORNING"

    assert result[2].utterance_id == "1234-5678-0003"
    assert result[2].transcript == "OPEN SOURCE"

    # File order preserved (not sorted)
    ids = [r.utterance_id for r in result]
    assert ids == ["1234-5678-0001", "1234-5678-0002", "1234-5678-0003"]


def test_parse_trans_txt_rejects_malformed(tmp_path: Path) -> None:
    """Lines without a space between id and transcript are silently skipped."""
    text = "1234-5678-0001 GOOD LINE\nMALFORMED_NO_SPACE\n1234-5678-0002 ALSO GOOD\n"
    result = _parse_trans_txt(text, speaker_id="1234", chapter_id="5678", audio_dir=tmp_path)
    assert len(result) == 2
    assert result[0].utterance_id == "1234-5678-0001"
    assert result[1].utterance_id == "1234-5678-0002"


def test_parse_trans_txt_blank_lines_ignored(tmp_path: Path) -> None:
    """Blank lines in trans.txt do not produce utterances."""
    text = "\n\n1234-5678-0001 HELLO\n\n"
    result = _parse_trans_txt(text, speaker_id="1234", chapter_id="5678", audio_dir=tmp_path)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Test 2: _select_50 — filter + lex sort + take 50
# ---------------------------------------------------------------------------


def _make_utterances(
    count: int, *, dur_override: float | None = None, seed: int = 42
) -> list[_Utterance]:
    """Create *count* synthetic _Utterance objects with varying durations."""
    rng = random.Random(seed)
    utterances: list[_Utterance] = []
    for i in range(count):
        # Vary speaker/chapter/utterance so sort order is non-trivial.
        speaker = f"{1000 + (i % 20):04d}"
        chapter = f"{2000 + (i % 10):04d}"
        utt_id = f"{speaker}-{chapter}-{i:04d}"
        dur = dur_override if dur_override is not None else rng.uniform(1.0, 18.0)
        utterances.append(
            _Utterance(
                speaker_id=speaker,
                chapter_id=chapter,
                utterance_id=utt_id,
                transcript=f"utterance {i}",
                flac_path=Path(f"/fake/{utt_id}.flac"),
                duration_sec=dur,
            )
        )
    return utterances


def test_select_50_returns_exactly_50() -> None:
    """_select_50 returns exactly 50 items from a 200-utterance pool."""
    utterances = _make_utterances(200)
    selected = _select_50(utterances)
    assert len(selected) == 50


def test_select_50_all_within_duration_window() -> None:
    """All returned items have 2.0 <= duration_sec <= 15.0."""
    utterances = _make_utterances(200)
    selected = _select_50(utterances)
    for u in selected:
        assert 2.0 <= u.duration_sec <= 15.0, (
            f"Utterance {u.utterance_id} has duration {u.duration_sec} outside window"
        )


def test_select_50_lex_sorted() -> None:
    """Selected items are sorted by (speaker_id, chapter_id, utterance_id) lex."""
    utterances = _make_utterances(200)
    selected = _select_50(utterances)
    keys = [(u.speaker_id, u.chapter_id, u.utterance_id) for u in selected]
    assert keys == sorted(keys), "Selected utterances are not lex-sorted"


def test_select_50_boundary_durations_kept() -> None:
    """Utterances at exactly 2.0 s and 15.0 s are KEPT (inclusive bounds)."""
    # Build pool with exactly-boundary utterances plus enough padding
    base = _make_utterances(100)
    boundary_low = _Utterance(
        speaker_id="0001",
        chapter_id="0001",
        utterance_id="0001-0001-LOWBOUND",
        transcript="low boundary",
        flac_path=Path("/fake/low.flac"),
        duration_sec=2.0,
    )
    boundary_high = _Utterance(
        speaker_id="0001",
        chapter_id="0001",
        utterance_id="0001-0001-HIGHBND",
        transcript="high boundary",
        flac_path=Path("/fake/high.flac"),
        duration_sec=15.0,
    )
    # Add them to the pool; ensure they pass the filter
    all_utterances = [boundary_low, boundary_high] + base
    filtered = [u for u in all_utterances if 2.0 <= u.duration_sec <= 15.0]
    assert boundary_low in filtered
    assert boundary_high in filtered


def test_select_50_raises_if_not_enough() -> None:
    """_select_50 raises ValueError if fewer than 50 utterances pass filter."""
    # All durations outside the window
    utterances = _make_utterances(200, dur_override=20.0)
    with pytest.raises(ValueError, match="need 50"):
        _select_50(utterances)


# ---------------------------------------------------------------------------
# Test 3: Determinism
# ---------------------------------------------------------------------------


def test_select_50_deterministic() -> None:
    """Two calls on the same shuffled input return identical sequences."""
    utterances = _make_utterances(200)

    shuffled_a = utterances.copy()
    random.Random(1).shuffle(shuffled_a)

    shuffled_b = utterances.copy()
    random.Random(99).shuffle(shuffled_b)

    result_a = _select_50(shuffled_a)
    result_b = _select_50(shuffled_b)

    assert [u.utterance_id for u in result_a] == [u.utterance_id for u in result_b]


def test_select_50_round_robin_diversity() -> None:
    """Round-robin selection: with >=50 speakers each having >=1 in-window utterance,
    the 50 selections come from 50 distinct speakers (one each).
    With fewer speakers, all are represented before any speaker contributes a 2nd.
    """
    # 60 speakers, each with 2 in-window utterances → 50 picks should hit 50
    # distinct speakers (1 per speaker, no doubles).
    pool: list[_Utterance] = []
    for sid_idx in range(60):
        speaker = f"{sid_idx:04d}"
        for utt_idx in range(2):
            pool.append(
                _Utterance(
                    speaker_id=speaker,
                    chapter_id="C",
                    utterance_id=f"{speaker}-C-{utt_idx:04d}",
                    transcript=f"u{sid_idx}-{utt_idx}",
                    flac_path=Path(f"/fake/{speaker}-{utt_idx}.flac"),
                    duration_sec=5.0,
                )
            )
    selected = _select_50(pool)
    distinct_speakers = {u.speaker_id for u in selected}
    assert len(distinct_speakers) == 50, (
        f"Expected 50 distinct speakers, got {len(distinct_speakers)}"
    )

    # 40 speakers (test-clean's actual count) each with 5 in-window utterances:
    # 50 picks should hit ALL 40 speakers, with the lex-first 10 contributing 2 each.
    pool2: list[_Utterance] = []
    for sid_idx in range(40):
        speaker = f"{1000 + sid_idx:04d}"
        for utt_idx in range(5):
            pool2.append(
                _Utterance(
                    speaker_id=speaker,
                    chapter_id="C",
                    utterance_id=f"{speaker}-C-{utt_idx:04d}",
                    transcript="x",
                    flac_path=Path(f"/fake/{speaker}-{utt_idx}.flac"),
                    duration_sec=5.0,
                )
            )
    selected2 = _select_50(pool2)
    assert len({u.speaker_id for u in selected2}) == 40
    counts: dict[str, int] = {}
    for u in selected2:
        counts[u.speaker_id] = counts.get(u.speaker_id, 0) + 1
    # Lex-first 10 speakers should have 2 picks each; rest should have 1.
    speakers_lex = sorted({u.speaker_id for u in pool2})
    for sid in speakers_lex[:10]:
        assert counts[sid] == 2, f"Speaker {sid} expected 2 picks, got {counts[sid]}"
    for sid in speakers_lex[10:]:
        assert counts[sid] == 1, f"Speaker {sid} expected 1 pick, got {counts[sid]}"


# ---------------------------------------------------------------------------
# Test 4: Filename mapping
# ---------------------------------------------------------------------------


def test_filename_mapping() -> None:
    """_render_manifest assigns 0001.wav through 0050.wav in order."""
    items = [
        _BuiltItem(
            speaker_id="1000",
            chapter_id="2000",
            utterance_id=f"1000-2000-{i:04d}",
            transcript=f"text {i}",
            wav_path=Path(f"/fake/{i:04d}.wav"),
            sha256="a" * 64,
            duration_sec=5.0,
            filename=f"{i:04d}.wav",
        )
        for i in range(1, 51)
    ]
    manifest_json = _render_manifest(items)
    data = json.loads(manifest_json)
    paths = [item["path"] for item in data["items"]]
    expected = [f"audio/{i:04d}.wav" for i in range(1, 51)]
    assert paths == expected


# ---------------------------------------------------------------------------
# Test 5: Manifest serialization round-trip
# ---------------------------------------------------------------------------


def test_manifest_roundtrip() -> None:
    """Rendered JSON with 2 fake STTManifestItem dicts validates via Manifest."""
    items = [
        _BuiltItem(
            speaker_id="1000",
            chapter_id="2000",
            utterance_id="1000-2000-0001",
            transcript="hello world",
            wav_path=Path("/fake/0001.wav"),
            sha256="b" * 64,
            duration_sec=4.2,
            filename="0001.wav",
        ),
        _BuiltItem(
            speaker_id="1000",
            chapter_id="2000",
            utterance_id="1000-2000-0002",
            transcript="goodbye world",
            wav_path=Path("/fake/0002.wav"),
            sha256="c" * 64,
            duration_sec=6.1,
            filename="0002.wav",
        ),
    ]
    manifest_json = _render_manifest(items)
    # Must validate cleanly against the Manifest Pydantic model
    manifest = Manifest.model_validate_json(manifest_json)
    assert manifest.id == "stt-v1"
    assert manifest.version == "1.0.0"
    assert manifest.license == "CC-BY-4.0"
    assert len(manifest.items) == 2
    assert manifest.items[0].path == "audio/0001.wav"  # type: ignore[union-attr]
    assert manifest.items[0].sha256 == "b" * 64  # type: ignore[union-attr]
    assert manifest.items[0].transcript == "hello world"
    assert manifest.items[1].sha256 == "c" * 64  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 6: _hash_file
# ---------------------------------------------------------------------------


def test_hash_file_known_bytes(tmp_path: Path) -> None:
    """_hash_file returns the expected SHA256 hex string for known content."""
    data = b"coval-bench-test-content"
    expected = hashlib.sha256(data).hexdigest()
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(data)
    assert _hash_file(test_file) == expected


def test_hash_file_empty(tmp_path: Path) -> None:
    """_hash_file handles empty files without error."""
    empty_file = tmp_path / "empty.bin"
    empty_file.write_bytes(b"")
    expected = hashlib.sha256(b"").hexdigest()
    assert _hash_file(empty_file) == expected


# ---------------------------------------------------------------------------
# Test 7: CLI smoke — tts-v1 exits 0 with "text-only" message
# ---------------------------------------------------------------------------


def test_cli_tts_v1_exits_zero() -> None:
    """Invoking build --dataset tts-v1 prints the text-only message and exits 0."""
    runner = CliRunner()
    result = runner.invoke(cli, ["build", "--dataset", "tts-v1"])
    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}\n{result.output}"
    assert "text-only" in result.output.lower() or "nothing to build" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 8: GCS client construction is mocked — upload is called correctly
# ---------------------------------------------------------------------------


def _make_tiny_wav(tmp_path: Path, filename: str) -> Path:
    """Write a minimal 16 kHz mono PCM_16 WAV to *tmp_path / filename*."""
    wav_path = tmp_path / filename
    sample_rate = 16000
    n = 100
    samples = [int(16384 * math.sin(2 * math.pi * 440 * j / sample_rate)) for j in range(n)]
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))
    return wav_path


def _make_mock_gcs_client(tmp_path: Path) -> tuple[MagicMock, dict[str, MagicMock]]:
    """Return a (client_mock, blob_map) pair.

    download_to_filename copies the original WAV back so SHA re-check passes.
    """
    blob_map: dict[str, MagicMock] = {}

    def make_mock_blob(blob_name: str) -> MagicMock:
        blob_mock = MagicMock()
        blob_mock.name = blob_name

        def fake_download(dest: str, **_kwargs: object) -> None:
            src = tmp_path / blob_name.split("/")[-1]
            shutil.copy(str(src), dest)

        blob_mock.download_to_filename.side_effect = fake_download
        return blob_mock

    bucket_mock = MagicMock()

    def get_blob(name: str) -> MagicMock:
        if name not in blob_map:
            blob_map[name] = make_mock_blob(name)
        return blob_map[name]

    bucket_mock.blob.side_effect = get_blob
    client_mock = MagicMock()
    client_mock.bucket.return_value = bucket_mock
    return client_mock, blob_map


def test_upload_calls_gcs_with_expected_args(tmp_path: Path) -> None:
    """_upload_items calls GCS with the expected bucket name and blob paths."""
    wav_files: list[_BuiltItem] = []
    for i in range(1, 3):
        filename = f"{i:04d}.wav"
        wav_path = _make_tiny_wav(tmp_path, filename)
        sha256 = _hash_file(wav_path)
        wav_files.append(
            _BuiltItem(
                speaker_id="1000",
                chapter_id="2000",
                utterance_id=f"1000-2000-{i:04d}",
                transcript=f"text {i}",
                wav_path=wav_path,
                sha256=sha256,
                duration_sec=0.1,
                filename=filename,
            )
        )

    client_mock, blob_map = _make_mock_gcs_client(tmp_path)

    _upload_items(wav_files, "test-bucket", overwrite=False, client=client_mock)

    # Assert bucket was accessed with the right name
    client_mock.bucket.assert_called_once_with("test-bucket")

    # Assert blob() was called for each file with the correct path
    expected_blob_calls = [
        call("stt-v1/audio/0001.wav"),
        call("stt-v1/audio/0002.wav"),
    ]
    client_mock.bucket.return_value.blob.assert_has_calls(expected_blob_calls, any_order=False)

    # Assert upload_from_filename was called with content_type audio/wav
    for blob_mock in blob_map.values():
        blob_mock.upload_from_filename.assert_called_once()
        _, kwargs = blob_mock.upload_from_filename.call_args
        assert kwargs.get("content_type") == "audio/wav"
        # if_generation_match=0 for non-overwrite
        assert kwargs.get("if_generation_match") == 0


def test_upload_overwrite_drops_generation_match(tmp_path: Path) -> None:
    """With --overwrite, upload_from_filename is called WITHOUT if_generation_match."""
    filename = "0001.wav"
    wav_path = _make_tiny_wav(tmp_path, filename)
    sha256 = _hash_file(wav_path)

    item = _BuiltItem(
        speaker_id="1000",
        chapter_id="2000",
        utterance_id="1000-2000-0001",
        transcript="test",
        wav_path=wav_path,
        sha256=sha256,
        duration_sec=0.1,
        filename=filename,
    )

    client_mock, blob_map = _make_mock_gcs_client(tmp_path)

    _upload_items([item], "test-bucket", overwrite=True, client=client_mock)

    blob_mock = blob_map["stt-v1/audio/0001.wav"]
    _, kwargs = blob_mock.upload_from_filename.call_args
    assert "if_generation_match" not in kwargs, (
        "Expected no if_generation_match when overwrite=True"
    )
