# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""build_dataset.py — offline tool to build and upload dataset manifests.

## Algorithm (ADR-020 — LibriSpeech test-clean, deterministic 50)

Downloads LibriSpeech ``test-clean``, deterministically picks 50 utterances,
transcodes to 16 kHz mono PCM_16 WAV, SHA256-hashes each file, uploads to
GCS, and regenerates ``stt-v1.json``.

**Deterministic subset selection contract (ADR-020, non-negotiable):**

1. Download the LibriSpeech ``test-clean`` tar from OpenSLR::

       https://www.openslr.org/resources/12/test-clean.tar.gz

2. Extract into ``~/.cache/coval-bench/librispeech/``.

3. Walk ``LibriSpeech-test-clean/test-clean/<speaker_id>/<chapter_id>/``
   and parse every ``.trans.txt`` to build utterance records.

4. Decode FLAC header (``soundfile.info``) to get ``duration_sec``.

5. Filter to ``2.0 <= duration_sec <= 15.0`` (inclusive both ends).

6. Sort the filtered list by ``(speaker_id, chapter_id, utterance_id)``
   — **all three are LEXICOGRAPHIC STRING SORTS**.  Do NOT int-cast.
   This is the reproducibility contract.

7. Take the first 50.

8. Transcode each to 16 kHz mono PCM_16 WAV; compute SHA256.

9. Upload to ``gs://<bucket>/stt-v1/audio/0001.wav`` etc.  Use
   ``if_generation_match=0`` so existing objects are refused (v1 freeze).
   Pass ``--overwrite`` to drop this precondition.

10. Re-download each blob and verify SHA256 (upload integrity check).

11. Write the regenerated manifest JSON.

``--dry-run`` executes steps 1–8 (download, parse, filter, sort, transcode,
hash) but skips 9–11.  Prints the manifest to stdout instead of writing it.

Usage::

    uv run python -m coval_bench.datasets.scripts.build_dataset build \\
        --dataset stt-v1 \\
        --bucket coval-benchmarks-datasets \\
        --dry-run

    uv run python -m coval_bench.datasets.scripts.build_dataset build \\
        --dataset stt-v1 \\
        --bucket coval-benchmarks-datasets
"""

from __future__ import annotations

import hashlib
import importlib.resources as _impres
import json
import logging
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click
import soundfile as sf

if TYPE_CHECKING:
    from google.cloud import storage as _gcs_storage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENSLR_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
_EXPECTED_SIZE_BYTES = 346_663_984  # ~346 MB — used as a cached-copy sanity check
_CACHE_ROOT = Path.home() / ".cache" / "coval-bench" / "librispeech"
_ARCHIVE_NAME = "test-clean.tar.gz"
_EXTRACT_DIR_NAME = "LibriSpeech"
_CHUNK_SIZE = 1024 * 1024  # 1 MB
_LOG_EVERY_BYTES = 10 * 1024 * 1024  # log progress every 10 MB
_TARGET_SR = 16_000
_NUM_UTTERANCES = 50
_DURATION_MIN = 2.0
_DURATION_MAX = 15.0

# GCS object path for uploads: stt-v1/audio/0001.wav etc.
_GCS_AUDIO_PREFIX = "stt-v1/audio"


# ---------------------------------------------------------------------------
# Dataclass for an utterance record (internal only)
# ---------------------------------------------------------------------------


@dataclass
class _Utterance:
    """Raw utterance record, before transcoding."""

    speaker_id: str
    chapter_id: str
    utterance_id: str
    transcript: str
    flac_path: Path
    duration_sec: float = 0.0


@dataclass
class _BuiltItem:
    """Utterance after transcoding + hashing, ready for manifest / upload."""

    speaker_id: str
    chapter_id: str
    utterance_id: str
    transcript: str
    wav_path: Path
    sha256: str
    duration_sec: float
    filename: str  # e.g. "0001.wav"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_archive(cache_root: Path) -> Path:
    """Download test-clean.tar.gz to *cache_root* (streaming, skip if cached).

    Skips the download if a file with the expected size already exists at the
    destination path — this is cheap and avoids an extra HEAD request.

    Returns the path to the local archive.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    dest = cache_root / _ARCHIVE_NAME

    if dest.exists() and dest.stat().st_size == _EXPECTED_SIZE_BYTES:
        logger.info("Archive already cached at %s (%d bytes)", dest, dest.stat().st_size)
        return dest

    logger.info("Downloading %s → %s", _OPENSLR_URL, dest)
    req = urllib.request.Request(  # noqa: S310 (audited: hardcoded https OpenSLR URL)
        _OPENSLR_URL,
        headers={"User-Agent": "coval-bench/build-dataset"},
    )
    downloaded = 0
    last_logged = 0
    with urllib.request.urlopen(req) as response, dest.open("wb") as out:  # noqa: S310 (audited: hardcoded https OpenSLR URL)
        while True:
            chunk = response.read(_CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if downloaded - last_logged >= _LOG_EVERY_BYTES:
                logger.info("Downloaded %d MB …", downloaded // (1024 * 1024))
                last_logged = downloaded

    logger.info("Download complete: %d bytes written to %s", downloaded, dest)
    return dest


def _extract_archive(archive_path: Path, cache_root: Path) -> Path:
    """Extract *archive_path* into *cache_root* (skip if already extracted).

    Returns the path to the extracted ``LibriSpeech`` directory.
    """
    extract_dir = cache_root / _EXTRACT_DIR_NAME
    if extract_dir.exists():
        logger.info("Archive already extracted at %s", extract_dir)
        return extract_dir

    logger.info("Extracting %s → %s", archive_path, cache_root)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(path=cache_root)  # noqa: S202
    logger.info("Extraction complete")
    return extract_dir


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_trans_txt(
    text: str, speaker_id: str, chapter_id: str, audio_dir: Path
) -> list[_Utterance]:
    """Parse a LibriSpeech ``.trans.txt`` file.

    Each non-blank line has the form::

        <utterance_id> <transcript text>

    Lines that do not contain a space are silently skipped (malformed).

    Args:
        text:        Raw file contents.
        speaker_id:  Parent speaker directory name (string, not int).
        chapter_id:  Parent chapter directory name (string, not int).
        audio_dir:   Directory that contains the FLAC files.

    Returns:
        List of :class:`_Utterance` in file order (no sorting here).
    """
    utterances: list[_Utterance] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if " " not in line:
            # Malformed — no space between utterance_id and transcript
            logger.warning("Skipping malformed trans.txt line: %r", line)
            continue
        utt_id, transcript = line.split(" ", 1)
        flac_path = audio_dir / f"{utt_id}.flac"
        utterances.append(
            _Utterance(
                speaker_id=speaker_id,
                chapter_id=chapter_id,
                utterance_id=utt_id,
                transcript=transcript,
                flac_path=flac_path,
            )
        )
    return utterances


def _enumerate_utterances(librispeech_dir: Path) -> list[_Utterance]:
    """Walk test-clean tree and return all utterances with header-decoded durations.

    LibriSpeech layout::

        LibriSpeech/test-clean/<speaker_id>/<chapter_id>/
            <utt_id>.flac
            <speaker_id>-<chapter_id>.trans.txt

    ``soundfile.info(path)`` is used to read FLAC header metadata
    (duration_sec = frames / samplerate) without fully decoding the audio.
    """
    test_clean = librispeech_dir / "test-clean"
    utterances: list[_Utterance] = []

    for speaker_dir in sorted(test_clean.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            for trans_file in chapter_dir.glob("*.trans.txt"):
                text = trans_file.read_text(encoding="utf-8")
                batch = _parse_trans_txt(
                    text,
                    speaker_id=speaker_dir.name,
                    chapter_id=chapter_dir.name,
                    audio_dir=chapter_dir,
                )
                for utt in batch:
                    if not utt.flac_path.exists():
                        logger.warning("FLAC not found (skipping): %s", utt.flac_path)
                        continue
                    info = sf.info(str(utt.flac_path))
                    utt.duration_sec = info.frames / info.samplerate
                utterances.extend(batch)

    logger.info("Enumerated %d utterances total", len(utterances))
    return utterances


# ---------------------------------------------------------------------------
# Selection logic (ADR-020 determinism contract)
# ---------------------------------------------------------------------------


def _select_50(utterances: list[_Utterance]) -> list[_Utterance]:
    """Filter, sort, and round-robin-select 50 utterances per ADR-020.

    Filtering:   2.0 <= duration_sec <= 15.0 (inclusive both ends).
    Sorting:     (speaker_id, chapter_id, utterance_id) — ALL THREE are
                 LEXICOGRAPHIC STRING SORTS.  Never int-cast.
    Selection:   ROUND-ROBIN by speaker_id.  Group filtered utterances by
                 speaker (preserving lex order within each group), then take
                 one utterance per speaker in lex order, looping back to the
                 first speaker after the last; continue until 50 are picked.

                 This guarantees maximum speaker diversity: every speaker
                 contributes >=1 utterance before any contributes a second.
                 Test-clean has 40 speakers (20F/20M), so the first 40 picks
                 are 1-per-speaker; picks 41–50 cycle back through the first
                 10 speakers in lex order.

                 The output list itself is then re-sorted by
                 (speaker_id, chapter_id, utterance_id) so filenames
                 0001.wav..0050.wav are in lex order, not pick order — this
                 matters because the manifest is committed to git and a
                 stable file ordering keeps diffs sane across rebuilds.

    Raises:
        ValueError: If fewer than 50 utterances pass the filter.
    """
    filtered = [u for u in utterances if _DURATION_MIN <= u.duration_sec <= _DURATION_MAX]
    # Deterministic lex sort — do NOT cast speaker_id/chapter_id/utterance_id to int.
    filtered.sort(key=lambda u: (u.speaker_id, u.chapter_id, u.utterance_id))
    if len(filtered) < _NUM_UTTERANCES:
        raise ValueError(
            f"Only {len(filtered)} utterances in [{_DURATION_MIN}, {_DURATION_MAX}]s window; "
            f"need {_NUM_UTTERANCES}."
        )

    by_speaker: dict[str, list[_Utterance]] = {}
    for utt in filtered:
        by_speaker.setdefault(utt.speaker_id, []).append(utt)
    speakers_lex = sorted(by_speaker.keys())

    picked: list[_Utterance] = []
    round_idx = 0
    while len(picked) < _NUM_UTTERANCES:
        any_added = False
        for sid in speakers_lex:
            if round_idx < len(by_speaker[sid]):
                picked.append(by_speaker[sid][round_idx])
                any_added = True
                if len(picked) == _NUM_UTTERANCES:
                    break
        if not any_added:  # pragma: no cover  (already gated by len(filtered) check above)
            raise ValueError("Exhausted all utterances before reaching 50; should not happen.")
        round_idx += 1

    picked.sort(key=lambda u: (u.speaker_id, u.chapter_id, u.utterance_id))
    return picked


# ---------------------------------------------------------------------------
# Transcode + hash
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str:
    """Return the lowercase hex SHA256 digest of the file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _transcode_to_wav(src_flac: Path, dest_wav: Path) -> None:
    """Transcode *src_flac* to 16 kHz mono PCM_16 WAV at *dest_wav*.

    LibriSpeech is already 16 kHz mono, so this is mostly a format conversion.
    If the source sample rate differs, soundfile will read it at its native
    rate and we re-write it; in practice this path is never taken for test-clean.
    """
    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    data, samplerate = sf.read(str(src_flac), dtype="int16", always_2d=False)
    if samplerate != _TARGET_SR:
        logger.warning(
            "Unexpected sample rate %d Hz for %s; expected 16000. "
            "Audio written as-is at %d Hz — add resampling if needed.",
            samplerate,
            src_flac.name,
            samplerate,
        )
    sf.write(str(dest_wav), data, _TARGET_SR, subtype="PCM_16")


def _build_items(selected: list[_Utterance], work_dir: Path) -> list[_BuiltItem]:
    """Transcode all selected utterances and return their :class:`_BuiltItem` records."""
    items: list[_BuiltItem] = []
    for idx, utt in enumerate(selected, start=1):
        filename = f"{idx:04d}.wav"
        wav_path = work_dir / filename
        _transcode_to_wav(utt.flac_path, wav_path)
        sha256 = _hash_file(wav_path)
        items.append(
            _BuiltItem(
                speaker_id=utt.speaker_id,
                chapter_id=utt.chapter_id,
                utterance_id=utt.utterance_id,
                transcript=utt.transcript,
                wav_path=wav_path,
                sha256=sha256,
                duration_sec=utt.duration_sec,
                filename=filename,
            )
        )
        logger.info(
            "[%d/%d] %s  sha256=%s…  %.2f s",
            idx,
            _NUM_UTTERANCES,
            filename,
            sha256[:16],
            utt.duration_sec,
        )
    return items


# ---------------------------------------------------------------------------
# GCS upload + verify
# ---------------------------------------------------------------------------


def _gcs_client() -> _gcs_storage.Client:
    """Create a GCS client using ADC (no service-account key file)."""
    from google.cloud import storage

    return storage.Client()


def _upload_items(
    items: list[_BuiltItem],
    bucket_name: str,
    *,
    overwrite: bool,
    client: _gcs_storage.Client | None = None,
) -> None:
    """Upload WAV files to GCS and verify SHA256 of each uploaded blob.

    Uses ``if_generation_match=0`` to refuse overwriting existing objects
    (v1 freeze contract).  Pass *overwrite=True* to drop this precondition.

    After upload, re-downloads each blob and recomputes SHA256 as a cheap
    integrity guard against silent upload corruption.
    """
    if client is None:
        client = _gcs_client()

    bucket = client.bucket(bucket_name)

    for item in items:
        blob_name = f"{_GCS_AUDIO_PREFIX}/{item.filename}"
        blob = bucket.blob(blob_name)
        blob.content_type = "audio/wav"

        logger.info("Uploading gs://%s/%s …", bucket_name, blob_name)

        if overwrite:
            blob.upload_from_filename(str(item.wav_path), content_type="audio/wav")
        else:
            # if_generation_match=0 → fail fast if object already exists
            blob.upload_from_filename(
                str(item.wav_path),
                content_type="audio/wav",
                if_generation_match=0,
            )

        # Integrity verification: re-download and re-hash
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            blob.download_to_filename(str(tmp_path))
            actual = _hash_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        if actual != item.sha256:
            raise RuntimeError(
                f"Upload integrity failure for {blob_name}: expected={item.sha256} got={actual}"
            )
        logger.info("Verified %s ✓", blob_name)


# ---------------------------------------------------------------------------
# Manifest rendering
# ---------------------------------------------------------------------------


def _render_manifest(items: list[_BuiltItem]) -> str:
    """Render the stt-v1 manifest as a JSON string (trailing newline)."""
    manifest_dict = {
        "_license": "Apache-2.0",
        "id": "stt-v1",
        "version": "1.0.0",
        "license": "CC-BY-4.0",
        "source": "LibriSpeech test-clean (OpenSLR-12)",
        "items": [
            {
                "path": f"audio/{item.filename}",
                "sha256": item.sha256,
                "transcript": item.transcript,
                "duration_sec": round(item.duration_sec, 6),
                "speaker_id": item.speaker_id,
                "chapter_id": item.chapter_id,
                "utterance_id": item.utterance_id,
            }
            for item in items
        ],
    }
    return json.dumps(manifest_dict, indent=2, ensure_ascii=False) + "\n"


def _manifest_path() -> Path:
    """Return the path to the packaged stt-v1.json manifest file."""
    # Resolve via importlib.resources so it works both from source and wheel.
    pkg = _impres.files("coval_bench.datasets.manifests")
    ref = pkg.joinpath("stt-v1.json")
    # In editable installs the traversable is backed by a real Path.
    resolved = Path(str(ref))
    return resolved


def _write_manifest(json_text: str) -> Path:
    """Write *json_text* to the packaged stt-v1.json location."""
    dest = _manifest_path()
    dest.write_text(json_text, encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Speaker gender parsing
# ---------------------------------------------------------------------------


def _parse_speaker_genders(librispeech_dir: Path) -> dict[str, str]:
    """Parse ``SPEAKERS.TXT`` and return ``{speaker_id: "F"|"M"}`` mapping.

    Format of relevant lines::

        ID  | SEX | SUBSET | ...

    Lines starting with ``;`` are comments.
    """
    speakers_txt = librispeech_dir / "SPEAKERS.TXT"
    if not speakers_txt.exists():
        logger.warning("SPEAKERS.TXT not found at %s; gender counts unavailable", speakers_txt)
        return {}
    genders: dict[str, str] = {}
    for line in speakers_txt.read_text(encoding="utf-8").splitlines():
        if line.startswith(";") or not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        speaker_id = parts[0].strip()
        sex = parts[1].strip().upper()
        genders[speaker_id] = sex
    return genders


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(
    *,
    downloaded_bytes: int,
    total_utterances: int,
    filtered_utterances: int,
    items: list[_BuiltItem],
    genders: dict[str, str],
    bucket: str,
    manifest_path: Path | None,
    dry_run: bool,
) -> None:
    """Print the build summary to stdout via click.echo."""
    distinct_speakers = {item.speaker_id for item in items}
    f_count = sum(1 for s in distinct_speakers if genders.get(s) == "F")
    m_count = sum(1 for s in distinct_speakers if genders.get(s) == "M")

    click.echo("")
    click.echo("─" * 60)
    click.echo(f"Downloaded:  {downloaded_bytes:,} bytes from OpenSLR")
    click.echo(
        f"Filtered:    {filtered_utterances}/{total_utterances} utterances "
        f"in [{_DURATION_MIN}, {_DURATION_MAX}]s window"
    )
    click.echo(f"Selected:    {len(items)}")
    click.echo(f"Speakers:    {len(distinct_speakers)} ({f_count}F / {m_count}M)")
    if dry_run:
        click.echo("Uploaded:    (dry-run — skipped)")
        click.echo("Manifest:    (dry-run — printed to stdout instead)")
    else:
        click.echo(
            f"Uploaded:    gs://{bucket}/{_GCS_AUDIO_PREFIX}/0001.wav .. {_NUM_UTTERANCES:04d}.wav"
        )
        if manifest_path is not None:
            click.echo(f"Manifest:    {manifest_path} ({manifest_path.stat().st_size} bytes)")
    click.echo("─" * 60)


# ---------------------------------------------------------------------------
# stt-v1 build
# ---------------------------------------------------------------------------


def _build_stt_v1(
    bucket: str,
    *,
    dry_run: bool,
    overwrite: bool,
    cache_root: Path = _CACHE_ROOT,
) -> None:
    """End-to-end stt-v1 dataset build (ADR-020)."""
    # ---- Step 1: Download archive ----
    archive_path = _download_archive(cache_root)
    downloaded_bytes = archive_path.stat().st_size

    # ---- Step 2: Extract ----
    librispeech_dir = _extract_archive(archive_path, cache_root)

    # ---- Steps 3–4: Walk + duration decode ----
    utterances = _enumerate_utterances(librispeech_dir)
    total_utterances = len(utterances)

    # ---- Steps 5–7: Filter + sort (lex) + take 50 ----
    selected = _select_50(utterances)
    filtered_utterances = sum(
        1 for u in utterances if _DURATION_MIN <= u.duration_sec <= _DURATION_MAX
    )

    # ---- Step 8: Transcode + hash ----
    genders = _parse_speaker_genders(librispeech_dir)

    with tempfile.TemporaryDirectory(prefix="coval-bench-wav-") as work_dir_str:
        work_dir = Path(work_dir_str)
        items = _build_items(selected, work_dir)

        # Render the manifest JSON now (needed for dry-run stdout output too)
        manifest_json = _render_manifest(items)

        if dry_run:
            # Print manifest to stdout and exit without GCS writes.
            click.echo(manifest_json, nl=False)
            _print_summary(
                downloaded_bytes=downloaded_bytes,
                total_utterances=total_utterances,
                filtered_utterances=filtered_utterances,
                items=items,
                genders=genders,
                bucket=bucket,
                manifest_path=None,
                dry_run=True,
            )
            return

        # ---- Steps 9–10: Upload + verify ----
        _upload_items(items, bucket, overwrite=overwrite)

    # ---- Step 11: Write manifest ----
    manifest_dest = _write_manifest(manifest_json)

    _print_summary(
        downloaded_bytes=downloaded_bytes,
        total_utterances=total_utterances,
        filtered_utterances=filtered_utterances,
        items=items,
        genders=genders,
        bucket=bucket,
        manifest_path=manifest_dest,
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Coval dataset builder — download, transcode, hash, upload."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.option(
    "--dataset",
    type=click.Choice(["stt-v1", "tts-v1"]),
    required=True,
    help="Dataset identifier to build.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help=(
        "Steps 1–8 only (download, parse, filter, sort, transcode, hash). "
        "Prints the manifest to stdout. Makes NO GCS writes."
    ),
)
@click.option(
    "--bucket",
    default="coval-benchmarks-datasets",
    show_default=True,
    help="Target GCS bucket name.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help=(
        "Allow overwriting existing GCS objects and the manifest file. "
        "By default, the v1 freeze contract refuses to overwrite."
    ),
)
def build(dataset: str, dry_run: bool, bucket: str, overwrite: bool) -> None:
    """Download LibriSpeech test-clean, deterministically pick 50 utterances
    in [2.0s, 15.0s], transcode to 16kHz mono PCM_16 WAV, hash, upload to
    gs://<bucket>/<dataset>/audio/, write manifest. ADR-020.

    --dry-run: steps 1–8 (download/parse/select/transcode/hash) but no GCS
    writes. Prints the manifest JSON to stdout.

    ## Deterministic subset selection (ADR-020)

    \b
    Sort key:  (speaker_id, chapter_id, utterance_id) — ALL STRING (lex) sorts.
    Filter:    2.0 <= duration_sec <= 15.0 (inclusive).
    Take:      first 50 from the sorted, filtered list.
    """
    if dataset == "tts-v1":
        click.echo("tts-v1 is text-only and committed; nothing to build.")
        return

    # dataset == "stt-v1"
    _build_stt_v1(bucket, dry_run=dry_run, overwrite=overwrite)


if __name__ == "__main__":
    cli()
