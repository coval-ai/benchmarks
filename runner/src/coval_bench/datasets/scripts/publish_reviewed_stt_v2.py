# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: MInDS-14 reviewed subset → WAV export → ``stt-v2.json`` → optional GCS upload.

Steps (single ``publish`` command):

1. Decode reviewed rows from Hugging Face ``PolyAI/minds14`` ``en-US`` ``train`` into
   16 kHz mono PCM_16 WAV files under ``--wav-dir``.
2. SHA256 each WAV file and read ``duration_sec`` from the exported WAV (via ``soundfile``).
3. Emit a manifest matching :class:`coval_bench.datasets.manifest.Manifest` with
   ``transcript`` from each review entry's ``approved_transcript``.
4. Optionally upload WAVs to ``gs://<bucket>/stt-v2/<path>`` with post-upload
   integrity check (same pattern as ``build_dataset.py``).

Runner usage after this::

    DATASET_ID=stt-v2 DATASET_BUCKET=<bucket> DATABASE_URL=… \\
        uv run python -m coval_bench run --kind stt

Example::

    uv run --project runner \\
        --with datasets --with torch --with torchcodec \\
        python -m coval_bench.datasets.scripts.publish_reviewed_stt_v2 publish \\
        --review-json review_results.json \\
        --wav-dir build/stt-v2 \\
        --upload \\
        --bucket coval-benchmarks-datasets
"""

from __future__ import annotations

import hashlib
import importlib.resources as _impres
import json
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import soundfile as sf

from coval_bench.datasets.manifest import Manifest
from coval_bench.datasets.scripts.export_minds14_reviewed_wavs import (
    _paths_from_review_file,
    _review_index,
    export_reviewed_wavs,
)

if TYPE_CHECKING:
    from google.cloud import storage as _gcs_storage

logger = logging.getLogger(__name__)

_DATASET_ID = "stt-v2"
_DEFAULT_BUCKET = "coval-benchmarks-datasets"


def _packaged_manifest_path() -> Path:
    ref = _impres.files("coval_bench.datasets.manifests").joinpath(f"{_DATASET_ID}.json")
    return Path(str(ref))


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _wav_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def _build_manifest_dict(
    *,
    review_by_path: dict[str, Any],
    wav_root: Path,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for path_str, row in sorted(review_by_path.items(), key=lambda kv: _review_index(kv[1])):
        wav_path = wav_root / path_str
        if not wav_path.is_file():
            raise click.ClickException(f"Missing WAV for manifest: {wav_path}")
        transcript = str(row.get("approved_transcript", "")).strip()
        if not transcript:
            raise click.ClickException(f"Missing approved_transcript for {path_str!r}")
        sha256 = _hash_file(wav_path)
        duration_sec = _wav_duration_sec(wav_path)
        if duration_sec <= 0:
            raise click.ClickException(f"Non-positive duration for {path_str!r}")
        entry: dict[str, Any] = {
            "path": path_str,
            "sha256": sha256,
            "transcript": transcript,
            "duration_sec": round(duration_sec, 6),
        }
        inn = row.get("intent_name")
        if inn is not None and str(inn).strip():
            entry["intent_name"] = str(inn).strip()
        if "intent_class" in row and row["intent_class"] is not None:
            entry["intent_class"] = int(row["intent_class"])
        gn = row.get("gender")
        entry["gender"] = str(gn).strip() if gn is not None and str(gn).strip() else "unspecified"
        lang = row.get("language")
        if lang is not None and str(lang).strip():
            entry["language"] = str(lang).strip().lower()
        else:
            entry["language"] = "english"
        items.append(entry)

    return {
        "_license": "Apache-2.0",
        "id": _DATASET_ID,
        "version": "1.0.0",
        "license": "PolyAI MInDS-14; verify redistribution terms before publishing",
        "source": "PolyAI MInDS-14 en-US reviewed subset",
        "items": items,
    }


def _write_manifest_json(manifest_dict: dict[str, Any], dest: Path) -> None:
    text = json.dumps(manifest_dict, indent=2, ensure_ascii=False) + "\n"
    Manifest.model_validate_json(text)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    click.echo(f"Wrote manifest ({len(manifest_dict['items'])} items): {dest}")


def _gcs_client() -> _gcs_storage.Client:
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import storage

    try:
        return storage.Client()
    except DefaultCredentialsError as exc:
        raise click.ClickException(
            "Google Cloud Application Default Credentials not found. For local upload:\n"
            "  gcloud auth application-default login\n"
            "or set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON key path.\n"
            "https://cloud.google.com/docs/authentication/external/set-up-adc"
        ) from exc


def _upload_wavs(
    wav_root: Path,
    bucket_name: str,
    manifest_items: Sequence[dict[str, Any]],
    *,
    overwrite: bool,
    client: _gcs_storage.Client | None = None,
) -> None:
    """Upload manifest WAV paths under *wav_root* to ``stt-v2/<relative path>``."""
    if client is None:
        client = _gcs_client()
    bucket = client.bucket(bucket_name)
    if not manifest_items:
        raise click.ClickException("Manifest has no items; nothing to upload")

    for item in manifest_items:
        rel = str(item["path"])
        expected_sha = str(item["sha256"])
        local = wav_root / rel
        if not local.is_file():
            raise click.ClickException(f"Missing WAV for upload: {local}")
        disk_sha = _hash_file(local)
        if disk_sha != expected_sha:
            raise click.ClickException(
                f"{local}: on-disk SHA256 {disk_sha} != manifest {expected_sha}; "
                "refresh the manifest before upload"
            )

        blob_name = f"{_DATASET_ID}/{rel}"
        blob = bucket.blob(blob_name)
        blob.content_type = "audio/wav"

        logger.info("Uploading gs://%s/%s", bucket_name, blob_name)
        if overwrite:
            blob.upload_from_filename(str(local), content_type="audio/wav")
        else:
            blob.upload_from_filename(
                str(local),
                content_type="audio/wav",
                if_generation_match=0,
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            blob.download_to_filename(str(tmp_path))
            actual = _hash_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        if actual != disk_sha:
            raise click.ClickException(
                f"Upload integrity failure for {blob_name}: expected={disk_sha} got={actual}"
            )
        logger.info("Verified %s ✓", blob_name)


@click.group()
def cli() -> None:
    """Build stt-v2 manifest + WAVs from reviewed MInDS-14 JSON."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command("publish")
@click.option(
    "--review-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Review JSON (object keyed by path; approved_transcript required).",
)
@click.option(
    "--wav-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("build") / _DATASET_ID,
    show_default=True,
    help="Directory for exported WAV trees (mirror HF path).",
)
@click.option(
    "--manifest",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Manifest output. Default: packaged coval_bench.datasets.manifests/stt-v2.json",
)
@click.option(
    "--skip-export",
    is_flag=True,
    default=False,
    help="Assume WAVs already exist under --wav-dir; only hash + manifest (+ optional upload).",
)
@click.option(
    "--upload",
    is_flag=True,
    default=False,
    help="Upload WAVs to gs://<bucket>/stt-v2/ after writing manifest.",
)
@click.option(
    "--bucket",
    default=_DEFAULT_BUCKET,
    show_default=True,
    help="GCS bucket (--upload only). Must match SETTINGS dataset_bucket.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Allow overwriting existing GCS objects (--upload only).",
)
def publish(
    review_json: Path,
    wav_dir: Path,
    manifest: Path | None,
    skip_export: bool,
    upload: bool,
    bucket: str,
    overwrite: bool,
) -> None:
    """Export WAVs → build stt-v2.json from on-disk bytes → optionally upload."""
    if skip_export:
        review_by_path = _paths_from_review_file(review_json)
    else:
        review_by_path, n = export_reviewed_wavs(review_json, wav_dir)
        click.echo(f"Exported {n} WAVs under {wav_dir}")

    manifest_dict = _build_manifest_dict(review_by_path=review_by_path, wav_root=wav_dir)

    manifest_dest = manifest if manifest is not None else _packaged_manifest_path()
    _write_manifest_json(manifest_dict, manifest_dest)

    if upload:
        _upload_wavs(
            wav_dir,
            bucket,
            manifest_dict["items"],
            overwrite=overwrite,
        )

    click.echo("")
    click.echo("─" * 60)
    click.echo(f"Dataset: {_DATASET_ID}")
    click.echo(f"WAV dir: {wav_dir.resolve()}")
    click.echo(f"Manifest: {manifest_dest.resolve()}")
    if upload:
        click.echo(f"Uploaded: gs://{bucket}/{_DATASET_ID}/…")
    else:
        click.echo("Upload:    (skipped — pass --upload to push to GCS)")
    click.echo("Run benchmarks with: DATASET_ID=stt-v2 DATASET_BUCKET=" + bucket)
    click.echo("─" * 60)


if __name__ == "__main__":
    cli()
