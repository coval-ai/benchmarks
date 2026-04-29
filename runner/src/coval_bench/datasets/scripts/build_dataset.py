# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""build_dataset.py — offline tool to build and upload dataset manifests.

**Phase 2 status:** CLI surface is defined; algorithm is documented.
GCS upload and actual file processing are deferred to Phase 4.

## Algorithm (ADR-020 — LibriSpeech test-clean, deterministic 50)

1. Download the LibriSpeech `test-clean` tar from OpenSLR::

       https://www.openslr.org/resources/12/test-clean.tar.gz

2. Extract and enumerate all utterances (FLAC + corresponding .trans.txt).

3. Filter to utterances in the 2.0 s – 15.0 s duration window.

4. Sort by ``(speaker_id, chapter_id, utterance_id)`` — all three are
   lexicographic string sorts to guarantee identical ordering on any platform.

5. Take the first 50 utterances from the sorted, filtered list.

6. Transcode each to 16 kHz mono PCM_16 WAV (via ``soundfile`` + ``librosa``).

7. Compute SHA256 for each WAV.

8. (Phase 4 only) Upload to ``gs://<bucket>/stt-v1/audio/0001.wav`` etc.

9. (Phase 4 only) Write ``stt-v1.json`` with all 50 items and commit.

``--dry-run`` executes steps 1–7 (parse, select, hash) but skips 8–9.

Usage::

    uv run python -m coval_bench.datasets.scripts.build_dataset build \\
        --dataset stt-v1 \\
        --bucket coval-benchmarks-datasets \\
        [--dry-run]

    uv run python -m coval_bench.datasets.scripts.build_dataset build \\
        --dataset tts-v1 \\
        --bucket coval-benchmarks-datasets \\
        --dry-run
"""

from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """Coval dataset builder — download, transcode, hash, upload."""


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
        "Parse, select, and hash files but do NOT upload to GCS or write "
        "the manifest.  Safe to run without GCS write credentials."
    ),
)
@click.option(
    "--bucket",
    default="coval-benchmarks-datasets",
    show_default=True,
    help="Target GCS bucket name.",
)
def build(dataset: str, dry_run: bool, bucket: str) -> None:
    """Download LibriSpeech test-clean, deterministically pick 50 utterances
    in [2.0s, 15.0s], transcode to 16kHz mono PCM_16 WAV, hash, upload to
    gs://<bucket>/<dataset>/audio/, write manifest. ADR-020.

    --dry-run: parse + select + hash but do NOT upload or write manifest.

    ## Deterministic subset selection (ADR-020)

    Utterances are selected from LibriSpeech test-clean as follows:

    \b
    1. Parse all .trans.txt files to build a list of (speaker_id, chapter_id,
       utterance_id, flac_path, transcript) tuples.
    2. Decode each FLAC to determine duration_sec.
    3. Discard utterances outside [2.0, 15.0] seconds.
    4. Sort remaining utterances by (speaker_id, chapter_id, utterance_id)
       — all string sorts, lexicographic, platform-stable.
    5. Take the first 50 from the sorted list.
    6. Transcode each to 16kHz mono PCM_16 WAV.
    7. Compute SHA256 of the WAV.

    This process is fully deterministic given the same source archive.
    """
    # Phase 2 stub — implementation deferred to Phase 4.
    if dry_run:
        click.echo(f"[dry-run] Would build dataset '{dataset}' → gs://{bucket}/{dataset}/")
        click.echo("[dry-run] Phase 4 will implement download, transcode, hash, upload.")
    else:
        click.echo(
            f"Dataset '{dataset}' build is not yet implemented (Phase 4). "
            "Use --dry-run to verify the CLI surface."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
