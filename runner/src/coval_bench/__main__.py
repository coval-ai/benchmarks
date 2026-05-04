# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for coval-bench.

The ``coval-bench run`` subcommand is the entrypoint for the Cloud Run Job::

    CMD ["python", "-m", "coval_bench", "run"]

It calls :func:`coval_bench.runner.run_benchmarks` and exits non-zero when the
run status is FAILED.
"""

import asyncio
import json
import sys

import click

from coval_bench import __version__
from coval_bench.db.cli import db_check, db_migrate
from coval_bench.migrations.import_legacy import import_legacy_cli


@click.group()
@click.version_option(version=__version__, prog_name="coval-bench")
def cli() -> None:
    """Coval voice-AI benchmarks runner."""


@cli.command(name="run")
@click.option(
    "--kind",
    type=click.Choice(["stt", "tts", "both"]),
    default="both",
    show_default=True,
    help="Which benchmark(s) to run.",
)
@click.option(
    "--smoke",
    is_flag=True,
    default=False,
    help="Run a single dataset item — for local dev.",
)
def run(kind: str, smoke: bool) -> None:
    """Execute one benchmark run — invoked by Cloud Run Job."""
    from coval_bench.config import get_settings
    from coval_bench.db.models import RunStatus
    from coval_bench.runner.orchestrator import RunSummary, run_benchmarks

    settings = get_settings()
    summary: RunSummary = asyncio.run(
        run_benchmarks(settings=settings, benchmark_kind=kind, smoke=smoke)  # type: ignore[arg-type]
    )
    click.echo(summary.model_dump_json(indent=2))
    if summary.status == str(RunStatus.FAILED):
        raise click.ClickException("run failed")


@cli.group()
def db() -> None:
    """Database management commands."""


db.add_command(db_migrate, name="migrate")
db.add_command(db_check, name="db-check")


@cli.group()
def migrate() -> None:
    """One-shot data migrations."""


migrate.add_command(import_legacy_cli, name="import-legacy")


@cli.command(name="tts-smoke")
@click.option("--provider", required=True, help="TTS provider name (e.g. openai, cartesia).")
@click.option("--model", required=True, help="Model ID for the provider (e.g. tts-1-hd).")
@click.option("--voice", required=True, help="Voice ID for the provider.")
@click.option(
    "--text",
    default="Hello world.",
    show_default=True,
    help="Text to synthesize.",
)
def tts_smoke(provider: str, model: str, voice: str, text: str) -> None:
    """Probe a single (provider, model, voice) against the real TTS upstream.

    Emits a structured single-line JSON to stdout. Exits 0 on success, 1 on
    provider error, 2 if *provider* is not registered. Does NOT write to the
    database — pure provider call.
    """
    from typing import Any

    from coval_bench.config import get_settings
    from coval_bench.providers.tts import TTS_PROVIDERS

    registry: dict[str, Any] = dict(TTS_PROVIDERS)
    provider_cls = registry.get(provider)
    if provider_cls is None:
        click.echo(
            f"Unknown TTS provider: {provider!r}. Known: {sorted(registry.keys())}",
            err=True,
        )
        sys.exit(2)

    settings = get_settings()
    instance = provider_cls(settings=settings, model=model, voice=voice)
    result = asyncio.run(instance.synthesize(text))

    audio_path_str: str | None = str(result.audio_path) if result.audio_path else None
    audio_bytes: int | None = None
    if result.audio_path is not None and result.audio_path.exists():
        audio_bytes = result.audio_path.stat().st_size

    ok = (
        result.error is None
        and result.audio_path is not None
        and audio_bytes is not None
        and audio_bytes > 0
    )

    click.echo(
        json.dumps(
            {
                "event": "tts_smoke",
                "provider": provider,
                "model": model,
                "voice": voice,
                "ttfa_ms": result.ttfa_ms,
                "audio_path": audio_path_str,
                "audio_bytes": audio_bytes,
                "error": result.error,
                "ok": ok,
            }
        )
    )
    sys.exit(0 if ok else 1)


@cli.command(name="stt-preview")
@click.option(
    "--dataset-id",
    envvar="DATASET_ID",
    default="stt-v2",
    show_default=True,
    help="Packaged manifest id / GCS prefix (stt-v1, stt-v2, …).",
)
@click.option(
    "--bucket",
    envvar="DATASET_BUCKET",
    default="coval-benchmarks-datasets",
    show_default=True,
    help="GCS bucket containing /<dataset-id>/… WAV objects.",
)
@click.option("--provider", default="deepgram", show_default=True, help="STT provider key.")
@click.option("--model", default="nova-2", show_default=True, help="Provider model id.")
@click.option(
    "--limit",
    default=1,
    show_default=True,
    type=int,
    help="Max utterances to transcribe (from start of manifest order).",
)
def stt_preview(dataset_id: str, bucket: str, provider: str, model: str, limit: int) -> None:
    """Load corpus audio from GCS + call one STT upstream — **no database**.

    Uses anonymous reads when ``GOOGLE_APPLICATION_CREDENTIALS`` is unset (public bucket).

    Requires a valid ``<PROVIDER>_API_KEY`` in the environment or ``.env``. ``DATABASE_URL``
    is optional; a harmless placeholder is used if unset.

    Emits one JSON object to stdout. Exit code ``0`` = every item succeeded.
    """
    from coval_bench.stt_preview import run_stt_preview

    exit_code, payload = asyncio.run(
        run_stt_preview(
            dataset_id=dataset_id,
            dataset_bucket=bucket,
            provider=provider,
            model=model,
            limit=max(1, limit),
        )
    )
    click.echo(json.dumps(payload, indent=2))
    if exit_code != 0:
        hint: str | None = payload.get("error") if isinstance(payload, dict) else None
        raise click.ClickException(
            hint or f"stt-preview failed (exit {exit_code}); see JSON above for details."
        )


if __name__ == "__main__":
    cli()
