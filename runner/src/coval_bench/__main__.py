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

# Backstop so a stalled connection can't hang a smoke probe forever. Loose enough
# for a cold dedicated endpoint (handshake + cold inference); the production paths
# use the orchestrator's tighter per-call timeouts instead.
_SMOKE_TIMEOUT_S = 120


@click.group()
@click.version_option(version=__version__, prog_name="coval-bench")
def cli() -> None:
    """Coval voice-AI benchmarks runner."""
    from coval_bench.config import get_settings
    from coval_bench.logging import configure_logging

    configure_logging(level=get_settings().log_level)


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
    click.echo(summary.model_dump_json())
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
@click.option("--model", required=True, help="Model ID for the provider (e.g. gpt-4o-mini-tts).")
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
    try:
        result = asyncio.run(asyncio.wait_for(instance.synthesize(text), timeout=_SMOKE_TIMEOUT_S))
    except TimeoutError:
        click.echo(
            json.dumps(
                {
                    "event": "tts_smoke",
                    "provider": provider,
                    "model": model,
                    "voice": voice,
                    "ttfa_ms": None,
                    "audio_path": None,
                    "audio_bytes": None,
                    "error": f"timed out after {_SMOKE_TIMEOUT_S}s",
                    "ok": False,
                }
            )
        )
        sys.exit(1)

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


@cli.command(name="probe")
@click.option(
    "--only",
    "selectors",
    multiple=True,
    required=True,
    help="provider/model to probe (repeatable), e.g. baseten/whisper-large-v3. "
    "Runs regardless of registry status.",
)
@click.option("--samples", default=20, show_default=True, type=int, help="Dataset items per model.")
@click.option(
    "--concurrency",
    default=1,
    show_default=True,
    type=int,
    help="Concurrent items; keep at 1 for a single pinned replica.",
)
def probe(selectors: tuple[str, ...], samples: int, concurrency: int) -> None:
    """Run dataset samples through selected models WITHOUT persisting (logs only).

    For private latency checks of PENDING models from the production (in-region)
    environment: writes nothing to the DB or public API, emits per-metric
    aggregates as a single JSON line to stdout. Exits 2 if a selector matches no
    registered model.
    """
    from coval_bench.config import get_settings
    from coval_bench.registries import MODEL_REGISTRY
    from coval_bench.runner.probe import run_probe

    models = [m for m in MODEL_REGISTRY if f"{m.provider}/{m.model}" in set(selectors)]
    matched = {f"{m.provider}/{m.model}" for m in models}
    missing = sorted(set(selectors) - matched)
    if missing:
        known = sorted({f"{m.provider}/{m.model}" for m in MODEL_REGISTRY})
        click.echo(f"Unknown model(s): {missing}. Known: {known}", err=True)
        sys.exit(2)

    settings = get_settings()
    results = asyncio.run(
        run_probe(settings=settings, models=models, sample_size=samples, concurrency=concurrency)
    )
    click.echo(json.dumps({"event": "probe_results", "samples": samples, "results": results}))


@cli.command(name="stt-smoke")
@click.option("--provider", required=True, help="STT provider name (e.g. deepgram, baseten).")
@click.option("--model", required=True, help="Model ID for the provider (e.g. nova-3).")
@click.option(
    "--wav",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="16 kHz mono PCM16 WAV to transcribe.",
)
def stt_smoke(provider: str, model: str, wav: str) -> None:
    """Probe a single (provider, model) against the real STT upstream.

    Streams *wav* in real time and emits a single-line JSON to stdout with the
    timing measurements. Exits 0 on success, 1 on provider error, 2 if
    *provider* is not registered. Does NOT write to the database.
    """
    import wave
    from typing import Any

    from coval_bench.config import get_settings
    from coval_bench.providers.stt import STT_PROVIDERS

    registry: dict[str, Any] = dict(STT_PROVIDERS)
    provider_cls = registry.get(provider)
    if provider_cls is None:
        click.echo(
            f"Unknown STT provider: {provider!r}. Known: {sorted(registry.keys())}",
            err=True,
        )
        sys.exit(2)

    settings = get_settings()
    kwargs: dict[str, Any] = {
        "api_key": getattr(settings, f"{provider}_api_key", None),
        "model": model,
    }
    if provider == "google":
        kwargs["project_id"] = settings.google_project_id
    elif provider == "baseten":
        kwargs["ws_url"] = settings.baseten_whisper_url
    instance = provider_cls(**kwargs)

    with wave.open(wav, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            click.echo("WAV must be 16 kHz, mono, 16-bit PCM", err=True)
            sys.exit(1)
        pcm = w.readframes(w.getnframes())

    try:
        result = asyncio.run(
            asyncio.wait_for(instance.measure_ttft(pcm, 1, 2, 16000, 0.1), timeout=_SMOKE_TIMEOUT_S)
        )
    except TimeoutError:
        click.echo(
            json.dumps(
                {
                    "event": "stt_smoke",
                    "provider": provider,
                    "model": model,
                    "ttft_seconds": None,
                    "audio_to_final_seconds": None,
                    "transcript": None,
                    "error": f"timed out after {_SMOKE_TIMEOUT_S}s",
                    "ok": False,
                }
            )
        )
        sys.exit(1)
    ok = result.error is None and result.complete_transcript is not None

    click.echo(
        json.dumps(
            {
                "event": "stt_smoke",
                "provider": provider,
                "model": model,
                "ttft_seconds": result.ttft_seconds,
                "audio_to_final_seconds": result.audio_to_final_seconds,
                "transcript": result.complete_transcript,
                "error": result.error,
                "ok": ok,
            }
        )
    )
    sys.exit(0 if ok else 1)


@cli.group()
def arena() -> None:
    """Voice Arena maintenance commands."""


@arena.command(name="snapshot")
@click.option(
    "--metric",
    default="naturalness",
    show_default=True,
    help="Metric to rate (MVP: naturalness).",
)
@click.option(
    "--bootstrap-rounds",
    default=1000,
    show_default=True,
    type=int,
    help="Bootstrap resamples for the confidence interval (0 disables CIs).",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    type=int,
    help="RNG seed; keeps the bootstrap CI reproducible.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Run even if another snapshot is in progress (bypass the advisory lock).",
)
def arena_snapshot(metric: str, bootstrap_rounds: int, seed: int, force: bool) -> None:
    """Refit Davidson-BT ratings from all votes and persist a leaderboard board."""
    from coval_bench.arena.rating import ConvergenceError
    from coval_bench.arena.snapshot import run_snapshot
    from coval_bench.config import get_settings
    from coval_bench.db.arena_store import ArenaStore
    from coval_bench.db.conn import lifespan_pool

    async def _run() -> int | None:
        settings = get_settings()
        async with lifespan_pool(settings) as pool:
            result = await run_snapshot(
                ArenaStore(pool),
                metric_name=metric,
                bootstrap_rounds=bootstrap_rounds,
                seed=seed,
                force=force,
            )
        return None if result is None else len(result.models)

    try:
        written = asyncio.run(_run())
    except ConvergenceError as exc:
        click.echo(
            json.dumps(
                {
                    "event": "arena_snapshot",
                    "metric": metric,
                    "error": "convergence_failed",
                    "detail": str(exc),
                }
            )
        )
        sys.exit(1)
    if written is None:
        click.echo(json.dumps({"event": "arena_snapshot", "metric": metric, "skipped": True}))
    else:
        click.echo(json.dumps({"event": "arena_snapshot", "metric": metric, "models": written}))


@arena.command(name="seed-battles")
@click.option(
    "--per-domain",
    default=1,
    show_default=True,
    type=click.IntRange(min=0),
    help="Demo battles to generate per domain (uses real TTS credits).",
)
def arena_seed_battles(per_domain: int) -> None:
    """Generate real demo battles from example prompts so a labeler can vote."""
    from coval_bench.arena.prompts import EXAMPLE_PROMPTS
    from coval_bench.arena.seed import seed_demo_battles
    from coval_bench.config import get_settings
    from coval_bench.db.arena_store import ArenaStore
    from coval_bench.db.conn import lifespan_pool
    from coval_bench.db.models import Battle

    async def _run() -> list[Battle]:
        settings = get_settings()
        async with lifespan_pool(settings) as pool:
            return await seed_demo_battles(settings, ArenaStore(pool), per_domain=per_domain)

    battles = asyncio.run(_run())
    attempted = sum(min(per_domain, len(prompts)) for prompts in EXAMPLE_PROMPTS.values())
    click.echo(
        json.dumps(
            {
                "event": "arena_seed_battles",
                "attempted": attempted,
                "created": len(battles),
                "skipped": attempted - len(battles),
                "battles": [{"id": str(b.id), "domain": b.domain} for b in battles],
            }
        )
    )


@arena.command(name="tune-scale")
@click.option(
    "--out",
    default="tune-scale.html",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="HTML loss-curve output path.",
)
@click.option(
    "--scales",
    default="50,100,150,200,250",
    show_default=True,
    help="Comma-separated candidate SCALE values.",
)
@click.option("--battles", default=2000, show_default=True, type=click.IntRange(min=1))
@click.option("--refit-every", default=100, show_default=True, type=click.IntRange(min=1))
@click.option("--replications", default=3, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--bootstrap-rounds",
    default=100,
    show_default=True,
    type=click.IntRange(min=0),
    help="Per-refit bootstrap rounds (drives the CI term in pairing).",
)
@click.option(
    "--elo-spread",
    default=800.0,
    show_default=True,
    type=float,
    help="True top-to-bottom Elo gap of the synthetic roster.",
)
@click.option(
    "--k",
    default=1.0,
    show_default=True,
    type=float,
    help="Confidently-wrong penalty weight in the loss.",
)
@click.option("--seed", default=0, show_default=True, type=int)
def arena_tune_scale(
    out: str,
    scales: str,
    battles: int,
    refit_every: int,
    replications: int,
    bootstrap_rounds: int,
    elo_spread: float,
    k: float,
    seed: int,
) -> None:
    """Offline: simulate battles to pick the pairing SCALE minimizing penalized CE."""
    from pathlib import Path

    from coval_bench.arena.tune_scale import render_loss_curve, tune_scale

    try:
        scale_values = [float(s) for s in scales.split(",") if s.strip()]
    except ValueError as exc:
        raise click.BadParameter(f"--scales must be comma-separated numbers: {exc}") from exc
    if not scale_values:
        raise click.BadParameter("--scales must contain at least one value")
    results = tune_scale(
        scales=scale_values,
        n_battles=battles,
        refit_every=refit_every,
        replications=replications,
        bootstrap_rounds=bootstrap_rounds,
        elo_spread=elo_spread,
        k=k,
        seed=seed,
    )
    Path(out).write_text(render_loss_curve(results), encoding="utf-8")
    best = min(results, key=lambda r: r.loss)
    for r in results:
        marker = " <-- best" if r is best else ""
        click.echo(f"SCALE {r.scale:6g}   L {r.loss:.4f}{marker}")
    click.echo(json.dumps({"event": "arena_tune_scale", "out": out, "best_scale": best.scale}))


@arena.command(name="cooccurrence")
@click.option(
    "--out",
    default="arena-cooccurrence.html",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="HTML output path.",
)
@click.option(
    "--metric",
    default="naturalness",
    show_default=True,
    help="Board metric used to Elo-order the axis.",
)
@click.option(
    "--domain",
    default="all",
    show_default=True,
    help="Board domain used to Elo-order the axis.",
)
def arena_cooccurrence(out: str, metric: str, domain: str) -> None:
    """Render the who-battled-whom heatmap (admin: reveals model identities)."""
    from pathlib import Path

    from coval_bench.arena.monitoring import build_cooccurrence, render_cooccurrence
    from coval_bench.config import get_settings
    from coval_bench.db.arena_store import ArenaStore
    from coval_bench.db.conn import lifespan_pool

    async def _run() -> tuple[int, int]:
        settings = get_settings()
        async with lifespan_pool(settings) as pool:
            store = ArenaStore(pool)
            rows = await store.get_cooccurrence_counts()
            ratings = await store.get_latest_ratings(metric_name=metric, domain=domain)
        elos = {key: r.rating_elo for key, r in ratings.items()}
        cooc = build_cooccurrence(rows, elos)
        Path(out).write_text(render_cooccurrence(cooc), encoding="utf-8")
        return cooc.total, len(cooc.labels)

    total, models = asyncio.run(_run())
    click.echo(
        json.dumps({"event": "arena_cooccurrence", "out": out, "battles": total, "models": models})
    )


@arena.command(name="convergence")
@click.option(
    "--out",
    default="arena-convergence.html",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="HTML output path.",
)
@click.option("--metric", default="naturalness", show_default=True)
@click.option("--domain", default="all", show_default=True)
@click.option(
    "--threshold",
    default=9.0,
    show_default=True,
    type=float,
    help="CI half-width (Elo) counted as converged.",
)
def arena_convergence(out: str, metric: str, domain: str, threshold: float) -> None:
    """Render CI-vs-votes convergence per model from snapshot history (admin)."""
    from pathlib import Path

    from coval_bench.arena.monitoring import build_convergence, render_convergence
    from coval_bench.config import get_settings
    from coval_bench.db.arena_store import ArenaStore
    from coval_bench.db.conn import lifespan_pool

    async def _run() -> int:
        settings = get_settings()
        async with lifespan_pool(settings) as pool:
            rows = await ArenaStore(pool).get_snapshot_history(metric_name=metric, domain=domain)
        conv = build_convergence(rows, threshold=threshold)
        Path(out).write_text(render_convergence(conv), encoding="utf-8")
        return len(conv.series)

    models = asyncio.run(_run())
    click.echo(json.dumps({"event": "arena_convergence", "out": out, "models": models}))


if __name__ == "__main__":
    cli()
