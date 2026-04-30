# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for coval-bench.

The ``coval-bench run`` subcommand is the entrypoint for the Cloud Run Job::

    CMD ["python", "-m", "coval_bench", "run"]

It calls :func:`coval_bench.runner.run_benchmarks` and exits non-zero when the
run status is FAILED.
"""

import asyncio

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


if __name__ == "__main__":
    cli()
