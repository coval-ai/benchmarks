# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""CLI for the standardized dataset builder.

Two ways to build:
- ``coval-build-dataset <name>`` — a registered dataset (``_SPECS``) or a scaffolded
  adapter file in ``adapters/``.
- ``coval-build-dataset --hf <path>`` — any Hugging Face audio dataset, read live via
  ``hf_source`` (auto-detects columns; ``--audio-col``/``--text-col`` override, and
  ``--balance <col>`` picks the balance dimension). If it can't resolve the dataset it
  scaffolds a starter adapter to complete by hand.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import click
import structlog

from coval_bench.datasets.scripts import hf_source
from coval_bench.datasets.scripts.framework import Clip, DatasetSpec, run_build
from coval_bench.datasets.scripts.s2s_v1 import S2S_V1

_SPECS: dict[str, DatasetSpec] = {S2S_V1.dataset_id: S2S_V1}
_CACHE_ROOT = Path.home() / ".cache" / "coval-bench"
_ADAPTERS_DIR = Path(__file__).parent / "adapters"


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def _meta_dim(col: str) -> Callable[[Clip], object]:
    """A balance-dimension accessor for meta column *col* (None when absent/empty)."""
    return lambda clip: clip.meta.get(col) or None


def _load_adapter_spec(name: str) -> DatasetSpec | None:
    """Load ``SPEC`` from a scaffolded handwritten adapter ``adapters/<name>.py``."""
    path = _ADAPTERS_DIR / f"{name}.py"
    if not path.exists():
        return None
    module_spec = importlib.util.spec_from_file_location(f"_adapter_{name}", path)
    if module_spec is None or module_spec.loader is None:
        return None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return cast(DatasetSpec, module.SPEC)


def _hf_spec(
    hf_path: str,
    *,
    config: str | None,
    split: str | None,
    audio_col: str | None,
    text_col: str | None,
    balance_cols: tuple[str, ...],
    num: int,
    dur_min: float,
    dur_max: float,
    dataset_id: str | None,
) -> DatasetSpec:
    """Resolve *hf_path* into a DatasetSpec; scaffold an adapter if it can't be read."""
    ds_id = dataset_id or f"{hf_path.split('/')[-1].lower()}-v1"
    try:
        config, splits = hf_source.resolve_splits(hf_path, config=config, split=split)
        audio_col, text_col, duration_col = hf_source.detect_columns(
            hf_path, config, splits[0], audio_col=audio_col, text_col=text_col
        )
    except hf_source.HFNeedsChoice as exc:
        raise click.UsageError(str(exc)) from exc
    except (hf_source.HFUnsupported, hf_source.HFAmbiguous) as exc:
        _ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
        dest = _ADAPTERS_DIR / f"{ds_id}.py"
        hf_source.scaffold_adapter(hf_path, dest, detected=str(exc))
        raise click.UsageError(
            f"{exc}\nScaffolded {dest} — complete it, then run: coval-build-dataset {ds_id}"
        ) from exc
    download, parse, fetch = hf_source.make_source(
        hf_path, config, splits, audio_col, text_col, duration_col
    )
    return DatasetSpec(
        dataset_id=ds_id,
        cache_name=ds_id,
        download=download,
        parse=parse,
        dur_min=dur_min,
        dur_max=dur_max,
        min_words=3,
        num=num,
        dedup_key=lambda clip: clip.audio_path.name,
        balance_dims=tuple(_meta_dim(col) for col in balance_cols),
        license="see-source",
        source=hf_path,
        needs_vad_offset=False,
        fetch=fetch,
    )


@click.command()
@click.argument("dataset", required=False)
@click.option("--hf", "hf_path", default=None, help="HF dataset path, e.g. bosonai/WildASR.")
@click.option("--config", default=None, help="HF config (required if the dataset has several).")
@click.option(
    "--split", default=None, help="HF split (default: test if standard, else all facets)."
)
@click.option("--audio-col", default=None, help="Override the audio column.")
@click.option("--text-col", default=None, help="Override the transcript column.")
@click.option("--balance", "balance_cols", multiple=True, help="Meta column(s) to balance across.")
@click.option("--num", type=int, default=50, show_default=True, help="Number of clips.")
@click.option("--dur-min", type=float, default=2.0, show_default=True, help="Min clip seconds.")
@click.option("--dur-max", type=float, default=10.0, show_default=True, help="Max clip seconds.")
@click.option("--dataset-id", default=None, help="Manifest id (default derived from --hf).")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Build + print manifest; no GCS writes."
)
@click.option(
    "--bucket", default="coval-benchmarks-datasets", show_default=True, help="Target GCS bucket."
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing GCS objects / manifest (off by default — v1 freeze).",
)
def build(
    dataset: str | None,
    hf_path: str | None,
    config: str | None,
    split: str | None,
    audio_col: str | None,
    text_col: str | None,
    balance_cols: tuple[str, ...],
    num: int,
    dur_min: float,
    dur_max: float,
    dataset_id: str | None,
    dry_run: bool,
    bucket: str,
    overwrite: bool,
) -> None:
    """Build a dataset through the standardized framework and write its manifest."""
    _configure_logging()
    if hf_path:
        spec = _hf_spec(
            hf_path,
            config=config,
            split=split,
            audio_col=audio_col,
            text_col=text_col,
            balance_cols=balance_cols,
            num=num,
            dur_min=dur_min,
            dur_max=dur_max,
            dataset_id=dataset_id,
        )
    elif dataset and dataset in _SPECS:
        spec = _SPECS[dataset]
    elif dataset and (adapter := _load_adapter_spec(dataset)) is not None:
        spec = adapter
    else:
        raise click.UsageError(
            f"pass --hf <path>, a registered dataset {sorted(_SPECS)}, or a scaffolded adapter name"
        )
    run_build(
        spec,
        bucket=bucket,
        dry_run=dry_run,
        overwrite=overwrite,
        cache_root=_CACHE_ROOT / spec.cache_name,
    )


if __name__ == "__main__":
    build()
