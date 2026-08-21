# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Commands for running, validating, and summarizing private benchmark artifacts."""

import json
from pathlib import Path

import click
import httpx
from pydantic import SecretStr, ValidationError

from coval_bench.preprocessing.artifacts import canonical_json_bytes
from coval_bench.preprocessing.benchmarking.candidates import (
    CANDIDATES,
    DEEPGRAM_NOVA_3_CANDIDATE_ID,
    deepgram_nova_3_candidate,
)
from coval_bench.preprocessing.benchmarking.contracts import HostedBillingPolicy
from coval_bench.preprocessing.benchmarking.privacy import validate_private_evidence_path
from coval_bench.preprocessing.benchmarking.reporting import (
    TimestampBenchmarkBundleV1,
    build_report,
    merge_benchmark_bundles,
    select_candidate_subset,
)


@click.group(name="timestamp-benchmark")
def timestamp_benchmark() -> None:
    """Validate and summarize timestamp model benchmark evidence."""


@timestamp_benchmark.command(name="candidates")
def candidates() -> None:
    """Print the pinned public candidate registry."""
    click.echo(
        json.dumps(
            [candidate.model_dump(mode="json") for candidate in CANDIDATES],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


@timestamp_benchmark.command(name="discover-deepgram-version")
@click.option("--audio", "audio_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--deepgram-api-key",
    envvar="DEEPGRAM_API_KEY",
    required=True,
    hide_input=True,
)
def discover_deepgram_version(audio_path: Path, deepgram_api_key: str) -> None:
    """Resolve Nova-3's current exact version before registering and running it."""
    from coval_bench.preprocessing.benchmarking.adapters import (
        discover_deepgram_nova_3_version,
    )

    source = validate_private_evidence_path(audio_path)
    try:
        model_version = discover_deepgram_nova_3_version(
            audio_path=source,
            api_key=SecretStr(deepgram_api_key),
        )
    except (OSError, ValueError, httpx.HTTPError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "candidate_spec": deepgram_nova_3_candidate(model_version).model_dump(mode="json"),
                "event": "deepgram_nova_3_version_discovered",
                "model_version": model_version,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


@timestamp_benchmark.command(name="build-word-manifest")
@click.option(
    "--dataset-manifest",
    "dataset_manifest_path",
    required=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--selection-manifest",
    "selection_manifest_path",
    required=True,
    type=click.Path(path_type=Path),
)
@click.option("--candidate-id", required=True)
@click.option("--deepgram-model-version", default=None)
@click.option("--benchmark-id", required=True)
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
def build_word_manifest(
    dataset_manifest_path: Path,
    selection_manifest_path: Path,
    candidate_id: str,
    deepgram_model_version: str | None,
    benchmark_id: str,
    output_path: Path,
) -> None:
    """Build a private word-run manifest from a frozen audio-only selection."""
    from coval_bench.preprocessing.benchmarking.runtime import (
        PrivateAudioSelectionV1,
        PublicSTTManifestV1,
        build_word_agreement_manifest,
    )

    selection_source = validate_private_evidence_path(selection_manifest_path)
    destination = validate_private_evidence_path(output_path)
    try:
        dataset_manifest = PublicSTTManifestV1.model_validate_json(
            dataset_manifest_path.read_bytes()
        )
        selection_manifest = PrivateAudioSelectionV1.model_validate_json(
            selection_source.read_bytes()
        )
        candidate_spec = None
        if candidate_id == DEEPGRAM_NOVA_3_CANDIDATE_ID:
            if deepgram_model_version is None:
                raise ValueError("Deepgram requires --deepgram-model-version from the probe")
            candidate_spec = deepgram_nova_3_candidate(deepgram_model_version)
        elif deepgram_model_version is not None:
            raise ValueError("--deepgram-model-version is only valid for the Deepgram candidate")
        manifest = build_word_agreement_manifest(
            dataset_manifest=dataset_manifest,
            selection_manifest=selection_manifest,
            candidate_id=candidate_id,
            benchmark_id=benchmark_id,
            candidate_spec=candidate_spec,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_json_bytes(manifest))
    except (OSError, RuntimeError, ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "clips": len(manifest.clips),
                "event": "timestamp_word_manifest_built",
                "output": str(destination),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


@timestamp_benchmark.command(name="run-word-agreement")
@click.option("--manifest", "manifest_path", required=True, type=click.Path(path_type=Path))
@click.option("--bundle-output", required=True, type=click.Path(path_type=Path))
@click.option("--operational-output", required=True, type=click.Path(path_type=Path))
@click.option("--google-project-id", default=None)
@click.option(
    "--deepgram-api-key",
    envvar="DEEPGRAM_API_KEY",
    default=None,
    hide_input=True,
)
@click.option("--device", default="cpu", show_default=True)
@click.option("--local-files-only", is_flag=True)
@click.option("--audio-minute-cost-usd", type=click.FloatRange(min=0), default=None)
@click.option("--hosted-price-reference", default=None)
@click.option(
    "--hosted-billing-policy",
    type=click.Choice(
        ["exact-audio-duration-v1", "per-request-ceil-1000ms-v1"],
        case_sensitive=True,
    ),
    default=None,
)
def run_word_agreement_command(
    manifest_path: Path,
    bundle_output: Path,
    operational_output: Path,
    google_project_id: str | None,
    deepgram_api_key: str | None,
    device: str,
    local_files_only: bool,
    audio_minute_cost_usd: float | None,
    hosted_price_reference: str | None,
    hosted_billing_policy: HostedBillingPolicy | None,
) -> None:
    """Run one registered word candidate over a private local manifest."""
    from coval_bench.preprocessing.benchmarking.runtime import (
        WordAgreementRunManifestV1,
        run_word_agreement,
    )

    source = validate_private_evidence_path(manifest_path)
    bundle_destination = validate_private_evidence_path(bundle_output)
    operational_destination = validate_private_evidence_path(operational_output)
    try:
        manifest = WordAgreementRunManifestV1.model_validate_json(source.read_bytes())
        bundle, operational = run_word_agreement(
            manifest,
            google_project_id=google_project_id,
            deepgram_api_key=(
                SecretStr(deepgram_api_key) if deepgram_api_key is not None else None
            ),
            device=device,
            local_files_only=local_files_only,
            audio_minute_cost_usd=audio_minute_cost_usd,
            hosted_price_reference=hosted_price_reference,
            hosted_billing_policy=hosted_billing_policy,
        )
        bundle_destination.parent.mkdir(parents=True, exist_ok=True)
        operational_destination.parent.mkdir(parents=True, exist_ok=True)
        bundle_destination.write_bytes(canonical_json_bytes(bundle))
        operational_destination.write_bytes(canonical_json_bytes(operational))
    except (OSError, RuntimeError, ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "bundle_output": str(bundle_destination),
                "event": "timestamp_word_agreement_completed",
                "operational_output": str(operational_destination),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


@timestamp_benchmark.command(name="run-phoneme-agreement")
@click.option("--manifest", "manifest_path", required=True, type=click.Path(path_type=Path))
@click.option("--bundle-output", required=True, type=click.Path(path_type=Path))
@click.option("--operational-output", required=True, type=click.Path(path_type=Path))
@click.option("--device", default="cpu", show_default=True)
@click.option("--local-files-only", is_flag=True)
@click.option("--gpu-hour-cost-usd", type=click.FloatRange(min=0), default=None)
def run_phoneme_agreement_command(
    manifest_path: Path,
    bundle_output: Path,
    operational_output: Path,
    device: str,
    local_files_only: bool,
    gpu_hour_cost_usd: float | None,
) -> None:
    """Run pinned audio-only phone candidates over a private local manifest."""
    from coval_bench.preprocessing.benchmarking.runtime import (
        PhonemeAgreementRunManifestV1,
        run_phoneme_agreement,
    )

    source = validate_private_evidence_path(manifest_path)
    bundle_destination = validate_private_evidence_path(bundle_output)
    operational_destination = validate_private_evidence_path(operational_output)
    try:
        manifest = PhonemeAgreementRunManifestV1.model_validate_json(source.read_bytes())
        bundle, operational = run_phoneme_agreement(
            manifest,
            device=device,
            local_files_only=local_files_only,
            gpu_hour_cost_usd=gpu_hour_cost_usd,
        )
        bundle_destination.parent.mkdir(parents=True, exist_ok=True)
        operational_destination.parent.mkdir(parents=True, exist_ok=True)
        bundle_destination.write_bytes(canonical_json_bytes(bundle))
        operational_destination.write_bytes(canonical_json_bytes(operational))
    except (OSError, ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "bundle_output": str(bundle_destination),
                "event": "timestamp_phoneme_agreement_completed",
                "operational_output": str(operational_destination),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


@timestamp_benchmark.command(name="merge")
@click.option(
    "--input",
    "input_paths",
    required=True,
    multiple=True,
    type=click.Path(path_type=Path),
)
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--benchmark-id", required=True)
def merge(input_paths: tuple[Path, ...], output_path: Path, benchmark_id: str) -> None:
    """Merge isolated candidate bundles over the same private case matrix."""
    sources = tuple(
        validate_private_evidence_path(path, allow_invented_fixture=True) for path in input_paths
    )
    destination = validate_private_evidence_path(output_path)
    try:
        bundles = tuple(
            TimestampBenchmarkBundleV1.model_validate_json(source.read_bytes())
            for source in sources
        )
        merged = merge_benchmark_bundles(bundles, benchmark_id=benchmark_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_json_bytes(merged))
    except (OSError, ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "candidates": len(merged.candidate_ids),
                "event": "timestamp_benchmark_bundles_merged",
                "output": str(destination),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


@timestamp_benchmark.command(name="subset")
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
@click.option("--candidate-id", "candidate_ids", required=True, multiple=True)
@click.option("--benchmark-id", required=True)
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
def subset(
    input_path: Path,
    candidate_ids: tuple[str, ...],
    benchmark_id: str,
    output_path: Path,
) -> None:
    """Create a private report bundle containing only the requested candidates."""
    source = validate_private_evidence_path(input_path, allow_invented_fixture=True)
    destination = validate_private_evidence_path(output_path)
    try:
        bundle = TimestampBenchmarkBundleV1.model_validate_json(source.read_bytes())
        selected = select_candidate_subset(
            bundle,
            candidate_ids=candidate_ids,
            benchmark_id=benchmark_id,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_json_bytes(selected))
    except (OSError, ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "candidates": len(selected.candidate_ids),
                "event": "timestamp_benchmark_subset_created",
                "output": str(destination),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


@timestamp_benchmark.command(name="validate")
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
def validate(input_path: Path) -> None:
    """Validate one private benchmark bundle without running model code."""
    path = validate_private_evidence_path(input_path, allow_invented_fixture=True)
    try:
        bundle = TimestampBenchmarkBundleV1.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "benchmark_id": bundle.benchmark_id,
                "cases": len(bundle.cases),
                "event": "timestamp_benchmark_valid",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


@timestamp_benchmark.command(name="summarize")
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
def summarize(input_path: Path, output_path: Path) -> None:
    """Build a deterministic private accuracy or agreement report."""
    source = validate_private_evidence_path(input_path, allow_invented_fixture=True)
    destination = validate_private_evidence_path(output_path)
    try:
        bundle = TimestampBenchmarkBundleV1.model_validate_json(source.read_bytes())
        report = build_report(bundle)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_json_bytes(report))
    except (OSError, ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "event": "timestamp_benchmark_summarized",
                "observations": len(report.observations),
                "output": str(destination),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
