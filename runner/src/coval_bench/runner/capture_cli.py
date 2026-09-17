# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Operator commands for inspecting and replaying durable normalized captures."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import click
from google.cloud import storage

from coval_bench.config import Settings, get_settings
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.models import RunStatus
from coval_bench.db.writer import RunWriter
from coval_bench.runner.capture import (
    CaptureEnvelope,
    FinalizedReceipt,
    RunManifest,
    RunSeal,
    identity_digest,
    list_envelope_uris,
    load_envelope,
    missing_expected_envelope_ids,
    preflight_capture_storage,
    read_receipt,
    read_run_state,
    run_prefix,
    upload_run_state,
)
from coval_bench.runner.normalized import CaptureOutcome, replay_capture


def _storage(settings: Settings) -> tuple[storage.Client, str]:
    bucket = settings.benchmark_artifact_bucket
    if not bucket:
        raise click.ClickException("BENCHMARK_ARTIFACT_BUCKET is not configured")
    return storage.Client(), bucket


def _json(value: object) -> None:
    click.echo(json.dumps(value, sort_keys=True, default=str))


@click.group(name="capture")
def capture() -> None:
    """Inspect or recover immutable normalized-capture envelopes."""


@capture.command(name="status")
@click.option("--run-id", type=click.IntRange(min=1), default=None)
@click.option("--limit", type=click.IntRange(min=1, max=1000), default=100, show_default=True)
@click.option("--cursor", type=str, default=None, help="Opaque next cursor from an earlier page.")
def capture_status(run_id: int | None, limit: int, cursor: str | None) -> None:
    """Report a bounded capture backlog without printing private payloads."""
    client, bucket = _storage(get_settings())
    uris, next_cursor = list_envelope_uris(
        client, bucket, run_id=run_id, limit=limit, cursor=cursor
    )
    counts = {"complete": 0, "pending": 0, "invalid": 0, "missing_expected": 0}
    items: list[dict[str, object]] = []
    for uri in uris:
        try:
            envelope = load_envelope(client, bucket, uri)
            complete = read_receipt(client, bucket, envelope) is not None
            stage = "complete" if complete else "pending"
            counts[stage] += 1
            items.append(
                {
                    "run_id": envelope.identity.run_id,
                    "identity": identity_digest(envelope.identity),
                    "envelope": envelope.envelope_digest(),
                    "stage": stage,
                }
            )
        except (ValueError, TypeError):
            counts["invalid"] += 1
            items.append({"object": uri.rsplit("/", 1)[-1], "stage": "invalid"})
    report: dict[str, object] = {
        "run_id": run_id,
        "counts": counts,
        "items": items,
        "next_cursor": next_cursor,
    }
    if run_id is not None:
        manifest_value = read_run_state(client, bucket, run_id, "manifest")
        missing_expected: list[str] = []
        if manifest_value is not None:
            manifest = RunManifest.model_validate(manifest_value)
            missing_expected = missing_expected_envelope_ids(
                client, bucket, run_id, manifest.expected_capture_ids
            )
            counts["missing_expected"] = len(missing_expected)
        report["run"] = {
            kind: read_run_state(client, bucket, run_id, kind) is not None
            for kind in ("manifest", "seal", "finalized")
        }
        report["missing_expected_capture_ids"] = missing_expected
    _json(report)
    if counts["invalid"]:
        raise click.ClickException("one or more capture objects failed validation")


def _expected_envelopes(
    client: storage.Client,
    bucket: str,
    run_id: int,
    expected_capture_ids: list[str],
) -> tuple[list[tuple[CaptureEnvelope, bool]], bool]:
    """Load at most two envelope objects for each manifest identity."""
    found: list[tuple[CaptureEnvelope, bool]] = []
    exact = True
    for expected_id in expected_capture_ids:
        prefix = f"{run_prefix(run_id)}/{expected_id}/envelope/"
        blobs = list(client.list_blobs(bucket, prefix=prefix, max_results=2))
        if len(blobs) != 1:
            exact = False
            continue
        envelope = load_envelope(client, bucket, blobs[0].name)
        if envelope.identity.run_id != run_id or identity_digest(envelope.identity) != expected_id:
            exact = False
            continue
        found.append((envelope, read_receipt(client, bucket, envelope) is not None))
    return found, exact


async def _recover(
    *,
    settings: Settings,
    client: storage.Client,
    bucket: str,
    run_id: int,
    limit: int,
    cursor: str | None,
    abandoned: bool,
) -> dict[str, object]:
    manifest_value = read_run_state(client, bucket, run_id, "manifest")
    if manifest_value is None:
        raise click.ClickException(f"capture manifest for run {run_id} is missing")
    manifest = RunManifest.model_validate(manifest_value)
    seal_value = read_run_state(client, bucket, run_id, "seal")
    if seal_value is None and not abandoned:
        raise click.ClickException(
            "run is unsealed; pass --abandoned only after confirming provider work stopped"
        )

    preflight_capture_storage(client, bucket)
    uris, next_cursor = list_envelope_uris(
        client, bucket, run_id=run_id, limit=limit, cursor=cursor
    )
    outcomes: dict[str, int] = {"already_completed": 0}
    async with lifespan_pool(settings) as pool:
        writer = RunWriter(pool)
        await writer.preflight_required_capture_schema()
        for uri in uris:
            envelope = load_envelope(client, bucket, uri)
            if read_receipt(client, bucket, envelope) is not None:
                name = "already_completed"
            else:
                outcome = await replay_capture(
                    writer=writer,
                    storage_client=client,
                    bucket=bucket,
                    envelope=envelope,
                )
                name = str(outcome)
            outcomes[name] = outcomes.get(name, 0) + 1

        all_envelopes, exact_envelopes = _expected_envelopes(
            client, bucket, run_id, manifest.expected_capture_ids
        )
        identities = {identity_digest(envelope.identity) for envelope, _complete in all_envelopes}
        complete = all(complete for _envelope, complete in all_envelopes)
        expected_complete = (
            exact_envelopes
            and identities == set(manifest.expected_capture_ids)
            and len(all_envelopes) == len(identities)
            and complete
        )
        finalized = False
        if next_cursor is None and expected_complete:
            if seal_value is None:
                run = await writer.get_run(run_id)
                finished_at = run.finished_at or datetime.now(UTC)
                status = run.status if run.finished_at is not None else RunStatus.PARTIAL
                error = run.error if run.finished_at is not None else "abandoned normalized capture"
                seal = RunSeal(
                    run_id=run_id,
                    intended_status=str(status),
                    stored_status=str(status),
                    finished_at=finished_at,
                    error=error,
                    expected_capture_ids=manifest.expected_capture_ids,
                )
                _, seal_sha256 = upload_run_state(client, bucket, run_id, "seal", seal)
                await writer.finish_run_exact(
                    run_id,
                    status=status,
                    finished_at=finished_at,
                    error=error,
                )
            else:
                seal = RunSeal.model_validate(seal_value)
                _, seal_digest = upload_run_state(client, bucket, run_id, "seal", seal)
                seal_sha256 = seal_digest
                intended = RunStatus(seal.intended_status)
                finish_error = None if seal.error == "normalized capture pending" else seal.error
                await writer.finish_run_exact(
                    run_id,
                    status=intended,
                    finished_at=seal.finished_at,
                    error=finish_error,
                    allow_capture_recovery=True,
                )
            upload_run_state(
                client,
                bucket,
                run_id,
                "finalized",
                FinalizedReceipt(run_id=run_id, seal_sha256=seal_sha256),
            )
            finalized = True

    incomplete = sum(
        count
        for name, count in outcomes.items()
        if name
        in {
            str(CaptureOutcome.PENDING),
            str(CaptureOutcome.CONFLICT),
            str(CaptureOutcome.UNACKNOWLEDGED),
        }
    )
    missing_expected = sorted(set(manifest.expected_capture_ids) - identities)
    if missing_expected:
        outcomes["missing_expected"] = len(missing_expected)
        incomplete += len(missing_expected)
    return {
        "run_id": run_id,
        "processed": len(uris),
        "outcomes": outcomes,
        "expected": len(manifest.expected_capture_ids),
        "discovered": len(identities),
        "missing_expected_capture_ids": missing_expected,
        "finalized": finalized,
        "next_cursor": next_cursor,
        "incomplete": incomplete,
    }


@capture.command(name="recover")
@click.option("--run-id", type=click.IntRange(min=1), required=True)
@click.option("--limit", type=click.IntRange(min=1, max=1000), default=100, show_default=True)
@click.option("--cursor", type=str, default=None, help="Opaque next cursor from an earlier page.")
@click.option(
    "--abandoned",
    is_flag=True,
    help="Allow replay/finalization of an unsealed run after provider work has stopped.",
)
def capture_recover(run_id: int, limit: int, cursor: str | None, abandoned: bool) -> None:
    """Replay a bounded page and finalize a complete sealed run."""
    settings = get_settings()
    client, bucket = _storage(settings)
    report = asyncio.run(
        _recover(
            settings=settings,
            client=client,
            bucket=bucket,
            run_id=run_id,
            limit=limit,
            cursor=cursor,
            abandoned=abandoned,
        )
    )
    _json(report)
    if report["incomplete"] or (report["next_cursor"] is None and not report["finalized"]):
        raise click.ClickException("capture recovery remains incomplete")
