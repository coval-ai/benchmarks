# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the one-shot WER error-type backfill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
import pytest
from httpx import AsyncClient

from coval_bench.datasets.loader import _load_manifest
from coval_bench.datasets.manifest import STTManifestItem
from coval_bench.metrics.wer import compute_wer, normalize_text
from coval_bench.migrations.backfill_wer_breakdown import backfill
from tests.api.conftest import _insert_result, _insert_run, _make_db_url, _refresh_mv

_DATASET = "stt-wildasr-clean"
_ITEM = _load_manifest(_DATASET).items[0]
assert isinstance(_ITEM, STTManifestItem)
_REFERENCE = _ITEM.transcript
_FILENAME = Path(_ITEM.path).name
_WORDS = normalize_text(_REFERENCE).split()

# One hypothesis per error type, derived from the real manifest transcript.
_HYPOTHESES = {
    "deletion": " ".join(_WORDS[1:]),
    "insertion": " ".join([*_WORDS, "orange"]),
    "substitution": " ".join(["orange", *_WORDS[1:]]),
}


async def _insert_legacy_wer(postgresql: Any, run_id: int, hypothesis: str, **kwargs: Any) -> int:
    """A pre-0013 row: scored transcript present, split columns null."""
    defaults: dict[str, Any] = {
        "audio_filename": _FILENAME,
        "transcript": hypothesis,
        "metric_value": compute_wer(_REFERENCE, hypothesis).wer_percentage,
    }
    defaults.update(kwargs)
    return await _insert_result(postgresql, run_id, **defaults)


def _run_backfill(postgresql: Any, **kwargs: Any) -> dict[str, int]:
    with psycopg.connect(_make_db_url(postgresql)) as conn:
        return backfill(conn, **kwargs)


async def _split_of(postgresql: Any, row_id: int) -> tuple[Any, Any, Any]:
    aconn = await psycopg.AsyncConnection.connect(_make_db_url(postgresql), autocommit=True)
    try:
        cur = await aconn.execute(
            "SELECT wer_insertions_pct, wer_deletions_pct, wer_substitutions_pct"
            " FROM benchmarks_v2.results WHERE id = %s",
            (row_id,),
        )
        row = await cur.fetchone()
        assert row is not None
        return row
    finally:
        await aconn.close()


async def test_backfill_types_each_error_and_reconciles(postgresql: Any) -> None:
    run_id = await _insert_run(postgresql, dataset_id=_DATASET)
    ids = {
        kind: await _insert_legacy_wer(postgresql, run_id, hyp) for kind, hyp in _HYPOTHESES.items()
    }

    counts = _run_backfill(postgresql)
    assert counts == {"updated": 3, "missing_reference": 0, "score_drift": 0}

    for kind, row_id in ids.items():
        ins, dele, sub = await _split_of(postgresql, row_id)
        total = compute_wer(_REFERENCE, _HYPOTHESES[kind]).wer_percentage
        expected = {"insertion": ins, "deletion": dele, "substitution": sub}
        assert expected[kind] == pytest.approx(total)
        assert ins + dele + sub == pytest.approx(total)


async def test_backfill_skips_rows_it_cannot_prove(postgresql: Any) -> None:
    run_id = await _insert_run(postgresql, dataset_id=_DATASET)
    drifted = await _insert_legacy_wer(
        postgresql, run_id, _HYPOTHESES["deletion"], metric_value=99.0
    )
    unknown_clip = await _insert_legacy_wer(
        postgresql, run_id, _HYPOTHESES["deletion"], audio_filename="nope.wav"
    )
    unknown_run = await _insert_run(postgresql, dataset_id="stt-v0-retired")
    unknown_dataset = await _insert_legacy_wer(postgresql, unknown_run, _HYPOTHESES["deletion"])
    tts_run = await _insert_run(postgresql, dataset_id="tts-v1")
    tts = await _insert_legacy_wer(postgresql, tts_run, "whatever", benchmark="TTS")

    counts = _run_backfill(postgresql)
    assert counts == {"updated": 0, "missing_reference": 2, "score_drift": 1}
    for row_id in (drifted, unknown_clip, unknown_dataset, tts):
        assert await _split_of(postgresql, row_id) == (None, None, None)


async def test_backfill_is_idempotent_and_dry_run_writes_nothing(
    postgresql: Any,
) -> None:
    run_id = await _insert_run(postgresql, dataset_id=_DATASET)
    row_id = await _insert_legacy_wer(postgresql, run_id, _HYPOTHESES["insertion"])

    assert _run_backfill(postgresql, dry_run=True)["updated"] == 1
    assert await _split_of(postgresql, row_id) == (None, None, None)

    assert _run_backfill(postgresql)["updated"] == 1
    assert _run_backfill(postgresql)["updated"] == 0


async def test_backfilled_rows_reach_aggregates(client: AsyncClient, postgresql: Any) -> None:
    run_id = await _insert_run(postgresql, dataset_id=_DATASET)
    await _insert_legacy_wer(postgresql, run_id, _HYPOTHESES["substitution"])

    _run_backfill(postgresql)
    await _refresh_mv(postgresql)

    response = await client.get("/v1/results/aggregates", params={"benchmark": "STT"})
    s = response.json()["model_stats"][0]
    parts = ("wer_insertions_pct", "wer_deletions_pct", "wer_substitutions_pct")
    assert sum(s[k] for k in parts) == pytest.approx(s["avg_value"])
    assert s["wer_substitutions_pct"] == pytest.approx(s["avg_value"])
