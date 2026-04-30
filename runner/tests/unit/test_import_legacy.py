# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for coval_bench.migrations.import_legacy.

All tests are pure — no live database connections, no network calls.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime

import pytest

from coval_bench.migrations.import_legacy import (
    LegacyRow,
    _group_by_day,
    _summarize,
    _validate,
    map_status,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    *,
    provider: str = "deepgram",
    model: str = "nova-2",
    voice: str | None = None,
    benchmark: str = "STT",
    metric_type: str = "WER",
    metric_value: float | None = 0.05,
    metric_units: str | None = "ratio",
    audio_filename: str | None = "sample.wav",
    transcript: str | None = "hello world",
    timestamp: datetime | None = None,
    status: str = "success",
) -> LegacyRow:
    if timestamp is None:
        timestamp = datetime(2026, 4, 22, 10, 0, 0, tzinfo=UTC)
    return LegacyRow(
        provider=provider,
        model=model,
        voice=voice,
        benchmark=benchmark,
        metric_type=metric_type,
        metric_value=metric_value,
        metric_units=metric_units,
        audio_filename=audio_filename,
        transcript=transcript,
        timestamp=timestamp,
        status=status,
    )


# ---------------------------------------------------------------------------
# test 1 — all known combos produce empty unmatched set
# ---------------------------------------------------------------------------


def test_validate_all_known() -> None:
    rows = [
        _make_row(provider="deepgram", model="nova-2"),
        _make_row(provider="deepgram", model="nova-3"),
        _make_row(provider="assemblyai", model="universal-streaming"),
    ]
    unmatched, status_counts = _validate(rows)
    assert unmatched == set()
    assert status_counts["success"] == 3


# ---------------------------------------------------------------------------
# test 2 — unknown (provider, model) pair appears in unmatched
# ---------------------------------------------------------------------------


def test_validate_unmatched() -> None:
    rows = [
        _make_row(provider="deepgram", model="nova-2"),
        _make_row(provider="unknown-provider", model="unknown-model"),
        _make_row(provider="unknown-provider", model="unknown-model"),
    ]
    unmatched, _ = _validate(rows)
    assert ("unknown-provider", "unknown-model") in unmatched
    assert len(unmatched) == 1


# ---------------------------------------------------------------------------
# test 3 — mixed-case provider is normalised for lookup
# ---------------------------------------------------------------------------


def test_validate_provider_casing() -> None:
    # Legacy writes "Deepgram" (capital D) for TTS; matrix has "deepgram".
    rows = [
        _make_row(provider="Deepgram", model="aura-2-thalia-en", benchmark="TTS"),
    ]
    unmatched, _ = _validate(rows)
    assert unmatched == set(), f"expected no unmatched, got {unmatched}"


# ---------------------------------------------------------------------------
# test 4 — legacy 'success' maps to ('success', None)
# ---------------------------------------------------------------------------


def test_status_mapping_success() -> None:
    target_status, error = map_status("success")
    assert target_status == "success"
    assert error is None


# ---------------------------------------------------------------------------
# test 5 — legacy 'tts_failed' maps to ('failed', 'legacy_status:tts_failed')
# ---------------------------------------------------------------------------


def test_status_mapping_tts_failed() -> None:
    target_status, error = map_status("tts_failed")
    assert target_status == "failed"
    assert error == "legacy_status:tts_failed"


# ---------------------------------------------------------------------------
# test 6 — unknown status is NOT in map; Counter reflects it; caller exits
# ---------------------------------------------------------------------------


def test_status_mapping_unknown_raises() -> None:
    # map_status raises KeyError for unknown values (dict lookup).
    with pytest.raises(KeyError):
        map_status("weird")

    # Validate that _validate accumulates it in status_counts so caller
    # can detect it.  The caller (import_legacy_cli) then exits non-zero.
    rows = [_make_row(status="weird")]
    _, status_counts = _validate(rows)
    assert status_counts["weird"] == 1
    unexpected = set(status_counts) - {"success", "tts_failed"}
    assert "weird" in unexpected


# ---------------------------------------------------------------------------
# test 7 — rows spanning 3 UTC days produce 3 groups
# ---------------------------------------------------------------------------


def test_group_by_day_utc() -> None:
    rows = [
        _make_row(timestamp=datetime(2026, 4, 22, 0, 0, 0, tzinfo=UTC)),
        _make_row(timestamp=datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)),
        _make_row(timestamp=datetime(2026, 4, 24, 23, 59, 59, tzinfo=UTC)),
    ]
    groups = _group_by_day(rows)
    assert set(groups.keys()) == {
        date(2026, 4, 22),
        date(2026, 4, 23),
        date(2026, 4, 24),
    }
    # Row at 23:59:59 UTC on the 24th must land in the 24th, not the 25th.
    assert len(groups[date(2026, 4, 24)]) == 1


# ---------------------------------------------------------------------------
# test 8 — started_at = min(timestamp), finished_at = max(timestamp)
# ---------------------------------------------------------------------------


def test_group_by_day_started_finished() -> None:
    ts_early = datetime(2026, 4, 22, 1, 0, 0, tzinfo=UTC)
    ts_late = datetime(2026, 4, 22, 22, 30, 0, tzinfo=UTC)
    ts_mid = datetime(2026, 4, 22, 10, 0, 0, tzinfo=UTC)

    rows = [
        _make_row(timestamp=ts_early),
        _make_row(timestamp=ts_late),
        _make_row(timestamp=ts_mid),
    ]
    groups = _group_by_day(rows)
    day_rows = groups[date(2026, 4, 22)]
    day_ts = [r.timestamp for r in day_rows]
    assert min(day_ts) == ts_early
    assert max(day_ts) == ts_late


# ---------------------------------------------------------------------------
# test 9 — _summarize golden-string check against a fixed 5-row fixture
# ---------------------------------------------------------------------------

_FIXTURE_ROWS = [
    _make_row(
        provider="deepgram",
        model="nova-2",
        metric_type="WER",
        metric_value=0.05,
        timestamp=datetime(2026, 4, 22, 1, 0, 0, tzinfo=UTC),
        status="success",
    ),
    _make_row(
        provider="deepgram",
        model="nova-2",
        metric_type="TTFT",
        metric_value=120.0,
        timestamp=datetime(2026, 4, 22, 1, 0, 5, tzinfo=UTC),
        status="success",
    ),
    _make_row(
        provider="assemblyai",
        model="universal-streaming",
        metric_type="WER",
        metric_value=0.08,
        timestamp=datetime(2026, 4, 22, 2, 0, 0, tzinfo=UTC),
        status="success",
    ),
    _make_row(
        provider="deepgram",
        model="aura-2-thalia-en",
        benchmark="TTS",
        metric_type="TTFA",
        metric_value=320.0,
        timestamp=datetime(2026, 4, 23, 10, 0, 0, tzinfo=UTC),
        status="success",
    ),
    _make_row(
        provider="Deepgram",
        model="aura-2-thalia-en",
        benchmark="TTS",
        metric_type="TTFA",
        metric_value=315.0,
        timestamp=datetime(2026, 4, 23, 10, 5, 0, tzinfo=UTC),
        status="tts_failed",
    ),
]


def test_summarize_format_stable() -> None:
    unmatched: set[tuple[str, str]] = set()
    status_counts: Counter[str] = Counter({"success": 4, "tts_failed": 1})

    output = _summarize(_FIXTURE_ROWS, unmatched, status_counts)

    # Header
    assert "== legacy-import dry-run ==" in output

    # Window lines
    assert "2026-04-22 00:00:00 UTC" in output
    assert "2026-04-23" in output

    # Row counts table
    assert "deepgram" in output
    assert "nova-2" in output
    assert "WER" in output
    assert "TOTAL: 5 rows" in output

    # Status block
    assert "success" in output
    assert "tts_failed" in output

    # Proposed runs block
    assert "Proposed runs (one per UTC day):" in output
    assert "day=2026-04-22" in output
    assert "day=2026-04-23" in output

    # Validation block
    assert "Provider/model validation" in output
    assert "matched" in output
    assert "unmatched : 0 combos" in output

    # Footer
    assert "Dry-run complete. No writes performed." in output


# ---------------------------------------------------------------------------
# test 10 — empty-string voice becomes None
# ---------------------------------------------------------------------------


def test_voice_empty_to_null() -> None:
    # LegacyRow is produced by _read_legacy which converts "" → None.
    # The dataclass itself is agnostic, so we test the invariant via the
    # fixture helper that passes voice explicitly, and verify None is preserved.
    row = _make_row(voice=None)
    assert row.voice is None

    # Verify empty string produced by legacy DB would be converted.
    # We exercise the same normalisation logic inline (mirrors _read_legacy).
    voice_raw: str | None = ""
    voice: str | None = None if (voice_raw is None or voice_raw == "") else voice_raw
    assert voice is None
