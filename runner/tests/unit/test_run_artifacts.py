# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from coval_bench.db.models import Benchmark, Result, ResultStatus
from coval_bench.runner.artifacts import write_run_artifact


def test_write_run_artifact_sanitizes_transcripts_and_groups_failures(tmp_path: Path) -> None:
    results = [
        Result(
            run_id=7,
            provider="deepgram",
            model="nova-3",
            benchmark=Benchmark.STT,
            metric_type="WER",
            metric_value=4.2,
            metric_units="percent",
            audio_filename="0001.wav",
            transcript="hello world",
            status=ResultStatus.SUCCESS,
            error=None,
        ),
        Result(
            run_id=7,
            provider="deepgram",
            model="nova-3",
            benchmark=Benchmark.STT,
            metric_type="TTFS",
            metric_value=None,
            metric_units="seconds",
            audio_filename="0001.wav",
            transcript="hello world",
            status=ResultStatus.FAILED,
            error="timeout",
        ),
        Result(
            run_id=7,
            provider="deepgram",
            model="nova-3",
            benchmark=Benchmark.STT,
            metric_type="TTFT",
            metric_value=None,
            metric_units="seconds",
            audio_filename="0002.wav",
            transcript="second transcript",
            status=ResultStatus.FAILED,
            error="timeout",
        ),
    ]

    path = write_run_artifact(
        artifact_dir=tmp_path,
        run_id=7,
        runner_sha="abc",
        dataset_id="stt-v3",
        dataset_sha256="deadbeef",
        benchmark_kind="stt",
        smoke=False,
        scheduled_at=datetime(2026, 7, 16, 10, 0, tzinfo=UTC),
        started_at=datetime(2026, 7, 16, 10, 1, tzinfo=UTC),
        finished_at=datetime(2026, 7, 16, 10, 2, tzinfo=UTC),
        status="partial",
        results=results,
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["record_type"] for r in records] == [
        "run",
        "result",
        "result",
        "result",
        "summary",
    ]
    assert records[0]["dataset_id"] == "stt-v3"
    assert "hello world" not in path.read_text()
    assert records[1]["transcript_sha256"]
    assert records[1]["transcript_chars"] == 11
    assert records[-1]["status"] == "partial"
    assert records[-1]["success_count"] == 1
    assert records[-1]["fail_count"] == 2
    assert records[-1]["failure_buckets"] == [
        {
            "provider": "deepgram",
            "model": "nova-3",
            "metric_type": "TTFS",
            "error": "timeout",
            "count": 1,
        },
        {
            "provider": "deepgram",
            "model": "nova-3",
            "metric_type": "TTFT",
            "error": "timeout",
            "count": 1,
        },
    ]
