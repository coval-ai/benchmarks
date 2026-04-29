# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the coval-bench test suite."""

from __future__ import annotations

import pytest

from coval_bench.config import Settings


@pytest.fixture(autouse=False)
def override_settings(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Settings:
    """Return a Settings instance with safe test defaults.

    Patches ``get_settings`` so that any code calling it during a test
    receives the test settings rather than reading real env vars.
    """
    test_settings = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",  # type: ignore[arg-type]
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        log_level="DEBUG",
    )
    monkeypatch.setattr("coval_bench.config.get_settings", lambda: test_settings)
    return test_settings
