# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
from pytest import CaptureFixture, MonkeyPatch


def load_methodology_marker_check() -> ModuleType:
    script_path = Path(__file__).parents[2] / "scripts" / "methodology_marker_check.py"
    spec = importlib.util.spec_from_file_location("methodology_marker_check", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


methodology_marker_check = load_methodology_marker_check()
missing_methodology_marker = methodology_marker_check.missing_methodology_marker
main = methodology_marker_check.main


def test_metric_change_requires_methodology_marker() -> None:
    assert missing_methodology_marker(["runner/src/coval_bench/metrics/wer.py"]) == [
        "runner/src/coval_bench/metrics/wer.py"
    ]


def test_dataset_change_requires_methodology_marker() -> None:
    assert missing_methodology_marker(["runner/src/coval_bench/datasets/manifest.py"]) == [
        "runner/src/coval_bench/datasets/manifest.py"
    ]


def test_docs_methodology_change_requires_marker() -> None:
    assert missing_methodology_marker(["docs/methodology.md"]) == ["docs/methodology.md"]


def test_marker_satisfies_sensitive_changes() -> None:
    assert (
        missing_methodology_marker(
            [
                "runner/src/coval_bench/metrics/ttft.py",
                "web/lib/config/methodologyChanges.ts",
            ]
        )
        == []
    )


def test_provider_orchestrator_changes_do_not_warn() -> None:
    assert missing_methodology_marker(["runner/src/coval_bench/runner/orchestrator.py"]) == []


def test_git_diff_failure_surfaces_instead_of_passing_clean(monkeypatch: MonkeyPatch) -> None:
    def raise_git_error(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["git", "diff"])

    monkeypatch.setattr(methodology_marker_check.subprocess, "run", raise_git_error)

    with pytest.raises(RuntimeError, match="Unable to list changed files against origin/main"):
        methodology_marker_check.list_changed_files("origin/main")


def test_cli_prints_warning_for_missing_marker(capsys: CaptureFixture[str]) -> None:
    assert main(["--changed-file", "runner/src/coval_bench/metrics/ttfa.py"]) == 0

    output = capsys.readouterr().out
    assert "::warning::Methodology-sensitive files changed" in output
    assert "runner/src/coval_bench/metrics/ttfa.py" in output


def test_cli_prints_noop_message_when_marker_is_present(capsys: CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "--changed-file",
                "runner/src/coval_bench/metrics/ttfa.py",
                "--changed-file",
                "web/lib/config/methodologyChanges.ts",
            ]
        )
        == 0
    )

    assert capsys.readouterr().out == "No methodology marker warning needed.\n"
