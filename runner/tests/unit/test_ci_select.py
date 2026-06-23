# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import ModuleType


def load_ci_select() -> ModuleType:
    script_path = Path(__file__).parents[2] / "scripts" / "ci_select.py"
    spec = importlib.util.spec_from_file_location("ci_select", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci_select = load_ci_select()
classify_changed_files = ci_select.classify_changed_files
main = ci_select.main


def test_runner_changes_select_runner_only() -> None:
    assert classify_changed_files(["runner/src/coval_bench/metrics/wer.py"]) == {
        "run_runner": True,
        "run_web": False,
        "methodology_marker": False,
        "methodology_sensitive": True,
    }


def test_web_changes_select_web_only() -> None:
    assert classify_changed_files(["web/app/page.tsx"]) == {
        "run_runner": False,
        "run_web": True,
        "methodology_marker": False,
        "methodology_sensitive": False,
    }


def test_ci_guardrail_changes_select_runner_and_web() -> None:
    assert classify_changed_files([".github/workflows/ci.yml"]) == {
        "run_runner": True,
        "run_web": True,
        "methodology_marker": False,
        "methodology_sensitive": False,
    }


def test_empty_diff_fails_open_to_runner_and_web() -> None:
    assert classify_changed_files([]) == {
        "run_runner": True,
        "run_web": True,
        "methodology_marker": False,
        "methodology_sensitive": False,
    }


def test_dataset_changes_are_methodology_sensitive() -> None:
    assert classify_changed_files(["runner/src/coval_bench/datasets/manifest.py"]) == {
        "run_runner": True,
        "run_web": False,
        "methodology_marker": False,
        "methodology_sensitive": True,
    }


def test_methodology_marker_is_tracked_separately() -> None:
    assert classify_changed_files(["web/lib/config/methodologyChanges.ts"]) == {
        "run_runner": False,
        "run_web": True,
        "methodology_marker": True,
        "methodology_sensitive": False,
    }


def test_cli_writes_github_outputs(tmp_path: Path) -> None:
    output_path = tmp_path / "github-output.txt"

    assert (
        main(
            [
                "--changed-file=web/package.json",
                "--github-output",
                str(output_path),
            ]
        )
        == 0
    )
    assert output_path.read_text(encoding="utf-8") == (
        "run_runner=false\nrun_web=true\nmethodology_sensitive=false\nmethodology_marker=false\n"
    )
