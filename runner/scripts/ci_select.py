# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

RUNNER_PREFIXES = ("runner/",)
WEB_PREFIXES = ("web/",)

CI_GUARDRAIL_FILES = {
    ".github/workflows/ci.yml",
    "runner/scripts/ci_select.py",
}

RUNNER_GUARDRAIL_FILES = CI_GUARDRAIL_FILES | {
    "runner/pyproject.toml",
    "runner/uv.lock",
}

WEB_GUARDRAIL_FILES = CI_GUARDRAIL_FILES | {
    "web/package.json",
    "web/pnpm-lock.yaml",
    "web/pnpm-workspace.yaml",
    "web/next.config.ts",
    "web/eslint.config.mjs",
    "web/tsconfig.json",
}

METHODOLOGY_SENSITIVE_PREFIXES = ("runner/src/coval_bench/datasets/",)
METHODOLOGY_SENSITIVE_FILES = {
    "docs/methodology.md",
    "runner/src/coval_bench/metrics/ttfa.py",
    "runner/src/coval_bench/metrics/ttfs.py",
    "runner/src/coval_bench/metrics/ttft.py",
    "runner/src/coval_bench/metrics/wer.py",
}
METHODOLOGY_MARKER_FILE = "web/lib/config/methodologyChanges.ts"


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def is_runner_relevant(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized.startswith(RUNNER_PREFIXES) or normalized in RUNNER_GUARDRAIL_FILES


def is_web_relevant(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized.startswith(WEB_PREFIXES) or normalized in WEB_GUARDRAIL_FILES


def is_methodology_sensitive(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized in METHODOLOGY_SENSITIVE_FILES or normalized.startswith(
        METHODOLOGY_SENSITIVE_PREFIXES
    )


def classify_changed_files(changed_files: Iterable[str]) -> dict[str, bool]:
    files = [normalized for path in changed_files if (normalized := normalize_path(path))]

    # Fail open when the diff is unavailable or empty. A selector problem should
    # cost CI time, not silently skip the only relevant guardrail.
    if not files:
        return {
            "run_runner": True,
            "run_web": True,
            "methodology_sensitive": False,
            "methodology_marker": False,
        }

    return {
        "run_runner": any(is_runner_relevant(path) for path in files),
        "run_web": any(is_web_relevant(path) for path in files),
        "methodology_sensitive": any(is_methodology_sensitive(path) for path in files),
        "methodology_marker": METHODOLOGY_MARKER_FILE in files,
    }


def list_changed_files(base: str) -> list[str]:
    git = shutil.which("git") or "git"

    try:
        result = subprocess.run(  # noqa: S603,S607
            [git, "diff", "--name-only", f"{base}...HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    return [line for line in result.stdout.splitlines() if line]


def write_github_outputs(outputs: dict[str, bool], output_path: str | None) -> None:
    if not output_path:
        return

    lines = [f"{key}={str(value).lower()}" for key, value in outputs.items()]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select benchmarks CI guardrails from changed files."
    )
    parser.add_argument("--base", help="Base git ref to diff against, e.g. origin/main")
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="Explicit changed file path. Repeatable; bypasses git diff when present.",
    )
    parser.add_argument("--github-output", help="Path to a GitHub output file.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    changed_files = args.changed_file or list_changed_files(args.base or "origin/main")
    outputs = classify_changed_files(changed_files)

    for key, value in outputs.items():
        print(f"{key}={str(value).lower()}")

    write_github_outputs(outputs, args.github_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
