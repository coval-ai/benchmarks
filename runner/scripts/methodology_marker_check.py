# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence

METHODOLOGY_MARKER_FILE = "web/lib/config/methodologyChanges.ts"
METHODOLOGY_SENSITIVE_PREFIXES = ("runner/src/coval_bench/datasets/",)
METHODOLOGY_SENSITIVE_FILES = {
    "docs/methodology.md",
    "runner/src/coval_bench/metrics/ttfa.py",
    "runner/src/coval_bench/metrics/ttfs.py",
    "runner/src/coval_bench/metrics/ttft.py",
    "runner/src/coval_bench/metrics/wer.py",
}


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def is_methodology_sensitive(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized in METHODOLOGY_SENSITIVE_FILES or normalized.startswith(
        METHODOLOGY_SENSITIVE_PREFIXES
    )


def missing_methodology_marker(changed_files: Iterable[str]) -> list[str]:
    files = [normalized for path in changed_files if (normalized := normalize_path(path))]
    sensitive_files = [path for path in files if is_methodology_sensitive(path)]

    if not sensitive_files or METHODOLOGY_MARKER_FILE in files:
        return []

    return sensitive_files


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


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warn when methodology-sensitive changes lack a marker."
    )
    parser.add_argument("--base", help="Base git ref to diff against, e.g. origin/main")
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="Explicit changed file path. Repeatable; bypasses git diff when present.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    changed_files = args.changed_file or list_changed_files(args.base or "origin/main")
    missing_marker_files = missing_methodology_marker(changed_files)

    if missing_marker_files:
        print(
            "::warning::Methodology-sensitive files changed but "
            f"{METHODOLOGY_MARKER_FILE} was not updated. If this PR changes how a "
            "metric is computed or the eval dataset, add a methodology marker so "
            "the timeline charts explain the step."
        )
        print("Changed methodology-sensitive files:")
        for path in missing_marker_files:
            print(path)
    else:
        print("No methodology marker warning needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
