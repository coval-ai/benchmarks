# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail closed when real benchmark evidence is pointed into the public repository."""

from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_INVENTED_FIXTURE_PATH = (
    _REPOSITORY_ROOT
    / "runner"
    / "tests"
    / "fixtures"
    / "preprocessing"
    / "benchmarking"
    / "invented-word-ground-truth.json"
)


def validate_private_evidence_path(
    path: Path,
    *,
    allow_invented_fixture: bool = False,
) -> Path:
    """Reject any benchmarks source checkout, not just the checkout importing this code."""
    resolved = path.expanduser().resolve()
    fixture_read = allow_invented_fixture and resolved == _INVENTED_FIXTURE_PATH
    source_checkout = next(
        (
            ancestor
            for ancestor in (resolved, *resolved.parents)
            if (ancestor / ".git").exists()
            or (
                (ancestor / "runner" / "src" / "coval_bench").is_dir()
                and (ancestor / "runner" / "pyproject.toml").is_file()
            )
        ),
        None,
    )
    if source_checkout is not None and not fixture_read:
        raise ValueError(
            "real benchmark inputs and detailed reports must stay outside the repository"
        )
    return resolved
