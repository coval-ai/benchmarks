# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Definition identity shared by saved dashboard statistics and their readers."""

from __future__ import annotations

import hashlib
import json

from coval_bench.registries.metrics import METRIC_VALUE_CONTRACTS

DEFINITION_REVISION = 1


def _canonical(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=str)
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def aggregation_fingerprint() -> str:
    """Identify the registered value and aggregation rules used by stored statistics."""
    contracts = [
        _canonical(contract.model_dump(mode="python"))
        for _, contract in sorted(
            METRIC_VALUE_CONTRACTS.items(), key=lambda entry: (str(entry[0][0]), entry[0][1])
        )
    ]
    payload = json.dumps(
        {"revision": DEFINITION_REVISION, "contracts": contracts},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()
