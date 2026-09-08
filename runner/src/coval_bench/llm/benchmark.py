# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Coval run settings shared by every LLM benchmark agent."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

DEFAULT_PERSONA_ID = "PN3xgmsqeLDjsNNEA2e55e"
ITERATION_COUNT = 1
TEMPLATE_MANAGED = ("agent_ids", "persona_ids", "test_set_ids", "metric_ids", "iteration_count")
_TEMPLATE_PATCH_KEYS = {
    "agent_ids": "agent_id",
    "persona_ids": "persona_id",
    "test_set_ids": "test_set_id",
}


def run_template_body(
    display_name: str, agent_id: str, test_set_id: str, metric_id: str
) -> dict[str, Any]:
    return {
        "display_name": display_name,
        "agent_ids": [agent_id],
        "persona_ids": [DEFAULT_PERSONA_ID],
        "test_set_ids": [test_set_id],
        "metric_ids": [metric_id],
        "iteration_count": ITERATION_COUNT,
    }


def template_patch_body(wanted: dict[str, Any], paths: Iterable[str]) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for path in paths:
        singular = _TEMPLATE_PATCH_KEYS.get(path)
        if singular is None:
            body[path] = wanted[path]
            continue
        (only,) = wanted[path]
        body[singular] = only
    return body
