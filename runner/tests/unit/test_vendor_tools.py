# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The tool definitions the platforms are configured with, built from a Corral tools.json."""

from __future__ import annotations

import json

from coval_bench.mocktools.dispatch import vendor_tool_definitions

TOOLS = json.dumps(
    {
        "tools": [
            {
                "name": "lookup",
                "description": "Look up the detail the caller gave.",
                "parameters": {"value": "string"},
                "character_error_rate": {"value": {"card-1": "4111111111111111"}},
            }
        ],
        "responses": {"lookup": {"received": True}},
    }
)


def test_each_tool_becomes_a_json_schema_function_with_every_argument_required() -> None:
    assert vendor_tool_definitions(TOOLS) == [
        {
            "name": "lookup",
            "description": "Look up the detail the caller gave.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        }
    ]


def test_the_right_values_never_reach_a_platform() -> None:
    rendered = json.dumps(vendor_tool_definitions(TOOLS))
    assert "character_error_rate" not in rendered
    assert "4111111111111111" not in rendered
