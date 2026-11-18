# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic tool definitions for contract, worker, and mock API tests."""

import json

SYNTHETIC_SUITE = "synthetic"
TOOL_DEFINITIONS = [
    {
        "name": "lookup_patient",
        "description": "Look up a fixture record.",
        "parameters": {
            "type": "object",
            "properties": {"phone": {"type": "string"}},
            "required": ["phone"],
        },
    },
    {
        "name": "check_availability",
        "description": "Check a fixture slot.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "appointment_type": {"type": "string", "enum": ["cleaning", "checkup"]},
            },
            "required": ["date", "appointment_type"],
        },
    },
    {
        "name": "book_appointment",
        "description": "Book a fixture slot.",
        "parameters": {
            "type": "object",
            "properties": {"slot": {"type": "integer"}},
            "required": ["slot"],
        },
    },
    {
        "name": "cancel_appointment",
        "description": "Cancel a fixture booking.",
        "parameters": {
            "type": "object",
            "properties": {"booking_id": {"type": "string"}},
            "required": ["booking_id"],
        },
    },
]
TOOL_DEFINITIONS_JSON = json.dumps(TOOL_DEFINITIONS)
