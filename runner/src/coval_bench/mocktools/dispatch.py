# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool handlers built from the contract, not written per tool.

``tool-definitions.json`` is what the platforms are configured with, so it is
also what this service serves. Deriving the handlers from it means the two can
never drift: a tool the contract declares is answerable here, a tool it does not
is a 404, and a fixture keyed on an argument the schema has no field for is a
startup error rather than a seed that silently never matches.

The dispatcher is total. An unknown tool, a missing required argument and a
seeded 500 all return a response and are recorded, because each is a thing the
agent did and the suite grades how it recovers. Raising instead would lose the
row and, worse, turn the agent's mistake into our outage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from coval_bench.contracts import read_contract_file, read_private_fixture
from coval_bench.mocktools.fixtures import MockFixtures, parse_fixtures
from coval_bench.mocktools.resolver import Resolution, resolve

TOOL_DEFINITIONS_FILE = "tool-definitions.json"

# JSON Schema types the contract can declare, mapped to what the transport
# actually delivers once the body is parsed.
_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "array": list,
    "object": dict,
}


def _type_matches(value: object, declared: str) -> bool:
    """Whether a supplied value has the shape the contract declared for it."""
    expected = _JSON_TYPES.get(declared)
    if expected is None:
        return True
    # `bool` subclasses `int`, so True would otherwise pass as a number.
    if declared in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, expected)


@dataclass(frozen=True)
class ToolSpec:
    """One tool's callable surface, read off its contract definition."""

    name: str
    properties: frozenset[str]
    required: frozenset[str]
    # Declared JSON types, so a value of the wrong shape is a rejected call rather
    # than one silently coerced into matching a seed.
    types: dict[str, str] = field(default_factory=dict)
    # Declared value sets, kept for keyterm extraction: an enum in the contract is
    # domain vocabulary the STT has to hear correctly.
    enums: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    """What the service returns for one call, and how it got there."""

    response: dict[str, Any]
    http_status: int
    resolution: Resolution | None = None


def parse_tool_specs(raw: str | bytes) -> dict[str, ToolSpec]:
    """Read tool definitions into specs, keyed by tool name."""
    specs: dict[str, ToolSpec] = {}
    for entry in json.loads(raw):
        parameters = entry.get("parameters") or {}
        properties = parameters.get("properties") or {}
        enums = {
            name: tuple(schema["enum"])
            for name, schema in properties.items()
            if isinstance(schema, dict) and schema.get("enum")
        }
        specs[entry["name"]] = ToolSpec(
            name=entry["name"],
            properties=frozenset(properties),
            required=frozenset(parameters.get("required") or ()),
            types={
                name: schema["type"]
                for name, schema in properties.items()
                if isinstance(schema, dict) and isinstance(schema.get("type"), str)
            },
            enums=enums,
        )
    return specs


def load_tool_specs(suite: str) -> dict[str, ToolSpec]:
    """Read the suite's committed tool definitions."""
    return parse_tool_specs(read_contract_file(suite, TOOL_DEFINITIONS_FILE))


class Dispatcher:
    """Answers tool calls for one suite from one set of fixtures."""

    def __init__(self, specs: dict[str, ToolSpec], fixtures: MockFixtures) -> None:
        """Bind fixtures to specs, refusing any disagreement between them."""
        _assert_fixtures_cover_contract(specs, fixtures)
        self._specs = specs
        self._fixtures = fixtures

    @property
    def tools(self) -> frozenset[str]:
        """The tool names this dispatcher answers."""
        return frozenset(self._specs)

    def call(self, tool: str, args: dict[str, Any]) -> Outcome:
        """Answer one tool call. Never raises; every outcome is a recordable response."""
        spec = self._specs.get(tool)
        if spec is None:
            return Outcome(
                response={"error": "unknown_tool", "tool": tool},
                http_status=404,
            )
        missing = sorted(spec.required - args.keys())
        if missing:
            return Outcome(
                response={"error": "missing_required_arguments", "missing": missing},
                http_status=422,
            )
        mistyped = sorted(
            key
            for key, value in args.items()
            if key in spec.types and not _type_matches(value, spec.types[key])
        )
        if mistyped:
            # `2065550180` is not `"2065550180"`. Stringifying it would find the
            # seeded patient and hide a tool call the agent got wrong.
            return Outcome(
                response={
                    "error": "invalid_argument_types",
                    "invalid": [{"name": key, "expected": spec.types[key]} for key in mistyped],
                },
                http_status=422,
            )
        resolution = resolve(
            self._fixtures.tools[tool], args, categorical_keys=frozenset(spec.enums)
        )
        return Outcome(
            response=resolution.seed.response,
            http_status=resolution.seed.http_status,
            resolution=resolution,
        )


def _assert_fixtures_cover_contract(specs: dict[str, ToolSpec], fixtures: MockFixtures) -> None:
    """Fail at startup on any drift between the contract and the seeded world."""
    missing = sorted(specs.keys() - fixtures.tools.keys())
    if missing:
        raise ValueError(f"tools declared in the contract but unseeded: {', '.join(missing)}")
    extra = sorted(fixtures.tools.keys() - specs.keys())
    if extra:
        raise ValueError(f"tools seeded but absent from the contract: {', '.join(extra)}")
    for name, fixture in fixtures.tools.items():
        allowed = specs[name].properties
        for seed in (*fixture.seeds, fixture.fallback):
            unknown = sorted(seed.match.keys() - allowed)
            if unknown:
                raise ValueError(
                    f"seed {name}/{seed.id} matches on arguments the contract "
                    f"does not declare: {', '.join(unknown)}"
                )


def build_dispatcher(suite: str) -> Dispatcher:
    """Build a suite's dispatcher from its committed contract and private fixtures.

    Raises ``FileNotFoundError`` where the fixtures are not installed, and
    ``ValueError`` where they disagree with the contract.
    """
    return Dispatcher(load_tool_specs(suite), parse_fixtures(read_private_fixture(suite)))
