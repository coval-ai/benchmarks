# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark contracts: the pinned stack and the per-suite task definition.

Everything here ships inside the wheel and is read with ``importlib.resources``,
the same way ``datasets/manifests`` is. That is not incidental: the mock-tool
router runs inside the API container, which is built from ``runner/`` and copies
only ``src/``, so a contract living at the repo root would not exist at request
time.

What is a contract here, and what is not
----------------------------------------
``tool-definitions.json`` and ``stack.json`` are contracts in the ordinary sense:
an interface both sides bind to. The agent is configured against the tool names
and argument schemas; the mock service dispatches from the same file. They are
public, because a reader cannot audit a published result without them.

``system-prompt.txt`` and ``first-message.txt`` are not contracts, they are the
agent's configuration. Public for the same auditing reason.

``<suite>/_private/`` is the evaluator: the scenario scripts, their assertions,
and the mock fixtures derived from them. Never committed. Publishing the answers
would let a platform optimise for the test rather than the task.

Two rules the layout encodes:

* ``stack.json`` and the public suite files are byte-identical across every
  variant. A variant may not restate a pinned value; it references this.
* ``platforms/`` holds the only permitted difference, published for audit.

The ``_``-prefix convention
---------------------------
JSON has no comments, so rationale lives beside the value it explains under
keys starting with ``_`` (``_why``, ``_verify``, ``_limitation``). Models accept
those and reject anything else, so a mistyped *key* still fails loudly:

    {"modle": "gpt-4.1"}   ->  ValidationError: field 'model' missing

That is the failure worth catching. A mistyped *value* is caught later by the
vendor rejecting it at apply time; a mistyped key would silently drop the pin.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "Stack",
    "load_stack",
    "contract_sha256",
    "read_contract_file",
    "has_private_contract",
]

_CONTRACTS_PACKAGE = "coval_bench.contracts"

# Files that make up a suite contract, in a fixed order so the hash is stable.
# Missing entries are skipped, so the hash is meaningful before the contract is
# complete and changes when a file is added.
PUBLIC_CONTRACT_FILES: tuple[str, ...] = (
    "system-prompt.txt",
    "first-message.txt",
    "tool-definitions.json",
)

# Hashed but never committed. Including them means the published hash identifies
# exactly which fixtures a run used without revealing what they contain; a
# checkout without them produces a different hash, which is why the pinned test
# skips rather than fails when the directory is absent.
PRIVATE_CONTRACT_FILES: tuple[str, ...] = ("_private/mock-tools.json",)


class _Annotated(BaseModel):
    """Base for contract models: required fields, plus ``_``-prefixed notes."""

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_validator(mode="after")
    def _only_underscore_extras(self) -> _Annotated:
        extras = self.__pydantic_extra__ or {}
        unknown = sorted(k for k in extras if not k.startswith("_"))
        if unknown:
            raise ValueError(
                f"unknown key(s) {unknown} in {type(self).__name__}; "
                "rationale keys must start with '_'"
            )
        return self


class LlmPin(_Annotated):
    provider: str
    model: str
    temperature: float


class SttPin(_Annotated):
    provider: str
    model: str
    fallback_model: str
    keyterms_source: str


class TtsPin(_Annotated):
    provider: str
    model: str
    voice_id: str
    voice_name: str


class TurnTakingPin(_Annotated):
    end_of_turn_target_ms: int


class MediaPin(_Annotated):
    codec: Literal["PCMU", "L16"]
    sample_rate_hz: int


class PlatformBehaviourPin(_Annotated):
    native_auto_hangup: bool
    vendor_post_call_analysis: bool


class Stack(_Annotated):
    """The pinned component layer. One stack for every variant.

    Transport is deliberately absent: it is declared per variant on the
    registry row, not pinned here.
    """

    llm: LlmPin
    stt: SttPin
    tts: TtsPin
    turn_taking: TurnTakingPin
    media: MediaPin
    platform_behaviour: PlatformBehaviourPin


def _read_bytes(*parts: str) -> bytes:
    ref = importlib.resources.files(_CONTRACTS_PACKAGE)
    for part in parts:
        ref = ref.joinpath(part)
    return ref.read_bytes()


def load_stack() -> Stack:
    """Parse and validate ``contracts/stack.json``."""
    return Stack.model_validate(json.loads(_read_bytes("stack.json")))


def read_contract_file(suite: str, filename: str) -> str:
    """Read one file from a suite contract as text."""
    return _read_bytes(suite, filename).decode()


def contract_sha256(suite: str) -> str:
    """One SHA-256 over the pinned stack, the public suite files, and the fixtures.

    Mirrors ``_dataset_sha256`` in the S2S fetcher: a single number that
    identifies exactly what was run, published beside every result. Covers
    ``stack.json`` too, because changing a pin changes what the agent is just
    as surely as changing its prompt.
    """
    digest = hashlib.sha256()
    digest.update(_read_bytes("stack.json"))
    for filename in (*PUBLIC_CONTRACT_FILES, *PRIVATE_CONTRACT_FILES):
        try:
            digest.update(_read_bytes(suite, *filename.split("/")))
        except (FileNotFoundError, NotADirectoryError):
            continue
    return digest.hexdigest()


def has_private_contract(suite: str) -> bool:
    """Whether the evaluator is present locally.

    False in a fresh checkout and in CI, since ``_private/`` is never committed.
    Callers that need the fixtures should skip rather than fail.
    """
    try:
        _read_bytes(suite, "_private", "mock-tools.json")
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def stack_as_dict(stack: Stack) -> dict[str, Any]:
    """Serialise without the ``_``-prefixed rationale keys.

    Use this when handing pins to a vendor API. The rationale is for readers of
    the repo, not for request bodies.
    """
    raw = stack.model_dump(mode="json")

    # A JSON tree is genuinely arbitrary; Any is the honest annotation here.
    def strip(node: Any) -> Any:  # noqa: ANN401
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if not k.startswith("_")}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return cast("dict[str, Any]", strip(raw))
