# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The contract must not change by accident, and a dropped pin must fail loudly."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import coval_bench.contracts as contracts_module
from coval_bench.contracts import (
    LlmPin,
    Stack,
    contract_sha256,
    has_private_contract,
    load_stack,
    public_contract_sha256,
    read_contract_file,
    read_private_fixture,
    register_fixture_provider,
    stack_as_dict,
)
from coval_bench.variants.platforms import redact_identifiers

# Bump deliberately, in the same commit that changes a committed contract file or
# a pin. A failure here means someone edited the contract without versioning it,
# which would silently repoint every published number at a different agent.
#
# This pins the *public* hash. The full hash also covers `_private/`, which is
# never committed, so its value differs between a machine holding the fixtures
# and CI, which never does — pinning that would be red for exactly the people
# doing the work and vacuous everywhere else.
EXPECTED_PUBLIC_CONTRACT_SHA = "146ac237d7ef007c"


def test_stack_loads_and_pins_are_what_the_design_doc_says() -> None:
    stack = load_stack()
    assert (stack.llm.provider, stack.llm.model, stack.llm.temperature) == ("openai", "gpt-4.1", 0)
    assert (stack.stt.provider, stack.stt.model, stack.stt.fallback_model) == (
        "deepgram",
        "nova-3",
        "general",
    )
    assert stack.tts.model == "eleven_flash_v2"
    assert stack.tts.voice_id == "EXAVITQu4vr4xnSDxMaL"
    assert (stack.media.codec, stack.media.sample_rate_hz) == ("PCMU", 8000)
    assert stack.turn_taking.end_of_turn_target_ms == 800
    # Both must be off or the benchmark measures the vendor's rubric and timer.
    assert stack.platform_behaviour.native_auto_hangup is False
    assert stack.platform_behaviour.vendor_post_call_analysis is False


def test_public_contract_hash_is_pinned() -> None:
    assert public_contract_sha256("dental").startswith(EXPECTED_PUBLIC_CONTRACT_SHA), (
        "The dental contract or the pinned stack changed. If that was deliberate, "
        "update EXPECTED_PUBLIC_CONTRACT_SHA in the same commit."
    )


def test_the_published_hash_also_covers_the_private_fixtures() -> None:
    """What ran is identified by the fixtures too, not just the committed files.

    Skips in CI and a fresh checkout, where `_private/` does not exist.
    """
    if not has_private_contract("dental"):
        pytest.skip("private fixtures are not installed")
    assert contract_sha256("dental") != public_contract_sha256("dental")


def test_rationale_keys_are_allowed_and_survive_load() -> None:
    stack = load_stack()
    assert (stack.llm.__pydantic_extra__ or {}).get("_why")


def test_mistyped_key_is_rejected_rather_than_dropping_the_pin() -> None:
    # The dangerous case: `model` goes missing and the platform keeps its default.
    with pytest.raises(ValidationError):
        LlmPin.model_validate({"provider": "openai", "modle": "gpt-4.1", "temperature": 0})


def test_unknown_non_underscore_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must start with '_'"):
        LlmPin.model_validate(
            {"provider": "openai", "model": "gpt-4.1", "temperature": 0, "notes": "x"}
        )


def test_codec_is_constrained_to_what_coval_accepts() -> None:
    with pytest.raises(ValidationError):
        Stack.model_validate(
            {
                **load_stack().model_dump(mode="json"),
                "media": {"codec": "OPUS", "sample_rate_hz": 48000},
            }
        )


def test_vendor_payload_excludes_rationale() -> None:
    payload = stack_as_dict(load_stack())
    assert all(not key.startswith("_") for section in payload.values() for key in section)
    assert payload["llm"] == {"provider": "openai", "model": "gpt-4.1", "temperature": 0}


def test_identifiers_are_withheld_from_the_committed_dumps() -> None:
    """This repo is public. A live dial target must never reach it."""
    agent = json.loads(read_contract_file("dental", "_source/coval-agent.json"))
    assert agent["phone_number"] == "[REDACTED]"
    assert agent["id"] == "[REDACTED]"
    assert agent["customer_agent_id"] == "[REDACTED]"
    # Methodology stays: a reader needs to know which voice was used.
    assert agent["metadata"]["voice_id"]
    assert agent["display_name"]


def test_redact_identifiers_leaves_methodology_alone() -> None:
    found: list[str] = []
    out = redact_identifiers(
        {"id": "abc", "display_name": "keep", "nested": [{"phone_number": "sip:x@y"}]}, found
    )
    assert out == {
        "id": "[REDACTED]",
        "display_name": "keep",
        "nested": [{"phone_number": "[REDACTED]"}],
    }
    assert found == ["id", "nested[0].phone_number"]


# --- the fixture fallback chain ---------------------------------------------
#
# `_private/` is gitignored and the image is built from a git checkout, so the
# deployed service can never carry the seeded world. It reads it from elsewhere,
# and that elsewhere registers itself here.

SEEDED = b'{"tools": {}}'


@pytest.fixture
def no_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty chain, so one test cannot leak a provider into the next."""
    monkeypatch.setattr(contracts_module, "_FIXTURE_PROVIDERS", [])


@pytest.fixture
def checkout_without_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkout where only `_private/` is missing, as in CI and in the image."""
    real = contracts_module._read_bytes

    def missing_private(*parts: str) -> bytes:
        if "_private" in parts:
            raise FileNotFoundError("/".join(parts))
        return real(*parts)

    monkeypatch.setattr(contracts_module, "_read_bytes", missing_private)


def test_a_provider_answers_when_the_checkout_has_none(
    no_providers: None, checkout_without_fixtures: None
) -> None:
    register_fixture_provider(lambda suite: SEEDED)
    assert read_private_fixture("dental") == SEEDED


def test_the_local_checkout_wins_over_a_provider(no_providers: None) -> None:
    """Editing the file is what a developer expects to take effect."""
    if not has_private_contract("dental"):
        pytest.skip("local fixtures are not installed")
    register_fixture_provider(lambda suite: SEEDED)
    assert read_private_fixture("dental") != SEEDED


def test_a_provider_without_the_suite_falls_through(
    no_providers: None, checkout_without_fixtures: None
) -> None:
    def absent(suite: str) -> bytes:
        raise FileNotFoundError(suite)

    register_fixture_provider(absent)
    register_fixture_provider(lambda suite: SEEDED)
    assert read_private_fixture("dental") == SEEDED


def test_an_empty_chain_still_raises(no_providers: None, checkout_without_fixtures: None) -> None:
    with pytest.raises(FileNotFoundError):
        read_private_fixture("dental")


def test_the_published_hash_is_the_same_whichever_source_supplied_the_bytes(
    no_providers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason the fallback lives in the reader and not in the mock service.

    `contract_sha256` reads through the same function. A fallback wired only into
    dispatch would leave production serving the right fixtures while publishing a
    hash computed as though it had none — a different "what ran" number on every
    result, and nothing to notice it.
    """
    if not has_private_contract("dental"):
        pytest.skip("local fixtures are not installed")
    from_disk = contract_sha256("dental")
    seeded = read_private_fixture("dental")

    real = contracts_module._read_bytes

    def missing_private(*parts: str) -> bytes:
        if "_private" in parts:
            raise FileNotFoundError("/".join(parts))
        return real(*parts)

    monkeypatch.setattr(contracts_module, "_read_bytes", missing_private)
    register_fixture_provider(lambda suite: seeded)
    assert contract_sha256("dental") == from_disk
    assert contract_sha256("dental") != public_contract_sha256("dental")


def test_redact_catches_the_mock_tools_header_but_not_the_correlation_one() -> None:
    from coval_bench.variants.platforms import redact

    found: list[str] = []
    out = redact(
        {
            "server": {
                "headers": {"X-Mock-Tools-Key": "s3cr3t", "X-Coval-Simulation-Id": "{{x}}"},
            },
            "credentialIds": ["cred_1"],
        },
        found,
    )
    assert out["server"]["headers"] == {
        "X-Mock-Tools-Key": "[REDACTED]",
        "X-Coval-Simulation-Id": "{{x}}",
    }
    assert out["credentialIds"] == ["cred_1"]
    assert found == ["server.headers.X-Mock-Tools-Key"]
