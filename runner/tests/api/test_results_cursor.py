# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for authenticated v2 results cursors."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from coval_bench.api.results_cursor import (
    MAX_CURSOR_LENGTH,
    CursorError,
    CursorKeyError,
    decode,
    encode,
)

KEY = SecretStr("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
OTHER_KEY = SecretStr("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=")
PAYLOAD = {
    "v": 1,
    "fingerprint": "a" * 64,
    "since": None,
    "until": datetime(2026, 9, 22, tzinfo=UTC).isoformat(),
    "anchor_time": datetime(2026, 9, 22, tzinfo=UTC).isoformat(),
    "anchor_id": "00000000-0000-4000-8000-000000000001",
}


def test_roundtrip_is_opaque_and_cross_instance() -> None:
    token = encode(PAYLOAD, KEY)
    assert token.startswith("v1.")
    assert decode(token, SecretStr(KEY.get_secret_value())) == PAYLOAD


@pytest.mark.parametrize(
    "token,key",
    [
        ("legacy-plaintext", KEY),
        (encode(PAYLOAD, KEY), OTHER_KEY),
        (encode(PAYLOAD, KEY)[:-1] + "x", KEY),
        ("v1.not-a-fernet-token", KEY),
    ],
)
def test_wrong_tampered_legacy_and_malformed_tokens_are_rejected(
    token: str, key: SecretStr
) -> None:
    with pytest.raises(CursorError):
        decode(token, key)


def test_length_and_missing_key_fail_closed() -> None:
    with pytest.raises(CursorError):
        decode("v1." + "x" * MAX_CURSOR_LENGTH, KEY)
    with pytest.raises(CursorKeyError):
        encode(PAYLOAD, None)
    with pytest.raises(CursorKeyError):
        decode("v1." + "x", None)


def test_payload_is_authenticated_not_plain_json() -> None:
    token = encode(PAYLOAD, KEY)
    encoded = token.removeprefix("v1.")
    with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
        json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
