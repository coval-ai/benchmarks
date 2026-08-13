# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Classification of provider failures, and the missing-credential check."""

from __future__ import annotations

import pytest

from coval_bench.arena.provider_health import (
    KeyFailure,
    classify_failure,
    unconfigured_providers,
)
from coval_bench.config import Settings


class TestClassify:
    def test_payment_required_is_credit(self) -> None:
        assert classify_failure(402, "HTTP 402: balance exhausted") is KeyFailure.CREDIT

    @pytest.mark.parametrize("status", [401, 403])
    def test_unauthorized_and_forbidden_are_auth(self, status: int) -> None:
        assert classify_failure(status, "nope") is KeyFailure.AUTH

    def test_bare_429_is_a_rate_limit(self) -> None:
        assert classify_failure(429, "HTTP 429: slow down") is KeyFailure.RATE_LIMIT

    def test_429_naming_quota_is_credit(self) -> None:
        # Several providers report exhaustion with a rate-limit status; reading that as
        # transient would keep a dead key in rotation.
        assert classify_failure(429, "HTTP 429: quota exceeded for this month") is KeyFailure.CREDIT

    def test_server_error_implicates_nothing(self) -> None:
        assert classify_failure(500, "internal error") is None

    def test_the_live_minimax_wording_is_credit(self) -> None:
        # Verbatim from prod: minimax reports errors in websocket frames, so there is no
        # status code and only the wording identifies it.
        assert (
            classify_failure(None, "task_failed: [1008] insufficient balance") is KeyFailure.CREDIT
        )

    def test_prose_is_read_only_when_no_status_is_reported(self) -> None:
        assert classify_failure(None, "Invalid API key provided") is KeyFailure.AUTH
        assert classify_failure(None, "connection reset by peer") is None

    def test_no_error_at_all_is_not_a_key_failure(self) -> None:
        assert classify_failure(None, None) is None


class TestUnconfiguredProviders:
    """A provider with no key mounted here fails every call, so it leaves the roster
    rather than being swapped on every request. Answered from PROVIDER_ENV, not from the
    wording of 26 different constructor errors."""

    def test_a_provider_authenticating_another_way_is_never_excluded(self) -> None:
        # No PROVIDER_ENV entry means there is no key to be missing (e.g. ADC).
        assert unconfigured_providers(Settings(), ["not-a-registered-provider"]) == frozenset()

    def test_an_unconfigured_provider_is_excluded(self) -> None:
        settings = Settings(minimax_api_key=None, elevenlabs_api_key="set")
        assert unconfigured_providers(settings, ["minimax", "elevenlabs"]) == frozenset({"minimax"})

    def test_a_configured_roster_excludes_nobody(self) -> None:
        settings = Settings(minimax_api_key="set", elevenlabs_api_key="set")
        assert unconfigured_providers(settings, ["minimax", "elevenlabs"]) == frozenset()
