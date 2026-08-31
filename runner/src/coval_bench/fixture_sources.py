# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Where the seeded world comes from when it is not in the checkout.

``contracts`` deliberately knows nothing about storage; it exposes a seam and
this module fills it. Keeping the two apart is what lets the fixtures move to a
different store later without touching the contract or the published hash.

The object path mirrors the layout on disk, so it is a mechanical function of
the suite name and nothing has to be remembered:

    disk  contracts/<suite>/_private/mock-tools.json
    gcs   gs://<bucket>/contracts/<suite>/mock-tools.json
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import structlog

from coval_bench import gcs
from coval_bench.contracts import FixtureProvider, register_fixture_provider

if TYPE_CHECKING:
    from google.cloud import storage

    from coval_bench.config import Settings

logger = structlog.get_logger("coval_bench.fixtures")

FIXTURE_OBJECT = "contracts/{suite}/mock-tools.json"


def gcs_fixture_provider(
    bucket: str, *, storage_client: storage.Client | None = None
) -> FixtureProvider:
    """A provider reading each suite's fixtures from *bucket*, fetched once.

    Memoised on purpose, and not for the round trip — on the happy path there is
    exactly one fetch, at startup. The seeded world must not change under a
    running benchmark: the contract hash is computed over these bytes and
    published beside every result, so an object edited mid-run would leave some
    results citing a hash that no longer describes what they ran against. A
    fixture update takes effect on the next start, which is also when the hash is
    next computed.
    """

    @functools.cache
    def provider(suite: str) -> bytes:
        key = FIXTURE_OBJECT.format(suite=suite)
        data = gcs.read_bytes(bucket, key, storage_client=storage_client)
        if data is None:
            # Not an error: the chain moves on to the next provider.
            raise FileNotFoundError(f"gs://{bucket}/{key}")
        logger.info("mock_fixtures_loaded", suite=suite, bucket=bucket, bytes=len(data))
        return data

    return provider


def install_fixture_providers(settings: Settings) -> None:
    """Register whatever sources this deployment is configured for.

    Called once per process, by each entry point that reads fixtures: the API
    serves them, and the runner hashes them.
    """
    if settings.mock_fixtures_bucket:
        register_fixture_provider(gcs_fixture_provider(settings.mock_fixtures_bucket))
