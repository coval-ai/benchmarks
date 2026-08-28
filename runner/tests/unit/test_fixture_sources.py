# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The GCS source that fills the fixture seam on a deployment."""

from __future__ import annotations

from typing import Any

import pytest
from google.api_core.exceptions import NotFound

import coval_bench.contracts as contracts_module
from coval_bench.config import Settings
from coval_bench.fixture_sources import (
    FIXTURE_OBJECT,
    gcs_fixture_provider,
    install_fixture_providers,
)

SEEDED = b'{"tools": {}}'
BUCKET = "coval-benchmarks-contracts"


class _Blob:
    def __init__(self, store: dict[str, bytes], key: str) -> None:
        self._store = store
        self._key = key
        self.downloads = 0

    def download_as_bytes(self) -> bytes:
        if self._key not in self._store:
            raise NotFound(self._key)  # type: ignore[no-untyped-call]
        self.downloads += 1
        return self._store[self._key]


class _FakeClient:
    """Just enough of storage.Client, and it counts round trips."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects
        self.blobs: list[_Blob] = []

    def bucket(self, name: str) -> _FakeClient:
        assert name == BUCKET
        return self

    def blob(self, key: str) -> _Blob:
        blob = _Blob(self._objects, key)
        self.blobs.append(blob)
        return blob

    @property
    def downloads(self) -> int:
        return sum(b.downloads for b in self.blobs)


@pytest.fixture
def no_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contracts_module, "_FIXTURE_PROVIDERS", [])


def _client(**objects: bytes) -> Any:
    return _FakeClient({FIXTURE_OBJECT.format(suite=s): b for s, b in objects.items()})


def test_the_object_path_mirrors_the_layout_on_disk() -> None:
    assert FIXTURE_OBJECT.format(suite="dental") == "contracts/dental/mock-tools.json"


def test_a_seeded_suite_is_returned() -> None:
    provider = gcs_fixture_provider(BUCKET, storage_client=_client(dental=SEEDED))
    assert provider("dental") == SEEDED


def test_a_missing_object_raises_file_not_found_so_the_chain_continues() -> None:
    """Not an error: another provider may carry the suite."""
    provider = gcs_fixture_provider(BUCKET, storage_client=_client(dental=SEEDED))
    with pytest.raises(FileNotFoundError):
        provider("insurance")


def test_the_fetch_is_memoised() -> None:
    """The seeded world must not change under a running benchmark.

    The contract hash is computed over these bytes and published beside every
    result, so a mid-run change would silently invalidate it.
    """
    client = _client(dental=SEEDED)
    provider = gcs_fixture_provider(BUCKET, storage_client=client)
    for _ in range(3):
        assert provider("dental") == SEEDED
    assert client.downloads == 1


def test_nothing_is_registered_without_a_bucket(no_providers: None) -> None:
    """Every developer machine: local fixtures only, no cloud call attempted."""
    install_fixture_providers(Settings(mock_fixtures_bucket=""))
    assert contracts_module._FIXTURE_PROVIDERS == []


def test_a_configured_bucket_registers_one_provider(no_providers: None) -> None:
    install_fixture_providers(Settings(mock_fixtures_bucket=BUCKET))
    assert len(contracts_module._FIXTURE_PROVIDERS) == 1


# --- every entry point, not just the API ------------------------------------


def test_the_cli_installs_the_same_providers_the_api_does(
    no_providers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner hashes contracts too.

    `pull-contract` prints the hash today and the voice-agent ingest will write
    it onto result rows. A runner that could not see the fixtures would publish
    the hash of a run that had none — the one thing the hash exists to prevent.
    """
    from click.testing import CliRunner

    from coval_bench.__main__ import cli
    from coval_bench.config import get_settings

    monkeypatch.setenv("MOCK_FIXTURES_BUCKET", BUCKET)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    monkeypatch.setenv("DATASET_BUCKET", "b")
    monkeypatch.setenv("DATASET_ID", "d")
    get_settings.cache_clear()
    try:
        # A subcommand's help runs the group callback first, then exits before
        # the subcommand does anything. `--help` on the *group* would not: click
        # handles that eagerly and never reaches the callback.
        result = CliRunner().invoke(cli, ["pull-contract", "--help"])
        assert result.exit_code == 0, result.output
        assert len(contracts_module._FIXTURE_PROVIDERS) == 1
    finally:
        get_settings.cache_clear()
