# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the ``db migrate`` command."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from alembic import command as alembic_command
from alembic.util.exc import CommandError
from click.testing import CliRunner

from coval_bench.db import cli


@pytest.fixture
def upgrade(monkeypatch: pytest.MonkeyPatch) -> Mock:
    import coval_bench.config

    settings = SimpleNamespace(database_url="postgresql://fixture")
    monkeypatch.setattr(coval_bench.config, "get_settings", Mock(return_value=settings))
    upgrade = Mock()
    monkeypatch.setattr(alembic_command, "upgrade", upgrade)
    return upgrade


def test_default_target_is_head(upgrade: Mock) -> None:
    result = CliRunner().invoke(cli.db_migrate, [])
    assert result.exit_code == 0, result.output
    assert upgrade.call_args.args[1] == "head"


def test_explicit_target_is_passed_through(upgrade: Mock) -> None:
    result = CliRunner().invoke(cli.db_migrate, ["--revision", "20261007_0044"])
    assert result.exit_code == 0, result.output
    assert upgrade.call_args.args[1] == "20261007_0044"


def test_invalid_explicit_target_does_not_fall_back(upgrade: Mock) -> None:
    upgrade.side_effect = CommandError("unknown revision")
    result = CliRunner().invoke(cli.db_migrate, ["--revision", "unknown"])
    assert result.exit_code != 0
    assert upgrade.call_count == 1
    assert upgrade.call_args.args[1] == "unknown"
