# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the staged Alembic migration gate."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from alembic import command as alembic_command
from alembic.config import Config
from alembic.util.exc import CommandError
from click import ClickException
from click.testing import CliRunner

from coval_bench.db import cli


@pytest.fixture
def graph(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock, Mock]:
    cap = SimpleNamespace(revision="20261005_0043")
    old = SimpleNamespace(revision="20260929_0042")
    newer = SimpleNamespace(revision="20261007_0044")
    divergent = SimpleNamespace(revision="divergent")
    divergent = SimpleNamespace(revision="20260930_0043")
    script = MagicMock()
    script.get_revision.side_effect = lambda revision: {
        "20261005_0043": cap,
        "20261007_0044": newer,
        "20260929_0042": old,
        "divergent": divergent,
        "20260930_0043": divergent,
    }.get(revision)

    def iterate(revision: str, _: str) -> list[SimpleNamespace]:
        if revision == "20261007_0044":
            return [newer, cap]
        if revision == "20261005_0043":
            return [cap, old]
        return [divergent] if revision == "20260930_0043" else [old]

    script.iterate_revisions.side_effect = iterate
    monkeypatch.setattr(cli, "ScriptDirectory", MagicMock(from_config=Mock(return_value=script)))
    engine = MagicMock()
    connection = MagicMock()
    engine.connect.return_value.__enter__.return_value = connection
    monkeypatch.setattr(cli, "create_engine", Mock(return_value=engine))
    context = MagicMock()
    monkeypatch.setattr(cli, "MigrationContext", MagicMock(configure=Mock(return_value=context)))
    return script, context, engine


@pytest.mark.parametrize(
    ("heads", "expected"),
    [
        ((), "20261005_0043"),
        (("20260929_0042",), "20261005_0043"),
        (("20261005_0043",), None),
        (("20261007_0044",), None),
        (("20260930_0043",), "error"),
        (("unknown",), "error"),
        (("divergent",), "error"),
        (("20261005_0043", "20261007_0044"), "error"),
    ],
)
def test_default_target_uses_graph_ancestry(
    graph: tuple[Mock, Mock, Mock],
    heads: tuple[str, ...],
    expected: str | None,
) -> None:
    _, context, _ = graph
    context.get_current_heads.return_value = heads
    cfg = Config()
    if expected == "error":
        with pytest.raises(ClickException):
            cli._default_migration_target(cfg, "postgresql://fixture")
    else:
        assert cli._default_migration_target(cfg, "postgresql://fixture") == expected


def test_explicit_target_sets_cleanup_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    import coval_bench.config

    upgrade = Mock()
    settings = SimpleNamespace(database_url="postgresql://fixture")
    monkeypatch.setattr(coval_bench.config, "get_settings", Mock(return_value=settings))
    monkeypatch.setattr(alembic_command, "upgrade", upgrade)
    result = CliRunner().invoke(cli.db_migrate, ["--revision", "20261007_0044"])
    assert result.exit_code == 0, result.output
    cfg = upgrade.call_args.args[0]
    assert cfg.attributes["allow_metric_code_cleanup"] is True
    assert upgrade.call_args.args[1] == "20261007_0044"


def test_invalid_explicit_target_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    import coval_bench.config

    monkeypatch.setattr(
        coval_bench.config,
        "get_settings",
        Mock(return_value=SimpleNamespace(database_url="postgresql://fixture")),
    )
    upgrade = Mock(side_effect=CommandError("unknown revision"))
    monkeypatch.setattr(alembic_command, "upgrade", upgrade)
    result = CliRunner().invoke(cli.db_migrate, ["--revision", "unknown"])
    assert result.exit_code != 0
    assert upgrade.call_count == 1
    assert upgrade.call_args.args[1] == "unknown"


def test_default_cap_does_not_enable_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    import coval_bench.config

    monkeypatch.setattr(
        coval_bench.config,
        "get_settings",
        Mock(return_value=SimpleNamespace(database_url="postgresql://fixture")),
    )
    monkeypatch.setattr(cli, "_default_migration_target", Mock(return_value="20261005_0043"))
    upgrade = Mock()
    monkeypatch.setattr(alembic_command, "upgrade", upgrade)
    result = CliRunner().invoke(cli.db_migrate, [])
    assert result.exit_code == 0, result.output
    cfg = upgrade.call_args.args[0]
    assert "allow_metric_code_cleanup" not in cfg.attributes
    assert upgrade.call_args.args[1] == "20261005_0043"
