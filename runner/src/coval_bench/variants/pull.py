# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pull a live suite contract out of Coval and, optionally, one platform.

The contract is *extracted*, not authored: the scenarios and the agent config
already exist upstream, so this command dumps them into the packaged
``contracts/`` tree where they become diffable, hashable artifacts that ship
inside the wheel.

Coval is always the source for the suite. A platform is optional and resolved
through ``variants.platforms.FETCHERS``, so no vendor is special here and
``--platform`` accepts any registered name.

Where a dump lands matters. ``_source/`` holds provenance that is committed;
``_private/`` holds the evaluator and is never committed. Promoting either into
a contract file is a deliberate human step, so a re-pull cannot silently rewrite
a contract that a published run already used.

Usage::

    export COVAL_API_KEY=...
    export <PLATFORM>_API_KEY=...        # only when --platform is given

    coval-bench pull-contract \\
        --coval-agent-id <22-char id> \\
        --test-set-id <8-char id> \\
        --platform <name> --platform-agent-id <id>

Read-only against every API. Writes only inside ``--out-root``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import click
import httpx
import structlog

from coval_bench.contracts import contract_sha256
from coval_bench.variants.platforms import FETCHERS, fetch_platform, redact_identifiers

logger = structlog.get_logger("coval_bench.variants.pull")

COVAL_API_BASE = "https://api.coval.dev/v1"
PAGE_SIZE = 100
# A pathological filter or a huge test set should stop the loop, not spin it.
MAX_PAGES = 50
TIMEOUT_SECONDS = 30.0

# Coval does not guarantee one canonical display-name field across resources,
# so match on any of these rather than asserting a single key.
NAME_FIELDS = ("display_name", "name", "title")


def _name_of(item: dict[str, Any]) -> str:
    for field in NAME_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def _write(path: Path, payload: Any) -> None:  # noqa: ANN401
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload)
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    click.echo(f"  wrote {path}")


def _coval(api_key: str) -> httpx.Client:
    return httpx.Client(
        base_url=COVAL_API_BASE,
        headers={"X-API-Key": api_key},
        timeout=TIMEOUT_SECONDS,
    )


def _paginate(
    client: httpx.Client, path: str, key: str, params: dict[str, Any]
) -> list[dict[str, Any]]:
    """Collect every page of a Coval list endpoint."""
    items: list[dict[str, Any]] = []
    page_token: str | None = None
    for _ in range(MAX_PAGES):
        query = {**params, "page_size": PAGE_SIZE}
        if page_token:
            query["page_token"] = page_token
        response = client.get(path, params=query)
        response.raise_for_status()
        body = cast("dict[str, Any]", response.json())
        items.extend(cast("list[dict[str, Any]]", body.get(key, [])))
        page_token = body.get("next_page_token") or None
        if not page_token:
            return items
    raise click.ClickException(f"{path} exceeded {MAX_PAGES} pages; narrow the filter")


def _find_test_set(client: httpx.Client, wanted: str) -> dict[str, Any]:
    test_sets = _paginate(client, "/test-sets", "test_sets", {})
    needle = wanted.strip().lower()
    exact = [t for t in test_sets if _name_of(t).strip().lower() == needle]
    partial = [t for t in test_sets if needle in _name_of(t).strip().lower()]
    candidates = exact or partial
    if not candidates:
        available = ", ".join(sorted(_name_of(t) for t in test_sets if _name_of(t))) or "(none)"
        raise click.ClickException(f"No test set matching {wanted!r}. Available: {available}")
    if len(candidates) > 1:
        names = ", ".join(f"{_name_of(t)} ({t.get('id')})" for t in candidates)
        raise click.ClickException(f"{wanted!r} is ambiguous: {names}. Pass --test-set-id instead.")
    return candidates[0]


@click.command(name="pull-contract")
@click.option("--coval-agent-id", required=True, help="Coval agent whose config to dump.")
@click.option(
    "--test-set-name",
    default="dental",
    show_default=True,
    help="Test set to dump, matched by name.",
)
@click.option("--test-set-id", default=None, help="Exact test set id; skips the name lookup.")
@click.option(
    "--platform",
    type=click.Choice(sorted(FETCHERS)),
    default=None,
    help="Also dump one platform's live agent config.",
)
@click.option("--platform-agent-id", default=None, help="That platform's agent id.")
@click.option(
    "--out-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path(__file__).resolve().parent.parent / "contracts",
    show_default=True,
    help="Contracts package dir. Ships in the wheel; the API container reads it at runtime.",
)
def pull_contract(
    coval_agent_id: str,
    test_set_name: str,
    test_set_id: str | None,
    platform: str | None,
    platform_agent_id: str | None,
    out_root: Path,
) -> None:
    """Dump the live Coval agent and test set, and optionally one platform's agent."""
    coval_api_key = os.environ.get("COVAL_API_KEY")
    if not coval_api_key:
        raise click.ClickException("COVAL_API_KEY is not set")

    source_dir = out_root / "dental" / "_source"
    private_dir = out_root / "dental" / "_private"
    prompts: dict[str, str] = {}

    with _coval(coval_api_key) as client:
        click.echo(f"Coval agent {coval_agent_id}")
        response = client.get(f"/agents/{coval_agent_id}")
        response.raise_for_status()
        agent = cast("dict[str, Any]", response.json()).get("agent", {})
        # The dump is committed, and this repo is public. Identifiers go out;
        # the prompt and settings stay, because they are what drift detection
        # is for.
        stripped: list[str] = []
        _write(source_dir / "coval-agent.json", redact_identifiers(agent, stripped))

        coval_prompt = agent.get("prompt") or ""
        if isinstance(coval_prompt, str) and coval_prompt.strip():
            prompts["coval"] = coval_prompt
            _write(source_dir / "coval-prompt.txt", coval_prompt)
        click.echo(
            f"  model_type={agent.get('model_type')} phone_number={agent.get('phone_number')}"
        )

        if test_set_id is None:
            test_set = _find_test_set(client, test_set_name)
            test_set_id = cast("str", test_set.get("id"))
            click.echo(f"Test set {_name_of(test_set)!r} -> {test_set_id}")
            _write(source_dir / "test-set.json", redact_identifiers(test_set, stripped))

        click.echo(f"  withheld {len(stripped)} identifier(s) from _source/")

        # The value must be double-quoted. The OpenAPI example shows it bare
        # (`test_set_id=abc12345`), but the API rejects that with INVALID_ARGUMENT.
        test_cases = _paginate(
            client, "/test-cases", "test_cases", {"filter": f'test_set_id="{test_set_id}"'}
        )
        _write(private_dir / "test-cases.json", test_cases)
        click.echo(
            f"  {len(test_cases)} test cases -> _private/ (the evaluator is never committed)"
        )

    if platform:
        if not platform_agent_id:
            raise click.ClickException("--platform-agent-id is required with --platform")
        click.echo(f"Platform {platform}: agent {platform_agent_id}")
        try:
            config, redacted_paths = fetch_platform(platform, platform_agent_id)
        except (KeyError, RuntimeError) as exc:
            raise click.ClickException(str(exc)) from exc

        # The live agent config is the platform artifact, not a contract file.
        _write(out_root / "platforms" / f"{platform}-agent.json", config.raw)
        if redacted_paths:
            click.echo(
                f"  redacted {len(redacted_paths)} secret field(s): " + ", ".join(redacted_paths)
            )
        else:
            click.echo("  no secret-shaped fields found; nothing redacted")

        if config.system_prompt:
            prompts[platform] = config.system_prompt
            _write(source_dir / f"{platform}-prompt.txt", config.system_prompt)
        if config.first_message:
            _write(source_dir / "first-message.txt", config.first_message)
        _write(source_dir / f"{platform}-tools.json", config.tools)
        names = [t.get("function", {}).get("name") or t.get("name") for t in config.tools]
        click.echo(f"  {len(config.tools)} tools: {', '.join(n for n in names if n) or '(none)'}")

    # Coval stores a prompt per agent, but for a dialled voice agent that field is
    # descriptive: the prompt the agent actually runs lives with the platform. A
    # divergence means the judges are shown different instructions than the agent
    # followed, so it is surfaced rather than silently reconciled.
    if platform and len(prompts) == 2:
        if prompts["coval"].strip() == prompts[platform].strip():
            click.echo(f"\nPrompts match: Coval's copy == the prompt {platform} runs.")
        else:
            click.echo(
                f"\n!! Prompts DIFFER (coval={len(prompts['coval'])} chars, "
                f"{platform}={len(prompts[platform])} chars). "
                f"The {platform} copy is the one the agent actually ran."
            )

    click.echo(f"\nDumped to {source_dir}. Review, then promote to contract files by hand.")
    click.echo(f"Contract sha256 (stack + dental): {contract_sha256('dental')}")
