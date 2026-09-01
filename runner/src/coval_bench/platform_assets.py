# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import click
import httpx
from pydantic import BaseModel, Field

from coval_bench.assets import SecretRef
from coval_bench.contracts import contract_sha256, read_contract_file
from coval_bench.variants.platforms import redact

TOOL_TIMEOUT_SECONDS = 20
SIMULATION_HEADER_TEMPLATE = "{{coval-simulation-id}}"
TOOLS_PATH = "model.tools"
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class SyncError(RuntimeError):
    pass


class PlatformAgentSpec(BaseModel, frozen=True, extra="forbid"):
    key: str = Field(min_length=1)
    platform: str
    suite: str
    agent_id: SecretRef
    api_key: SecretRef
    mock_secret: SecretRef


class AgentClient(Protocol):
    def __enter__(self) -> AgentClient: ...
    def __exit__(self, *exc: object) -> None: ...
    def get_agent(self, agent_id: str) -> dict[str, Any]: ...
    def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]: ...


Renderer = Callable[[PlatformAgentSpec, str, str], dict[str, Any]]
ClientFactory = Callable[[str, str], AgentClient]


@dataclass(frozen=True)
class Platform:
    name: str
    api_base: str
    render: Renderer
    client: ClientFactory


@dataclass
class Plan:
    update: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"update={sorted(self.update)} unchanged={sorted(self.unchanged)}"


def render_vapi_tools(
    definitions: list[dict[str, Any]], mock_base_url: str, secret: str
) -> list[dict[str, Any]]:
    server = {
        "url": f"{mock_base_url.rstrip('/')}/mock/vapi",
        "timeoutSeconds": TOOL_TIMEOUT_SECONDS,
        "headers": {
            "X-Mock-Tools-Key": secret,
            "X-Coval-Simulation-Id": SIMULATION_HEADER_TEMPLATE,
        },
    }
    return [
        {
            "type": "function",
            "async": False,
            "function": {
                "name": definition["name"],
                "description": definition["description"],
                "parameters": definition["parameters"],
            },
            "server": server,
        }
        for definition in definitions
    ]


def render_vapi(spec: PlatformAgentSpec, mock_base_url: str, secret: str) -> dict[str, Any]:
    definitions = json.loads(read_contract_file(spec.suite, "tool-definitions.json"))
    return {TOOLS_PATH: render_vapi_tools(definitions, mock_base_url, secret)}


class VapiClient:
    def __init__(
        self, api_key: str, base_url: str, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_TIMEOUT,
            transport=transport,
        )

    def __enter__(self) -> VapiClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        response = self._client.request(method, path, json=payload)
        if response.status_code >= 400:
            raise SyncError(f"{method} {path} -> {response.status_code}: {response.text[:400]}")
        parsed: dict[str, Any] = response.json()
        return parsed

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self._request("GET", f"/assistant/{agent_id}")

    def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"/assistant/{agent_id}", body)


PLATFORMS: dict[str, Platform] = {
    "vapi": Platform(
        name="vapi", api_base="https://api.vapi.ai", render=render_vapi, client=VapiClient
    ),
}

AGENTS: tuple[PlatformAgentSpec, ...] = (
    PlatformAgentSpec(
        key="vapi-dental",
        platform="vapi",
        suite="dental",
        agent_id=SecretRef(
            name="VAPI_DENTAL_ASSISTANT_ID",
            purpose="the Vapi assistant id for the dental suite",
        ),
        api_key=SecretRef(name="VAPI_API_KEY", purpose="the Vapi API key for the dental org"),
        mock_secret=SecretRef(
            name="MOCK_TOOLS_SECRET",
            purpose="the shared secret the mock tool endpoint requires on X-Mock-Tools-Key",
        ),
    ),
)


def spec_for(key: str) -> PlatformAgentSpec:
    for spec in AGENTS:
        if spec.key == key:
            return spec
    known = ", ".join(sorted(s.key for s in AGENTS))
    raise KeyError(f"unknown platform agent {key!r}; known: {known}")


def platform_for(spec: PlatformAgentSpec) -> Platform:
    platform = PLATFORMS.get(spec.platform)
    if platform is None:
        known = ", ".join(sorted(PLATFORMS))
        raise SyncError(f"unknown platform {spec.platform!r}; known: {known}")
    return platform


def _get_path(body: dict[str, Any], path: str) -> Any:  # noqa: ANN401
    node: Any = body
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _set_path(body: dict[str, Any], path: str, value: Any) -> None:  # noqa: ANN401
    parts = path.split(".")
    node = body
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def desired(spec: PlatformAgentSpec, mock_base_url: str) -> dict[str, Any]:
    return platform_for(spec).render(spec, mock_base_url, spec.mock_secret.resolve())


def plan(live: dict[str, Any], wanted: dict[str, Any]) -> Plan:
    result = Plan()
    for path, value in wanted.items():
        current = _get_path(live, path)
        if current == value:
            result.unchanged.append(path)
        else:
            result.update[path] = (current, value)
    return result


def patch_body(live: dict[str, Any], wanted: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for path, value in wanted.items():
        top = path.split(".")[0]
        if top not in body:
            body[top] = copy.deepcopy(live.get(top)) if isinstance(live.get(top), dict) else {}
        if "." in path:
            _set_path(body, path, value)
        else:
            body[top] = value
    return body


def apply(client: AgentClient, spec: PlatformAgentSpec, wanted: dict[str, Any]) -> Plan:
    agent_id = spec.agent_id.resolve()
    live = client.get_agent(agent_id)
    result = plan(live, wanted)
    if result.update:
        client.update_agent(agent_id, patch_body(live, wanted))
    return result


_agent_option = click.option(
    "--agent", "agent_key", type=click.Choice(sorted(s.key for s in AGENTS)), required=True
)
_mock_base_url_option = click.option(
    "--mock-base-url",
    envvar="MOCK_TOOLS_BASE_URL",
    required=True,
    help="Where the mock tool service is reachable, e.g. https://api.example.com",
)
_api_base_option = click.option(
    "--api-base", default=None, help="Override the platform's API base URL."
)


@click.group()
def platform_assets() -> None:
    """Platform agents managed as code."""


@platform_assets.command(name="plan")
@_agent_option
@_mock_base_url_option
@_api_base_option
def assets_plan(agent_key: str, mock_base_url: str, api_base: str | None) -> None:
    """Show what apply would change; writes nothing."""
    spec = spec_for(agent_key)
    platform = platform_for(spec)
    wanted = desired(spec, mock_base_url)
    with platform.client(spec.api_key.resolve(), api_base or platform.api_base) as client:
        result = plan(client.get_agent(spec.agent_id.resolve()), wanted)
    click.echo(result.summary())
    for path, (current, value) in result.update.items():
        found: list[str] = []
        click.echo(f"\n{path}:")
        click.echo(f"  live:    {json.dumps(redact(current, found), sort_keys=True)[:400]}")
        click.echo(f"  desired: {json.dumps(redact(value, found), sort_keys=True)[:400]}")


@platform_assets.command(name="apply")
@_agent_option
@_mock_base_url_option
@_api_base_option
@click.option("--yes", is_flag=True, default=False, help="Skip the confirmation prompt.")
def assets_apply(agent_key: str, mock_base_url: str, api_base: str | None, yes: bool) -> None:
    """Patch the managed fields on the live agent."""
    spec = spec_for(agent_key)
    platform = platform_for(spec)
    wanted = desired(spec, mock_base_url)
    with platform.client(spec.api_key.resolve(), api_base or platform.api_base) as client:
        preview = plan(client.get_agent(spec.agent_id.resolve()), wanted)
        click.echo(preview.summary())
        if not preview.update:
            return
        if not yes and not click.confirm(f"Patch {sorted(preview.update)} on {agent_key}?"):
            raise click.ClickException("aborted")
        result = apply(client, spec, wanted)
    click.echo(f"patched {sorted(result.update)}")
    click.echo(f"contract sha256 ({spec.suite}): {contract_sha256(spec.suite)}")
