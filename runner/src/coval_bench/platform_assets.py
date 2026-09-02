# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import click
import httpx
from pydantic import BaseModel, Field

from coval_bench.assets import SecretRef
from coval_bench.contracts import contract_sha256, load_stack, read_contract_file
from coval_bench.variants.platforms import redact

TOOL_TIMEOUT_SECONDS = 20
SIMULATION_HEADER_TEMPLATE = "{{coval-simulation-id}}"
TOOLS_PATH = "model.tools"
COVAL_API_BASE = "https://api.coval.dev/v1"
COVAL_MODEL_TYPE = "MODEL_TYPE_VOICE"
COVAL_PROMPT_FILE = "_source/coval-prompt.txt"
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

COVAL_API_KEY = SecretRef(
    name="COVAL_API_KEY", purpose="the Coval API key for the benchmarking org"
)


class SyncError(RuntimeError):
    pass


class PlatformAgentSpec(BaseModel, frozen=True, extra="forbid"):
    key: str = Field(min_length=1)
    platform: str
    suite: str
    agent_id: SecretRef
    api_key: SecretRef
    mock_secret: SecretRef
    dial_target: SecretRef


class AgentClient(Protocol):
    def __enter__(self) -> AgentClient: ...
    def __exit__(self, *exc: object) -> None: ...
    def get_agent(self, agent_id: str) -> dict[str, Any]: ...
    def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]: ...


Renderer = Callable[[PlatformAgentSpec, str, str], dict[str, Any]]
ClientFactory = Callable[[str, str], AgentClient]
Prepare = Callable[[AgentClient, PlatformAgentSpec, bool], list[str]]
Probe = Callable[[dict[str, Any], dict[str, Any], str, str], tuple[str, dict[str, str], Any]]


@dataclass(frozen=True)
class Platform:
    name: str
    api_base: str
    render: Renderer
    client: ClientFactory
    probe: Probe
    prepare: Prepare | None = None


@dataclass
class Plan:
    update: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)
    prepared: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"prepare={self.prepared} update={sorted(self.update)} "
            f"unchanged={sorted(self.unchanged)}"
        )


def _unwrap(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


class _JsonClient:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str],
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url, headers=headers, timeout=_TIMEOUT, transport=transport
        )

    def __enter__(self) -> _JsonClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._client.request(method, path, json=payload, params=params)
        if response.status_code >= 400:
            raise SyncError(f"{method} {path} -> {response.status_code}: {response.text[:400]}")
        if not response.content:
            return {}
        parsed: dict[str, Any] = response.json()
        return parsed


# --- vapi ------------------------------------------------------------------


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


def probe_vapi(
    tool: dict[str, Any], args: dict[str, Any], secret: str, simulation_id: str
) -> tuple[str, dict[str, str], Any]:
    headers = {"X-Mock-Tools-Key": secret, "X-Coval-Simulation-Id": simulation_id}
    body = {
        "message": {
            "type": "tool-calls",
            "toolCallList": [
                {
                    "id": f"{simulation_id}-1",
                    "type": "function",
                    "function": {"name": tool["function"]["name"], "arguments": json.dumps(args)},
                }
            ],
        }
    }
    return str(tool["server"]["url"]), headers, body


class VapiClient(_JsonClient):
    def __init__(
        self, api_key: str, base_url: str, transport: httpx.BaseTransport | None = None
    ) -> None:
        super().__init__(base_url, {"Authorization": f"Bearer {api_key}"}, transport)

    def __enter__(self) -> VapiClient:
        return self

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self._request("GET", f"/assistant/{agent_id}")

    def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"/assistant/{agent_id}", body)


# --- the table -------------------------------------------------------------


PLATFORMS: dict[str, Platform] = {
    "vapi": Platform(
        name="vapi",
        api_base="https://api.vapi.ai",
        render=render_vapi,
        client=VapiClient,
        probe=probe_vapi,
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
        dial_target=SecretRef(
            name="VAPI_DENTAL_DIAL_TARGET",
            purpose="the sip: URI Coval dials to reach the Vapi dental assistant",
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


def apply(
    client: AgentClient, spec: PlatformAgentSpec, wanted: dict[str, Any], *, dry_run: bool = False
) -> Plan:
    platform = platform_for(spec)
    prepared = platform.prepare(client, spec, dry_run) if platform.prepare else []
    agent_id = spec.agent_id.resolve()
    live = client.get_agent(agent_id)
    result = plan(live, wanted)
    result.prepared = prepared
    if result.update and not dry_run:
        client.update_agent(agent_id, patch_body(live, wanted))
    return result


# --- coval side ------------------------------------------------------------


class CovalClient(_JsonClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = COVAL_API_BASE,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(base_url, {"X-API-Key": api_key}, transport)

    def __enter__(self) -> CovalClient:
        return self

    def _pages(self, path: str, key: str) -> Iterator[dict[str, Any]]:
        token: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": 100}
            if token:
                params["page_token"] = token
            payload = self._request("GET", path, params=params)
            yield from payload.get(key, [])
            token = payload.get("next_page_token") or None
            if not token:
                return

    def find_agent(self, customer_agent_id: str) -> dict[str, Any] | None:
        for agent in self._pages("/agents", "agents"):
            if agent.get("customer_agent_id") == customer_agent_id:
                return agent
        return None

    def create_agent(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._request("POST", "/agents", body)
        agent = payload.get("agent")
        return agent if isinstance(agent, dict) else payload

    def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._request("PATCH", f"/agents/{agent_id}", body)
        agent = payload.get("agent")
        return agent if isinstance(agent, dict) else payload

    def launch_run(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._request("POST", "/runs", body)
        run = payload.get("run")
        return run if isinstance(run, dict) else payload

    def get_run(self, run_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/runs/{run_id}")
        run = payload.get("run")
        return run if isinstance(run, dict) else payload


COVAL_MANAGED = ("phone_number", "prompt", "metadata", "attributes")


def coval_agent_body(spec: PlatformAgentSpec) -> dict[str, Any]:
    """The Coval-side agent every variant registers: same call, only the dial target differs."""
    stack = load_stack()
    return {
        "display_name": spec.key,
        "customer_agent_id": spec.key,
        "model_type": COVAL_MODEL_TYPE,
        "phone_number": spec.dial_target.resolve(),
        "prompt": read_contract_file(spec.suite, COVAL_PROMPT_FILE),
        "language": "en",
        "metadata": {"audio_codec": stack.media.codec},
        "attributes": {
            "platform": spec.platform,
            "suite": spec.suite,
            "contract_sha256": contract_sha256(spec.suite),
        },
        "tags": ["orchestration", spec.suite, spec.platform],
    }


def register(
    client: CovalClient, spec: PlatformAgentSpec, *, dry_run: bool = False
) -> tuple[str, Plan]:
    """Create or reconcile the Coval agent that dials this variant; returns its id."""
    wanted = coval_agent_body(spec)
    live = client.find_agent(spec.key)
    if live is None:
        result = Plan(update={path: (None, wanted[path]) for path in COVAL_MANAGED})
        if dry_run:
            return "", result
        return str(client.create_agent(wanted)["id"]), result
    result = plan(live, {path: wanted[path] for path in COVAL_MANAGED})
    if result.update and not dry_run:
        client.update_agent(str(live["id"]), {path: wanted[path] for path in result.update})
    return str(live["id"]), result


def drift(client: AgentClient, spec: PlatformAgentSpec, mock_base_url: str) -> list[str]:
    """What the live platform agent still differs from the contract by; empty means in sync."""
    result = apply(client, spec, desired(spec, mock_base_url), dry_run=True)
    return sorted(result.update) + result.prepared


def launch_body(
    agent_id: str,
    spec: PlatformAgentSpec,
    persona_id: str,
    test_set_id: str,
    metric_ids: tuple[str, ...],
    sample: int,
    seed: int,
) -> dict[str, Any]:
    sha = contract_sha256(spec.suite)
    return {
        "agent_id": agent_id,
        "persona_id": persona_id,
        "test_set_id": test_set_id,
        "metric_ids": list(metric_ids),
        "options": {
            "iteration_count": 1,
            "concurrency": 1,
            "sub_sample_size": sample,
            "sub_sample_seed": seed,
        },
        "metadata": {
            "display_name": f"{spec.key} e2e {sha[:8]}",
            "customer_metadata": {
                "arm": spec.key,
                "platform": spec.platform,
                "contract_sha256": sha,
            },
        },
    }


# --- cli -------------------------------------------------------------------


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
_coval_base_option = click.option(
    "--coval-api-base", envvar="COVAL_API_BASE", default=COVAL_API_BASE, show_default=True
)


def _echo_plan(result: Plan) -> None:
    click.echo(result.summary())
    for path, (current, value) in result.update.items():
        found: list[str] = []
        click.echo(f"\n{path}:")
        click.echo(f"  live:    {json.dumps(redact(current, found), sort_keys=True)[:400]}")
        click.echo(f"  desired: {json.dumps(redact(value, found), sort_keys=True)[:400]}")


@click.group()
def platform_assets() -> None:
    """Platform agents managed as code."""


@platform_assets.command(name="plan")
@_agent_option
@_mock_base_url_option
@_api_base_option
def assets_plan(agent_key: str, mock_base_url: str, api_base: str | None) -> None:
    """Show what apply would change on the platform; writes nothing."""
    spec = spec_for(agent_key)
    platform = platform_for(spec)
    wanted = desired(spec, mock_base_url)
    with platform.client(spec.api_key.resolve(), api_base or platform.api_base) as client:
        _echo_plan(apply(client, spec, wanted, dry_run=True))


@platform_assets.command(name="apply")
@_agent_option
@_mock_base_url_option
@_api_base_option
@click.option("--yes", is_flag=True, default=False, help="Skip the confirmation prompt.")
def assets_apply(agent_key: str, mock_base_url: str, api_base: str | None, yes: bool) -> None:
    """Prepare the platform's prerequisites, then patch the managed fields on the live agent."""
    spec = spec_for(agent_key)
    platform = platform_for(spec)
    wanted = desired(spec, mock_base_url)
    with platform.client(spec.api_key.resolve(), api_base or platform.api_base) as client:
        preview = apply(client, spec, wanted, dry_run=True)
        click.echo(preview.summary())
        if not preview.update and not preview.prepared:
            return
        pending = sorted(preview.update) + preview.prepared
        if not yes and not click.confirm(f"Apply {pending} on {agent_key}?"):
            raise click.ClickException("aborted")
        result = apply(client, spec, wanted)
    click.echo(f"prepared {result.prepared}")
    click.echo(f"patched {sorted(result.update)}")
    click.echo(f"contract sha256 ({spec.suite}): {contract_sha256(spec.suite)}")


@platform_assets.command(name="smoke")
@_agent_option
@_mock_base_url_option
@click.option("--tool", default="lookup_patient", show_default=True)
@click.option("--arg", "args", multiple=True, help="key=value, repeatable.")
def assets_smoke(agent_key: str, mock_base_url: str, tool: str, args: tuple[str, ...]) -> None:
    """Fire one tool call at /mock exactly as this platform would, and print the answer."""
    spec = spec_for(agent_key)
    platform = platform_for(spec)
    rendered = desired(spec, mock_base_url)
    tools = rendered.get(TOOLS_PATH) or rendered.get("tools") or []
    wanted = next(
        (t for t in tools if (t.get("function") or t.get("webhook") or {}).get("name") == tool),
        None,
    )
    if wanted is None:
        raise click.ClickException(f"{tool!r} is not in the {spec.suite} contract")
    call_args = dict(item.split("=", 1) for item in args)
    simulation_id = f"smoke-{spec.key}-{int(time.time())}"
    url, headers, body = platform.probe(
        wanted, call_args, spec.mock_secret.resolve(), simulation_id
    )
    response = httpx.post(url, headers=headers, json=body, timeout=_TIMEOUT)
    click.echo(f"POST {url} -> {response.status_code}")
    click.echo(response.text[:800])
    click.echo(f"simulation_id sent: {simulation_id}")


@platform_assets.command(name="register")
@_agent_option
@_coval_base_option
@click.option("--yes", is_flag=True, default=False, help="Skip the confirmation prompt.")
def assets_register(agent_key: str, coval_api_base: str, yes: bool) -> None:
    """Create or reconcile the Coval agent that dials this variant."""
    spec = spec_for(agent_key)
    with CovalClient(COVAL_API_KEY.resolve(), coval_api_base) as client:
        agent_id, preview = register(client, spec, dry_run=True)
        _echo_plan(preview)
        if not preview.update:
            click.echo(f"coval agent {agent_id} already matches")
            return
        if not yes and not click.confirm(
            f"Write {sorted(preview.update)} to Coval for {agent_key}?"
        ):
            raise click.ClickException("aborted")
        agent_id, _ = register(client, spec)
    click.echo(f"coval agent id: {agent_id}")


@platform_assets.command(name="launch")
@_agent_option
@_mock_base_url_option
@_api_base_option
@_coval_base_option
@click.option("--persona-id", envvar="COVAL_DENTAL_PERSONA_ID", required=True)
@click.option("--test-set-id", envvar="COVAL_DENTAL_TEST_SET_ID", required=True)
@click.option("--metric-id", "metric_ids", multiple=True, envvar="COVAL_DENTAL_METRIC_IDS")
@click.option("--sample", default=1, show_default=True, help="Test cases to run; 0 = all.")
@click.option("--seed", default=847293, show_default=True)
@click.option("--wait/--no-wait", default=False, help="Poll until the run is terminal.")
@click.option(
    "--allow-drift", is_flag=True, default=False, help="Launch even if the platform agent drifts."
)
def assets_launch(
    agent_key: str,
    mock_base_url: str,
    api_base: str | None,
    coval_api_base: str,
    persona_id: str,
    test_set_id: str,
    metric_ids: tuple[str, ...],
    sample: int,
    seed: int,
    wait: bool,
    allow_drift: bool,
) -> None:
    """Launch one Coval run against this variant's registered agent, once the platform matches."""
    spec = spec_for(agent_key)
    platform = platform_for(spec)
    with platform.client(spec.api_key.resolve(), api_base or platform.api_base) as client:
        pending = drift(client, spec, mock_base_url)
    if pending:
        message = f"{agent_key} drifts from the contract: {pending}; run apply first"
        if not allow_drift:
            raise click.ClickException(message)
        click.echo(f"warning: {message}")
    with CovalClient(COVAL_API_KEY.resolve(), coval_api_base) as client:
        agent = client.find_agent(spec.key)
        if agent is None:
            raise click.ClickException(f"no Coval agent for {agent_key}; run register first")
        run = client.launch_run(
            launch_body(str(agent["id"]), spec, persona_id, test_set_id, metric_ids, sample, seed)
        )
        run_id = str(run.get("run_id") or run.get("id"))
        click.echo(f"run_id: {run_id} status: {run.get('status')}")
        click.echo("open this run id in the Coval UI to watch the simulation")
        while wait and run.get("status") not in {"COMPLETED", "FAILED", "CANCELLED", "DELETED"}:
            time.sleep(30)
            run = client.get_run(run_id)
            click.echo(f"status: {run.get('status')} progress: {run.get('progress')}")
