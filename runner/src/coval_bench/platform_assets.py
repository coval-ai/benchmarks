# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import click
import httpx
from pydantic import BaseModel, Field

from coval_bench.assets import SecretRef
from coval_bench.config import Settings
from coval_bench.fixture_sources import install_fixture_providers
from coval_bench.mocktools.codecs import PRESET_CALLER, PRESET_SIMULATION, Correlation, codec_for
from coval_bench.scenarios import (
    Stack,
    contract_sha256,
    has_private_contract,
    load_stack,
    read_contract_file,
)
from coval_bench.variants.platforms import read_retell, redact, retell_engine

TOOL_TIMEOUT_SECONDS = 20
SIMULATION_HEADER_TEMPLATE = "{{coval-simulation-id}}"
TOOLS_PATH = "model.tools"
TELNYX_SIP_SUFFIX = ".sip.telnyx.com"
TELNYX_SIP_RECEIVE = "from_anyone"
COVAL_API_BASE = "https://api.coval.dev/v1"
COVAL_MODEL_TYPE = "MODEL_TYPE_VOICE"
COVAL_PROMPT_FILE = "_source/coval-prompt.txt"
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

COVAL_API_KEY = SecretRef(
    name="COVAL_API_KEY", purpose="the Coval API key for the benchmarking org"
)


TELNYX_SECRETS: dict[str, SecretRef] = {
    "coval-bench-openai": SecretRef(
        name="OPENAI_API_KEY", purpose="the OpenAI key Telnyx bills LLM calls to"
    ),
    "coval-bench-elevenlabs": SecretRef(
        name="ELEVENLABS_API_KEY", purpose="the ElevenLabs key Telnyx synthesises with"
    ),
    "coval-bench-mock": SecretRef(
        name="MOCK_TOOLS_SECRET", purpose="the shared secret the mock tool endpoint requires"
    ),
}
TELNYX_LLM_REF = "coval-bench-openai"
TELNYX_TTS_REF = "coval-bench-elevenlabs"
TELNYX_MOCK_REF = "coval-bench-mock"


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


ToolRenderer = Callable[[list[dict[str, Any]], str, str], list[dict[str, Any]]]
Pin = Callable[[Stack], Any]
Canon = Callable[[Any], Any]
ClientFactory = Callable[[str, str], AgentClient]
Prepare = Callable[[AgentClient, PlatformAgentSpec, bool], list[str]]


@dataclass(frozen=True)
class Platform:
    """One vendor: where its API is, how a tool is spelled, and which paths carry the pin."""

    name: str
    api_base: str
    client: ClientFactory
    tools_path: str
    render_tools: ToolRenderer
    pins: Mapping[str, Pin] = field(default_factory=dict)
    canon: Mapping[str, Canon] = field(default_factory=dict)
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


# --- telnyx ----------------------------------------------------------------


def _mustache_secret(identifier: str) -> str:
    return f"{{{{#integration_secret}}}}{identifier}{{{{/integration_secret}}}}"


def render_telnyx_tools(
    definitions: list[dict[str, Any]], mock_base_url: str, _secret: str
) -> list[dict[str, Any]]:
    base = mock_base_url.rstrip("/")
    return [
        {
            "type": "webhook",
            "webhook": {
                "name": definition["name"],
                "description": definition["description"],
                "url": f"{base}/mock/telnyx/{definition['name']}",
                "method": "POST",
                "headers": [
                    {"name": "X-Mock-Tools-Key", "value": _mustache_secret(TELNYX_MOCK_REF)}
                ],
                "body_parameters": definition["parameters"],
                "preset_body_fields": {
                    PRESET_SIMULATION: "{{coval_simulation_id}}",
                    PRESET_CALLER: "{{telnyx_end_user_target}}",
                },
                "timeout_ms": TOOL_TIMEOUT_SECONDS * 1000,
                "async": False,
            },
        }
        for definition in definitions
    ]


TELNYX_PINS: dict[str, Pin] = {
    "model": lambda stack: f"{stack.llm.provider}/{stack.llm.model}",
    "llm_api_key_ref": lambda _stack: TELNYX_LLM_REF,
    "transcription.model": lambda stack: f"{stack.stt.provider}/{stack.stt.model}",
    "transcription.language": lambda _stack: "en",
    "voice_settings.voice": lambda stack: f"ElevenLabs.{stack.tts.model}.{stack.tts.voice_id}",
    "voice_settings.api_key_ref": lambda _stack: TELNYX_TTS_REF,
    "voice_settings.expressive_mode": lambda _stack: False,
    "tool_ids": lambda _stack: [],
    "telephony_settings.recording_settings.enabled": (
        lambda stack: stack.platform_behaviour.vendor_post_call_analysis
    ),
    "interruption_settings.start_speaking_plan.wait_seconds": (
        lambda stack: stack.turn_taking.end_of_turn_target_ms / 1000
    ),
}


TELNYX_SERVER_TOOL_KEYS = frozenset({"tool_id", "shared", "timeout_ms"})


def _telnyx_canon_tools(live: Any) -> Any:  # noqa: ANN401
    if not isinstance(live, list):
        return live
    return [
        {k: v for k, v in tool.items() if k not in TELNYX_SERVER_TOOL_KEYS}
        if isinstance(tool, dict)
        else tool
        for tool in live
    ]


TELNYX_CANON: dict[str, Canon] = {
    "tools": _telnyx_canon_tools,
    "tool_ids": lambda live: live or [],
}


def sip_subdomain(dial_target: str) -> str:
    """The Telnyx subdomain a ``sip:user@<sub>.sip.telnyx.com`` target names."""
    host = dial_target.split("@", 1)[-1].split(":", 1)[0].lower()
    if not dial_target.lower().startswith("sip:") or not host.endswith(TELNYX_SIP_SUFFIX):
        raise SyncError(f"{dial_target!r} is not a sip:...@<sub>{TELNYX_SIP_SUFFIX} target")
    return host.removesuffix(TELNYX_SIP_SUFFIX)


class TelnyxClient(_JsonClient):
    def __init__(
        self, api_key: str, base_url: str, transport: httpx.BaseTransport | None = None
    ) -> None:
        super().__init__(base_url, {"Authorization": f"Bearer {api_key}"}, transport)

    def __enter__(self) -> TelnyxClient:
        return self

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return _unwrap(self._request("GET", f"/ai/assistants/{agent_id}"))

    def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return _unwrap(self._request("POST", f"/ai/assistants/{agent_id}", body))

    def secret_identifiers(self) -> set[str]:
        payload = self._request("GET", "/integration_secrets", params={"page[size]": 250})
        return {str(item.get("identifier")) for item in payload.get("data", [])}

    def create_secret(self, identifier: str, token: str) -> None:
        self._request(
            "POST",
            "/integration_secrets",
            {"identifier": identifier, "type": "bearer", "token": token},
        )

    def get_texml_app(self, app_id: str) -> dict[str, Any]:
        return _unwrap(self._request("GET", f"/texml_applications/{app_id}"))

    def set_sip_subdomain(self, app_id: str, subdomain: str) -> None:
        self._request(
            "PATCH",
            f"/texml_applications/{app_id}",
            {
                "inbound": {
                    "sip_subdomain": subdomain,
                    "sip_subdomain_receive_settings": TELNYX_SIP_RECEIVE,
                }
            },
        )


def prepare_telnyx(client: AgentClient, spec: PlatformAgentSpec, dry_run: bool) -> list[str]:
    """Ensure the integration secrets and SIP subdomain exist before the assistant is patched."""
    if not isinstance(client, TelnyxClient):
        raise SyncError("telnyx prepare needs a TelnyxClient")
    pending: list[str] = []
    present = client.secret_identifiers()
    for identifier, ref in TELNYX_SECRETS.items():
        if identifier in present:
            continue
        pending.append(f"integration_secret:{identifier}")
        if not dry_run:
            client.create_secret(identifier, ref.resolve())
    wanted = sip_subdomain(spec.dial_target.resolve())
    live = client.get_agent(spec.agent_id.resolve())
    app_id = str(_get_path(live, "telephony_settings.default_texml_app_id") or "")
    if not app_id:
        raise SyncError("the assistant has no telephony_settings.default_texml_app_id")
    inbound = _get_path(client.get_texml_app(app_id), "inbound") or {}
    current = (inbound.get("sip_subdomain"), inbound.get("sip_subdomain_receive_settings"))
    if current != (wanted, TELNYX_SIP_RECEIVE):
        pending.append(f"sip_subdomain:{app_id}={wanted}:{TELNYX_SIP_RECEIVE}")
        if not dry_run:
            client.set_sip_subdomain(app_id, wanted)
    return pending


# --- retell ----------------------------------------------------------------

RETELL_LLM_FIELDS = frozenset(
    {"general_tools", "general_prompt", "begin_message", "model", "model_temperature"}
)
RETELL_CALLER_TEMPLATE = "{{user_number}}"
RETELL_ROUTE_VERSION = "latest"
RETELL_VOICE_ID = "11labs-Cimo"
RETELL_SIP_HOST = "sip.retellai.com"

RETELL_PINS: dict[str, Pin] = {
    "model": lambda stack: stack.llm.model,
    "model_temperature": lambda stack: stack.llm.temperature,
    "voice_model": lambda stack: stack.tts.model,
    "voice_id": lambda _stack: RETELL_VOICE_ID,
    "stt_mode": lambda _stack: "fast",
    "language": lambda _stack: "en-US",
    "post_call_analysis_model": (
        lambda stack: "gpt-4.1" if stack.platform_behaviour.vendor_post_call_analysis else None
    ),
}


def render_retell_tools(
    definitions: list[dict[str, Any]], mock_base_url: str, secret: str
) -> list[dict[str, Any]]:
    base = mock_base_url.rstrip("/")
    return [
        {
            "type": "custom",
            "name": definition["name"],
            "description": definition["description"],
            "url": f"{base}/mock/retell/{definition['name']}",
            "method": "POST",
            "headers": {
                "X-Mock-Tools-Key": secret,
                "X-Coval-Simulation-Id": SIMULATION_HEADER_TEMPLATE,
                "X-Coval-Caller-Number": RETELL_CALLER_TEMPLATE,
            },
            "parameters": definition["parameters"],
            "args_at_root": True,
            "speak_during_execution": False,
            "speak_after_execution": True,
            "timeout_ms": TOOL_TIMEOUT_SECONDS * 1000,
            "max_retry": 0,
        }
        for definition in definitions
    ]


class RetellClient(_JsonClient):
    """Presents the agent and the Retell LLM it answers with as one object.

    Retell keeps prompt, greeting and tools on the LLM, and the agent only points
    at it, so a read merges the two and a write routes each field to its owner.
    """

    def __init__(
        self, api_key: str, base_url: str, transport: httpx.BaseTransport | None = None
    ) -> None:
        super().__init__(base_url, {"Authorization": f"Bearer {api_key}"}, transport)
        self._engines: dict[str, tuple[str, int | None]] = {}

    def __enter__(self) -> RetellClient:
        return self

    def _read(self, agent_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            agent, llm = read_retell(
                lambda path, params: self._request("GET", path, params=params), agent_id
            )
        except RuntimeError as exc:
            raise SyncError(str(exc)) from exc
        self._engines[agent_id] = retell_engine(agent)
        return agent, llm

    @staticmethod
    def _llm_params(version: int | None) -> dict[str, Any] | None:
        return {"version": version} if version is not None else None

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        agent, llm = self._read(agent_id)
        return {**agent, **{key: llm.get(key) for key in RETELL_LLM_FIELDS}}

    def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        if agent_id not in self._engines:
            self._read(agent_id)
        llm_id, version = self._engines[agent_id]
        llm_part = {k: v for k, v in body.items() if k in RETELL_LLM_FIELDS}
        agent_part = {k: v for k, v in body.items() if k not in RETELL_LLM_FIELDS}
        merged: dict[str, Any] = {}
        if llm_part:
            llm = self._request(
                "PATCH",
                f"/update-retell-llm/{llm_id}",
                llm_part,
                params=self._llm_params(version),
            )
            merged.update({key: llm.get(key) for key in RETELL_LLM_FIELDS})
        if agent_part:
            merged = {**self._request("PATCH", f"/update-agent/{agent_id}", agent_part), **merged}
        return merged

    def inbound_agents(self, number: str) -> list[dict[str, Any]]:
        payload = self._request("GET", f"/get-phone-number/{number}")
        return list(payload.get("inbound_agents") or [])

    def bind_inbound(self, number: str, agents: list[dict[str, Any]]) -> None:
        self._request("PATCH", f"/update-phone-number/{number}", {"inbound_agents": agents})


def retell_number(dial_target: str) -> str:
    """The imported number a ``sip:+E164@sip.retellai.com`` target routes on."""
    user, _, host = dial_target.partition("@")
    number = user.lower().removeprefix("sip:")
    if (
        not dial_target.lower().startswith("sip:")
        or host.split(":", 1)[0].lower() != RETELL_SIP_HOST
    ):
        raise SyncError(f"{dial_target!r} is not a sip:+E164@{RETELL_SIP_HOST} target")
    if not number.startswith("+") or not number[1:].isdigit():
        raise SyncError(
            f"{dial_target!r} does not carry an E.164 number; Retell routes inbound by number"
        )
    return number


def prepare_retell(client: AgentClient, spec: PlatformAgentSpec, dry_run: bool) -> list[str]:
    """Ensure the dialled number answers with this agent's newest draft, so apply reaches calls."""
    if not isinstance(client, RetellClient):
        raise SyncError("retell prepare needs a RetellClient")
    number = retell_number(spec.dial_target.resolve())
    agent_id = spec.agent_id.resolve()
    wanted = [{"agent_id": agent_id, "agent_version": RETELL_ROUTE_VERSION, "weight": 1}]
    live = client.inbound_agents(number)
    if live == wanted:
        return []
    foreign = {str(a.get("agent_id")) for a in live if isinstance(a, dict)} - {agent_id}
    if foreign:
        raise SyncError(f"{number} answers with {sorted(foreign)}; not ours to repoint")
    if not dry_run:
        client.bind_inbound(number, wanted)
    return [f"route:{number}={agent_id}@{RETELL_ROUTE_VERSION}"]


# --- the table -------------------------------------------------------------


PLATFORMS: dict[str, Platform] = {
    "vapi": Platform(
        name="vapi",
        api_base="https://api.vapi.ai",
        client=VapiClient,
        tools_path=TOOLS_PATH,
        render_tools=render_vapi_tools,
    ),
    "telnyx": Platform(
        name="telnyx",
        api_base="https://api.telnyx.com/v2",
        client=TelnyxClient,
        tools_path="tools",
        render_tools=render_telnyx_tools,
        pins=TELNYX_PINS,
        canon=TELNYX_CANON,
        prepare=prepare_telnyx,
    ),
    "retell": Platform(
        name="retell",
        api_base="https://api.retellai.com",
        client=RetellClient,
        tools_path="general_tools",
        render_tools=render_retell_tools,
        pins=RETELL_PINS,
        prepare=prepare_retell,
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
    PlatformAgentSpec(
        key="telnyx-dental",
        platform="telnyx",
        suite="dental",
        agent_id=SecretRef(
            name="TELNYX_DENTAL_ASSISTANT_ID",
            purpose="the Telnyx assistant id for the dental suite",
        ),
        api_key=SecretRef(
            name="TELNYX_API_KEY", purpose="the Telnyx v2 API key for the benchmark account"
        ),
        mock_secret=SecretRef(
            name="MOCK_TOOLS_SECRET",
            purpose="the shared secret the mock tool endpoint requires on X-Mock-Tools-Key",
        ),
        dial_target=SecretRef(
            name="TELNYX_DENTAL_DIAL_TARGET",
            purpose="the sip:...@<sub>.sip.telnyx.com URI Coval dials; names the subdomain",
        ),
    ),
    PlatformAgentSpec(
        key="retell-dental",
        platform="retell",
        suite="dental",
        agent_id=SecretRef(
            name="RETELL_DENTAL_AGENT_ID",
            purpose="the Retell agent id for the dental suite; its LLM is found from the agent",
        ),
        api_key=SecretRef(
            name="RETELL_API_KEY", purpose="the Retell API key for the benchmark workspace"
        ),
        mock_secret=SecretRef(
            name="MOCK_TOOLS_SECRET",
            purpose="the shared secret the mock tool endpoint requires on X-Mock-Tools-Key",
        ),
        dial_target=SecretRef(
            name="RETELL_DENTAL_DIAL_TARGET",
            purpose="the sip:+E164@sip.retellai.com URI Coval dials; names the imported number",
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
    """Every managed path and the value the contract says it should hold."""
    platform = platform_for(spec)
    stack = load_stack()
    definitions = json.loads(read_contract_file(spec.suite, "tool-definitions.json"))
    wanted: dict[str, Any] = {path: pin(stack) for path, pin in platform.pins.items()}
    wanted[platform.tools_path] = platform.render_tools(
        definitions, mock_base_url, spec.mock_secret.resolve()
    )
    return wanted


def plan(
    live: dict[str, Any], wanted: dict[str, Any], canon: Mapping[str, Canon] | None = None
) -> Plan:
    """Compare each managed path, after the platform's canonicaliser strips what the vendor adds."""
    result = Plan()
    for path, value in wanted.items():
        current = _get_path(live, path)
        if canon and path in canon:
            current = canon[path](current)
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
    agent_id = spec.agent_id.resolve()
    live = client.get_agent(agent_id)
    result = plan(live, wanted, platform.canon)
    result.prepared = platform.prepare(client, spec, dry_run) if platform.prepare else []
    if result.update and not dry_run:
        client.update_agent(agent_id, patch_body(live, {p: wanted[p] for p in result.update}))
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
        payload = self._request(
            "GET",
            "/agents",
            params={"filter": f'customer_agent_id="{customer_agent_id}"', "page_size": 1},
        )
        for agent in payload.get("agents") or []:
            if isinstance(agent, dict) and agent.get("customer_agent_id") == customer_agent_id:
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


COVAL_MANAGED = ("phone_number", "prompt", "metadata", "attributes", "display_name", "tags")


def published_contract_sha256(suite: str) -> str:
    """The hash a run or agent may be labelled with; refuses when the fixtures are unreadable here.

    ``contract_sha256`` falls back to the public files alone, which would label the
    run with a number that does not describe the seeded world the mock answered from.
    """
    if not has_private_contract(suite):
        raise SyncError(
            f"no mock fixtures for {suite!r} here (checkout or MOCK_FIXTURES_BUCKET); "
            "the contract hash would omit the seeded world"
        )
    return contract_sha256(suite)


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
            "contract_sha256": published_contract_sha256(spec.suite),
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
    if live.get("model_type") != COVAL_MODEL_TYPE:
        raise SyncError(
            f"coval agent {live.get('id')} is {live.get('model_type')!r}, not {COVAL_MODEL_TYPE}; "
            "model_type cannot be patched, so this record is not ours to reconcile"
        )
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
    test_case_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    sha = published_contract_sha256(spec.suite)
    options: dict[str, Any] = {"iteration_count": 1, "concurrency": 1}
    if test_case_ids:
        options["test_case_ids"] = list(test_case_ids)
    else:
        options["sub_sample_size"] = sample
        options["sub_sample_seed"] = seed
    return {
        "agent_id": agent_id,
        "persona_id": persona_id,
        "test_set_id": test_set_id,
        "metric_ids": list(metric_ids),
        "options": options,
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
    codec = codec_for(spec.platform)
    definitions = json.loads(read_contract_file(spec.suite, "tool-definitions.json"))
    if tool not in {d["name"] for d in definitions}:
        raise click.ClickException(f"{tool!r} is not in the {spec.suite} contract")
    call_args = dict(item.split("=", 1) for item in args)
    simulation_id = f"smoke-{spec.key}-{int(time.time())}"
    request = codec.encode_request(
        tool, call_args, Correlation(simulation_id, "+15550100000", source="smoke")
    )
    headers = {**request.headers, "X-Mock-Tools-Key": spec.mock_secret.resolve()}
    url = f"{mock_base_url.rstrip('/')}{request.path}"
    response = httpx.post(url, headers=headers, json=request.body, timeout=_TIMEOUT)
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
    install_fixture_providers(Settings())
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
@click.option(
    "--test-case",
    "test_case_ids",
    multiple=True,
    help="Run exactly these test case ids instead of a sample.",
)
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
    test_case_ids: tuple[str, ...],
    wait: bool,
    allow_drift: bool,
) -> None:
    """Launch one Coval run against this variant, once both the platform and Coval agent match."""
    spec = spec_for(agent_key)
    install_fixture_providers(Settings())
    platform = platform_for(spec)
    with platform.client(spec.api_key.resolve(), api_base or platform.api_base) as client:
        pending = drift(client, spec, mock_base_url)
    with CovalClient(COVAL_API_KEY.resolve(), coval_api_base) as coval:
        agent_id, coval_plan = register(coval, spec, dry_run=True)
        if not agent_id:
            raise click.ClickException(f"no Coval agent for {agent_key}; run register first")
        pending += [f"coval:{path}" for path in sorted(coval_plan.update)]
        if pending:
            message = (
                f"{agent_key} drifts from the contract: {pending}; run apply and register first"
            )
            if not allow_drift:
                raise click.ClickException(message)
            click.echo(f"warning: {message}")
        run = coval.launch_run(
            launch_body(
                agent_id, spec, persona_id, test_set_id, metric_ids, sample, seed, test_case_ids
            )
        )
        run_id = str(run.get("run_id") or run.get("id"))
        click.echo(f"run_id: {run_id} status: {run.get('status')}")
        click.echo("open this run id in the Coval UI to watch the simulation")
        while wait and run.get("status") not in {"COMPLETED", "FAILED", "CANCELLED", "DELETED"}:
            time.sleep(30)
            run = coval.get_run(run_id)
            click.echo(f"status: {run.get('status')} progress: {run.get('progress')}")
