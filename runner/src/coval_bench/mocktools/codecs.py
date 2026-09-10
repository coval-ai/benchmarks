# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import structlog

from coval_bench.mocktools.dispatch import Outcome

logger = structlog.get_logger("coval_bench.mocktools")

__all__ = [
    "CODECS",
    "MAX_TOOL_CALLS",
    "PRESET_PREFIX",
    "Codec",
    "Correlation",
    "Request",
    "ToolCall",
    "codec_for",
]

SIMULATION_HEADER = "x-coval-simulation-id"
CALLER_HEADER = "x-coval-caller-number"
TELNYX_CALL_HEADER = "x-telnyx-call-control-id"
PRESET_PREFIX = "_coval_"
PRESET_SIMULATION = f"{PRESET_PREFIX}simulation_id"
PRESET_CALLER = f"{PRESET_PREFIX}caller_number"
MAX_TOOL_CALLS = 32

Body = dict[str, Any] | None


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class Correlation:
    simulation_id: str | None = None
    caller_number: str | None = None
    source: str = "none"


@dataclass(frozen=True)
class Request:
    """One tool call as the platform would put it on the wire, minus the shared secret."""

    path: str
    headers: dict[str, str]
    body: Any


@dataclass(frozen=True)
class Codec:
    name: str
    tool_in_path: bool
    decode: Callable[[Body, Mapping[str, str], str | None], list[ToolCall]]
    correlate: Callable[[Body, Mapping[str, str]], Correlation]
    encode: Callable[[list[ToolCall], list[Outcome]], tuple[Any, int]]
    encode_request: Callable[[str, dict[str, Any], Correlation], Request]


def _coval_header_values(correlation: Correlation) -> dict[str, str]:
    headers: dict[str, str] = {}
    if correlation.simulation_id:
        headers[SIMULATION_HEADER] = correlation.simulation_id
    if correlation.caller_number:
        headers[CALLER_HEADER] = correlation.caller_number
    return headers


def _coval_headers(headers: Mapping[str, str]) -> Correlation:
    simulation_id = headers.get(SIMULATION_HEADER)
    caller_number = headers.get(CALLER_HEADER)
    if simulation_id:
        return Correlation(simulation_id, caller_number, source="simulation_header")
    if caller_number:
        return Correlation(None, caller_number, source="caller_header")
    return Correlation()


def _as_args(body: object) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _generic_decode(body: Body, _headers: Mapping[str, str], tool: str | None) -> list[ToolCall]:
    return [ToolCall(tool=tool or "", args=_as_args(body))]


def _generic_encode(_calls: list[ToolCall], outcomes: list[Outcome]) -> tuple[Any, int]:
    return outcomes[0].response, 200


def _generic_request(tool: str, args: dict[str, Any], correlation: Correlation) -> Request:
    return Request(f"/mock/generic/{tool}", _coval_header_values(correlation), dict(args))


GENERIC = Codec(
    name="generic",
    tool_in_path=True,
    decode=_generic_decode,
    correlate=lambda _body, headers: _coval_headers(headers),
    encode=_generic_encode,
    encode_request=_generic_request,
)


def _telnyx_decode(body: Body, _headers: Mapping[str, str], tool: str | None) -> list[ToolCall]:
    args = {k: v for k, v in _as_args(body).items() if not k.startswith(PRESET_PREFIX)}
    return [ToolCall(tool=tool or "", args=args)]


def _rendered(value: object) -> str | None:
    if not isinstance(value, str) or not value or "{{" in value:
        return None
    return value


def _telnyx_correlate(body: Body, headers: Mapping[str, str]) -> Correlation:
    found = _coval_headers(headers)
    if found.source != "none":
        return found
    preset = _as_args(body)
    simulation_id = _rendered(preset.get(PRESET_SIMULATION))
    caller_number = _rendered(preset.get(PRESET_CALLER))
    if simulation_id:
        return Correlation(simulation_id, caller_number, source="telnyx_preset_body")
    if caller_number:
        return Correlation(None, caller_number, source="telnyx_preset_caller")
    if headers.get(TELNYX_CALL_HEADER):
        return Correlation(source="telnyx_call_control_id")
    return Correlation()


def _telnyx_request(tool: str, args: dict[str, Any], correlation: Correlation) -> Request:
    body = dict(args)
    if correlation.simulation_id:
        body[PRESET_SIMULATION] = correlation.simulation_id
    if correlation.caller_number:
        body[PRESET_CALLER] = correlation.caller_number
    return Request(f"/mock/telnyx/{tool}", {}, body)


TELNYX = Codec(
    name="telnyx",
    tool_in_path=True,
    decode=_telnyx_decode,
    correlate=_telnyx_correlate,
    encode=_generic_encode,
    encode_request=_telnyx_request,
)


def _vapi_arguments(raw: object) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
    return _as_args(raw)


def _vapi_decode(body: Body, _headers: Mapping[str, str], _tool: str | None) -> list[ToolCall]:
    message = body.get("message") if body else None
    raw_calls = message.get("toolCallList") if isinstance(message, dict) else None
    if not isinstance(raw_calls, list):
        return []
    calls: list[ToolCall] = []
    for entry in raw_calls[:MAX_TOOL_CALLS]:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if not isinstance(function, dict):
            function = {}
        raw = entry.get("arguments", entry.get("parameters", function.get("arguments")))
        calls.append(
            ToolCall(
                tool=str(entry.get("name") or function.get("name") or ""),
                args=_vapi_arguments(raw),
                call_id=str(entry["id"]) if entry.get("id") else None,
            )
        )
    return calls


def _vapi_correlate(body: Body, headers: Mapping[str, str]) -> Correlation:
    found = _coval_headers(headers)
    if found.source != "none":
        return found
    message = body.get("message") if body else None
    call = message.get("call") if isinstance(message, dict) else None
    logger.warning(
        "mock_tool_call_uncorrelated",
        codec="vapi",
        message_keys=sorted(message) if isinstance(message, dict) else None,
        call_keys=sorted(call) if isinstance(call, dict) else None,
    )
    return Correlation()


def _vapi_encode(calls: list[ToolCall], outcomes: list[Outcome]) -> tuple[Any, int]:
    return {
        "results": [
            {
                "name": call.tool,
                "toolCallId": call.call_id,
                "result": json.dumps(outcome.response, separators=(",", ":")),
            }
            for call, outcome in zip(calls, outcomes, strict=True)
        ]
    }, 200


def _vapi_request(tool: str, args: dict[str, Any], correlation: Correlation) -> Request:
    call_id = f"{correlation.simulation_id or 'call'}-1"
    body = {
        "message": {
            "type": "tool-calls",
            "toolCallList": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(args)},
                }
            ],
        }
    }
    return Request("/mock/vapi", _coval_header_values(correlation), body)


VAPI = Codec(
    name="vapi",
    tool_in_path=False,
    decode=_vapi_decode,
    correlate=_vapi_correlate,
    encode=_vapi_encode,
    encode_request=_vapi_request,
)


CODECS: dict[str, Codec] = {
    GENERIC.name: GENERIC,
    TELNYX.name: TELNYX,
    VAPI.name: VAPI,
}


def codec_for(platform: str) -> Codec:
    codec = CODECS.get(platform)
    if codec is None:
        known = ", ".join(sorted(CODECS))
        raise KeyError(f"unknown platform {platform!r}; known: {known}")
    return codec
