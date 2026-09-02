# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

import coval_bench.mocktools.codecs as codecs
from coval_bench.mocktools.codecs import (
    CODECS,
    GENERIC,
    MAX_TOOL_CALLS,
    PRESET_CALLER,
    PRESET_SIMULATION,
    TELNYX,
    VAPI,
    ToolCall,
    codec_for,
)
from coval_bench.mocktools.dispatch import Outcome

PHONE = "6025550182"
SIM = "sim_abc123"
NO_HEADERS: dict[str, str] = {}

FOUND = Outcome(response={"found": True, "patient_id": "P-1041"}, http_status=200)
UNKNOWN = Outcome(response={"error": "unknown_tool", "tool": "read_chart"}, http_status=404)


def _vapi_body(*entries: dict[str, Any], call: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"type": "tool-calls", "toolCallList": list(entries)}
    if call is not None:
        message["call"] = call
    return {"message": message}


# --- generic ---------------------------------------------------------------


def test_generic_decodes_bare_args_with_the_path_tool() -> None:
    calls = GENERIC.decode({"phone": PHONE}, NO_HEADERS, "lookup_patient")
    assert calls == [ToolCall(tool="lookup_patient", args={"phone": PHONE})]


def test_generic_decodes_an_empty_body_as_no_args() -> None:
    assert GENERIC.decode(None, NO_HEADERS, "lookup_patient")[0].args == {}


def test_generic_encodes_the_bare_response_at_200() -> None:
    payload, status = GENERIC.encode([ToolCall("lookup_patient", {})], [FOUND])
    assert (payload, status) == (FOUND.response, 200)


def test_generic_encodes_a_rejected_call_at_200_too() -> None:
    payload, status = GENERIC.encode([ToolCall("read_chart", {})], [UNKNOWN])
    assert status == 200
    assert payload["error"] == "unknown_tool"


def test_generic_correlates_from_the_coval_headers() -> None:
    found = GENERIC.correlate({}, {"x-coval-simulation-id": SIM, "x-coval-caller-number": PHONE})
    assert (found.simulation_id, found.caller_number, found.source) == (
        SIM,
        PHONE,
        "simulation_header",
    )


def test_generic_correlates_from_the_caller_alone() -> None:
    found = GENERIC.correlate({}, {"x-coval-caller-number": PHONE})
    assert (found.simulation_id, found.caller_number, found.source) == (
        None,
        PHONE,
        "caller_header",
    )


def test_generic_without_headers_is_uncorrelated() -> None:
    assert GENERIC.correlate({}, NO_HEADERS).source == "none"


# --- telnyx ----------------------------------------------------------------


def test_telnyx_shares_the_generic_response_shape_but_not_decode() -> None:
    assert TELNYX.decode is not GENERIC.decode
    assert TELNYX.encode is GENERIC.encode
    assert TELNYX.tool_in_path


def test_telnyx_strips_preset_body_fields_before_dispatch() -> None:
    body = {"phone": PHONE, PRESET_SIMULATION: SIM, PRESET_CALLER: "+14045550710"}
    (call,) = TELNYX.decode(body, NO_HEADERS, "lookup_patient")
    assert call == ToolCall(tool="lookup_patient", args={"phone": PHONE})


def test_telnyx_prefers_the_coval_header() -> None:
    found = TELNYX.correlate(
        {PRESET_SIMULATION: "other"},
        {"x-coval-simulation-id": SIM, "x-telnyx-call-control-id": "v3:abc"},
    )
    assert (found.simulation_id, found.source) == (SIM, "simulation_header")


def test_telnyx_correlates_from_the_preset_body() -> None:
    found = TELNYX.correlate({"phone": PHONE, PRESET_SIMULATION: SIM, PRESET_CALLER: "+1404"}, {})
    assert (found.simulation_id, found.caller_number, found.source) == (
        SIM,
        "+1404",
        "telnyx_preset_body",
    )


def test_telnyx_falls_back_to_the_preset_caller_alone() -> None:
    found = TELNYX.correlate({PRESET_CALLER: "+1404"}, {})
    assert (found.simulation_id, found.caller_number, found.source) == (
        None,
        "+1404",
        "telnyx_preset_caller",
    )


def test_telnyx_treats_an_unrendered_template_as_absent() -> None:
    found = TELNYX.correlate(
        {PRESET_SIMULATION: "{{coval_simulation_id}}", PRESET_CALLER: "+1404"}, {}
    )
    assert (found.simulation_id, found.source) == (None, "telnyx_preset_caller")


def test_telnyx_control_id_is_noted_but_never_stored_as_a_number() -> None:
    found = TELNYX.correlate({}, {"x-telnyx-call-control-id": "v3:abc"})
    assert found.source == "telnyx_call_control_id"
    assert found.caller_number is None
    assert found.simulation_id is None


# --- vapi decode -----------------------------------------------------------


def test_vapi_decodes_one_call() -> None:
    body = _vapi_body({"id": "tc_1", "name": "lookup_patient", "arguments": {"phone": PHONE}})
    assert VAPI.decode(body, NO_HEADERS, None) == [
        ToolCall(tool="lookup_patient", args={"phone": PHONE}, call_id="tc_1")
    ]


def test_vapi_accepts_parameters_as_the_arguments_key() -> None:
    body = _vapi_body({"id": "tc_1", "name": "lookup_patient", "parameters": {"phone": PHONE}})
    assert VAPI.decode(body, NO_HEADERS, None)[0].args == {"phone": PHONE}


def test_vapi_decodes_a_batch_in_order() -> None:
    body = _vapi_body(
        {"id": "tc_1", "name": "check_availability", "arguments": {"date": "2030-04-04"}},
        {"id": "tc_2", "name": "check_availability", "arguments": {"date": "2030-04-05"}},
    )
    calls = VAPI.decode(body, NO_HEADERS, None)
    assert [c.call_id for c in calls] == ["tc_1", "tc_2"]
    assert [c.args["date"] for c in calls] == ["2030-04-04", "2030-04-05"]


def test_vapi_keeps_a_call_with_no_id() -> None:
    body = _vapi_body({"name": "lookup_patient", "arguments": {"phone": PHONE}})
    assert VAPI.decode(body, NO_HEADERS, None)[0].call_id is None


def test_vapi_decodes_the_nested_function_shape_with_string_arguments() -> None:
    body = _vapi_body(
        {
            "id": "tc_1",
            "type": "function",
            "function": {"name": "lookup_patient", "arguments": json.dumps({"phone": PHONE})},
        }
    )
    assert VAPI.decode(body, NO_HEADERS, None) == [
        ToolCall(tool="lookup_patient", args={"phone": PHONE}, call_id="tc_1")
    ]


def test_vapi_parses_string_arguments_on_the_flat_shape_too() -> None:
    body = _vapi_body({"id": "tc_1", "name": "lookup_patient", "arguments": '{"phone": "1"}'})
    assert VAPI.decode(body, NO_HEADERS, None)[0].args == {"phone": "1"}


def test_vapi_treats_unparseable_arguments_as_none() -> None:
    body = _vapi_body({"id": "tc_1", "name": "lookup_patient", "arguments": "not json"})
    assert VAPI.decode(body, NO_HEADERS, None)[0].args == {}


def test_vapi_skips_entries_that_are_not_objects() -> None:
    body = _vapi_body({"id": "tc_1", "name": "lookup_patient", "arguments": {}})
    body["message"]["toolCallList"].insert(0, "garbage")
    calls = VAPI.decode(body, NO_HEADERS, None)
    assert [c.call_id for c in calls] == ["tc_1"]


@pytest.mark.parametrize(
    "body",
    [None, {}, {"message": {}}, {"message": {"toolCallList": "nope"}}, {"message": "x"}],
)
def test_vapi_decodes_a_missing_list_as_no_calls(body: dict[str, Any] | None) -> None:
    assert VAPI.decode(body, NO_HEADERS, None) == []


def test_vapi_caps_the_batch() -> None:
    entries = [
        {"id": f"tc_{i}", "name": "lookup_patient", "arguments": {}}
        for i in range(MAX_TOOL_CALLS + 8)
    ]
    assert len(VAPI.decode(_vapi_body(*entries), NO_HEADERS, None)) == MAX_TOOL_CALLS


# --- vapi encode -----------------------------------------------------------


def test_vapi_encodes_each_result_under_its_call_id_in_order() -> None:
    calls = [ToolCall("a", {}, "tc_1"), ToolCall("b", {}, "tc_2")]
    payload, status = VAPI.encode(calls, [FOUND, UNKNOWN])
    assert status == 200
    assert [r["toolCallId"] for r in payload["results"]] == ["tc_1", "tc_2"]
    assert [r["name"] for r in payload["results"]] == ["a", "b"]


def test_vapi_encodes_the_result_as_a_json_string() -> None:
    payload, _ = VAPI.encode([ToolCall("a", {}, "tc_1")], [FOUND])
    result = payload["results"][0]["result"]
    assert isinstance(result, str)
    assert json.loads(result) == FOUND.response


def test_vapi_encodes_an_error_inside_the_result_string() -> None:
    payload, _ = VAPI.encode([ToolCall("read_chart", {}, "tc_1")], [UNKNOWN])
    assert json.loads(payload["results"][0]["result"])["error"] == "unknown_tool"
    assert "error" not in payload["results"][0]


def test_vapi_encodes_a_missing_id_as_null() -> None:
    payload, _ = VAPI.encode([ToolCall("a", {})], [FOUND])
    assert payload["results"][0]["toolCallId"] is None


def test_vapi_encodes_an_empty_batch() -> None:
    assert VAPI.encode([], []) == ({"results": []}, 200)


def test_vapi_encode_refuses_a_cardinality_mismatch() -> None:
    with pytest.raises(ValueError):
        VAPI.encode([ToolCall("a", {}, "tc_1")], [FOUND, UNKNOWN])


# --- vapi correlate --------------------------------------------------------


def test_vapi_correlates_from_the_coval_header_when_present() -> None:
    found = VAPI.correlate(_vapi_body(), {"x-coval-simulation-id": SIM})
    assert (found.simulation_id, found.source) == (SIM, "simulation_header")


def test_vapi_without_a_header_logs_the_envelope_shape_and_not_its_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    monkeypatch.setattr(codecs, "logger", fake)
    body = _vapi_body(
        {"id": "tc_1", "name": "lookup_patient", "arguments": {"phone": PHONE}},
        call={"id": "call_9", "customer": {"number": PHONE}},
    )
    found = VAPI.correlate(body, NO_HEADERS)
    assert found.source == "none"
    fake.warning.assert_called_once()
    assert fake.warning.call_args.args == ("mock_tool_call_uncorrelated",)
    kwargs = fake.warning.call_args.kwargs
    assert kwargs["message_keys"] == ["call", "toolCallList", "type"]
    assert kwargs["call_keys"] == ["customer", "id"]
    assert PHONE not in repr(kwargs)
    assert "call_9" not in repr(kwargs)


# --- registry --------------------------------------------------------------


def test_codec_for_names_the_known_set_on_a_miss() -> None:
    with pytest.raises(KeyError, match="known: generic, telnyx, vapi"):
        codec_for("retell")


@pytest.mark.parametrize(
    ("name", "tool_in_path"),
    [("generic", True), ("telnyx", True), ("vapi", False)],
)
def test_each_codec_declares_where_the_tool_name_lives(name: str, tool_in_path: bool) -> None:
    assert codec_for(name).tool_in_path is tool_in_path
    assert CODECS[name].name == name
