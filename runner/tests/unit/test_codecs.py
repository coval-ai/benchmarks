# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
import structlog.testing

from coval_bench.mocktools.codecs import (
    CODECS,
    GENERIC,
    MAX_TOOL_CALLS,
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


def test_telnyx_shares_the_generic_wire_shape() -> None:
    assert TELNYX.decode is GENERIC.decode
    assert TELNYX.encode is GENERIC.encode
    assert TELNYX.tool_in_path


def test_telnyx_prefers_the_coval_header() -> None:
    found = TELNYX.correlate(
        {}, {"x-coval-simulation-id": SIM, "x-telnyx-call-control-id": "v3:abc"}
    )
    assert (found.simulation_id, found.source) == (SIM, "simulation_header")


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


def test_vapi_treats_non_object_arguments_as_none() -> None:
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
    assert payload == {
        "results": [
            {"toolCallId": "tc_1", "result": FOUND.response},
            {"toolCallId": "tc_2", "result": UNKNOWN.response},
        ]
    }


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


def test_vapi_without_a_header_logs_the_envelope_shape_and_not_its_values() -> None:
    body = _vapi_body(
        {"id": "tc_1", "name": "lookup_patient", "arguments": {"phone": PHONE}},
        call={"id": "call_9", "customer": {"number": PHONE}},
    )
    with structlog.testing.capture_logs() as captured:
        found = VAPI.correlate(body, NO_HEADERS)
    assert found.source == "none"
    events = [e for e in captured if e["event"] == "mock_tool_call_uncorrelated"]
    assert len(events) == 1
    assert events[0]["message_keys"] == ["call", "toolCallList", "type"]
    assert events[0]["call_keys"] == ["customer", "id"]
    assert PHONE not in repr(events[0])
    assert "call_9" not in repr(events[0])


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
