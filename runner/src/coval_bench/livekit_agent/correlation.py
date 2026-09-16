"""Recover the Coval simulation id and caller number from the SIP participant."""

from __future__ import annotations

from livekit import rtc

from coval_bench.mocktools.codecs import CALLER_HEADER, SIMULATION_HEADER, Correlation

PHONE_ATTRIBUTE = "sip.phoneNumber"


def _attribute(attributes: dict[str, str], *keys: str) -> str | None:
    lowered = {key.lower(): value for key, value in attributes.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value:
            return value
    return None


def from_attributes(attributes: dict[str, str]) -> Correlation:
    simulation_id = _attribute(attributes, f"sip.h.{SIMULATION_HEADER}")
    caller_number = _attribute(attributes, f"sip.h.{CALLER_HEADER}", PHONE_ATTRIBUTE)
    if simulation_id:
        return Correlation(simulation_id, caller_number, source="sip_header")
    if caller_number:
        return Correlation(None, caller_number, source="sip_phone")
    return Correlation()


def sip_participant(room: rtc.Room) -> rtc.RemoteParticipant | None:
    for participant in room.remote_participants.values():
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            return participant
    return next(iter(room.remote_participants.values()), None)
