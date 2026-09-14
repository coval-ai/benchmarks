"""Unit tests for coval_bench.runner.gate."""

from __future__ import annotations

import asyncio

import pytest

from coval_bench.runner.gate import ModelGate


class _Tracker:
    def __init__(self) -> None:
        self.active: dict[object, int] = {}
        self.peak: dict[object, int] = {}
        self.total_active = 0
        self.total_peak = 0

    async def run(self, gate: ModelGate, key: object, hold_s: float = 0.01) -> None:
        async with gate.slot(key):
            self.active[key] = self.active.get(key, 0) + 1
            self.total_active += 1
            self.peak[key] = max(self.peak.get(key, 0), self.active[key])
            self.total_peak = max(self.total_peak, self.total_active)
            await asyncio.sleep(hold_s)
            self.active[key] -= 1
            self.total_active -= 1


@pytest.mark.asyncio
async def test_same_key_never_overlaps() -> None:
    gate = ModelGate(8)
    tracker = _Tracker()
    await asyncio.gather(*(tracker.run(gate, ("baseten", "whisper")) for _ in range(6)))
    assert tracker.peak[("baseten", "whisper")] == 1


@pytest.mark.asyncio
async def test_different_keys_run_side_by_side() -> None:
    gate = ModelGate(8)
    tracker = _Tracker()
    keys = [("baseten", "whisper"), ("baseten", "qwen3-asr"), ("deepgram", "flux")]
    await asyncio.gather(*(tracker.run(gate, key) for key in keys for _ in range(3)))
    assert tracker.total_peak == len(keys)
    assert all(tracker.peak[key] == 1 for key in keys)


@pytest.mark.asyncio
async def test_global_cap_bounds_distinct_keys() -> None:
    gate = ModelGate(2)
    tracker = _Tracker()
    await asyncio.gather(*(tracker.run(gate, ("p", f"m{i}")) for i in range(6)))
    assert tracker.total_peak == 2


@pytest.mark.asyncio
async def test_shared_slot_counts_against_cap_only() -> None:
    gate = ModelGate(1)
    order: list[str] = []

    async def side_work() -> None:
        async with gate.shared():
            order.append("shared-in")
            await asyncio.sleep(0.01)
            order.append("shared-out")

    async def request() -> None:
        async with gate.slot(("p", "m")):
            order.append("slot-in")
            order.append("slot-out")

    await asyncio.gather(side_work(), request())
    assert order == ["shared-in", "shared-out", "slot-in", "slot-out"]


def test_cap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="cap"):
        ModelGate(0)
