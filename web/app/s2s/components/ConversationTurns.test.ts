// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import type { S2STurn } from "@/lib/audioSamples/s2sFeed";
import { activeTurnIndex } from "./ConversationTurns";

function turn(index: number, start?: number, end?: number): S2STurn {
  return { index, role: index % 2 === 0 ? "user" : "assistant", content: "x", start_offset: start, end_offset: end };
}

describe("activeTurnIndex", () => {
  const turns = [turn(0, 0, 2), turn(1, 2.5, 5), turn(2, 6, 9)];

  it("finds the turn holding the playhead", () => {
    expect(activeTurnIndex(turns, 1)).toBe(0);
    expect(activeTurnIndex(turns, 3)).toBe(1);
    expect(activeTurnIndex(turns, 7)).toBe(2);
  });

  it("keeps the last started turn while the playhead sits in a gap", () => {
    // 5.5s is past turn 1's end but before turn 2 starts — the pane should not
    // flicker back to nothing between utterances.
    expect(activeTurnIndex(turns, 5.5)).toBe(1);
  });

  it("is inactive before the first turn starts", () => {
    expect(activeTurnIndex([turn(0, 1.5, 2)], 0.5)).toBe(-1);
  });

  it("holds the final turn past the end of the audio", () => {
    expect(activeTurnIndex(turns, 99)).toBe(2);
  });

  it("treats a turn with no start as never active", () => {
    expect(activeTurnIndex([turn(0), turn(1)], 5)).toBe(-1);
  });

  it("still resolves when only some turns carry offsets", () => {
    expect(activeTurnIndex([turn(0), turn(1, 3)], 4)).toBe(1);
  });

  it("handles an open-ended turn (Coval omitted end_offset)", () => {
    expect(activeTurnIndex([turn(0, 1)], 500)).toBe(0);
  });
});
