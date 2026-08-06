// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { spokenTurns, visibleRecordings } from "./s2sFeed";
import type { S2SSampleApiResponse } from "@/lib/api/client";

describe("spokenTurns", () => {
  it("drops turns that aren't spoken dialogue", () => {
    const turns = spokenTurns([
      { index: 0, role: "user", content: "hello" },
      { index: 1, role: "tool", content: '{"name":"end_conversation"}' },
      { index: 2, role: "assistant", content: "hi" },
    ]);

    expect(turns.map((t) => t.role)).toEqual(["user", "assistant"]);
  });

  it("collapses a null offset to undefined, so the turn never reads as active at 0", () => {
    const turns = spokenTurns([
      { index: 0, role: "user", content: "hello", start_offset: null },
    ]);

    expect(turns[0]?.start_offset).toBeUndefined();
  });
});

describe("visibleRecordings", () => {
  it("distinguishes two models of the same provider", () => {
    const sample = {
      sample_id: "2026-07-30T00:00:00Z",
      test_case_id: "tc-1",
      recordings: ["shown-model", "hidden-model"].map((model) => ({
        provider: "acme",
        model,
        audio_path: `/v1/s2s/samples/2026-07-30T00:00:00Z/acme/${model}/audio`,
        coval_run_id: "run-1",
        sim_id: "sim-1",
        turns: [],
      })),
    } satisfies S2SSampleApiResponse;

    const kept = visibleRecordings(sample, new Set(["acme:shown-model"]));

    expect(kept.map((r) => r.model)).toEqual(["shown-model"]);
  });
});
