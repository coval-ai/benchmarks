// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { WER_BREAKDOWN_LABELS, werBreakdownOf } from "./werBreakdown";
import type { ModelStatEntry } from "../api/client";

const stat = (over: Partial<ModelStatEntry> = {}): ModelStatEntry =>
  ({
    provider: "deepgram",
    model: "nova-3",
    metric_type: "WER",
    avg_value: 6,
    stddev_value: 0,
    p25: 0,
    p50: 0,
    p75: 0,
    p90: 0,
    p95: 0,
    p99: 0,
    min_value: 0,
    max_value: 0,
    sample_count: 1,
    ...over,
  }) as ModelStatEntry;

describe("werBreakdownOf", () => {
  it("maps the three components and reconciles to avg_value", () => {
    const breakdown = werBreakdownOf(
      stat({
        wer_substitutions_pct: 3,
        wer_deletions_pct: 2,
        wer_insertions_pct: 1,
      })
    );
    expect(breakdown).toEqual({ substitutions: 3, deletions: 2, insertions: 1 });
    const total =
      breakdown!.substitutions + breakdown!.deletions + breakdown!.insertions;
    expect(total).toBeCloseTo(6);
  });

  it("keeps a genuine zero component rather than dropping the breakdown", () => {
    expect(
      werBreakdownOf(
        stat({
          wer_substitutions_pct: 6,
          wer_deletions_pct: 0,
          wer_insertions_pct: 0,
        })
      )
    ).toEqual({ substitutions: 6, deletions: 0, insertions: 0 });
  });

  it("is undefined when the API reports no split", () => {
    expect(werBreakdownOf(stat())).toBeUndefined();
  });

  it("is undefined when only part of the split is present", () => {
    expect(
      werBreakdownOf(stat({ wer_substitutions_pct: 3, wer_deletions_pct: 2 }))
    ).toBeUndefined();
  });

  it("labels every component exactly once", () => {
    expect(WER_BREAKDOWN_LABELS.map(([key]) => key)).toEqual([
      "substitutions",
      "deletions",
      "insertions",
    ]);
  });
});
