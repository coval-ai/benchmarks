/**
 * Unit tests for lib/aggregates.ts.
 * Uses the codegen'd ResultOut type from the FastAPI OpenAPI schema.
 */

import { describe, it, expect } from "vitest";
import { computeModelStats, statsByKey, makeStatLookup } from "./aggregates";
import type { Result } from "./aggregates";

let nextId = 1;

function makeRow(overrides: Partial<Result> = {}): Result {
  const id = nextId++;
  return {
    id,
    run_id: 1,
    provider: "deepgram",
    model: "nova-3",
    voice: null,
    benchmark: "STT",
    metric_type: "WER",
    metric_value: 0.1,
    metric_units: "%",
    audio_filename: "test.wav",
    created_at: "2026-04-29T00:00:00Z",
    status: "SUCCEEDED",
    ...overrides,
  };
}

describe("computeModelStats", () => {
  it("happy path: 6 rows for (deepgram, nova-3, WER) → one group with correct stats", () => {
    const values = [1, 2, 3, 4, 5, 6];
    const rows: Result[] = values.map((v) => makeRow({ metric_value: v }));

    const stats = computeModelStats(rows);

    expect(stats).toHaveLength(1);
    const s = stats[0]!;
    expect(s.provider).toBe("deepgram");
    expect(s.model).toBe("nova-3");
    expect(s.metric_type).toBe("WER");
    expect(s.sample_count).toBe(6);
    // avg = 3.5
    expect(s.avg_value).toBeCloseTo(3.5, 10);
    // min/max
    expect(s.min_value).toBe(1);
    expect(s.max_value).toBe(6);
    // sample stddev of [1,2,3,4,5,6] = sqrt(17.5/5) = sqrt(3.5) ≈ 1.8708
    expect(s.stddev_value).toBeCloseTo(Math.sqrt(3.5), 5);
  });

  it("status filter: 3 SUCCEEDED + 2 FAILED → only 3 in sample_count", () => {
    const rows: Result[] = [
      makeRow({ metric_value: 1, status: "SUCCEEDED" }),
      makeRow({ metric_value: 2, status: "SUCCEEDED" }),
      makeRow({ metric_value: 3, status: "SUCCEEDED" }),
      makeRow({ metric_value: 99, status: "FAILED" }),
      makeRow({ metric_value: 99, status: "FAILED" }),
    ];

    const stats = computeModelStats(rows);

    expect(stats).toHaveLength(1);
    expect(stats[0]!.sample_count).toBe(3);
    expect(stats[0]!.max_value).toBe(3);
  });

  it("defensive: null metric_value rows are filtered out (in case the API contract slips)", () => {
    // ResultOut.metric_value is typed `number` post-response_model sweep, but the function
    // keeps its defensive null-skip. Cast through unknown to exercise the runtime guard.
    const rows = [
      makeRow({ metric_value: 5 }),
      makeRow({ metric_value: null as unknown as number }),
      makeRow({ metric_value: null as unknown as number }),
    ];

    const stats = computeModelStats(rows);

    expect(stats).toHaveLength(1);
    expect(stats[0]!.sample_count).toBe(1);
    expect(stats[0]!.avg_value).toBe(5);
  });

  it("n=1 group: stddev coerces to 0 (mirrors Postgres COALESCE(STDDEV(...), 0))", () => {
    const rows: Result[] = [makeRow({ metric_value: 42 })];

    const stats = computeModelStats(rows);

    expect(stats[0]!.sample_count).toBe(1);
    expect(stats[0]!.stddev_value).toBe(0);
  });

  it("multiple metric_types per model: each becomes its own row", () => {
    const rows: Result[] = [
      makeRow({ metric_type: "WER", metric_value: 0.1 }),
      makeRow({ metric_type: "WER", metric_value: 0.2 }),
      makeRow({ metric_type: "TTFA", metric_value: 300 }),
      makeRow({ metric_type: "TTFA", metric_value: 400 }),
    ];

    const stats = computeModelStats(rows);

    expect(stats).toHaveLength(2);
    const types = stats.map((s) => s.metric_type).sort();
    expect(types).toEqual(["TTFA", "WER"]);
  });

  it("empty input returns []", () => {
    const stats = computeModelStats([]);
    expect(stats).toEqual([]);
  });

  it("memoization: calling twice with same array reference returns same reference", () => {
    const rows: Result[] = [makeRow({ metric_value: 1 })];
    const first = computeModelStats(rows);
    const second = computeModelStats(rows);
    expect(first).toBe(second);
  });

  it("percentile correctness: [1,2,3,4,5] → p50=3, p25=2, p75=4 (matches Postgres percentile_cont)", () => {
    const rows: Result[] = [1, 2, 3, 4, 5].map((v) => makeRow({ metric_value: v }));

    const stats = computeModelStats(rows);

    expect(stats[0]!.p50).toBe(3);
    expect(stats[0]!.p25).toBe(2);
    expect(stats[0]!.p75).toBe(4);
  });

  it("stddev correctness: [2,4,4,4,5,5,7,9] → sample stddev ≈ 2.138 (matches Postgres STDDEV)", () => {
    // Classic textbook example. Sample stddev = sqrt(33/[n-1]) where n=8 → sqrt(33/7) ≈ 2.1380899...
    const values = [2, 4, 4, 4, 5, 5, 7, 9];
    const rows: Result[] = values.map((v) => makeRow({ metric_value: v }));

    const stats = computeModelStats(rows);

    // Postgres STDDEV (sample, n-1 denom). Manual: mean=5, sum of squared deviations=32,
    // sample variance=32/7, stddev=sqrt(32/7) ≈ 2.138089935...
    expect(stats[0]!.stddev_value).toBeCloseTo(Math.sqrt(32 / 7), 5);
  });
});

describe("statsByKey", () => {
  it("builds a map keyed by metric_type|provider|model", () => {
    const rows: Result[] = [makeRow({ metric_value: 1 })];
    const stats = computeModelStats(rows);
    const map = statsByKey(stats);
    expect(map.has("WER|deepgram|nova-3")).toBe(true);
  });
});

describe("makeStatLookup", () => {
  it("looks up with provider", () => {
    const rows: Result[] = [makeRow({ metric_value: 1 })];
    const stats = computeModelStats(rows);
    const lookup = makeStatLookup(stats);
    expect(lookup("nova-3", "WER", "deepgram")).toBeDefined();
    expect(lookup("nova-3", "WER", "google")).toBeUndefined();
  });

  it("fallback lookup without provider", () => {
    const rows: Result[] = [makeRow({ metric_value: 1 })];
    const stats = computeModelStats(rows);
    const lookup = makeStatLookup(stats);
    expect(lookup("nova-3", "WER")).toBeDefined();
    expect(lookup("nova-3", "TTFA")).toBeUndefined();
  });
});
