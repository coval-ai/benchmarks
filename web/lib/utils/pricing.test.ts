// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { buildPricingMap, priceUnitShortLabel } from "./pricing";
import { formatUsd } from "./formatters";
import type { PricingEntry } from "@/lib/api/client";

const entry = (provider: string, model: string): PricingEntry => ({
  provider,
  model,
  normalized_usd: 6,
  basis: "list_price",
  native_rates: [{ billing_unit: "per_minute", rate_usd: 0.006 }],
  as_of: "2026-08-06",
  source_url: "https://example.com/pricing",
  history: [],
});

describe("buildPricingMap", () => {
  it("keys entries by the app-wide provider:model composite", () => {
    const map = buildPricingMap([entry("deepgram", "nova-3")]);
    expect(map.get("deepgram:nova-3")?.normalized_usd).toBe(6);
    expect(map.size).toBe(1);
  });

  it("returns an empty map for undefined input", () => {
    expect(buildPricingMap(undefined).size).toBe(0);
  });
});

describe("priceUnitShortLabel", () => {
  it("uses chars for TTS and minutes elsewhere", () => {
    expect(priceUnitShortLabel("tts")).toBe("$/1M chars");
    expect(priceUnitShortLabel("stt")).toBe("$/1k min");
    expect(priceUnitShortLabel("s2s")).toBe("$/1k min");
  });
});

describe("formatUsd", () => {
  it("keeps sensible significant figures with no trailing noise", () => {
    expect(formatUsd(0.42)).toBe("$0.42");
    expect(formatUsd(12.5)).toBe("$12.50");
    expect(formatUsd(1240)).toBe("$1,240");
    expect(formatUsd(0.0048)).toBe("$0.0048");
    expect(formatUsd(0.6)).toBe("$0.60");
    expect(formatUsd(6)).toBe("$6.00");
    expect(formatUsd(0)).toBe("$0");
  });
});
