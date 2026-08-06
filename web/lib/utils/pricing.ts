// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import type { PricingEntry } from "@/lib/api/client";
import { toModelKey } from "@/lib/utils/formatters";

/** O(1) price joins for chart/table hooks, keyed the app-wide "provider:model" way. */
export function buildPricingMap(
  entries: readonly PricingEntry[] | undefined
): Map<string, PricingEntry> {
  const map = new Map<string, PricingEntry>();
  for (const entry of entries ?? []) {
    map.set(toModelKey(entry.provider, entry.model), entry);
  }
  return map;
}

/** Column/axis label for the benchmark's display unit (AA convention). */
export function priceUnitShortLabel(tab: "tts" | "stt" | "s2s"): string {
  return tab === "tts" ? "$/1M chars" : "$/1k min";
}

/** Human wording of a native billing unit for tooltips. */
export function billingUnitLabel(unit: string): string {
  const labels: Record<string, string> = {
    per_minute: "per minute",
    per_second: "per second",
    per_hour: "per hour",
    per_1m_chars: "per 1M characters",
    per_1m_tokens_input: "per 1M input tokens",
    per_1m_tokens_output: "per 1M output tokens",
    per_request: "per request",
  };
  return labels[unit] ?? unit;
}
