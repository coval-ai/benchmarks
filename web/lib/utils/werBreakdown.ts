// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import type { ModelStatEntry } from "@/lib/api/client";

// Percentage points of the model's average WER; the three sum back to the
// total. Null (pre-migration rows) means fall back to the total alone.
export interface WerBreakdown {
  substitutions: number;
  deletions: number;
  insertions: number;
}

// Substitutions first: it dominates in practice, so the list reads largest-first.
export const WER_BREAKDOWN_LABELS: [keyof WerBreakdown, string][] = [
  ["substitutions", "Substitutions"],
  ["deletions", "Deletions"],
  ["insertions", "Insertions"],
];

// Declared here (not read off ModelStatEntry) so the web build never depends
// on the deployed API's schema: Vercel regenerates schema.ts from the live
// openapi.json at build time, and an API that doesn't serve the split yet
// must degrade the tooltip, not fail the build.
export type WerBreakdownFields = {
  wer_substitutions_pct?: number | null;
  wer_deletions_pct?: number | null;
  wer_insertions_pct?: number | null;
};

export function werBreakdownOf(
  stat: ModelStatEntry & WerBreakdownFields
): WerBreakdown | undefined {
  const { wer_substitutions_pct, wer_deletions_pct, wer_insertions_pct } = stat;
  if (
    wer_substitutions_pct == null ||
    wer_deletions_pct == null ||
    wer_insertions_pct == null
  )
    return undefined;
  return {
    substitutions: wer_substitutions_pct,
    deletions: wer_deletions_pct,
    insertions: wer_insertions_pct,
  };
}
