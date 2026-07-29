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

export function werBreakdownOf(stat: ModelStatEntry): WerBreakdown | undefined {
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
