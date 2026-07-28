// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import type { ModelStatEntry } from "@/lib/api/client";

// WER splits into insertions, deletions and substitutions. The API sends each
// as percentage points of the model's average WER, so the three sum back to the
// total — see runner migration 20260728_0013. A group whose rows predate that
// migration sends nulls, and every surface falls back to the total alone.
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
