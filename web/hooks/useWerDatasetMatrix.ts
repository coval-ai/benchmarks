// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useMemo, useRef } from "react";
import { useQueries } from "@tanstack/react-query";
import { aggregatesQueryOptions } from "@/lib/api/queries";
import { toModelKey } from "@/lib/utils/formatters";
import type { AggregatesQueryParams } from "@/lib/api/client";

// A radar polygon needs at least three axes; below this the chart can't draw.
export const MIN_RADAR_AXES = 3;

export function useWerDatasetMatrix(
  base: Pick<AggregatesQueryParams, "benchmark" | "window">,
  datasets: string[]
): { werByDataset: Map<string, Map<string, number>> | null; loading: boolean } {
  const results = useQueries({
    queries: datasets.map((dataset) =>
      aggregatesQueryOptions({ ...base, dataset })
    ),
  });

  const resultsKey = results
    .map((r) => `${r.dataUpdatedAt}:${r.isPlaceholderData}`)
    .join();

  // Each axis renders as soon as its own query lands — no waiting on the
  // slowest dataset. Placeholder (previous-window) rows are excluded so a
  // window switch can never mix old and new values across axes.
  const fresh = useMemo(() => {
    const matrix = new Map<string, Map<string, number>>();
    results.forEach((r, i) => {
      if (!r.data || r.isPlaceholderData) return;
      const byModel = new Map<string, number>();
      r.data.model_stats.forEach((s) => {
        if (s.metric_type !== "WER") return;
        byModel.set(toModelKey(s.provider, s.model), s.avg_value);
      });
      if (byModel.size > 0) matrix.set(datasets[i]!, byModel);
    });
    return matrix.size > 0 ? matrix : null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasets, resultsKey]);

  const loading = results.some((r) => r.isPending || r.isPlaceholderData);

  // While a refetch is in flight, keep showing the previous window's matrix
  // (dimmed by the caller) until the fresh one has enough axes to draw a
  // radar — the chart never blanks, and no render ever mixes windows.
  // Render-phase ref write is idempotent by construction: it only ever
  // stores the current render's own derived value.
  const last = useRef(fresh);
  if ((fresh?.size ?? 0) >= MIN_RADAR_AXES || !loading) last.current = fresh;

  return { werByDataset: last.current, loading };
}
