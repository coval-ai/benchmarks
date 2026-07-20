// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useMemo, useRef } from "react";
import { useQueries } from "@tanstack/react-query";
import { aggregatesQueryOptions } from "@/lib/api/queries";
import { toModelKey } from "@/lib/utils/formatters";
import type { AggregatesQueryParams } from "@/lib/api/client";

export function useWerDatasetMatrix(
  base: Pick<AggregatesQueryParams, "benchmark" | "window">,
  datasets: string[]
): { werByDataset: Map<string, Map<string, number>> | null; loading: boolean } {
  const results = useQueries({
    queries: datasets.map((dataset) =>
      aggregatesQueryOptions({ ...base, dataset })
    ),
  });

  // Settled = every query holds fresh data for the current params or ended in
  // a terminal error. The matrix only rebuilds on that boundary, so axes never
  // pop in one by one on first paint and a window switch never mixes
  // new-window values on fast axes with old-window values on slow ones — the
  // previous complete snapshot stays up (dimmed via `loading`) until the new
  // one lands whole.
  const settled = results.every(
    (r) => (r.data && !r.isPlaceholderData) || r.isError
  );
  const lastMatrixRef = useRef<Map<string, Map<string, number>> | null>(null);

  const werByDataset = useMemo(() => {
    if (!settled) return lastMatrixRef.current;
    const matrix = new Map<string, Map<string, number>>();
    results.forEach((r, i) => {
      if (!r.data) return;
      const byModel = new Map<string, number>();
      r.data.model_stats.forEach((s) => {
        if (s.metric_type !== "WER") return;
        byModel.set(toModelKey(s.provider, s.model), s.avg_value);
      });
      if (byModel.size > 0) matrix.set(datasets[i]!, byModel);
    });
    lastMatrixRef.current = matrix.size > 0 ? matrix : null;
    return lastMatrixRef.current;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settled, datasets, results.map((r) => r.dataUpdatedAt).join()]);

  return { werByDataset, loading: !settled };
}
