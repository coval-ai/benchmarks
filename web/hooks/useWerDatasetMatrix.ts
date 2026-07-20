// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useMemo } from "react";
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

  const werByDataset = useMemo(() => {
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
    return matrix.size > 0 ? matrix : null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasets, results.map((r) => r.dataUpdatedAt).join()]);

  const loading = results.some((r) => r.isPending || r.isPlaceholderData);

  return { werByDataset, loading };
}
