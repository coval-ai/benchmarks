// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useMemo } from "react";
import { useAggregatesByDatasetQuery } from "@/lib/api/queries";
import { toModelKey } from "@/lib/utils/formatters";
import type { AggregatesByDatasetQueryParams } from "@/lib/api/client";

// A radar polygon needs at least three axes; below this the chart can't draw.
export const MIN_RADAR_AXES = 3;

export function useWerDatasetMatrix(
  base: Pick<AggregatesByDatasetQueryParams, "benchmark" | "window">
): { werByDataset: Map<string, Map<string, number>> | null; loading: boolean } {
  const query = useAggregatesByDatasetQuery(base);

  // One batched response carries every dataset, so no render can mix windows
  // across axes. While a window switch is in flight the previous window's
  // response is served as placeholder data (dimmed by the caller), so the
  // chart never blanks.
  const werByDataset = useMemo(() => {
    const matrix = new Map<string, Map<string, number>>();
    query.data?.blocks.forEach((block) => {
      const byModel = new Map<string, number>();
      block.model_stats.forEach((s) => {
        if (s.metric_type !== "WER") return;
        byModel.set(toModelKey(s.provider, s.model), s.avg_value);
      });
      if (byModel.size > 0) matrix.set(block.dataset, byModel);
    });
    return matrix.size > 0 ? matrix : null;
  }, [query.data]);

  return {
    werByDataset,
    loading: query.isPending || query.isPlaceholderData,
  };
}
