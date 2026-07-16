// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";
import { useAggregatesQuery } from "@/lib/api/queries";
import { toModelKey } from "@/lib/utils/formatters";
import type { AggregatesQueryParams, ModelStatEntry } from "@/lib/api/client";

// A null datasetId is "pooled": the query dedupes onto the main aggregates
// fetch and werByModel stays null so callers fall back to their pooled data.
export function useDatasetScopedWer(
  base: Pick<AggregatesQueryParams, "benchmark" | "window">,
  datasetId: string | null
): { werByModel: Map<string, ModelStatEntry> | null; loading: boolean } {
  const query = useAggregatesQuery({
    ...base,
    ...(datasetId ? { dataset: datasetId } : {}),
  });

  const werByModel = useMemo(() => {
    if (!datasetId || (query.isError && !query.data)) return null;
    const map = new Map<string, ModelStatEntry>();
    (query.data?.model_stats ?? []).forEach((s) => {
      if (s.metric_type !== "WER") return;
      map.set(toModelKey(s.provider, s.model), s);
    });
    return map;
  }, [datasetId, query.data, query.isError]);

  const loading =
    datasetId !== null && (query.isLoading || query.isPlaceholderData);

  return { werByModel, loading };
}
