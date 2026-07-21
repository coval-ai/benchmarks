// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { getAggregates, getProviders } from "./client";
import type {
  AggregatesQueryParams,
  FetchOptions,
} from "./client";

export function aggregatesQueryOptions(params: AggregatesQueryParams) {
  return {
    queryKey: ["aggregates", params],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getAggregates(params, { signal } satisfies FetchOptions),
    // Toggling windows keeps the prior data up instead of flashing the skeleton.
    placeholderData: keepPreviousData,
  };
}

export function useAggregatesQuery(params: AggregatesQueryParams) {
  return useQuery(aggregatesQueryOptions(params));
}

export function useProvidersQuery() {
  return useQuery({
    queryKey: ["providers"],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getProviders({ signal } satisfies FetchOptions),
  });
}
