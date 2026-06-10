// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  getAggregates,
  getProviders,
  getRuns,
  getLeaderboard,
} from "./client";
import type {
  AggregatesQueryParams,
  FetchOptions,
} from "./client";
import type { paths } from "./generated/schema";

type RunsQueryParams = NonNullable<
  paths["/v1/runs"]["get"]["parameters"]["query"]
>;

type LeaderboardQueryParams = NonNullable<
  paths["/v1/leaderboard"]["get"]["parameters"]["query"]
>;

export function useAggregatesQuery(params: AggregatesQueryParams) {
  return useQuery({
    queryKey: ["aggregates", params],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getAggregates(params, { signal } satisfies FetchOptions),
    // Toggling windows keeps the prior data up instead of flashing the skeleton.
    placeholderData: keepPreviousData,
  });
}

export function useProvidersQuery() {
  return useQuery({
    queryKey: ["providers"],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getProviders({ signal } satisfies FetchOptions),
  });
}

export function useRunsQuery(params?: RunsQueryParams) {
  return useQuery({
    queryKey: ["runs", params],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getRuns(params, { signal } satisfies FetchOptions),
  });
}

export function useLeaderboardQuery(params: LeaderboardQueryParams) {
  return useQuery({
    queryKey: ["leaderboard", params],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getLeaderboard(params, { signal } satisfies FetchOptions),
  });
}
