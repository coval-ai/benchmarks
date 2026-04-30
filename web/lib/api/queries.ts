"use client";

import { useQuery } from "@tanstack/react-query";
import {
  getResults,
  getProviders,
  getRuns,
  getLeaderboard,
} from "./client";
import type {
  ResultsQueryParams,
  FetchOptions,
} from "./client";
import type { paths } from "./generated/schema";

type RunsQueryParams = NonNullable<
  paths["/v1/runs"]["get"]["parameters"]["query"]
>;

type LeaderboardQueryParams = NonNullable<
  paths["/v1/leaderboard"]["get"]["parameters"]["query"]
>;

export function useResultsQuery(params: ResultsQueryParams) {
  return useQuery({
    queryKey: ["results", params],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getResults(params, { signal } satisfies FetchOptions),
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
