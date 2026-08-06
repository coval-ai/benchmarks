// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useMemo } from "react";
import { keepPreviousData, useQueries, useQuery } from "@tanstack/react-query";
import {
  getAggregates,
  getAggregatesByDataset,
  getPricing,
  getProviders,
  getRuns,
  getS2SSample,
  getS2SSampleAudio,
  getS2SSampleIds,
} from "./client";
import type {
  AggregatesByDatasetQueryParams,
  AggregatesQueryParams,
  FetchOptions,
  S2SSampleRecording,
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

export function useAggregatesByDatasetQuery(
  params: AggregatesByDatasetQueryParams
) {
  return useQuery({
    queryKey: ["aggregates-by-dataset", params],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getAggregatesByDataset(params, { signal } satisfies FetchOptions),
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

// Prices change on the weekly collector cadence at most; an hour of staleness
// is invisible while sparing every navigation a refetch.
const PRICING_STALE_MS = 60 * 60 * 1000;

export function usePricingQuery(benchmark: "STT" | "TTS" | "S2S") {
  return useQuery({
    queryKey: ["pricing", benchmark],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getPricing(benchmark, { signal } satisfies FetchOptions),
    staleTime: PRICING_STALE_MS,
  });
}

export function useRunsQuery() {
  return useQuery({
    queryKey: ["runs"],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getRuns({ signal } satisfies FetchOptions),
  });
}

// Shortest interval a new sample can appear: the runner's S2S fetch period
// (s2s_fetch_period_seconds in config.py, default 10_800 = 3h).
const S2S_FETCH_PERIOD_MS = 10_800 * 1000;

// Signed audio URLs are minted for AUDIO_URL_TTL (10 minutes). Treated as stale
// before that so a returning tab replaces one, but deliberately NOT on a timer: a
// URL has to be valid at the instant of a click, and the click can never wait for
// one — a browser refuses to start playback that begins after an await.
const S2S_AUDIO_URL_STALE_MS = 8 * 60 * 1000;

export function useS2SSampleIdsQuery(enabled = true) {
  return useQuery({
    queryKey: ["s2s-samples", "index"],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getS2SSampleIds({ signal } satisfies FetchOptions),
    enabled,
    staleTime: S2S_FETCH_PERIOD_MS,
    refetchInterval: S2S_FETCH_PERIOD_MS,
  });
}

export function useS2SSampleQuery(sampleId: string | null) {
  return useQuery({
    queryKey: ["s2s-samples", "manifest", sampleId],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getS2SSample(sampleId!, { signal } satisfies FetchOptions),
    enabled: sampleId != null,
    // Swapping days keeps the prior sample up instead of flashing the skeleton.
    placeholderData: keepPreviousData,
    staleTime: S2S_FETCH_PERIOD_MS,
    refetchInterval: S2S_FETCH_PERIOD_MS,
  });
}

/**
 * A signed URL per recording, positionally aligned with `recordings`.
 *
 * `urls` is rebuilt only when a URL actually changes, not on every render: it
 * feeds the shared audio element's track list, and a fresh array each render would
 * re-fire the playback effects that watch it.
 */
export function useS2SSampleAudioUrls(recordings: readonly S2SSampleRecording[]): {
  urls: readonly string[];
  isPending: boolean;
  isError: boolean;
} {
  const results = useQueries({
    queries: recordings.map((recording) => ({
      queryKey: ["s2s-samples", "audio", recording.audio_path],
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        getS2SSampleAudio(recording.audio_path, { signal } satisfies FetchOptions),
      staleTime: S2S_AUDIO_URL_STALE_MS,
      // Overrides the app-wide default: a tab left in the background outlives its
      // URL, and refreshing on return replaces it before anyone presses play. A
      // periodic refetch would instead swap URLs during playback.
      refetchOnWindowFocus: true,
    })),
  });

  // Newline-joined: a URL cannot contain one, so the round-trip is lossless and the
  // string doubles as the memo key.
  const joined = results.map((result) => result.data?.url ?? "").join("\n");
  const count = results.length;
  const urls = useMemo(() => (count === 0 ? [] : joined.split("\n")), [joined, count]);

  return {
    urls,
    isPending: results.some((result) => result.isPending),
    isError: results.some((result) => result.isError),
  };
}
