// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useQuery } from "@tanstack/react-query";

// Generic reader for a modality's public sample bucket. The fetch job publishes
// one folder per tick (keyed by the timeline bucket timestamp) with a
// manifest.json, next to a rolling index.json listing ticks newest-first. The
// bucket is public-read, so the dashboard reads it directly — no backend hop.
// Each modality (s2s/tts/stt) wraps this with its own bucket + manifest type.

const BUCKET_HOST = "https://storage.googleapis.com";

export interface SampleFeedConfig {
  // Query-key namespace so modalities don't collide, e.g. "s2s-samples".
  name: string;
  bucket: string;
  prefix: string;
  // Shortest interval a new tick can appear = the runner's fetch period for
  // this modality. Drives staleTime and a background refetch backstop.
  refetchMs: number;
}

export function createSampleFeed<TManifest>(config: SampleFeedConfig) {
  const root = `${BUCKET_HOST}/${config.bucket}`;
  const indexUrl = `${root}/${config.prefix}/index.json`;
  const manifestUrl = (tick: string) =>
    `${root}/${config.prefix}/${encodeURIComponent(tick)}/manifest.json`;

  // Public URL for an object whose manifest path is bucket-root-relative.
  const objectUrl = (object: string): string => `${root}/${object}`;

  // Tick keys newest-first; [] before the first tick has published.
  const fetchIndex = async (signal?: AbortSignal): Promise<string[]> => {
    const res = await fetch(indexUrl, { signal, cache: "no-store" });
    if (res.status === 404) return [];
    if (!res.ok) throw new Error(`${config.name} index -> ${res.status}`);
    const data: unknown = await res.json();
    if (!Array.isArray(data) || !data.every((t) => typeof t === "string")) {
      throw new Error(`${config.name} index is not a list of tick strings`);
    }
    // Sort newest-first ourselves: backfill prepends older ticks out of order,
    // so index order isn't reliable. Tick keys are ISO, so lexical = chrono.
    return [...data].sort((a, b) => (a < b ? 1 : a > b ? -1 : 0));
  };

  // null when the tick has no manifest (e.g. a timeline bucket with no sample).
  const fetchManifest = async (
    tick: string,
    signal?: AbortSignal
  ): Promise<TManifest | null> => {
    const res = await fetch(manifestUrl(tick), { signal, cache: "no-store" });
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`${config.name} manifest ${tick} -> ${res.status}`);
    return (await res.json()) as TManifest;
  };

  const cadence = {
    staleTime: config.refetchMs,
    refetchInterval: config.refetchMs,
  } as const;

  const useIndexQuery = () =>
    useQuery({
      queryKey: [config.name, "index"],
      queryFn: ({ signal }: { signal: AbortSignal }) => fetchIndex(signal),
      ...cadence,
    });

  const useManifestQuery = (tick: string | null) =>
    useQuery({
      queryKey: [config.name, "manifest", tick],
      queryFn: ({ signal }: { signal: AbortSignal }) => fetchManifest(tick!, signal),
      enabled: tick != null,
      ...cadence,
    });

  return { objectUrl, fetchIndex, fetchManifest, useIndexQuery, useManifestQuery };
}
