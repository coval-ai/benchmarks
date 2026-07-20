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

// A real fetch failure, as distinct from a 404 (turned into empty index / null
// manifest). `kind`: network (fetch rejected — offline/DNS/CORS), http (bad
// status), parse (bad JSON) — so callers don't show a false "no sample" state.
export type SampleFetchErrorKind = "network" | "http" | "parse";

export class SampleFetchError extends Error {
  readonly kind: SampleFetchErrorKind;
  readonly status?: number;
  constructor(kind: SampleFetchErrorKind, message: string, status?: number) {
    super(message);
    this.name = "SampleFetchError";
    this.kind = kind;
    this.status = status;
  }
}

export interface SampleFeedConfig {
  // Query-key namespace so modalities don't collide, e.g. "s2s-samples".
  name: string;
  bucket: string;
  prefix: string;
  // Shortest interval a new tick can appear = the runner's fetch period for
  // this modality. Drives staleTime and a background refetch backstop.
  refetchMs: number;
}

// parseManifest validates+narrows the raw JSON into TManifest; it must throw on
// any unexpected shape so a malformed manifest surfaces as a fetch error rather
// than crashing a consumer that trusts the type.
export function createSampleFeed<TManifest>(
  config: SampleFeedConfig,
  parseManifest: (data: unknown) => TManifest
) {
  const root = `${BUCKET_HOST}/${config.bucket}`;
  const indexUrl = `${root}/${config.prefix}/index.json`;
  const manifestUrl = (tick: string) =>
    `${root}/${config.prefix}/${encodeURIComponent(tick)}/manifest.json`;

  // Public URL for an object whose manifest path is bucket-root-relative.
  const objectUrl = (object: string): string => `${root}/${object}`;

  // fetch() rejects on network/CORS failure; classify as network. Let an
  // aborted request (cancelled query / unmount) pass through untouched.
  const fetchOrThrow = async (
    url: string,
    label: string,
    signal?: AbortSignal
  ): Promise<Response> => {
    try {
      return await fetch(url, { signal, cache: "no-store" });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") throw err;
      throw new SampleFetchError("network", `${config.name} ${label}: network/CORS failure`);
    }
  };

  const readJson = async (res: Response, label: string): Promise<unknown> => {
    try {
      return await res.json();
    } catch {
      throw new SampleFetchError("parse", `${config.name} ${label}: malformed JSON`, res.status);
    }
  };

  // Tick keys newest-first; [] before the first tick has published.
  const fetchIndex = async (signal?: AbortSignal): Promise<string[]> => {
    const res = await fetchOrThrow(indexUrl, "index", signal);
    if (res.status === 404) return [];
    if (!res.ok) throw new SampleFetchError("http", `${config.name} index -> ${res.status}`, res.status);
    const data = await readJson(res, "index");
    if (!Array.isArray(data) || !data.every((t) => typeof t === "string")) {
      throw new SampleFetchError("parse", `${config.name} index is not a list of tick strings`, res.status);
    }
    // Sort newest-first ourselves: backfill prepends older ticks out of order,
    // so index order isn't reliable. Tick keys are ISO, so lexical = chrono.
    return [...data].sort((a, b) => (a < b ? 1 : a > b ? -1 : 0));
  };

  // null on 404 (no sample for this tick); any other failure throws so the
  // caller shows an error, not a false "no sample" state.
  const fetchManifest = async (
    tick: string,
    signal?: AbortSignal
  ): Promise<TManifest | null> => {
    const res = await fetchOrThrow(manifestUrl(tick), `manifest ${tick}`, signal);
    if (res.status === 404) return null;
    if (!res.ok) {
      throw new SampleFetchError("http", `${config.name} manifest ${tick} -> ${res.status}`, res.status);
    }
    const data = await readJson(res, `manifest ${tick}`);
    try {
      return parseManifest(data);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      throw new SampleFetchError("parse", `${config.name} manifest ${tick}: ${detail}`, res.status);
    }
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
