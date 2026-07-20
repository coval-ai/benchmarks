// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { createSampleFeed } from "./createSampleFeed";

export interface S2SSampleRecording {
  provider: string;
  model: string;
  object: string;
  coval_run_id: string;
  sim_id: string;
}

export interface S2SSampleManifest {
  bucket_at: string;
  test_case_id: string;
  transcript: string | null;
  input_audio_url: string | null;
  recordings: S2SSampleRecording[];
}

// Mirror the runner's s2s_fetch_period_seconds (config.py, default 10_800 = 3h).
const S2S_FETCH_PERIOD_MS = 10_800 * 1000;

export const s2sSampleFeed = createSampleFeed<S2SSampleManifest>({
  name: "s2s-samples",
  bucket: "coval-benchmarks-s2s-samples",
  prefix: "s2s-samples",
  refetchMs: S2S_FETCH_PERIOD_MS,
});

// Drop recordings whose provider isn't visible on the page (disabled catalogue).
export function visibleRecordings(
  manifest: S2SSampleManifest,
  visibleProviders: ReadonlySet<string>
): S2SSampleRecording[] {
  return manifest.recordings.filter((r) => visibleProviders.has(r.provider));
}
