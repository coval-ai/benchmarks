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

function asString(v: unknown, field: string): string {
  if (typeof v !== "string") throw new Error(`${field} must be a string`);
  return v;
}

function asNullableString(v: unknown, field: string): string | null {
  if (v === null) return null;
  return asString(v, field);
}

// Validate the full manifest shape before the UI touches it — a shallow object
// check would let `{}` through and crash visibleRecordings' .filter().
export function parseS2SManifest(data: unknown): S2SSampleManifest {
  if (typeof data !== "object" || data === null) {
    throw new Error("manifest must be an object");
  }
  const m = data as Record<string, unknown>;
  if (!Array.isArray(m.recordings)) {
    throw new Error("recordings must be an array");
  }
  const recordings = m.recordings.map((r, i): S2SSampleRecording => {
    if (typeof r !== "object" || r === null) {
      throw new Error(`recordings[${i}] must be an object`);
    }
    const rec = r as Record<string, unknown>;
    return {
      provider: asString(rec.provider, `recordings[${i}].provider`),
      model: asString(rec.model, `recordings[${i}].model`),
      object: asString(rec.object, `recordings[${i}].object`),
      coval_run_id: asString(rec.coval_run_id, `recordings[${i}].coval_run_id`),
      sim_id: asString(rec.sim_id, `recordings[${i}].sim_id`),
    };
  });
  return {
    bucket_at: asString(m.bucket_at, "bucket_at"),
    test_case_id: asString(m.test_case_id, "test_case_id"),
    transcript: asNullableString(m.transcript, "transcript"),
    input_audio_url: asNullableString(m.input_audio_url, "input_audio_url"),
    recordings,
  };
}

export const s2sSampleFeed = createSampleFeed<S2SSampleManifest>(
  {
    name: "s2s-samples",
    bucket: "coval-benchmarks-s2s-samples",
    prefix: "s2s-samples",
    refetchMs: S2S_FETCH_PERIOD_MS,
  },
  parseS2SManifest
);

// Drop recordings whose provider isn't visible on the page (disabled catalogue).
export function visibleRecordings(
  manifest: S2SSampleManifest,
  visibleProviders: ReadonlySet<string>
): S2SSampleRecording[] {
  return manifest.recordings.filter((r) => visibleProviders.has(r.provider));
}
