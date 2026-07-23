// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { createSampleFeed } from "./createSampleFeed";

// One turn of a multi-turn conversation (v2 manifests). `role` is "user"
// (the persona) or "assistant" (the agent); `index` is the turn's position.
export interface S2STurn {
  index: number;
  role: string;
  content: string;
}

export interface S2SSampleRecording {
  provider: string;
  model: string;
  object: string;
  coval_run_id: string;
  sim_id: string;
  // v2 only: the Coval agent behind the provider, and this agent's own
  // conversation transcript (transcripts diverge per agent).
  agent_id?: string;
  turns?: S2STurn[];
}

export interface S2SSampleManifest {
  // Absent on v1 (single-turn) manifests; 2 on multi-turn.
  schema_version?: number;
  bucket_at: string;
  test_case_id: string;
  // v2 only: the multi-turn test set + the sampled persona as a display label
  // (e.g. "Standard Male") — a label, not a selection axis. The persona id
  // stays internal to the sampler's composite key and is not written out.
  test_set_id?: string;
  persona_name?: string;
  // v1 single-turn fields: first user turn + its baked dataset clip. Null/absent
  // on v2 (the transcript lives per-recording as `turns`; no baked clip).
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

// Absent (undefined) and null both collapse to null — a v2 manifest omits the
// v1-only transcript/input_audio_url keys, and that must not fail the parse.
function asNullableString(v: unknown, field: string): string | null {
  if (v === null || v === undefined) return null;
  return asString(v, field);
}

// undefined -> undefined (optional field omitted); otherwise must be a string.
function asOptionalString(v: unknown, field: string): string | undefined {
  if (v === undefined) return undefined;
  return asString(v, field);
}

function parseTurns(v: unknown, field: string): S2STurn[] | undefined {
  if (v === undefined) return undefined;
  if (!Array.isArray(v)) throw new Error(`${field} must be an array`);
  return v.map((t, i): S2STurn => {
    if (typeof t !== "object" || t === null) {
      throw new Error(`${field}[${i}] must be an object`);
    }
    const turn = t as Record<string, unknown>;
    if (typeof turn.index !== "number") {
      throw new Error(`${field}[${i}].index must be a number`);
    }
    return {
      index: turn.index,
      role: asString(turn.role, `${field}[${i}].role`),
      content: asString(turn.content, `${field}[${i}].content`),
    };
  });
}

// Validate the full manifest shape before the UI touches it — a shallow object
// check would let `{}` through and crash visibleRecordings' .filter(). Tolerant
// of both single-turn (v1) and multi-turn (v2) shapes.
export function parseS2SManifest(data: unknown): S2SSampleManifest {
  if (typeof data !== "object" || data === null) {
    throw new Error("manifest must be an object");
  }
  const m = data as Record<string, unknown>;
  if (m.schema_version !== undefined && typeof m.schema_version !== "number") {
    throw new Error("schema_version must be a number");
  }
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
      agent_id: asOptionalString(rec.agent_id, `recordings[${i}].agent_id`),
      turns: parseTurns(rec.turns, `recordings[${i}].turns`),
    };
  });
  return {
    schema_version: m.schema_version as number | undefined,
    bucket_at: asString(m.bucket_at, "bucket_at"),
    test_case_id: asString(m.test_case_id, "test_case_id"),
    test_set_id: asOptionalString(m.test_set_id, "test_set_id"),
    persona_name: asOptionalString(m.persona_name, "persona_name"),
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

// True when a manifest carries the multi-turn shape (per-recording turns).
export function isMultiTurn(manifest: S2SSampleManifest): boolean {
  return (manifest.schema_version ?? 1) >= 2;
}

// Drop recordings whose provider isn't visible on the page (disabled catalogue).
export function visibleRecordings(
  manifest: S2SSampleManifest,
  visibleProviders: ReadonlySet<string>
): S2SSampleRecording[] {
  return manifest.recordings.filter((r) => visibleProviders.has(r.provider));
}
