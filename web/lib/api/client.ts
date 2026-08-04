/**
 * Typed fetch wrapper for the Coval Benchmarks API.
 * All shapes are derived from the OpenAPI codegen — see lib/api/generated/schema.ts.
 */

import type { components, paths } from "./generated/schema";
import { buildQueryString } from "./url";
import { tokenHeaders } from "./accessTokens";
import { normalizePlaygroundError, type PlaygroundApiError } from "@/lib/playground/schemas";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly statusText: string,
    public readonly body: unknown
  ) {
    super(`API ${status}: ${statusText}`);
    this.name = "ApiError";
  }
}

export class PlaygroundTtsError extends Error {
  constructor(public readonly payload: PlaygroundApiError) {
    super(payload.error);
    this.name = "PlaygroundTtsError";
  }
}

async function request<T>(path: string, init?: Parameters<typeof fetch>[1]): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      // Unlocks early-access models server-side; absent for public callers.
      ...tokenHeaders(),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      // ignore parse failure
    }
    throw new ApiError(res.status, res.statusText, body);
  }
  return (await res.json()) as T;
}

// Response and row types from codegen
export type ProvidersApiResponse = components["schemas"]["ProvidersResponse"];
export type ModelTagOut = components["schemas"]["ModelTagOut"];
export type TagCategoryOut = components["schemas"]["TagCategoryOut"];
export type AggregatesApiResponse = components["schemas"]["AggregatesResponse"];
export type AggregatesByDatasetApiResponse =
  components["schemas"]["AggregatesByDatasetResponse"];
export type ModelStatEntry = components["schemas"]["ModelStatEntry"];
export type SeriesPoint = components["schemas"]["SeriesPoint"];

// Query-param types from codegen
export type AggregatesQueryParams = NonNullable<
  paths["/v1/results/aggregates"]["get"]["parameters"]["query"]
>;
export type AggregatesByDatasetQueryParams = NonNullable<
  paths["/v1/results/aggregates/by-dataset"]["get"]["parameters"]["query"]
>;
export interface FetchOptions {
  signal?: AbortSignal;
}

export type PlaygroundTtsResponse = {
  audioBlob: Blob;
  ttfaMs: number | null;
  totalMs: number;
};

export async function postPlaygroundTts(
  body: { modelId: string; text: string },
  opts?: FetchOptions
): Promise<PlaygroundTtsResponse> {
  const t0 = performance.now();
  const res = await fetch(`/api/playground/tts`, {
    method: "POST",
    signal: opts?.signal,
    headers: {
      "Content-Type": "application/json",
      Accept: "audio/wav,application/json",
    },
    body: JSON.stringify({ model_id: body.modelId, text: body.text }),
  });

  if (!res.ok) {
    let parsed: unknown = null;
    try {
      parsed = await res.json();
    } catch {
      // ignore parse failure
    }
    throw new PlaygroundTtsError(normalizePlaygroundError(res.status, parsed));
  }

  const contentType = (res.headers.get("Content-Type") ?? "").toLowerCase();
  if (!contentType.startsWith("audio/")) {
    throw new PlaygroundTtsError(
      normalizePlaygroundError(502, {
        code: "UPSTREAM_ERROR",
        error: `Expected audio response, got ${contentType || "unknown"}.`,
      }),
    );
  }
  const blob = await res.blob();
  const totalMs = performance.now() - t0;
  const ttfaHeader = res.headers.get("X-TTFA-Ms");
  const ttfaMs = ttfaHeader ? Number(ttfaHeader) : null;
  return { audioBlob: blob, ttfaMs: Number.isFinite(ttfaMs) ? ttfaMs : null, totalMs };
}

export async function getAggregates(
  params: AggregatesQueryParams,
  opts?: FetchOptions
): Promise<AggregatesApiResponse> {
  const qs = buildQueryString(
    params as Record<string, string | number | boolean | null | undefined>
  );
  return request<AggregatesApiResponse>(`/v1/results/aggregates${qs}`, {
    signal: opts?.signal,
  });
}

export async function getAggregatesByDataset(
  params: AggregatesByDatasetQueryParams,
  opts?: FetchOptions
): Promise<AggregatesByDatasetApiResponse> {
  const qs = buildQueryString(
    params as Record<string, string | number | boolean | null | undefined>
  );
  return request<AggregatesByDatasetApiResponse>(
    `/v1/results/aggregates/by-dataset${qs}`,
    { signal: opts?.signal }
  );
}

export async function getProviders(
  opts?: FetchOptions
): Promise<ProvidersApiResponse> {
  return request<ProvidersApiResponse>("/v1/providers", { signal: opts?.signal });
}

// Declared by hand rather than taken from codegen: the committed schema predates
// these routes, and regenerating it here would sweep in every unrelated API change
// since it was last built. Refresh codegen on its own and swap these for
// components["schemas"]["S2SSample*"] then.
export interface S2SSampleTurn {
  index: number;
  role: string;
  content: string;
  start_offset?: number | null;
  end_offset?: number | null;
}

export interface S2SSampleRecording {
  provider: string;
  model: string;
  // An address on this API, not a storage URL: playing it mints a signed URL.
  audio_path: string;
  coval_run_id: string;
  sim_id: string;
  agent_id?: string | null;
  turns: S2SSampleTurn[];
}

export interface S2SSampleApiResponse {
  schema_version?: number | null;
  sample_id: string;
  test_case_id: string;
  test_set_id?: string | null;
  persona_name?: string | null;
  transcript?: string | null;
  recordings: S2SSampleRecording[];
}

export interface S2SSampleAudio {
  url: string;
  expires_at: string;
}

/** Sample ids newest-first; empty when the caller may see no S2S model at all. */
export async function getS2SSampleIds(opts?: FetchOptions): Promise<string[]> {
  return request<string[]>("/v1/s2s/samples", { signal: opts?.signal });
}

/**
 * One sample, carrying only the recordings this caller may hear.
 *
 * `null` on 404 — no sample was published for that tick, or every recording in it
 * is embargoed for this caller. Both mean the same thing to the card: nothing to
 * show here. Any other failure throws so the card says so instead of going quiet.
 */
export async function getS2SSample(
  sampleId: string,
  opts?: FetchOptions
): Promise<S2SSampleApiResponse | null> {
  try {
    return await request<S2SSampleApiResponse>(
      `/v1/s2s/samples/${encodeURIComponent(sampleId)}`,
      { signal: opts?.signal }
    );
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}

/** A freshly signed URL for one recording, from the `audio_path` its manifest gave. */
export async function getS2SSampleAudio(
  audioPath: string,
  opts?: FetchOptions
): Promise<S2SSampleAudio> {
  return request<S2SSampleAudio>(audioPath, { signal: opts?.signal });
}
