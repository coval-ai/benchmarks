/**
 * Typed fetch wrapper for the Coval Benchmarks API.
 * All shapes are derived from the OpenAPI codegen — see lib/api/generated/schema.ts.
 */

import type { components, paths } from "./generated/schema";
import { buildQueryString } from "./url";
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
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
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
export type ProviderInfo = components["schemas"]["ProviderInfo"];
export type ModelInfo = components["schemas"]["ModelInfo"];
export type ModelTagOut = components["schemas"]["ModelTagOut"];
export type TagCategoryOut = components["schemas"]["TagCategoryOut"];
export type LeaderboardApiResponse = components["schemas"]["LeaderboardResponse"];
export type LeaderboardEntry = components["schemas"]["LeaderboardEntry"];
export type RunRow = components["schemas"]["RunOut"];
export type RunsApiResponse = components["schemas"]["RunsResponse"];
export type AggregatesApiResponse = components["schemas"]["AggregatesResponse"];
export type ModelStatEntry = components["schemas"]["ModelStatEntry"];
export type SeriesPoint = components["schemas"]["SeriesPoint"];

// Query-param types from codegen
export type AggregatesQueryParams = NonNullable<
  paths["/v1/results/aggregates"]["get"]["parameters"]["query"]
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

export async function getProviders(
  opts?: FetchOptions
): Promise<ProvidersApiResponse> {
  return request<ProvidersApiResponse>("/v1/providers", { signal: opts?.signal });
}
