/**
 * Client-side aggregation logic that mirrors the legacy SQL `getModelStats` from
 * `benchmarking-web-app/lib/db.ts`.
 *
 * Filter: status IN ('SUCCEEDED', 'PARTIAL') AND metric_value IS NOT NULL.
 * Note: Phase 4.5 uses uppercase status enum (RUNNING|SUCCEEDED|PARTIAL|FAILED).
 * Legacy SQL used lowercase 'success'. The new contract is uppercase; we include
 * PARTIAL as well (partially-succeeded runs may still have valid individual results).
 *
 * Stddev: sample stddev (n-1 denominator), COALESCE to 0 for n=1 (undefined from d3).
 * Percentiles: linear interpolation via d3 quantile(), identical to Postgres percentile_cont.
 */

import { mean, deviation, quantile } from "d3";
import type { components } from "@/lib/api/generated/schema";

export type Result = components["schemas"]["ResultOut"];

export interface ModelStats {
  provider: string;
  model: string;
  metric_type: string;
  avg_value: number;
  stddev_value: number;
  p25: number;
  p50: number;
  p75: number;
  min_value: number;
  max_value: number;
  sample_count: number;
}

const INCLUDED_STATUSES = new Set<Result["status"]>(["SUCCEEDED", "PARTIAL"]);

/**
 * Mirrors `getModelStats` from legacy `lib/db.ts`.
 * Filter `status IN ('SUCCEEDED','PARTIAL')` + `metric_value IS NOT NULL`.
 * Stddev=sample (n-1), n=1→0. Percentiles=linear interp.
 *
 * GROUP BY (provider, model, metric_type), compute avg, stddev, p25/p50/p75, min, max, count.
 */
function compute(rows: readonly Result[]): ModelStats[] {
  // Group rows by (provider, model, metric_type), filtering as we go.
  const groups = new Map<string, { provider: string; model: string; metric_type: string; values: number[] }>();

  for (const row of rows) {
    // Drop rows with null/undefined metric_value.
    if (row.metric_value == null) continue;
    // Drop rows with disallowed status — keep only SUCCEEDED and PARTIAL.
    if (!INCLUDED_STATUSES.has(row.status)) continue;

    const key = `${row.provider}\x00${row.model}\x00${row.metric_type}`;
    let group = groups.get(key);
    if (!group) {
      group = { provider: row.provider, model: row.model, metric_type: row.metric_type, values: [] };
      groups.set(key, group);
    }
    group.values.push(row.metric_value);
  }

  const stats: ModelStats[] = [];

  for (const { provider, model, metric_type, values } of groups.values()) {
    if (values.length === 0) continue; // should never happen given the filter above

    // Sort values in place for quantile computation (quantile requires sorted input).
    values.sort((a, b) => a - b);

    const avg_value = mean(values) ?? 0;

    // sample stddev (n-1 denom). d3 deviation() returns undefined for n=1; coerce to 0.
    const stddev_value = deviation(values) ?? 0;

    // quantile() does linear interpolation — identical to Postgres percentile_cont.
    const p25 = quantile(values, 0.25) ?? 0;
    const p50 = quantile(values, 0.5) ?? 0;
    const p75 = quantile(values, 0.75) ?? 0;

    // min/max: values is sorted.
    const min_value = values[0] ?? 0;
    const max_value = values[values.length - 1] ?? 0;

    stats.push({
      provider,
      model,
      metric_type,
      avg_value,
      stddev_value,
      p25,
      p50,
      p75,
      min_value,
      max_value,
      sample_count: values.length,
    });
  }

  // Match Postgres ORDER BY provider, model, metric_type.
  stats.sort((a, b) => {
    if (a.provider !== b.provider) return a.provider.localeCompare(b.provider);
    if (a.model !== b.model) return a.model.localeCompare(b.model);
    return a.metric_type.localeCompare(b.metric_type);
  });

  return stats;
}

const cache = new WeakMap<readonly Result[], ModelStats[]>();

/**
 * Compute ModelStats from raw Result rows, memoized by array reference.
 * React Query returns the same reference until refetch, so this is a free per-render skip.
 */
export function computeModelStats(rows: readonly Result[]): ModelStats[] {
  const cached = cache.get(rows);
  if (cached) return cached;
  const result = compute(rows);
  cache.set(rows, result);
  return result;
}

/**
 * Build a Map keyed by `${metric_type}|${provider}|${model}` for O(1) lookup.
 */
export function statsByKey(stats: ModelStats[]): Map<string, ModelStats> {
  return new Map(stats.map((s) => [`${s.metric_type}|${s.provider}|${s.model}`, s]));
}

/**
 * Returns a lookup function for per-(model, metricType, provider) stats.
 * Supports both exact lookup (with provider) and fallback linear scan (without provider)
 * to preserve legacy hook behaviour that searched by (model, metric_type) only.
 */
export function makeStatLookup(stats: ModelStats[]) {
  const map = statsByKey(stats);
  return (model: string, metricType: string, provider?: string): ModelStats | undefined => {
    if (provider) return map.get(`${metricType}|${provider}|${model}`);
    // Legacy fallback: search by (model, metric_type) without provider.
    for (const s of stats) {
      if (s.model === model && s.metric_type === metricType) return s;
    }
    return undefined;
  };
}
