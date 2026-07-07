// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useCallback, useMemo, useState } from "react";
import { LatencyPercentile, ModelHeatmapData } from "@/types/benchmark.types";
import { normalizeModelName } from "@/lib/utils/formatters";
import { useActiveTab } from "@/hooks/useActiveTab";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";

interface ModelComparisonTableProps {
  data: ModelHeatmapData[];
  getProviderForModel: (model: string) => string;
}

type ColumnKey = "latency" | "avgWER" | "sampleCount";

const PERCENTILES: { key: LatencyPercentile; hint?: string }[] = [
  { key: "p0", hint: "fastest run" },
  { key: "p25" },
  { key: "p50", hint: "median" },
  { key: "p75" },
  { key: "p90" },
  { key: "p95" },
  { key: "p99" },
  { key: "p100", hint: "slowest run" }
];

const COLUMNS: {
  key: ColumnKey;
  label: string;
  bestDirection: "asc" | "desc";
}[] = [
  { key: "latency", label: "Latency", bestDirection: "asc" },
  { key: "avgWER", label: "Word error rate", bestDirection: "asc" },
  { key: "sampleCount", label: "Samples", bestDirection: "desc" }
];

// Relative position of a value across the models shown, 1 = best, 0 = worst.
const relativize = (v: number, values: number[], lowerIsBetter: boolean) => {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return 1;
  return lowerIsBetter ? (max - v) / (max - min) : (v - min) / (max - min);
};

const ModelComparisonTable: React.FC<ModelComparisonTableProps> = ({
  data,
  getProviderForModel
}) => {
  const activeTab = useActiveTab();
  const [percentileIdx, setPercentileIdx] = useState(2);
  const [sort, setSort] = useState<{ key: ColumnKey; direction: "asc" | "desc" }>(
    { key: "latency", direction: "asc" }
  );

  const percentile = (PERCENTILES[percentileIdx] ?? PERCENTILES[2])!;

  const value = useCallback(
    (row: ModelHeatmapData, key: ColumnKey): number =>
      key === "latency" ? row.latency[percentile.key] : row[key],
    [percentile.key]
  );

  const rows = useMemo(
    () =>
      [...data].sort((a, b) => {
        const delta = value(a, sort.key) - value(b, sort.key);
        return sort.direction === "asc" ? delta : -delta;
      }),
    [data, sort, value]
  );

  const handleSort = useCallback(
    (column: (typeof COLUMNS)[number]) => {
      const direction =
        sort.key === column.key
          ? sort.direction === "asc"
            ? "desc"
            : "asc"
          : column.bestDirection;
      capturePostHogEvent(POSTHOG_EVENTS.dashboardHeatmapSorted, {
        surface: `${activeTab}_dashboard`,
        mode: activeTab,
        metric: column.key,
        direction
      });
      setSort({ key: column.key, direction });
    },
    [activeTab, sort]
  );

  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-text-secondary">
        <p>No data available</p>
      </div>
    );
  }

  const columnValues = (key: ColumnKey) => rows.map((r) => value(r, key));
  const best = Object.fromEntries(
    COLUMNS.map((c) => [
      c.key,
      (c.bestDirection === "asc" ? Math.min : Math.max)(...columnValues(c.key))
    ])
  );

  const bar = (fraction: number) => (
    <div className="mt-1.5 h-1 w-full min-w-16 rounded-full bg-surface-secondary">
      <div
        className="h-full rounded-full bg-accent-teal"
        style={{ width: `${Math.max(2, fraction * 100)}%` }}
      />
    </div>
  );

  const cell = (row: ModelHeatmapData, key: ColumnKey, content: React.ReactNode) => (
    <td
      className={`py-3 pl-6 text-right align-top tabular-nums ${
        value(row, key) === best[key]
          ? "font-semibold text-text-primary"
          : "text-text-secondary"
      }`}
    >
      {content}
    </td>
  );

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-3 text-sm">
        <label htmlFor="latency-percentile" className="text-text-secondary">
          Latency percentile
        </label>
        <input
          id="latency-percentile"
          type="range"
          min={0}
          max={PERCENTILES.length - 1}
          step={1}
          value={percentileIdx}
          onChange={(e) => setPercentileIdx(Number(e.target.value))}
          className="w-44 accent-accent-teal"
          aria-valuetext={percentile.key}
        />
        <span className="tabular-nums font-medium text-text-primary">
          {percentile.key}
          {percentile.hint && (
            <span className="ml-1 font-normal text-text-tertiary">
              ({percentile.hint})
            </span>
          )}
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-border-primary text-text-tertiary">
              <th className="py-2 pr-4 text-left font-medium">Model</th>
              {COLUMNS.map((column) => (
                <th
                  key={column.key}
                  aria-sort={
                    sort.key === column.key
                      ? sort.direction === "asc"
                        ? "ascending"
                        : "descending"
                      : undefined
                  }
                  className="py-2 pl-6 text-right font-medium"
                >
                  <button
                    type="button"
                    onClick={() => handleSort(column)}
                    className={`whitespace-nowrap hover:text-text-primary transition-colors ${
                      sort.key === column.key ? "text-text-primary" : ""
                    }`}
                  >
                    {column.key === "latency"
                      ? `Latency (${percentile.key})`
                      : column.label}
                    <span className="inline-block w-3 text-xs">
                      {sort.key === column.key
                        ? sort.direction === "asc"
                          ? "↑"
                          : "↓"
                        : ""}
                    </span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.model}
                className="border-b border-border-secondary last:border-b-0 hover:bg-hover-bg"
              >
                <td className="py-3 pr-4 align-top">
                  <div className="font-medium text-text-primary">
                    {normalizeModelName(row.model)}
                  </div>
                  <div className="text-xs text-text-tertiary">
                    {getProviderForModel(row.model)}
                  </div>
                </td>
                {cell(
                  row,
                  "latency",
                  <>
                    {Math.round(row.latency[percentile.key])}
                    <span className="text-xs text-text-tertiary"> ms</span>
                    {bar(
                      relativize(
                        row.latency[percentile.key],
                        columnValues("latency"),
                        true
                      )
                    )}
                  </>
                )}
                {cell(
                  row,
                  "avgWER",
                  <>
                    {row.avgWER.toFixed(1)}
                    <span className="text-xs text-text-tertiary">
                      % ± {row.werStdDev.toFixed(1)}
                    </span>
                    {bar(relativize(row.avgWER, columnValues("avgWER"), true))}
                  </>
                )}
                {cell(
                  row,
                  "sampleCount",
                  <span className="text-xs">{row.sampleCount.toLocaleString()}</span>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ModelComparisonTable;
