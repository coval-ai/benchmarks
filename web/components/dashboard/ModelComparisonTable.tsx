// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useCallback, useMemo, useRef, useState } from "react";
import { Server } from "lucide-react";
import { useDedicatedInfoTip } from "@/components/shared/DedicatedInferenceInfo";
import { LatencyPercentile, ModelHeatmapData } from "@/types/benchmark.types";
import { normalizeModelName } from "@/lib/utils/formatters";
import WerDatasetSelect from "@/components/dashboard/WerDatasetSelect";
import { useActiveTab } from "@/hooks/useActiveTab";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";

interface ModelComparisonTableProps {
  data: ModelHeatmapData[];
  getProviderForModel: (model: string) => string;
  /** Dedicated-inference endpoints carry the server marker beside the name. */
  dedicatedModels?: Set<string>;
  percentileIdx: number;
  onPercentileChange: (idx: number) => void;
  werLabel?: string;
  werLoading?: boolean;
}

type ColumnKey = "latency" | "avgWER" | "sampleCount";

export const PERCENTILES: { key: LatencyPercentile; hint?: string }[] = [
  { key: "p0", hint: "fastest run" },
  { key: "p25" },
  { key: "p50", hint: "median" },
  { key: "p75" },
  { key: "p90" },
  { key: "p95" },
  { key: "p99" },
  { key: "p100", hint: "slowest run" }
];

// p95: the tail latency real-time voice users actually feel.
export const DEFAULT_PERCENTILE_IDX = 5;

const COLUMNS: {
  key: ColumnKey;
  label: string;
  bestDirection: "asc" | "desc";
}[] = [
  { key: "latency", label: "Latency", bestDirection: "asc" },
  { key: "avgWER", label: "Word error rate", bestDirection: "asc" },
  { key: "sampleCount", label: "Samples", bestDirection: "desc" }
];

const ModelComparisonTable: React.FC<ModelComparisonTableProps> = ({
  data,
  getProviderForModel,
  dedicatedModels,
  percentileIdx,
  onPercentileChange,
  werLabel,
  werLoading
}) => {
  const activeTab = useActiveTab();
  // The table scrolls horizontally, which clips anchored popovers, so the
  // dedicated explainer renders as an overlay on this unclipped wrapper.
  const tableWrapRef = useRef<HTMLDivElement>(null);
  const { iconHandlers: dedicatedIconHandlers, overlay: dedicatedOverlay } =
    useDedicatedInfoTip(tableWrapRef);
  const [sort, setSort] = useState<{ key: ColumnKey; direction: "asc" | "desc" }>(
    { key: "latency", direction: "asc" }
  );

  const percentile = (PERCENTILES[percentileIdx] ?? PERCENTILES[DEFAULT_PERCENTILE_IDX])!;

  // Latency-only benchmarks (S2S) ship rows without WER; hide that column.
  const hasWER = useMemo(() => data.some((d) => d.avgWER !== undefined), [data]);
  const columns = useMemo(
    () => (hasWER ? COLUMNS : COLUMNS.filter((c) => c.key !== "avgWER")),
    [hasWER]
  );

  // One pass per (data, percentile, sort) change: pull the selected latency
  // percentile, precompute the relative bar fractions, sort, and note the best
  // value per column.
  const { rows, best } = useMemo(() => {
    const span = (values: number[]): [number, number] => {
      const min = Math.min(...values);
      return [min, Math.max(...values) - min];
    };
    const latencyValues = data
      .map((d) => d.latency?.[percentile.key])
      .filter((v): v is number => v !== undefined);
    const [latencyMin, latencySpan] =
      latencyValues.length > 0 ? span(latencyValues) : [0, 0];
    const werValues = data
      .map((d) => d.avgWER)
      .filter((v): v is number => v !== undefined);
    const [werMin, werSpan] = werValues.length > 0 ? span(werValues) : [0, 0];
    const rel = (v: number, min: number, s: number) =>
      s === 0 ? 1 : (min + s - v) / s;

    const rows = data
      .map((d) => {
        const latency = d.latency?.[percentile.key];
        return {
          ...d,
          latency,
          latencyRel:
            latency !== undefined ? rel(latency, latencyMin, latencySpan) : 1,
          werRel:
            hasWER && d.avgWER !== undefined
              ? rel(d.avgWER, werMin, werSpan)
              : 1
        };
      })
      .sort((a, b) => {
        const [av, bv] = [a[sort.key], b[sort.key]];
        if (av === undefined || bv === undefined)
          return (av === undefined ? 1 : 0) - (bv === undefined ? 1 : 0);
        const delta = av - bv;
        return sort.direction === "asc" ? delta : -delta;
      });

    return {
      rows,
      best: {
        latency: latencyMin,
        avgWER: werMin,
        sampleCount: Math.max(...data.map((d) => d.sampleCount))
      }
    };
  }, [data, percentile.key, sort, hasWER]);

  type Row = (typeof rows)[number];

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

  const bar = (fraction: number) => (
    <div className="mt-1.5 h-1 w-full min-w-16 rounded-full bg-surface-secondary">
      <div
        className="h-full rounded-full bg-accent-blue"
        style={{ width: `${Math.max(2, fraction * 100)}%` }}
      />
    </div>
  );

  const cell = (row: Row, key: ColumnKey, content: React.ReactNode) => (
    <td
      className={`py-3 pl-6 text-right align-top tabular-nums ${
        row[key] === best[key]
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
          onChange={(e) => onPercentileChange(Number(e.target.value))}
          className="h-11 w-44 accent-accent-blue lg:h-auto"
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
        <WerDatasetSelect className="ml-auto" />
      </div>

      <div ref={tableWrapRef} className="relative">
        {dedicatedOverlay}
        <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-border-primary text-text-tertiary">
              <th className="py-2 pr-4 text-left font-medium">Model</th>
              {columns.map((column) => (
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
                    className={`min-h-11 whitespace-nowrap hover:text-text-primary transition-colors lg:min-h-0 ${
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
                    {column.key === "avgWER" && werLabel && (
                      <span className="block text-[10px] font-normal normal-case text-text-tertiary">
                        {werLabel}
                      </span>
                    )}
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
                  <div className="flex items-center gap-1.5 font-medium text-text-primary">
                    {normalizeModelName(row.model)}
                    {dedicatedModels?.has(row.model) && (
                      <button
                        type="button"
                        aria-label="About dedicated inference"
                        className="flex shrink-0 cursor-help items-center rounded-md p-1 -m-1 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40"
                        {...dedicatedIconHandlers}
                      >
                        <Server size={13} aria-hidden />
                      </button>
                    )}
                  </div>
                  <div className="text-xs text-text-tertiary">
                    {getProviderForModel(row.model)}
                  </div>
                </td>
                {cell(
                  row,
                  "latency",
                  row.latency !== undefined ? (
                    <>
                      {Math.round(row.latency)}
                      <span className="text-xs text-text-tertiary"> ms</span>
                      {bar(row.latencyRel)}
                    </>
                  ) : (
                    <span className="text-text-tertiary">Not applicable</span>
                  )
                )}
                {hasWER &&
                  cell(
                    row,
                    "avgWER",
                    <div
                      className={`transition-opacity ${werLoading ? "opacity-40" : ""}`}
                    >
                      {row.avgWER !== undefined ? (
                        <>
                          {row.avgWER.toFixed(1)}
                          <span className="text-xs text-text-tertiary">
                            % ± {(row.werStdDev ?? 0).toFixed(1)}
                          </span>
                          {bar(row.werRel)}
                        </>
                      ) : (
                        <span className="text-text-tertiary">—</span>
                      )}
                    </div>
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
    </div>
  );
};

export default React.memo(ModelComparisonTable);
