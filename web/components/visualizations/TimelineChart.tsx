// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
  Tooltip
} from "recharts";
import { getModelColor } from "@/lib/utils/colors";
import { formatDate, formatTime, getLocalTimeZoneAbbr } from "@/lib/utils/formatters";
import { metricDescriptions } from "@/lib/config/metrics";
import CustomTimelineTooltip from "@/components/charts/tooltips/TimelineTooltip";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import MetricToggle from "@/components/dashboard/MetricToggle";
import { useActiveTab } from "@/hooks/useActiveTab";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";

interface LegendEntry {
  value: string;
  color?: string;
  dataKey?: string;
}

// Custom legend: names are rendered in black (recharts colors them per-series
// by default), and items fill top-to-bottom within each column so the list
// reads alphabetically down each column rather than across rows.
const TimelineLegend: React.FC<{ payload?: LegendEntry[] }> = ({ payload }) => (
  <ul className="columns-2 gap-x-4 px-2 pt-5 sm:columns-3 sm:gap-x-6 lg:columns-4">
    {payload?.map((entry) => (
      <li
        key={entry.dataKey ?? entry.value}
        className="mb-1.5 flex items-start gap-1.5 text-xs leading-tight text-text-primary break-inside-avoid"
      >
        <span
          className="mt-0.5 inline-block w-3 h-3 shrink-0 rounded-[2px]"
          style={{ backgroundColor: entry.color }}
          aria-hidden="true"
        />
        <span>{entry.value}</span>
      </li>
    ))}
  </ul>
);

interface TooltipPayloadItem {
  dataKey: string;
  value: number;
  name: string;
  color: string;
}

interface PinnedTooltip {
  label: string;
  payload: TooltipPayloadItem[];
  x: number;
  y: number;
  flip: boolean;
}

const TimelineChart: React.FC = () => {
  const activeTab = useActiveTab();
  const {
    getModelsWithTimelineData,
    getWindowedTimelineData,
    getTimelineData,
    getCurrentTimeWindow,
    getTimelineTicks,
    formatChartLabel,
    getProviderForModel,
    getAvgLatencyMs,
    activeMetric: metric,
    dataTimeWindow,
  } = useDashboard();
  const trackChartHover = useChartHoverTracking("timeline");

  const chartRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState<PinnedTooltip | null>(null);

  // The metric is shared dashboard-wide, so clear any pinned tooltip whenever it
  // changes — whether from this chart's toggle or another section's.
  useEffect(() => {
    setPinned(null);
  }, [metric]);

  useEffect(() => {
    if (pinned === null) return;
    const onDocMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (pinnedRef.current?.contains(target)) return;
      const surface = chartRef.current?.querySelector(".recharts-surface");
      if (surface?.contains(target)) return;
      setPinned(null);
    };
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [pinned]);

  const themeColors = useThemeColors();
  const modelsWithData = getModelsWithTimelineData(metric);
  const windowedTimelineData = getWindowedTimelineData(metric);
  const currentTimeWindow = getCurrentTimeWindow();
  const tzAbbr = getLocalTimeZoneAbbr();
  const dateScale = dataTimeWindow !== "24h";
  const xAxisLabel = dateScale ? "Date" : tzAbbr ? `Time (${tzAbbr})` : "Time";

  const metricLabel = metric;
  const description =
    metricDescriptions[metric.toLowerCase() as keyof typeof metricDescriptions];

  // Headline: run-weighted average latency across selected models (same stat the
  // box plot reports).
  const avgValue = getAvgLatencyMs(metric);

  // Fix the Y axis to the max across the full dataset (not just the visible
  // window) so the scale stays consistent while panning. Rounded up to a clean
  // 0.5s step so the gridlines land on tidy values.
  const yAxisMax = useMemo(() => {
    let max = 0;
    for (const point of getTimelineData(metric)) {
      const record = point as Record<string, number>;
      for (const model of modelsWithData) {
        const value = record[`${model}_value`];
        if (typeof value === "number" && !Number.isNaN(value) && value > max) {
          max = value;
        }
      }
    }
    if (max === 0) return "dataMax" as const;
    const step = 500;
    return Math.ceil(max / step) * step;
  }, [getTimelineData, metric, modelsWithData]);

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8">
        <SectionHeader
          label={
            activeTab === "tts"
              ? "Latency Benchmarks"
              : "Performance Consistency"
          }
          description={description}
          stat={{
            label: `Average ${metricLabel}`,
            value: `${avgValue.toFixed(0)} ms`,
          }}
        />

        <MetricToggle />

        <div ref={chartRef} className="relative h-96" onMouseEnter={trackChartHover}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={windowedTimelineData}
              margin={{ top: 5, right: 8, left: 0, bottom: 5 }}
              onClick={(state) => {
                const lbl = state?.activeLabel;
                const coord = state?.activeCoordinate;
                const payload = (state?.activePayload ?? []) as TooltipPayloadItem[];
                const hasRows = payload.some(
                  (item) => typeof item?.value === "number" && item.value > 0
                );
                if (lbl == null || !coord || !hasRows) return;
                setPinned((cur) => {
                  if (cur && cur.label === lbl) return null;
                  const width = chartRef.current?.clientWidth ?? 0;
                  const height = chartRef.current?.clientHeight ?? 0;
                  const x = coord.x ?? 0;
                  const rawY = coord.y ?? 0;
                  const pad = 130;
                  const y =
                    height > pad * 2
                      ? Math.min(Math.max(rawY, pad), height - pad)
                      : rawY;
                  return {
                    label: lbl,
                    payload,
                    x,
                    y,
                    flip: width > 0 && x > width / 2,
                  };
                });
              }}
            >
              <CartesianGrid
                vertical={false}
                strokeDasharray="2 2"
                stroke={themeColors.grid}
              />
              <XAxis
                dataKey="timestamp"
                type="number"
                scale="time"
                domain={currentTimeWindow}
                ticks={getTimelineTicks()}
                allowDataOverflow={false}
                axisLine={false}
                tickLine={false}
                tick={{ fill: themeColors.axisText, fontSize: 12 }}
                tickFormatter={(value) =>
                  dateScale ? formatDate(value) : formatTime(value)
                }
                label={{
                  value: xAxisLabel,
                  position: "insideBottom",
                  offset: -5,
                  style: {
                    textAnchor: "middle",
                    fill: themeColors.axisText,
                    fontSize: "14px"
                  }
                }}
              />
              <YAxis
                width={40}
                axisLine={false}
                tickLine={false}
                tick={{ fill: themeColors.axisText, fontSize: 12 }}
                domain={[0, yAxisMax]}
                tickFormatter={(value) => `${(value / 1000).toFixed(1)}s`}
              />
              <Tooltip
                content={
                  <CustomTimelineTooltip
                    getProviderForModel={getProviderForModel}
                    showDate={dateScale}
                  />
                }
                active={pinned ? false : undefined}
              />
              {modelsWithData.length > 1 && (
                <Legend content={<TimelineLegend />} />
              )}
              {modelsWithData.map((model) => (
                <Line
                  key={model}
                  type="monotone"
                  dataKey={`${model}_value`}
                  stroke={getModelColor(model)}
                  strokeWidth={modelsWithData.length === 1 ? 3 : 2}
                  dot={false}
                  isAnimationActive={false}
                  activeDot={{
                    r: modelsWithData.length === 1 ? 7 : 6,
                    fill: getModelColor(model)
                  }}
                  connectNulls={false}
                  name={formatChartLabel(model, getProviderForModel(model))}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
          {pinned && (
            <div
              ref={pinnedRef}
              style={{
                position: "absolute",
                left: pinned.x,
                top: pinned.y,
                transform: pinned.flip
                  ? "translate(calc(-100% - 12px), -50%)"
                  : "translate(12px, -50%)",
                pointerEvents: "auto",
                zIndex: 20,
              }}
            >
              <CustomTimelineTooltip
                active
                payload={pinned.payload}
                label={pinned.label}
                getProviderForModel={getProviderForModel}
                showDate={dateScale}
              />
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default TimelineChart;
