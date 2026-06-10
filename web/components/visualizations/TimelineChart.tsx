// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo, useState } from "react";
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
// by default), and the items lay out in a grid that grows from two columns on
// mobile to more columns on larger screens.
const TimelineLegend: React.FC<{ payload?: LegendEntry[] }> = ({ payload }) => (
  <ul className="grid grid-cols-2 gap-x-4 gap-y-1.5 px-2 pt-5 sm:grid-cols-3 sm:gap-x-6 sm:gap-y-2 lg:grid-cols-4">
    {payload?.map((entry) => (
      <li
        key={entry.dataKey ?? entry.value}
        className="flex items-start gap-1.5 text-xs leading-tight text-text-primary"
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
    dataTimeWindow,
  } = useDashboard();
  const trackChartHover = useChartHoverTracking("timeline");

  // STT shows a TTFS / TTFT toggle; TTS is single-metric (TTFA). Defaulting to
  // TTFT for now; flip back to "TTFS" once enough TTFS data has accumulated.
  const [metric, setMetric] = useState<string>(
    activeTab === "stt" ? "TTFT" : "TTFA"
  );

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

  const avgValue = useMemo(() => {
    let sum = 0;
    let count = 0;
    for (const point of windowedTimelineData) {
      const record = point as Record<string, number>;
      for (const model of modelsWithData) {
        const value = record[`${model}_value`];
        if (typeof value === "number" && !Number.isNaN(value)) {
          sum += value;
          count += 1;
        }
      }
    }
    return count > 0 ? sum / count : 0;
  }, [windowedTimelineData, modelsWithData]);

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
            label: `Avg ${metricLabel}`,
            value: `${avgValue.toFixed(0)} ms`,
          }}
        />

        {activeTab === "stt" && (
          <div className="mb-4 inline-flex gap-0.5 rounded-lg bg-gray-100 p-0.5">
            {(["TTFS", "TTFT"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMetric(m)}
                className={
                  "rounded-md px-3 py-1 text-xs font-medium transition-colors " +
                  (metric === m
                    ? "bg-white text-text-primary shadow-sm"
                    : "text-gray-500 hover:text-text-primary")
                }
              >
                {m}
              </button>
            ))}
          </div>
        )}

        <div className="h-96" onMouseEnter={trackChartHover}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={windowedTimelineData}
              margin={{ top: 5, right: 8, left: 0, bottom: 5 }}
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
        </div>
      </Card>
    </div>
  );
};

export default TimelineChart;
