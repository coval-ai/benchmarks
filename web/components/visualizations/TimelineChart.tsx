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
  Tooltip,
  ReferenceLine
} from "recharts";
import { getModelColor } from "@/lib/utils/colors";
import { formatDate, formatTime, getLocalTimeZoneAbbr } from "@/lib/utils/formatters";
import { metricDescriptions } from "@/lib/config/metrics";
import {
  methodologyChanges,
  type MethodologyChange,
  type MethodologyMetricKey
} from "@/lib/config/methodologyChanges";
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

interface MarkerLabelProps {
  viewBox?: { x?: number; y?: number; width?: number; height?: number };
  change: MethodologyChange;
  onEnter: (change: MethodologyChange, x: number) => void;
  onLeave: () => void;
}

const MethodologyMarkerLabel: React.FC<MarkerLabelProps> = ({
  viewBox,
  change,
  onEnter,
  onLeave,
}) => {
  const cx = viewBox?.x ?? 0;
  const cy = (viewBox?.y ?? 0) + 9;
  return (
    <g
      transform={`translate(${cx}, ${cy})`}
      style={{ cursor: "pointer" }}
      onMouseEnter={() => onEnter(change, cx)}
      onMouseLeave={onLeave}
    >
      <rect x={-24} y={-14} width={48} height={42} fill="transparent" />
      <circle r={9} fill="#f59e0b" stroke="#ffffff" strokeWidth={1.5} />
      <text
        textAnchor="middle"
        y={4}
        fontSize={13}
        fontWeight={700}
        fill="#ffffff"
      >
        !
      </text>
    </g>
  );
};

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
  const [hoveredMarker, setHoveredMarker] = useState<{
    change: MethodologyChange;
    x: number;
    ts: number;
  } | null>(null);

  const currentTimeWindow = getCurrentTimeWindow();
  const [windowStart, windowEnd] = currentTimeWindow;

  // The metric is shared dashboard-wide, so clear any pinned tooltip whenever it
  // changes — whether from this chart's toggle or another section's. Also clear
  // the marker popover when the time window shifts, since a marker may scroll
  // out of range.
  useEffect(() => {
    setPinned(null);
    setHoveredMarker(null);
  }, [metric, windowStart, windowEnd]);

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
  const activeMetricKey = metric.toLowerCase() as MethodologyMetricKey;
  // With an explicit time (an offset-qualified "HH:MM:SS±hh:mm") pin the marker
  // to that exact instant; otherwise anchor date-only strings at UTC noon so the
  // marker lands on the intended calendar day in every inhabited timezone (UTC
  // midnight would shift to the previous day for the Americas).
  const methodologyMarkers = useMemo(
    () =>
      methodologyChanges
        .filter((c) => !c.metrics || c.metrics.includes(activeMetricKey))
        .map((c) => ({
          change: c,
          ts: new Date(
            c.time ? `${c.date}T${c.time}` : `${c.date}T12:00:00Z`
          ).getTime(),
        }))
        .filter((m) => m.ts >= windowStart && m.ts <= windowEnd),
    [activeMetricKey, windowStart, windowEnd]
  );
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
              {methodologyMarkers.map((m) => (
                <ReferenceLine
                  key={`${m.change.date}-${m.change.title}`}
                  x={m.ts}
                  stroke="#f59e0b"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  label={
                    <MethodologyMarkerLabel
                      change={m.change}
                      onEnter={(change, x) =>
                        setHoveredMarker({ change, x, ts: m.ts })
                      }
                      onLeave={() => setHoveredMarker(null)}
                    />
                  }
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
          {hoveredMarker &&
            (() => {
              const width = chartRef.current?.clientWidth ?? 0;
              const pad = 120;
              const left =
                width > pad * 2
                  ? Math.min(Math.max(hoveredMarker.x, pad), width - pad)
                  : hoveredMarker.x;
              return (
                <div
                  style={{
                    position: "absolute",
                    left,
                    top: 26,
                    transform: "translateX(-50%)",
                    zIndex: 25,
                    pointerEvents: "none",
                    maxWidth: 240,
                  }}
                >
                  <div
                    style={{
                      backgroundColor: "var(--color-surface-tooltip)",
                      border: "1px solid var(--color-border-secondary)",
                      borderRadius: "8px",
                      color: "var(--color-text-on-tooltip)",
                      padding: "12px",
                    }}
                  >
                    <p
                      style={{
                        margin: 0,
                        fontWeight: "bold",
                        fontSize: "12px",
                      }}
                    >
                      {hoveredMarker.change.title}
                    </p>
                    <p
                      style={{
                        margin: "4px 0 0",
                        fontSize: "10px",
                        color: "var(--color-text-on-tooltip-secondary)",
                      }}
                    >
                      Methodology change · {formatDate(hoveredMarker.ts)}
                    </p>
                    <p
                      style={{
                        margin: "8px 0 0",
                        fontSize: "11px",
                        lineHeight: 1.45,
                      }}
                    >
                      {hoveredMarker.change.detail}
                    </p>
                  </div>
                </div>
              );
            })()}
        </div>
      </Card>
    </div>
  );
};

export default TimelineChart;
