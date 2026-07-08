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
import MetricInfo from "@/components/shared/MetricInfo";
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

const Y_MAX_PRESETS = [500, 1000, 1500, 2000, 3000, 5000];

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
  const chartInstRef = useRef<React.ElementRef<typeof LineChart>>(null);
  const pinnedRef = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState<PinnedTooltip | null>(null);
  const [zoom, setZoom] = useState<{
    x?: [number, number];
    y?: [number, number];
  } | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    x: number;
    y: number;
    armX: boolean;
    armY: boolean;
  } | null>(null);
  const dragEndAtRef = useRef(0);
  const [dragging, setDragging] = useState(false);
  const [hoveredMarker, setHoveredMarker] = useState<{
    change: MethodologyChange;
    x: number;
    ts: number;
  } | null>(null);

  const currentTimeWindow = getCurrentTimeWindow();
  const [windowStart, windowEnd] = currentTimeWindow;

  // The metric and time window are shared dashboard-wide, so drop the zoom and
  // the marker popover whenever either shifts — the zoomed region may no longer
  // exist in the new data.
  useEffect(() => {
    setHoveredMarker(null);
    setZoom(null);
  }, [metric, windowStart, windowEnd]);

  // The pin is anchored in pixel space, so a zoom change (drag, Y-max select,
  // reset) moves the ground under it — drop it rather than mislabel a point.
  useEffect(() => {
    setPinned(null);
  }, [zoom, metric, windowStart, windowEnd]);

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
  const zoomX = zoom?.x;
  const xDomain = zoomX ?? currentTimeWindow;
  const zoomTicks = useMemo(
    () =>
      zoomX &&
      Array.from(
        { length: 6 },
        (_, i) => zoomX[0] + ((zoomX[1] - zoomX[0]) * i) / 5
      ),
    [zoomX]
  );
  const dateScale = dataTimeWindow !== "24h";
  const dateTicks =
    dateScale && !(zoomX && zoomX[1] - zoomX[0] <= 48 * 60 * 60 * 1000);
  const xAxisLabel = dateTicks ? "Date" : tzAbbr ? `Time (${tzAbbr})` : "Time";

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
      if (zoomX && (point.timestamp < zoomX[0] || point.timestamp > zoomX[1]))
        continue;
      const record = point as Record<string, number>;
      for (const model of modelsWithData) {
        const value = record[`${model}_value`];
        if (typeof value === "number" && !Number.isNaN(value) && value > max) {
          max = value;
        }
      }
    }
    if (max === 0) return "dataMax" as const;
    const step = max > 2000 ? 500 : max > 500 ? 250 : 50;
    return Math.ceil(max / step) * step;
  }, [getTimelineData, metric, modelsWithData, zoomX]);

  const yDomain: [number, number | "dataMax"] = zoom?.y ?? [0, yAxisMax];
  const ySpan =
    typeof yDomain[1] === "number" ? yDomain[1] - yDomain[0] : null;

  // Ticks land on a tidy step sized to the zoomed span; finer steps need a
  // second decimal to stay distinguishable.
  let yTicks: number[] | undefined;
  let yTickDecimals = 1;
  if (ySpan !== null && ySpan > 0) {
    const step =
      [50, 100, 200, 250, 500, 1000, 2000].find((s) => ySpan / s <= 6) ?? 2500;
    yTicks = [];
    for (
      let t = Math.ceil(yDomain[0] / step) * step;
      t <= yDomain[0] + ySpan + 1;
      t += step
    ) {
      yTicks.push(t);
    }
    yTickDecimals = step < 500 ? 2 : 1;
  }

  const yMaxSelectValue = !zoom?.y
    ? "auto"
    : zoom.y[0] === 0 && Y_MAX_PRESETS.includes(zoom.y[1])
      ? String(zoom.y[1])
      : "custom";

  // Plot-area box in container pixels, read off the chart instance (recharts
  // keeps it in state; mouse-event payloads omit it).
  const plotBox = () => {
    const offset = chartInstRef.current?.state.offset;
    if (!offset?.width || !offset.height) return null;
    return {
      left: offset.left ?? 0,
      top: offset.top ?? 0,
      width: offset.width,
      height: offset.height,
    };
  };

  const endDrag = () => {
    dragRef.current = null;
    setDragging(false);
    if (boxRef.current) boxRef.current.style.display = "none";
  };

  // Whether a data value sits inside the visible (possibly zoomed) Y range.
  // Recharts doesn't clip active dots, so out-of-view dots must not render.
  const inYView = (v?: number) =>
    v != null &&
    (typeof yDomain[1] !== "number" ||
      (v >= yDomain[0] && v <= yDomain[1]));

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
            label: (
              <MetricInfo metric={metric} align="right">{`Average ${metricLabel}`}</MetricInfo>
            ),
            value: `${avgValue.toFixed(0)} ms`,
          }}
        />

        <div className="flex items-start justify-between gap-2">
          <MetricToggle />
          <div className="mb-4 ml-auto flex items-center gap-2">
            {zoom && (
              <button
                type="button"
                onClick={() => setZoom(null)}
                className="rounded-md bg-surface-toggle-inactive px-3 py-1 text-xs font-medium text-text-secondary transition-colors hover:text-text-primary"
              >
                Reset zoom
              </button>
            )}
            <label className="flex items-center gap-1.5 text-xs font-medium text-text-secondary">
              Y max
              <select
                value={yMaxSelectValue}
                onChange={(e) => {
                  const v = e.target.value;
                  if (v === "custom") return;
                  setZoom((z) =>
                    v === "auto"
                      ? z?.x
                        ? { x: z.x }
                        : null
                      : { x: z?.x, y: [0, Number(v)] }
                  );
                }}
                className="rounded-md bg-surface-toggle-inactive px-2 py-1 text-xs font-medium text-text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40"
              >
                <option value="auto">Auto</option>
                {Y_MAX_PRESETS.map((v) => (
                  <option key={v} value={v}>{`${v / 1000}s`}</option>
                ))}
                {yMaxSelectValue === "custom" && (
                  <option value="custom">Custom</option>
                )}
              </select>
            </label>
          </div>
        </div>

        <div
          ref={chartRef}
          className="relative h-96 cursor-crosshair select-none"
          onMouseEnter={trackChartHover}
          onDoubleClick={() => {
            if (Date.now() - dragEndAtRef.current > 400) setZoom(null);
          }}
          onPointerDown={(e) => {
            if (e.pointerType !== "mouse" || e.button !== 0) return;
            const box = plotBox();
            const rect = chartRef.current?.getBoundingClientRect();
            if (!box || !rect) return;
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            if (
              x < box.left ||
              x > box.left + box.width ||
              y < box.top ||
              y > box.top + box.height
            )
              return;
            dragRef.current = { x, y, armX: false, armY: false };
          }}
          onPointerMove={(e) => {
            const start = dragRef.current;
            const box = plotBox();
            const rect = chartRef.current?.getBoundingClientRect();
            const el = boxRef.current;
            if (!start || !box || !rect || !el) return;
            const x = Math.min(
              Math.max(e.clientX - rect.left, box.left),
              box.left + box.width
            );
            const y = Math.min(
              Math.max(e.clientY - rect.top, box.top),
              box.top + box.height
            );
            start.armX = start.armX || Math.abs(x - start.x) > 12;
            start.armY = start.armY || Math.abs(y - start.y) > 12;
            if (!start.armX && !start.armY) {
              el.style.display = "none";
              return;
            }
            // Capture only once a real drag is in progress — capturing on
            // pointerdown would swallow the click recharts needs for pinning.
            chartRef.current?.setPointerCapture(e.pointerId);
            if (!dragging) setDragging(true);
            el.style.display = "block";
            el.style.left = `${start.armX ? Math.min(start.x, x) : box.left}px`;
            el.style.width = `${start.armX ? Math.abs(x - start.x) : box.width}px`;
            el.style.top = `${start.armY ? Math.min(start.y, y) : box.top}px`;
            el.style.height = `${start.armY ? Math.abs(y - start.y) : box.height}px`;
          }}
          onPointerUp={(e) => {
            const start = dragRef.current;
            const box = plotBox();
            const rect = chartRef.current?.getBoundingClientRect();
            endDrag();
            if (!start || !box || !rect || (!start.armX && !start.armY)) return;
            dragEndAtRef.current = Date.now();
            const x = Math.min(
              Math.max(e.clientX - rect.left, box.left),
              box.left + box.width
            );
            const y = Math.min(
              Math.max(e.clientY - rect.top, box.top),
              box.top + box.height
            );
            const applyX = start.armX && Math.abs(x - start.x) > 8;
            const applyY = start.armY && Math.abs(y - start.y) > 8;
            // Invert pixels through recharts' own y-scale so this works even
            // on the automatic domain, where yDomain[1] is still "dataMax".
            const yScale = (
              chartInstRef.current?.state.yAxisMap &&
              (Object.values(chartInstRef.current.state.yAxisMap)[0] as {
                scale?: { invert?: (n: number) => number };
              })
            )?.scale;
            let yRange = zoom?.y;
            if (applyY && yScale?.invert) {
              yRange = [
                Math.max(0, yScale.invert(Math.max(start.y, y))),
                yScale.invert(Math.min(start.y, y)),
              ];
            }
            if (!applyX && yRange === zoom?.y) return;
            const toX = (px: number) =>
              xDomain[0] +
              ((px - box.left) / box.width) * (xDomain[1] - xDomain[0]);
            setZoom({
              x: applyX
                ? [toX(Math.min(start.x, x)), toX(Math.max(start.x, x))]
                : zoom?.x,
              y: yRange,
            });
          }}
          onPointerCancel={endDrag}
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              ref={chartInstRef}
              data={windowedTimelineData}
              margin={{ top: 5, right: 8, left: 0, bottom: 5 }}
              onClick={(state) => {
                if (Date.now() - dragEndAtRef.current < 300) return;
                const lbl = state?.activeLabel;
                const coord = state?.activeCoordinate;
                const payload = (state?.activePayload ?? []) as TooltipPayloadItem[];
                const hasRows = payload.some(
                  (item) => typeof item?.value === "number" && item.value > 0
                );
                if (lbl == null || !coord || !hasRows) return;
                setPinned((cur) => {
                  if (cur) return null;
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
                domain={xDomain}
                ticks={zoomTicks ?? getTimelineTicks()}
                allowDataOverflow
                axisLine={false}
                tickLine={false}
                tick={{ fill: themeColors.axisText, fontSize: 12 }}
                tickFormatter={(value) =>
                  dateTicks ? formatDate(value) : formatTime(value)
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
                domain={yDomain}
                ticks={yTicks}
                allowDataOverflow
                tickFormatter={(value) =>
                  `${(value / 1000).toFixed(yTickDecimals)}s`
                }
              />
              <Tooltip
                content={
                  <CustomTimelineTooltip
                    getProviderForModel={getProviderForModel}
                    showDate={dateScale}
                    highlightRange={zoom?.y}
                    compact
                  />
                }
                active={pinned || dragging ? false : undefined}
                cursor={!dragging}
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
                  activeDot={
                    dragging
                      ? false
                      : (props: { cx?: number; cy?: number; value?: number }) =>
                          inYView(props.value) ? (
                            <circle
                              cx={props.cx}
                              cy={props.cy}
                              r={modelsWithData.length === 1 ? 7 : 6}
                              fill={getModelColor(model)}
                            />
                          ) : (
                            <g />
                          )
                  }
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
          <div
            ref={boxRef}
            className="pointer-events-none absolute z-10 hidden rounded-sm border border-dashed border-text-secondary/50 bg-text-secondary/10"
          />
          {pinned && (
            <div
              ref={pinnedRef}
              className="absolute z-20 cursor-auto shadow-lg"
              style={{
                left: pinned.x,
                top: pinned.y,
                transform: pinned.flip
                  ? "translate(calc(-100% - 12px), -50%)"
                  : "translate(12px, -50%)",
              }}
              onPointerDown={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                aria-label="Close pinned stats"
                onClick={() => setPinned(null)}
                className="absolute right-2 top-2 z-10 flex h-5 w-5 items-center justify-center rounded-full bg-surface-toggle-inactive text-xs leading-none text-text-secondary hover:text-text-primary"
              >
                ×
              </button>
              <CustomTimelineTooltip
                active
                payload={pinned.payload}
                label={pinned.label}
                getProviderForModel={getProviderForModel}
                showDate={dateScale}
                highlightRange={zoom?.y}
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
