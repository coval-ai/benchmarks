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
const TimelineLegend: React.FC<{
  payload?: LegendEntry[];
  dimmedKeys?: Set<string>;
}> = ({ payload, dimmedKeys }) => (
  <ul className="columns-2 gap-x-4 px-2 pt-5 sm:columns-3 sm:gap-x-6 lg:columns-4">
    {[...(payload ?? [])]
      .sort(
        (a, b) =>
          Number(dimmedKeys?.has(a.dataKey ?? "") ?? 0) -
          Number(dimmedKeys?.has(b.dataKey ?? "") ?? 0)
      )
      .map((entry) => {
        const dimmed = dimmedKeys?.has(entry.dataKey ?? "");
        return (
          <li
            key={entry.dataKey ?? entry.value}
            data-dimmed={dimmed || undefined}
            className={`mb-1.5 flex items-start gap-1.5 text-xs leading-tight text-text-primary break-inside-avoid${dimmed ? " opacity-35" : ""}`}
          >
            <span
              className="mt-0.5 inline-block w-3 h-3 shrink-0 rounded-[2px]"
              style={{ backgroundColor: entry.color }}
              aria-hidden="true"
            />
            <span>{entry.value}</span>
          </li>
        );
      })}
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

// Liang–Barsky: does the segment (x0,y0)->(x1,y1) touch the axis-aligned
// rectangle [xMin,xMax]x[yMin,yMax]? Used to tell whether any part of a series
// line falls inside the zoom crop, so the legend dims only fully-clipped ones.
function segmentIntersectsRect(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  xMin: number,
  yMin: number,
  xMax: number,
  yMax: number
): boolean {
  let t0 = 0;
  let t1 = 1;
  const dx = x1 - x0;
  const dy = y1 - y0;
  const clip = (p: number, q: number): boolean => {
    if (p === 0) return q >= 0; // parallel to this edge; inside iff q >= 0
    const r = q / p;
    if (p < 0) {
      if (r > t1) return false;
      if (r > t0) t0 = r;
    } else {
      if (r < t0) return false;
      if (r < t1) t1 = r;
    }
    return true;
  };
  return (
    clip(-dx, x0 - xMin) &&
    clip(dx, xMax - x0) &&
    clip(-dy, y0 - yMin) &&
    clip(dy, yMax - y0) &&
    t0 <= t1
  );
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
  const zoomY = zoom?.y;

  // A series dims only when the crop clips it fully off-chart. Recharts clips
  // the polyline to the X and Y domain together, so a series is visible if any
  // segment between consecutive points passes through the crop rectangle —
  // even when both endpoints sit outside it (e.g. a spike crossing the top, or
  // a line entering only at the left/right edge). Test each segment against the
  // rectangle with Liang–Barsky rather than checking points in isolation.
  const dimmedLegendKeys = useMemo(() => {
    const dimmed = new Set<string>();
    if (!zoomY) return dimmed;
    const [xLo, xHi] = zoomX ?? [windowStart, windowEnd];
    const [yLo, yHi] = zoomY;
    for (const model of modelsWithData) {
      const key = `${model}_value`;
      let prevT: number | null = null;
      let prevV = 0;
      let visible = false;
      for (const point of windowedTimelineData) {
        const value = (point as Record<string, number | undefined>)[key];
        if (typeof value !== "number" || Number.isNaN(value)) {
          prevT = null;
          continue;
        }
        const t = point.timestamp;
        if (
          prevT === null
            ? t >= xLo && t <= xHi && value >= yLo && value <= yHi
            : segmentIntersectsRect(prevT, prevV, t, value, xLo, yLo, xHi, yHi)
        ) {
          visible = true;
          break;
        }
        prevT = t;
        prevV = value;
      }
      if (!visible) dimmed.add(key);
    }
    return dimmed;
  }, [zoomY, zoomX, windowStart, windowEnd, windowedTimelineData, modelsWithData]);
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
          exportRows={() =>
            windowedTimelineData.map((point) => ({
              time: point.timestampLabel,
              ...Object.fromEntries(
                modelsWithData.map((model) => [
                  `${model}_${metric}_ms`,
                  point[`${model}_value`],
                ])
              ),
            }))
          }
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
          className="relative h-96 cursor-crosshair select-none touch-pan-y"
          onMouseEnter={trackChartHover}
          onDoubleClick={() => {
            if (Date.now() - dragEndAtRef.current > 400) setZoom(null);
          }}
          onPointerDown={(e) => {
            if (e.button !== 0) return;
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
            if (e.pointerType !== "mouse") setDragging(true);
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
            start.armY =
              start.armY ||
              (e.pointerType === "mouse" && Math.abs(y - start.y) > 12);
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
                    dimmedKeys={dimmedLegendKeys}
                    compact
                  />
                }
                active={pinned || dragging || hoveredMarker ? false : undefined}
                cursor={!dragging && !hoveredMarker}
              />
              {modelsWithData.length > 1 && (
                <Legend
                  content={<TimelineLegend dimmedKeys={dimmedLegendKeys} />}
                />
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
                dimmedKeys={dimmedLegendKeys}
              />
            </div>
          )}
          {hoveredMarker &&
            (() => {
              const width = chartRef.current?.clientWidth ?? 0;
              const pad = 132;
              const left =
                width > pad * 2
                  ? Math.min(Math.max(hoveredMarker.x, pad), width - pad)
                  : hoveredMarker.x;
              return (
                <div
                  role="tooltip"
                  className="pointer-events-none absolute z-30 w-64 -translate-x-1/2 rounded-lg border border-border-secondary bg-surface-tooltip px-3 py-2 text-left text-xs font-normal leading-snug text-[var(--color-text-on-tooltip)] shadow-md"
                  style={{ left, top: 26 }}
                >
                  <p className="font-semibold">{hoveredMarker.change.title}</p>
                  <p className="mt-0.5 text-[10px] text-[var(--color-text-on-tooltip-secondary)]">
                    Methodology change · {formatDate(hoveredMarker.ts)}
                  </p>
                  <p className="mt-1.5">{hoveredMarker.change.detail}</p>
                </div>
              );
            })()}
        </div>
      </Card>
    </div>
  );
};

export default TimelineChart;
