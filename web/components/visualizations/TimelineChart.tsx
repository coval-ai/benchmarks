// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  ReferenceLine,
  type TooltipPayloadEntry,
} from "recharts";
import { getModelColor } from "@/lib/utils/colors";
import { DedicatedInfoIcon } from "@/components/shared/DedicatedInferenceInfo";
import { formatDate, formatTime, getLocalTimeZoneAbbr } from "@/lib/utils/formatters";
import { metricDescriptions } from "@/lib/config/metrics";
import {
  methodologyChanges,
  type MethodologyChange,
  type MethodologyMetricKey
} from "@/lib/config/methodologyChanges";
import CustomTimelineTooltip from "@/components/charts/tooltips/TimelineTooltip";
import ChartInteractionLayer, {
  type ChartInteractionHandle,
} from "@/components/charts/ChartInteractionLayer";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import MetricInfo from "@/components/shared/MetricInfo";
import MetricToggle from "@/components/dashboard/MetricToggle";
import { useActiveTab } from "@/hooks/useActiveTab";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";
import { useMobileDetection } from "@/hooks/useMobileDetection";

interface LegendEntry {
  value: string;
  color?: string;
  dataKey?: string;
  dedicated?: boolean;
}

// Custom legend: names are rendered in black (recharts colors them per-series
// by default), and items fill top-to-bottom within each column so the list
// reads alphabetically down each column rather than across rows. Each entry
// toggles its model in and out of the chart, like a Filters sidebar chip.
export const TimelineLegend: React.FC<{
  payload?: LegendEntry[];
  /** Zoom-clipped series: dimmed and pushed to the end. */
  dimmedKeys?: Set<string>;
  /** Deselected models: dimmed in place so entries never move under the pointer mid-toggle. */
  hiddenKeys?: Set<string>;
  onToggle?: (dataKey: string) => void;
}> = ({ payload, dimmedKeys, hiddenKeys, onToggle }) => {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  return (
    <div className="mt-3 border-t border-border-primary sm:mt-0 sm:border-0">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((prev) => !prev)}
        className="group flex w-full items-center justify-between gap-4 px-2 py-3 text-left sm:hidden"
      >
        <span className="text-sm font-medium text-text-primary">
          Models{" "}
          <span className="font-mono text-xs text-text-tertiary">
            {payload?.length ?? 0}
          </span>
        </span>
        <span aria-hidden className="flex shrink-0 items-center gap-2">
          <span className="text-xs font-light text-text-tertiary">
            {open ? "Click to collapse" : "Click to expand"}
          </span>
          <span className="font-mono text-xl leading-none text-text-tertiary transition-colors group-hover:text-text-primary">
            {open ? "–" : "+"}
          </span>
        </span>
      </button>
      <div
        id={panelId}
        className={`grid transition-[grid-template-rows,visibility] duration-300 ease-in-out ${open ? "visible grid-rows-[1fr]" : "invisible grid-rows-[0fr]"} sm:visible sm:grid-rows-[1fr]`}
      >
        {/* The clip only matters for the mobile collapse animation and scroll;
            above sm it would cut off the dedicated-inference popover. */}
        <div className="overflow-hidden sm:overflow-visible">
          <ul className="grid auto-cols-max grid-flow-col grid-rows-4 gap-x-4 overflow-x-auto px-2 pb-1 sm:block sm:columns-3 sm:gap-x-6 sm:overflow-visible sm:pt-5 lg:columns-4">
    {[...(payload ?? [])]
      .sort(
        (a, b) =>
          Number(dimmedKeys?.has(a.dataKey ?? "") ?? 0) -
          Number(dimmedKeys?.has(b.dataKey ?? "") ?? 0)
      )
      .map((entry) => {
        const hidden = hiddenKeys?.has(entry.dataKey ?? "");
        const dimmed = dimmedKeys?.has(entry.dataKey ?? "") || hidden;
        return (
          <li
            key={entry.dataKey ?? entry.value}
            data-dimmed={dimmed || undefined}
            className="mb-0.5 flex items-start break-inside-avoid"
          >
            <button
              type="button"
              aria-pressed={!hidden}
              onClick={() => onToggle?.(entry.dataKey ?? "")}
              className={`flex min-h-11 flex-1 items-center gap-1.5 rounded-md px-1 py-2 text-left text-xs leading-tight text-text-primary transition-opacity hover:bg-surface-toggle-inactive focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40 sm:min-h-0 sm:items-start sm:py-1${dimmed ? " opacity-35" : ""}`}
            >
              <span
                className="mt-0.5 inline-block w-3 h-3 shrink-0 rounded-[2px]"
                style={{ backgroundColor: entry.color }}
                aria-hidden="true"
              />
              <span>{entry.value}</span>
            </button>
            {entry.dedicated && (
              <DedicatedInfoIcon className="min-h-11 w-8 sm:min-h-0 sm:py-1" />
            )}
          </li>
        );
      })}
          </ul>
        </div>
      </div>
    </div>
  );
};

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

// Scrubber playhead knob: a dot at the top of the reference line so the
// scrubbed instant reads at a glance on mobile, where there is no hover cursor.
const ScrubMarkerDot: React.FC<{
  viewBox?: { x?: number; y?: number };
  color: string;
}> = ({ viewBox, color }) => (
  <circle
    cx={viewBox?.x ?? 0}
    cy={viewBox?.y ?? 0}
    r={4}
    fill={color}
    stroke="var(--color-surface-primary)"
    strokeWidth={1.5}
  />
);

// A timeline point's timestamp (epoch ms) is the 3h bucket start, which the S2S
// sampler uses verbatim as the tick-folder key — so this reproduces that key to
// join a clicked tooltip row to its recording set.
function bucketTickKey(label: number): string {
  return new Date(label).toISOString().replace(/\.\d{3}Z$/, "Z");
}

interface PinnedTooltip {
  label: string;
  payload: TooltipPayloadEntry[];
  x: number;
  y: number;
  flip: boolean;
}

interface PlotBox {
  left: number;
  top: number;
  width: number;
  height: number;
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
  const isMobile = useMobileDetection();
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
    legendModels,
    toggleLegendModel,
    dedicatedModels,
    selectedModels,
    page,
    requestS2SPlay,
  } = useDashboard();
  const trackChartHover = useChartHoverTracking("timeline");

  const chartRef = useRef<HTMLDivElement>(null);
  const surfaceRef = useRef<SVGSVGElement>(null);
  const interactionRef = useRef<ChartInteractionHandle>(null);
  const pinnedRef = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState<PinnedTooltip | null>(null);
  const [mobileScrub, setMobileScrub] = useState<PinnedTooltip | null>(null);
  // Synchronous mirror for the gesture handlers: a tap's pointerup must not
  // depend on the pointerdown's state update having committed.
  const mobileScrubRef = useRef<PinnedTooltip | null>(null);
  const updateMobileScrub = useCallback((next: PinnedTooltip | null) => {
    mobileScrubRef.current = next;
    setMobileScrub(next);
  }, []);
  // Finger actively on the axis — the compact readout only shows live;
  // after lift-off the measurement persists as just the playhead line.
  const [axisScrubLive, setAxisScrubLive] = useState(false);
  const [interactionBox, setInteractionBox] = useState<PlotBox | null>(null);
  const [zoom, setZoom] = useState<{
    x?: [number, number];
    y?: [number, number];
  } | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const axisScrubRef = useRef<{
    x: number;
    moved: boolean;
    following: boolean;
    hadPin: boolean;
  } | null>(null);
  const dragRef = useRef<{
    x: number;
    y: number;
    armX: boolean;
    armY: boolean;
    box: PlotBox;
    left: number;
    top: number;
  } | null>(null);
  const dragEndAtRef = useRef(0);
  const [dragging, setDragging] = useState(false);
  const [hoveredMarker, setHoveredMarker] = useState<{
    change: MethodologyChange;
    x: number;
    ts: number;
  } | null>(null);

  const currentTimeWindow = useMemo(getCurrentTimeWindow, [getCurrentTimeWindow]);
  const [windowStart, windowEnd] = currentTimeWindow;

  // The metric and time window are shared dashboard-wide, so drop the zoom and
  // the marker popover whenever either shifts — the zoomed region may no longer
  // exist in the new data.
  useEffect(() => {
    setHoveredMarker(null);
    setZoom(null);
  }, [metric, windowStart, windowEnd]);

  // Pin and measurement are pixel-anchored; a zoom change moves the ground
  // under them — drop both rather than mislabel a point.
  useEffect(() => {
    setPinned(null);
    updateMobileScrub(null);
  }, [zoom, metric, windowStart, windowEnd, updateMobileScrub]);

  // A mobile chart has separate touch targets for the plot and the date axis.
  // A pin from a desktop-sized layout would otherwise sit over those targets.
  useEffect(() => {
    if (isMobile) setPinned(null);
  }, [isMobile]);

  useEffect(() => {
    if (pinned === null) return;
    const onDocMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (pinnedRef.current?.contains(target)) return;
      // A tap's synthesized mousedown must not kill the pin it just placed;
      // the mobile overlays own in-chart dismissal.
      if (isMobile && chartRef.current?.contains(target)) return;
      if (surfaceRef.current?.contains(target)) return;
      setPinned(null);
    };
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [pinned, isMobile]);

  // Touching or scrolling away dismisses the measurement and the open list.
  // Touch scrolls rarely synthesize mouse events, so listen to capture-phase
  // scroll + pointerdown; scrolling the pinned list itself keeps it open.
  const hasMobileMeasurement = mobileScrub !== null || pinned !== null;
  useEffect(() => {
    if (!isMobile || !hasMobileMeasurement) return;
    const onDocPointerDown = (e: Event) => {
      const target = e.target as Node;
      if (pinnedRef.current?.contains(target)) return;
      if (chartRef.current?.contains(target)) return;
      setPinned(null);
      updateMobileScrub(null);
    };
    const onScroll = (e: Event) => {
      if (pinnedRef.current?.contains(e.target as Node)) return;
      setPinned(null);
      updateMobileScrub(null);
    };
    document.addEventListener("pointerdown", onDocPointerDown);
    window.addEventListener("scroll", onScroll, { capture: true, passive: true });
    return () => {
      document.removeEventListener("pointerdown", onDocPointerDown);
      window.removeEventListener("scroll", onScroll, { capture: true });
    };
  }, [isMobile, hasMobileMeasurement, updateMobileScrub]);

  const themeColors = useThemeColors();
  // Dedicated endpoints chart in Latency Variation, never as timeline lines —
  // a line here would read as the shared fleet's polling cadence.
  const modelsWithData = useMemo(
    () =>
      getModelsWithTimelineData(metric).filter((m) => !dedicatedModels.has(m)),
    [getModelsWithTimelineData, metric, dedicatedModels]
  );
  const windowedTimelineData = useMemo(
    () => getWindowedTimelineData(metric),
    [getWindowedTimelineData, metric]
  );
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
  // S2S buckets are stored in UTC (runner + infra convention), so render the S2S
  // timeline in UTC -- a midnight-UTC point rendered in a local zone behind UTC
  // would slip to the previous day. STT/TTS keep the viewer's local time.
  const isS2S = activeTab === "s2s";
  const displayTz = isS2S ? "UTC" : undefined;
  const tzAbbr = isS2S ? "UTC" : getLocalTimeZoneAbbr();
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
  const timelineTicks = useMemo(getTimelineTicks, [getTimelineTicks]);
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

  const yDomain = useMemo<[number, number | "dataMax"]>(
    () => zoom?.y ?? [0, yAxisMax],
    [zoom?.y, yAxisMax]
  );
  const { yTicks, yTickDecimals } = useMemo(() => {
    const span =
      typeof yDomain[1] === "number" ? yDomain[1] - yDomain[0] : null;
    if (span === null || span <= 0) return { yTicks: undefined, yTickDecimals: 1 };
    const step =
      [50, 100, 200, 250, 500, 1000, 2000].find((s) => span / s <= 6) ?? 2500;
    const ticks = [];
    for (
      let t = Math.ceil(yDomain[0] / step) * step;
      t <= yDomain[0] + span + 1;
      t += step
    ) {
      ticks.push(t);
    }
    return { yTicks: ticks, yTickDecimals: step < 500 ? 2 : 1 };
  }, [yDomain]);

  const yMaxSelectValue = !zoom?.y
    ? "auto"
    : zoom.y[0] === 0 && Y_MAX_PRESETS.includes(zoom.y[1])
      ? String(zoom.y[1])
      : "custom";

  const plotBox = useCallback((): PlotBox | null => {
    const area = interactionRef.current?.plotArea;
    if (!area?.width || !area.height) return null;
    return {
      left: area.x,
      top: area.y,
      width: area.width,
      height: area.height,
    };
  }, []);

  // Recharts owns the exact plot offset (including the Y axis). Mirror it so
  // the mobile touch overlays line up with the visible plot and date axis.
  const syncInteractionBox = useCallback(() => {
    const next = plotBox();
    if (!next) return;
    setInteractionBox((current) =>
      current &&
      current.left === next.left &&
      current.top === next.top &&
      current.width === next.width &&
      current.height === next.height
        ? current
        : next
    );
  }, [plotBox]);

  useEffect(() => {
    if (!isMobile) return;
    const frame = requestAnimationFrame(syncInteractionBox);
    return () => cancelAnimationFrame(frame);
  }, [isMobile, syncInteractionBox, windowedTimelineData, zoom, yAxisMax]);

  const endDrag = () => {
    dragRef.current = null;
    setDragging(false);
    if (boxRef.current) boxRef.current.style.display = "none";
  };

  const startDrag = (
    e: React.PointerEvent<HTMLDivElement>,
    captureImmediately = false
  ) => {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    const box = interactionBox ?? plotBox();
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
    dragRef.current = {
      x,
      y,
      armX: false,
      armY: false,
      box,
      left: rect.left,
      top: rect.top,
    };
    if (captureImmediately) e.currentTarget.setPointerCapture(e.pointerId);
  };

  const moveDrag = (
    e: React.PointerEvent<HTMLDivElement>,
    mobilePlot = false
  ) => {
    const start = dragRef.current;
    const el = boxRef.current;
    if (!start || !el) return;
    const { box } = start;
    const rawX = e.clientX - start.left;
    const rawY = e.clientY - start.top;
    const x = Math.min(Math.max(rawX, box.left), box.left + box.width);
    const y = Math.min(Math.max(rawY, box.top), box.top + box.height);
    const dx = Math.abs(rawX - start.x);
    const dy = Math.abs(rawY - start.y);
    if (mobilePlot) {
      // The plot is exclusively the crop zone on mobile. Once the finger has
      // moved, any direction selects its corresponding X/Y range.
      start.armX = start.armX || dx > 8;
      start.armY = start.armY || dy > 8;
    } else {
      start.armX = start.armX || dx > 12;
      start.armY = start.armY || dy > 12;
    }
    if (!start.armX && !start.armY) {
      el.style.display = "none";
      return;
    }
    // Desktop waits for a real drag so a click can still pin its tooltip. The
    // mobile plot overlay never pins, so it safely captures from touch-down.
    if (!mobilePlot && !e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.setPointerCapture(e.pointerId);
    }
    if (!dragging) setDragging(true);
    el.style.display = "block";
    el.style.left = `${start.armX ? Math.min(start.x, x) : box.left}px`;
    el.style.width = `${start.armX ? Math.abs(x - start.x) : box.width}px`;
    el.style.top = `${start.armY ? Math.min(start.y, y) : box.top}px`;
    el.style.height = `${start.armY ? Math.abs(y - start.y) : box.height}px`;
  };

  const finishDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    const start = dragRef.current;
    endDrag();
    if (!start || (!start.armX && !start.armY)) return;
    const { box } = start;
    dragEndAtRef.current = Date.now();
    const x = Math.min(
      Math.max(e.clientX - start.left, box.left),
      box.left + box.width
    );
    const y = Math.min(
      Math.max(e.clientY - start.top, box.top),
      box.top + box.height
    );
    const applyX = start.armX && Math.abs(x - start.x) > 8;
    const applyY = start.armY && Math.abs(y - start.y) > 8;
    let yRange = zoom?.y;
    if (applyY) {
      const lo = Number(interactionRef.current?.yValueAt(Math.max(start.y, y)));
      const hi = Number(interactionRef.current?.yValueAt(Math.min(start.y, y)));
      if (Number.isFinite(lo) && Number.isFinite(hi)) {
        yRange = [Math.max(0, lo), hi];
      }
    }
    if (!applyX && yRange === zoom?.y) return;
    const xLo = Number(interactionRef.current?.xValueAt(Math.min(start.x, x)));
    const xHi = Number(interactionRef.current?.xValueAt(Math.max(start.x, x)));
    setZoom({
      x: applyX && Number.isFinite(xLo) && Number.isFinite(xHi) ? [xLo, xHi] : zoom?.x,
      y: yRange,
    });
  };

  const scrubDateAxis = (e: React.PointerEvent<HTMLDivElement>) => {
    const box = interactionBox ?? plotBox();
    const rect = chartRef.current?.getBoundingClientRect();
    if (!box || !rect || windowedTimelineData.length === 0) return;
    const x = Math.min(
      Math.max(e.clientX - rect.left, box.left),
      box.left + box.width
    );
    const timestamp =
      xDomain[0] + ((x - box.left) / box.width) * (xDomain[1] - xDomain[0]);
    const point = windowedTimelineData.reduce((nearest, candidate) =>
      Math.abs(candidate.timestamp - timestamp) <
      Math.abs(nearest.timestamp - timestamp)
        ? candidate
        : nearest
    );
    const payload = modelsWithData.flatMap((model) => {
      const value = point[`${model}_value`];
      return typeof value === "number"
        ? [{
            dataKey: `${model}_value`,
            graphicalItemId: `${model}_value`,
            value,
            name: formatChartLabel(model, getProviderForModel(model)),
            color: getModelColor(model),
          }]
        : [];
    });
    if (payload.length === 0) {
      updateMobileScrub(null);
      return;
    }
    updateMobileScrub({
      label: String(point.timestamp),
      payload,
      x,
      y: box.top + 12,
      flip: x > (chartRef.current?.clientWidth ?? 0) / 2,
    });
  };

  // A touch-down must not move an existing measurement — only real travel
  // does, so a tap can inspect the persisted spot wherever it lands.
  const startAxisScrub = (e: React.PointerEvent<HTMLDivElement>) => {
    const fresh = pinned == null && mobileScrubRef.current == null;
    axisScrubRef.current = {
      x: e.clientX,
      moved: false,
      following: fresh,
      hadPin: pinned != null,
    };
    if (fresh) {
      setAxisScrubLive(true);
      scrubDateAxis(e);
    }
  };

  const moveAxisScrub = (e: React.PointerEvent<HTMLDivElement>) => {
    const scrub = axisScrubRef.current;
    if (!scrub) return;
    if (!scrub.moved && Math.abs(e.clientX - scrub.x) > 8) {
      scrub.moved = true;
      scrub.following = true;
      // a real scrub takes over from the open list
      setPinned(null);
      setAxisScrubLive(true);
    }
    if (scrub.following) scrubDateAxis(e);
  };

  // Scrub lift keeps only the playhead; a tap toggles the full ranked list
  // for the persisted spot.
  const endAxisScrub = () => {
    const scrub = axisScrubRef.current;
    axisScrubRef.current = null;
    setAxisScrubLive(false);
    if (!scrub || scrub.moved) return;
    const measurement = mobileScrubRef.current;
    if (scrub.hadPin) {
      setPinned(null);
    } else if (measurement) {
      // anchor the list to the half of the plot the playhead is not in
      const box = interactionBox ?? plotBox();
      const onLeft = box
        ? measurement.x < box.left + box.width / 2
        : false;
      setPinned(
        box
          ? {
              ...measurement,
              x: onLeft ? box.left + box.width : box.left,
              y: box.top,
              flip: onLeft,
            }
          : measurement
      );
    }
  };

  // A touch hijacked into a native scroll fires pointercancel with no travel;
  // that must not read as a tap — scrolling away dismisses instead.
  const cancelAxisScrub = () => {
    axisScrubRef.current = null;
    setAxisScrubLive(false);
    updateMobileScrub(null);
  };

  // Whether a data value sits inside the visible (possibly zoomed) Y range.
  // Recharts doesn't clip active dots, so out-of-view dots must not render.
  const inYView = (v?: number) =>
    v != null &&
    (typeof yDomain[1] !== "number" ||
      (v >= yDomain[0] && v <= yDomain[1]));

  // The legend is rendered as a sibling BELOW the chart (not a recharts
  // <Legend>), so the plot keeps the full card height instead of being
  // squeezed — a 20-series legend inside the chart left only ~40px to plot on
  // mobile. Its universe is every data-backed model, chip-style: filtered-out
  // models stay listed but dim, and a click toggles them back in.
  const legendModelList = useMemo(
    () => getModelsWithTimelineData(metric, legendModels),
    [getModelsWithTimelineData, metric, legendModels]
  );
  const legendPayload: LegendEntry[] = legendModelList.map((model) => ({
    value: formatChartLabel(model, getProviderForModel(model)),
    color: getModelColor(model),
    dataKey: `${model}_value`,
    dedicated: dedicatedModels.has(model),
  }));
  // Dedicated models never draw here, so their entry reads permanently
  // "off" — they live on every other chart instead.
  const plottedKeys = new Set(modelsWithData);
  const legendHiddenKeys = new Set<string>();
  for (const model of legendModelList) {
    if (!plottedKeys.has(model)) legendHiddenKeys.add(`${model}_value`);
  }

  // Mobile has no hover cursor, so a vertical playhead marks the instant being
  // scrubbed or pinned — the timestamp both the compact readout and the pinned
  // list are reading from.
  const scrubMarkerTs = isMobile
    ? Number(mobileScrub?.label ?? pinned?.label ?? NaN)
    : NaN;

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
          hint={
            isMobile
              ? "Drag chart to zoom · swipe axis for values"
              : undefined
          }
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
                className="h-11 rounded-md bg-surface-toggle-inactive px-3 text-xs font-medium text-text-secondary transition-colors hover:text-text-primary lg:h-auto lg:py-1"
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
                className="h-11 rounded-md bg-surface-toggle-inactive px-3 text-xs font-medium text-text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40 lg:h-auto lg:px-2"
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
          data-export-frame
          className="relative h-96 cursor-crosshair select-none touch-pan-y"
          onMouseEnter={trackChartHover}
          onDoubleClick={() => {
            if (Date.now() - dragEndAtRef.current > 400) setZoom(null);
          }}
          onPointerDown={(e) => {
            if (e.pointerType !== "mouse") return;
            e.preventDefault();
            surfaceRef.current?.blur();
            startDrag(e);
          }}
          onPointerMove={(e) => {
            if (e.pointerType !== "mouse") return;
            moveDrag(e);
          }}
          onPointerUp={(e) => {
            if (e.pointerType !== "mouse") return;
            finishDrag(e);
          }}
          onPointerCancel={endDrag}
        >
          <ResponsiveContainer
            width="100%"
            height="100%"
            debounce={200}
            onResize={() => requestAnimationFrame(syncInteractionBox)}
          >
            <LineChart
              ref={surfaceRef}
              data={windowedTimelineData}
              accessibilityLayer
              margin={{ top: 5, right: 8, left: 0, bottom: 5 }}
              onClick={(state) => {
                if (Date.now() - dragEndAtRef.current < 300) return;
                const { activeLabel: lbl, activeCoordinate: coord } = state;
                const point = windowedTimelineData.find(
                  (candidate) => candidate.timestamp === Number(lbl)
                );
                const payload = point
                  ? modelsWithData.flatMap((model) => {
                      const value = point[`${model}_value`];
                      return typeof value === "number"
                        ? [{
                            dataKey: `${model}_value`,
                            graphicalItemId: `${model}_value`,
                            value,
                            name: formatChartLabel(model, getProviderForModel(model)),
                            color: getModelColor(model),
                          }]
                        : [];
                    })
                  : [];
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
                    label: String(lbl),
                    payload,
                    x,
                    y,
                    flip: width > 0 && x > width / 2,
                  };
                });
              }}
            >
              <ChartInteractionLayer ref={interactionRef} />
              <CartesianGrid
                xAxisId={0}
                yAxisId={0}
                vertical={false}
                strokeDasharray="2 2"
                stroke={themeColors.grid}
              />
              <XAxis
                dataKey="timestamp"
                type="number"
                scale="time"
                domain={xDomain}
                ticks={zoomTicks ?? timelineTicks}
                allowDataOverflow
                axisLine={false}
                tickLine={false}
                tick={{ fill: themeColors.axisText, fontSize: 12 }}
                tickFormatter={(value) =>
                  dateTicks ? formatDate(value, displayTz) : formatTime(value, displayTz)
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
                isAnimationActive={false}
                content={
                  <CustomTimelineTooltip
                    getProviderForModel={getProviderForModel}
                    showDate={dateScale}
                    dimmedKeys={dimmedLegendKeys}
                    compact
                  />
                }
                active={pinned || dragging || hoveredMarker || isMobile ? false : undefined}
                cursor={!dragging && !hoveredMarker && !isMobile}
              />
              {modelsWithData.map((model) => (
                <Line
                  key={model}
                  id={`${model}_value`}
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
              {Number.isFinite(scrubMarkerTs) && (
                <ReferenceLine
                  x={scrubMarkerTs}
                  stroke={themeColors.label}
                  strokeWidth={1.5}
                  strokeOpacity={0.65}
                  label={<ScrubMarkerDot color={themeColors.label} />}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
          {modelsWithData.length === 0 &&
            selectedModels.some((m) => dedicatedModels.has(m)) && (
              <p className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center px-8 text-center text-sm text-text-secondary">
                Dedicated inference endpoints do not show on the shared
                timeline. See Latency Variation below.
              </p>
            )}
          <div
            ref={boxRef}
            className="pointer-events-none absolute z-10 hidden rounded-sm border border-dashed border-text-secondary/50 bg-text-secondary/10"
          />
          {interactionBox && (
            <>
              {/* Mobile interaction is deliberately partitioned: dragging the
                  plot selects a crop, while the axis alone scrubs values. */}
              <div
                className="absolute z-[15] touch-none md:hidden"
                style={{
                  left: interactionBox.left,
                  top: interactionBox.top,
                  width: interactionBox.width,
                  height: interactionBox.height,
                }}
                onPointerDown={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setPinned(null);
                  updateMobileScrub(null);
                  startDrag(e, true);
                }}
                onPointerMove={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  moveDrag(e, true);
                }}
                onPointerUp={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  finishDrag(e);
                }}
                onPointerCancel={(e) => {
                  e.stopPropagation();
                  endDrag();
                }}
              />
              <div
                className="absolute z-[15] border-t border-dashed border-text-secondary/40 bg-surface-toggle-inactive/35 touch-pan-y md:hidden"
                style={{
                  left: interactionBox.left,
                  top: interactionBox.top + interactionBox.height,
                  width: interactionBox.width,
                  bottom: 0,
                }}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  startAxisScrub(e);
                }}
                onPointerMove={(e) => {
                  e.stopPropagation();
                  moveAxisScrub(e);
                }}
                onPointerUp={(e) => {
                  e.stopPropagation();
                  endAxisScrub();
                }}
                onPointerCancel={(e) => {
                  e.stopPropagation();
                  cancelAxisScrub();
                }}
              >
              </div>
            </>
          )}
          {mobileScrub && axisScrubLive && !pinned && (
            <div
              className="pointer-events-none absolute z-20 shadow-lg md:hidden"
              style={{
                left: mobileScrub.x,
                top: mobileScrub.y,
                transform: mobileScrub.flip
                  ? "translate(calc(-100% - 12px), 0)"
                  : "translate(12px, 0)",
              }}
            >
              <CustomTimelineTooltip
                active
                payload={mobileScrub.payload}
                label={mobileScrub.label}
                getProviderForModel={getProviderForModel}
                showDate={dateScale}
                dimmedKeys={dimmedLegendKeys}
                compact
                interactionHint="tap axis to see all"
              />
            </div>
          )}
          {pinned && (
            <div
              ref={pinnedRef}
              className="absolute z-20 cursor-auto shadow-lg"
              style={{
                left: pinned.x,
                top: pinned.y,
                transform: isMobile
                  ? pinned.flip
                    ? "translate(calc(-100% - 12px), 0)"
                    : "translate(12px, 0)"
                  : pinned.flip
                    ? "translate(calc(-100% - 12px), -50%)"
                    : "translate(12px, -50%)",
              }}
              onPointerDown={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                aria-label="Close pinned stats"
                onClick={() => setPinned(null)}
                className="absolute right-0 top-0 z-10 flex h-11 w-11 items-center justify-center rounded-full bg-surface-toggle-inactive text-lg leading-none text-text-secondary hover:text-text-primary md:right-2 md:top-2 md:h-5 md:w-5 md:text-xs"
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
                maxHeight={isMobile ? 106 : undefined}
                onModelClick={
                  page === "s2s"
                    ? (model, label) =>
                        requestS2SPlay(bucketTickKey(label), getProviderForModel(model))
                    : undefined
                }
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
                    Methodology change · {formatDate(hoveredMarker.ts, displayTz)}
                  </p>
                  <p className="mt-1.5">{hoveredMarker.change.detail}</p>
                </div>
              );
            })()}
        </div>
        {legendPayload.length > 0 && (
          <div data-chart-legend>
            <TimelineLegend
              payload={legendPayload}
              dimmedKeys={dimmedLegendKeys}
              hiddenKeys={legendHiddenKeys}
              onToggle={(dataKey) =>
                toggleLegendModel(dataKey.replace(/_value$/, ""))
              }
            />
          </div>
        )}
      </Card>
    </div>
  );
};

export default TimelineChart;
