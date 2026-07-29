// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  getRelativeCoordinate,
} from "recharts";
import { Info } from "lucide-react";
import type { ScatterDataPoint } from "@/types/benchmark.types";
import { getModelColor } from "@/lib/utils/colors";
import { normalizeModelName } from "@/lib/utils/formatters";
import { labelScatterDots } from "@/lib/utils/chartExport";
import CustomScatterTooltip from "@/components/charts/tooltips/ScatterTooltip";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import MetricToggle, { useMetricTab } from "@/components/dashboard/MetricToggle";
import MetricInfo from "@/components/shared/MetricInfo";
import { metricAboutNote } from "@/lib/config/metrics";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";
import { useActiveTab } from "@/hooks/useActiveTab";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";
import { useMobileDetection } from "@/hooks/useMobileDetection";
import ChartInteractionLayer, {
  type ChartInteractionHandle,
} from "@/components/charts/ChartInteractionLayer";

// Human transcription accuracy is 2–4% WER under optimal conditions (per our
// ASR benchmarks doc); the band's ceiling puts human-level-or-better inside
// the zone. Y axis is whole percent.
const HUMAN_WER_CEILING = 4;
// Median human conversational turn-taking gap. X axis is ms. Only meaningful
// against TTFS — humans have no token-streaming analogue, so the zone hides
// on TTFT. TODO(bench-405): confirm the canonical figure with product.
const HUMAN_LATENCY_MS = 200;

// The frontier's plot-edge extensions meet the fastest/most-accurate models
// at up to a right angle, which no x-monotone curve interpolation can round —
// so the path is built by hand: straight runs with quadratic corners. The
// radius is capped by the adjacent segments, so gentle interior bends read as
// one smooth curve while a single dominant model gets a visibly rounded knee.
const PARETO_CORNER_RADIUS = 16;
const roundedFrontierPath = (points: { x: number; y: number }[]) => {
  const pts = points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  const first = pts[0];
  if (!first || pts.length < 2) return "";
  let d = `M${first.x},${first.y}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [a, p, b] = [pts[i - 1]!, pts[i]!, pts[i + 1]!];
    const la = Math.hypot(p.x - a.x, p.y - a.y) || 1;
    const lb = Math.hypot(b.x - p.x, b.y - p.y) || 1;
    const r = Math.min(PARETO_CORNER_RADIUS, la / 2, lb / 2);
    d +=
      `L${p.x - ((p.x - a.x) / la) * r},${p.y - ((p.y - a.y) / la) * r}` +
      `Q${p.x},${p.y} ${p.x + ((b.x - p.x) / lb) * r},${p.y + ((b.y - p.y) / lb) * r}`;
  }
  const last = pts[pts.length - 1]!;
  return `${d}L${last.x},${last.y}`;
};

const LatencyAccuracySection: React.FC = () => {
  const { selectedModels, getScatterData, activeMetric: metric, dedicatedModels } = useDashboard();

  const activeTab = useActiveTab();
  const themeColors = useThemeColors();
  const trackChartHover = useChartHoverTracking("scatter");
  const metricTab = useMetricTab();

  const latencyLabel = metric;
  const scatterData = useMemo(
    () => getScatterData(metric),
    [getScatterData, metric]
  );

  const isMobile = useMobileDetection();
  const sortedData = useMemo(
    () => [...scatterData].sort((a, b) => a.x - b.x || a.y - b.y),
    [scatterData]
  );
  const paretoData = useMemo(() => {
    let minY = Infinity;
    return sortedData.filter((p) => (p.y < minY ? ((minY = p.y), true) : false));
  }, [sortedData]);
  const [activeIdx, setActiveIdx] = useState(-1);
  const chartRef = useRef<HTMLDivElement>(null);
  const interactionRef = useRef<ChartInteractionHandle>(null);
  useEffect(() => setActiveIdx(-1), [sortedData]);
  const activePoint = sortedData[activeIdx];

  // Keyboard access uses one focusable container plus aria-activedescendant
  // instead of focusing the circles: Recharts recreates the shape nodes when
  // the selection re-renders, so DOM focus placed on a circle dies silently
  // with it. Focus never leaves the container; arrows move the selection and
  // the active option id, which survives any node swap. Handlers ignore keys
  // bubbling from the focused chart surface so its arrow navigation stays
  // independent.
  const onPlotKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.target !== e.currentTarget || sortedData.length === 0) return;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIdx((i) => Math.min(i + 1, sortedData.length - 1));
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Escape") {
      setActiveIdx(-1);
    }
  };

  useEffect(() => {
    if (!activePoint) return;
    const dismiss = (e: Event) => {
      if (!chartRef.current?.contains(e.target as Node)) setActiveIdx(-1);
    };
    document.addEventListener("pointerdown", dismiss);
    window.addEventListener("scroll", dismiss, { capture: true, passive: true });
    return () => {
      document.removeEventListener("pointerdown", dismiss);
      window.removeEventListener("scroll", dismiss, { capture: true });
    };
  }, [activePoint]);

  const description = {
    short: `Average ${latencyLabel} and WER per model`,
    detailed:
      "Every voice AI system faces a fundamental trade-off between speed and accuracy. Faster models might sacrifice precision to deliver quick responses, while more accurate models may take additional processing time to ensure correct results. Choose the model that offers the best balance for your specific use case.",
  };

  // Overall run-weighted average, matching the mean over all raw measurements
  const avgLatency = useMemo(() => {
    const totalRuns = scatterData.reduce(
      (acc: number, item: ScatterDataPoint) => acc + item.count,
      0
    );
    if (totalRuns === 0) return 0;
    const sum = scatterData.reduce(
      (acc: number, item: ScatterDataPoint) => acc + item.x * item.count,
      0
    );
    return sum / totalRuns;
  }, [scatterData]);

  // Y domain rounded up to the next 2% step, ticks every 2%
  const { yMax, yTicks } = useMemo(() => {
    const maxWER = scatterData.reduce(
      (acc: number, item: ScatterDataPoint) => Math.max(acc, item.y),
      0
    );
    const max = Math.max(2, Math.ceil((maxWER * 1.1) / 2) * 2);
    const ticks = [];
    for (let t = 0; t <= max; t += 2) ticks.push(t);
    return { yMax: max, yTicks: ticks };
  }, [scatterData]);

  // X ticks at a "nice" step computed from the data, domain rounded up to the last tick
  const { xMax, xTicks } = useMemo(() => {
    const maxLatency = scatterData.reduce(
      (acc: number, item: ScatterDataPoint) => Math.max(acc, item.x),
      0
    );
    const raw = (maxLatency * 1.05) / 5;
    const pow = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    const step =
      [1, 2, 2.5, 5, 10].map((m) => m * pow).find((s) => s >= raw) ?? pow;
    const max = Math.ceil((maxLatency * 1.05) / step) * step;
    const ticks = [];
    for (let t = 0; t <= max; t += step) ticks.push(t);
    return { xMax: max, xTicks: ticks };
  }, [scatterData]);

  // The non-dominated set is rarely convex, so a curve through every one of
  // its points would dip and flatten between them. The drawn curve is its
  // lower convex hull instead — every chord point is achievable by splitting
  // traffic between the two neighboring models, so the hull is the boundary
  // of achievable trade-offs. Non-dominated models off the hull keep their
  // full-opacity highlight.
  const paretoLine = useMemo(() => {
    const hull: ScatterDataPoint[] = [];
    for (const p of paretoData) {
      let a = hull[hull.length - 1];
      let o = hull[hull.length - 2];
      while (o && a && (a.x - o.x) * (p.y - o.y) - (a.y - o.y) * (p.x - o.x) <= 0) {
        hull.pop();
        a = hull[hull.length - 1];
        o = hull[hull.length - 2];
      }
      hull.push(p);
    }
    const first = hull[0];
    const last = hull[hull.length - 1];
    if (!first || !last) return [];
    return [{ ...first, y: yMax }, ...hull, { ...last, x: xMax }];
  }, [paretoData, xMax, yMax]);

  const scrub = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!sortedData.length) return;
    const { relativeX } = getRelativeCoordinate(e);
    const xValue = Number(interactionRef.current?.xValueAt(relativeX));
    if (!Number.isFinite(xValue)) return;
    setActiveIdx(
      sortedData.reduce(
        (best, p, i) =>
          Math.abs(p.x - xValue) < Math.abs((sortedData[best] ?? p).x - xValue)
            ? i
            : best,
        0
      )
    );
  };

  // WER-based: never rendered on S2S (no WER metric).
  if (activeTab === "s2s") return null;

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8">
        <SectionHeader
          label="Latency vs Accuracy"
          description={description}
          note={metricAboutNote(metric)}
          exportNote={metricTab}
          exportXLabel={`Average ${latencyLabel}`}
          exportAnnotate={(clone) => {
            // A tapped point leaves crosshairs, an enlarged dot and dimmed
            // neighbors in the SVG; the export is selection-neutral, so strip
            // that state and restore each dot's frontier opacity. Circles pair
            // with data by (cx, cy) order, the same pairing labelScatterDots
            // uses.
            clone
              .querySelectorAll(".recharts-reference-line")
              .forEach((el) => el.remove());
            const attr = (el: Element, name: string) => Number(el.getAttribute(name));
            const byPosition = Array.from(
              clone.querySelectorAll("[data-export-point]")
            ).sort((a, b) => attr(a, "cx") - attr(b, "cx") || attr(a, "cy") - attr(b, "cy"));
            const ordered = [...scatterData].sort((a, b) => a.x - b.x || b.y - a.y);
            byPosition.forEach((circle, i) => {
              circle.setAttribute("r", "6");
              circle.removeAttribute("stroke");
              const point = ordered[i];
              circle.setAttribute(
                "fill-opacity",
                point && paretoData.includes(point) ? "1" : "0.5"
              );
            });
            const nameCounts = new Map<string, number>();
            for (const { model } of scatterData) {
              const name = normalizeModelName(model);
              nameCounts.set(name, (nameCounts.get(name) ?? 0) + 1);
            }
            labelScatterDots(
              clone,
              scatterData.map(({ model, provider, x, y }) => {
                const name = normalizeModelName(model);
                return {
                  x,
                  y,
                  color: getModelColor(model),
                  label:
                    (nameCounts.get(name) ?? 0) > 1
                      ? `${name} (${provider})`
                      : name,
                };
              }),
              themeColors
            );
          }}
          exportRows={() =>
            scatterData.map(({ model, provider, benchmark, x, y, count }) => ({
              model,
              provider,
              benchmark,
              [`avg_${metric}_ms`]: x,
              avg_wer_percent: y,
              runs: count,
            }))
          }
          stat={{
            label: (
              <MetricInfo metric={metric} align="right">{`Avg ${latencyLabel}`}</MetricInfo>
            ),
            value: `${avgLatency.toFixed(0)} ms`,
          }}
        />

        <div className="flex flex-wrap items-center">
          <MetricToggle />
          <ul data-chart-legend className="mb-4 ml-auto flex items-center gap-4">
            {activeTab === "stt" && metric === "TTFS" && (
              <li
                className="flex items-center gap-1.5 whitespace-nowrap text-xs"
                style={{ color: themeColors.textSecondary }}
              >
                <span
                  className="inline-block h-3 w-3 rounded-[2px]"
                  style={{ backgroundColor: themeColors.zoneStroke }}
                  aria-hidden="true"
                />
                <MetricInfo metric="human-parity" align="right">
                  Human-parity zone{" "}
                  <Info size={12} aria-hidden="true" className="inline align-[-2px]" />
                </MetricInfo>
              </li>
            )}
            <li
              className="flex items-center gap-1.5 whitespace-nowrap text-xs"
              style={{ color: themeColors.textSecondary }}
            >
              <span
                className="inline-block w-4 border-t-2 border-dashed"
                style={{ borderColor: themeColors.axisText }}
                aria-hidden="true"
              />
              <MetricInfo metric="pareto" align="right">
                Pareto frontier{" "}
                <Info size={12} aria-hidden="true" className="inline align-[-2px]" />
              </MetricInfo>
            </li>
          </ul>
        </div>

        <div
          ref={chartRef}
          data-export-frame
          className="relative h-64 select-none"
          role="listbox"
          aria-label={`Models by ${latencyLabel} and WER; arrow keys move between points`}
          tabIndex={0}
          aria-activedescendant={
            activeIdx >= 0 ? `latency-scatter-point-${activeIdx}` : undefined
          }
          onKeyDown={onPlotKeyDown}
          onMouseEnter={trackChartHover}
          onPointerDown={isMobile ? scrub : undefined}
          onPointerMove={
            isMobile
              ? (e) => (e.pointerType === "touch" || e.buttons > 0) && scrub(e)
              : undefined
          }
          onPointerCancel={isMobile ? () => setActiveIdx(-1) : undefined}
          style={isMobile ? { touchAction: "pan-y" } : undefined}
        >
          {activePoint && (
            <div
              className={`absolute top-2 z-10 max-w-[40%] text-xs ${
                activePoint.x > xMax / 2 ? "left-12" : "right-2"
              }`}
              onPointerDown={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                aria-label="Close model details"
                onClick={() => setActiveIdx(-1)}
                className="absolute -right-2 -top-2 z-10 flex h-11 w-11 items-center justify-center rounded-full bg-surface-toggle-inactive text-lg text-text-secondary"
              >
                ×
              </button>
              <CustomScatterTooltip
                active
                payload={[{
                  payload: activePoint,
                  name: activePoint.model,
                  value: activePoint.x,
                  dataKey: "x",
                  graphicalItemId: activePoint.model,
                }]}
                activeTab={activeTab}
                metric={metric}
                dedicatedModels={dedicatedModels}
              />
            </div>
          )}
          <ResponsiveContainer width="100%" height="100%" debounce={200}>
            <ScatterChart
              margin={{ top: 10, right: 8, left: 0, bottom: 0 }}
              accessibilityLayer
            >
              <ChartInteractionLayer ref={interactionRef} />
              <CartesianGrid
                xAxisId={0}
                yAxisId={0}
                stroke={themeColors.grid}
                strokeDasharray="2 2"
              />
              {activeTab === "stt" && metric === "TTFS" && (
                <ReferenceArea
                  x1={0}
                  x2={HUMAN_LATENCY_MS}
                  y1={0}
                  y2={HUMAN_WER_CEILING}
                  fill={themeColors.zoneFill}
                  fillOpacity={1}
                  stroke={themeColors.zoneStroke}
                  strokeWidth={1}
                  ifOverflow="hidden"
                />
              )}
              <XAxis
                dataKey="x"
                type="number"
                name={latencyLabel}
                domain={[0, xMax]}
                ticks={xTicks}
                axisLine={false}
                tickLine={false}
                tick={{ fill: themeColors.axisText, fontSize: 12 }}
                tickFormatter={(value) => `${parseFloat((Number(value) / 1000).toFixed(2))}s`}
              />
              <YAxis
                dataKey="y"
                type="number"
                name="WER (%)"
                width={40}
                domain={[0, yMax]}
                ticks={yTicks}
                axisLine={false}
                tickLine={false}
                tick={{ fill: themeColors.axisText, fontSize: 12 }}
                tickFormatter={(value) => `${value}%`}
              />
              {!isMobile && (
                <Tooltip
                  content={
                    <CustomScatterTooltip
                      activeTab={activeTab}
                      metric={metric}
                      dedicatedModels={dedicatedModels}
                    />
                  }
                  isAnimationActive={false}
                  // Recharts defaults the wrapper to pointer-events: none,
                  // which would make the dedicated badge's explainer inert.
                  wrapperStyle={{ pointerEvents: "auto" }}
                />
              )}
              {activePoint && (
                <ReferenceLine x={activePoint.x} stroke={themeColors.axisText} strokeDasharray="3 3" />
              )}
              {activePoint && (
                <ReferenceLine y={activePoint.y} stroke={themeColors.axisText} strokeDasharray="3 3" />
              )}
              {paretoLine.length > 0 && (
                <Scatter
                  data={paretoLine}
                  line={(props: { points: { x: number; y: number }[] }) => (
                    <path
                      d={roundedFrontierPath(props.points)}
                      fill="none"
                      stroke={themeColors.axisText}
                      strokeWidth={1.5}
                      strokeDasharray="4 4"
                    />
                  )}
                  shape={() => <g />}
                  tooltipType="none"
                  isAnimationActive={false}
                />
              )}
              {selectedModels.map((model: string) => (
                <Scatter
                  key={model}
                  data={scatterData.filter(
                    (item: ScatterDataPoint) => item.model === model
                  )}
                  fill={getModelColor(model)}
                  name={model}
                  isAnimationActive={false}
                  shape={(props: { cx?: number; cy?: number; fill?: string; payload?: ScatterDataPoint }) => {
                    const idx = sortedData.indexOf(props.payload!);
                    return (
                      <circle
                        data-export-point
                        id={`latency-scatter-point-${idx}`}
                        cx={props.cx}
                        cy={props.cy}
                        r={props.payload === activePoint ? 8 : 6}
                        fill={props.fill}
                        fillOpacity={
                          activePoint
                            ? props.payload !== activePoint
                              ? 0.35
                              : 1
                            : paretoData.includes(props.payload!)
                              ? 1
                              : 0.5
                        }
                        stroke={props.payload === activePoint ? themeColors.axisText : undefined}
                        strokeWidth={2}
                        role="option"
                        aria-selected={props.payload === activePoint}
                        aria-label={`${normalizeModelName(props.payload!.model)}: ${props.payload!.x.toFixed(0)}ms ${metric}, ${props.payload!.y.toFixed(1)}% WER${paretoData.includes(props.payload!) ? ", on Pareto frontier" : ""}`}
                        onClick={() => setActiveIdx(idx)}
                      />
                    );
                  }}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div
          className="mt-1 text-center font-mono text-sm"
          style={{ color: themeColors.axisText }}
        >
          <MetricInfo metric={metric}>{latencyLabel}</MetricInfo>
        </div>
      </Card>
    </div>
  );
};

export default LatencyAccuracySection;
