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
import MetricToggle from "@/components/dashboard/MetricToggle";
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

const LatencyAccuracySection: React.FC = () => {
  const { selectedModels, getScatterData, activeMetric: metric } = useDashboard();

  const activeTab = useActiveTab();
  const themeColors = useThemeColors();
  const trackChartHover = useChartHoverTracking("scatter");

  const latencyLabel = metric;
  const scatterData = useMemo(
    () => getScatterData(metric),
    [getScatterData, metric]
  );

  const isMobile = useMobileDetection();
  const sortedData = useMemo(
    () => [...scatterData].sort((a, b) => a.x - b.x),
    [scatterData]
  );
  const [activeIdx, setActiveIdx] = useState(-1);
  const chartRef = useRef<HTMLDivElement>(null);
  const interactionRef = useRef<ChartInteractionHandle>(null);
  useEffect(() => setActiveIdx(-1), [sortedData]);
  const activePoint = sortedData[activeIdx];

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
          exportXLabel={`Average ${latencyLabel}`}
          exportAnnotate={(clone) => {
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
          {activeTab === "stt" && metric === "TTFS" && (
            <ul data-chart-legend className="mb-4 ml-auto">
              <li
                className="flex items-center gap-1.5 text-xs"
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
            </ul>
          )}
        </div>

        <div
          ref={chartRef}
          data-export-frame
          className="relative h-64 select-none"
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
                  content={<CustomScatterTooltip activeTab={activeTab} metric={metric} />}
                  isAnimationActive={false}
                />
              )}
              {activePoint && (
                <ReferenceLine x={activePoint.x} stroke={themeColors.axisText} strokeDasharray="3 3" />
              )}
              {activePoint && (
                <ReferenceLine y={activePoint.y} stroke={themeColors.axisText} strokeDasharray="3 3" />
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
                  shape={(props: { cx?: number; cy?: number; fill?: string; payload?: ScatterDataPoint }) => (
                    <circle
                      data-export-point
                      cx={props.cx}
                      cy={props.cy}
                      r={props.payload === activePoint ? 8 : 6}
                      fill={props.fill}
                      fillOpacity={activePoint && props.payload !== activePoint ? 0.35 : 1}
                      stroke={props.payload === activePoint ? themeColors.axisText : undefined}
                      strokeWidth={2}
                      role="button"
                      tabIndex={0}
                      aria-label={`${normalizeModelName(props.payload!.model)}: ${props.payload!.x.toFixed(0)}ms ${metric}, ${props.payload!.y.toFixed(1)}% WER`}
                      onClick={() => setActiveIdx(sortedData.indexOf(props.payload!))}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          setActiveIdx(sortedData.indexOf(props.payload!));
                        }
                      }}
                    />
                  )}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div
          className="mt-1 text-center text-sm"
          style={{ color: themeColors.axisText }}
        >
          <MetricInfo metric={metric}>{latencyLabel}</MetricInfo>
        </div>
      </Card>
    </div>
  );
};

export default LatencyAccuracySection;
