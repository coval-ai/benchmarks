// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useCallback, useMemo, useState } from "react";
import { type LabelProps } from "recharts";
import { getModelColor } from "@/lib/utils/colors";
import { normalizeModelName, parseModelKey } from "@/lib/utils/formatters";
import BoxPlot from "@/components/charts/d3/BoxPlot";
import QualityMetricBars from "@/components/charts/QualityMetricBars";
import { TtfaBreakdownTooltip } from "@/components/charts/tooltips/BarTooltip";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import MetricInfo from "@/components/shared/MetricInfo";
import MetricToggle, { useMetricTab } from "@/components/dashboard/MetricToggle";
import { metricAboutNote } from "@/lib/config/metrics";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";

// Segments follow each model's palette color (the same one the Filters
// sidebar and every other chart use); texture carries the split — solid is
// the network roundtrip, hatched the leading silence.
function silencePatternId(model: string): string {
  return `ttfa-var-silence-${model.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}

const TTFA_SEGMENTS = [
  { dataKey: "roundtrip", fill: (model: string) => getModelColor(model) },
  { dataKey: "silence", fill: (model: string) => `url(#${silencePatternId(model)})` },
] as const;

const BoxPlotSection: React.FC = () => {
  const {
    page,
    boxPlotDescription,
    latencyLabel,
    getBoxPlotData,
    getProviderForModel,
    dedicatedModels,
    crossRegionModels,
    isMobile,
    activeMetric,
    ttfaBreakdownBars,
  } = useDashboard();
  const trackChartHover = useChartHoverTracking("box_plot");
  const metricTab = useMetricTab();
  const themeColors = useThemeColors();

  // TTS gains a second view: the distribution boxes, or the same models as
  // stacked bars splitting average TTFA into roundtrip + leading silence.
  // The toggle appears only once component rows exist in the served window.
  const [view, setView] = useState<"distribution" | "breakdown">("distribution");
  const hasBreakdown = page === "tts" && ttfaBreakdownBars.length > 0;
  const showBreakdown = hasBreakdown && view === "breakdown";

  const boxPlotData = useMemo(
    () => getBoxPlotData(activeMetric),
    [getBoxPlotData, activeMetric]
  );

  // Headline: mean IQR across the models on the chart — how predictable the
  // field is, which is what this card is for, rather than how fast it is.
  const avgIqrMs = useMemo(() => {
    const widths = boxPlotData.data
      .map(({ quartiles }) => quartiles.q3 - quartiles.q1)
      .filter((width) => Number.isFinite(width) && width >= 0);
    return widths.length > 0
      ? widths.reduce((sum, width) => sum + width, 0) / widths.length
      : undefined;
  }, [boxPlotData]);

  // Headline for the breakdown view: how much of the field's average TTFA is
  // leading silence — the share a listener waits through after bytes arrive.
  const silenceShare = useMemo(() => {
    const totals = ttfaBreakdownBars.reduce(
      (acc, bar) => ({ silence: acc.silence + bar.silence, ttfa: acc.ttfa + bar.ttfa }),
      { silence: 0, ttfa: 0 }
    );
    return totals.ttfa > 0 ? (totals.silence / totals.ttfa) * 100 : undefined;
  }, [ttfaBreakdownBars]);

  // Total-TTFA labels ride the top of each stack; thin bars stay unlabeled
  // like the WER chart's, the tooltip still carrying the values.
  const totalBarLabel = useCallback(
    ({ x = 0, y = 0, width = 0, index = 0 }: LabelProps) => {
      const entry = ttfaBreakdownBars[index];
      if (!entry || Number(width) < 34) return <g />;
      return (
        <text
          x={Number(x) + Number(width) / 2}
          y={Number(y) - 8}
          textAnchor="middle"
          fill={themeColors.label}
          fontSize={12}
        >
          {`${Math.round(entry.ttfa)}`}
        </text>
      );
    },
    [ttfaBreakdownBars, themeColors.label]
  );

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8" onMouseEnter={trackChartHover}>
        <SectionHeader
          label="Latency Variation"
          description={
            showBreakdown
              ? {
                  short: `Average TTFA split into its parts, per model`,
                  detailed:
                    "TTFA is what a listener waits through, and it has two causes: the network roundtrip until audio starts arriving, and the leading silence a provider front-loads before the first audible sample. The stack makes the mix visible — two models with the same total can feel identical while one is network-bound and the other pads its output.",
                }
              : boxPlotDescription
          }
          note={metricAboutNote(activeMetric)}
          exportNote={showBreakdown ? "TTFA breakdown" : metricTab}
          exportXLabel={
            showBreakdown
              ? "Ranked by average TTFA (fastest first)"
              : "Ranked by median latency (fastest first)"
          }
          exportRows={() =>
            showBreakdown
              ? ttfaBreakdownBars.map(({ model, roundtrip, silence, ttfa }) => ({
                  model: parseModelKey(model).model,
                  provider: getProviderForModel(model),
                  avg_ttfa_ms: ttfa,
                  roundtrip_ms: roundtrip,
                  leading_silence_ms: silence,
                }))
              : boxPlotData.data.map(({ model, quartiles, stats }) => ({
                  model: parseModelKey(model).model,
                  provider: getProviderForModel(model),
                  metric: activeMetric,
                  whisker_low_ms: quartiles.min,
                  q1_ms: quartiles.q1,
                  median_ms: quartiles.median,
                  q3_ms: quartiles.q3,
                  whisker_high_ms: quartiles.max,
                  iqr_ms: quartiles.q3 - quartiles.q1,
                  iqr_pct_of_median:
                    quartiles.median > 0
                      ? ((quartiles.q3 - quartiles.q1) / quartiles.median) * 100
                      : undefined,
                  mean_ms: stats.mean,
                  std_dev_ms: stats.std,
                  p95_ms: stats.p95,
                  max_ms: stats.max,
                  runs: stats.count,
                }))
          }
          stat={
            showBreakdown
              ? silenceShare === undefined
                ? undefined
                : {
                    label: "Leading silence share of TTFA",
                    value: `${silenceShare.toFixed(0)}%`,
                  }
              : avgIqrMs === undefined
                ? undefined
                : {
                    label: (
                      <MetricInfo metric="iqr" align="right">{`Average ${latencyLabel} IQR`}</MetricInfo>
                    ),
                    value: `${avgIqrMs.toFixed(0)} ms`,
                  }
          }
        />

        <MetricToggle />
        {hasBreakdown && (
          <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2">
            <div className="inline-flex gap-0.5 rounded-lg bg-surface-toggle-inactive p-0.5">
              {(
                [
                  ["distribution", "Distribution"],
                  ["breakdown", "Breakdown"],
                ] as const
              ).map(([key, text]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setView(key)}
                  className={
                    "rounded-md px-4 py-3 text-sm sm:px-3 sm:py-1 sm:text-xs font-medium transition-colors " +
                    (view === key
                      ? "bg-surface-primary text-text-primary shadow-sm"
                      : "text-text-secondary hover:text-text-primary")
                  }
                >
                  {text}
                </button>
              ))}
            </div>
            {showBreakdown && (
              <div className="flex items-center gap-4 text-xs text-text-secondary">
                <span className="inline-flex items-center gap-1.5">
                  <span
                    className="h-2.5 w-2.5 rounded-sm"
                    style={{ backgroundColor: themeColors.axisText }}
                  />
                  Network roundtrip
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span
                    className="h-2.5 w-2.5 rounded-sm"
                    style={{
                      background: `repeating-linear-gradient(45deg, ${themeColors.axisText} 0 1.5px, transparent 1.5px 4px)`,
                    }}
                  />
                  Leading silence
                </span>
              </div>
            )}
          </div>
        )}

        {showBreakdown ? (
          <QualityMetricBars
            data={ttfaBreakdownBars}
            valueKey="ttfa"
            stackSegments={TTFA_SEGMENTS}
            svgDefs={
              <defs>
                {ttfaBreakdownBars.map(({ model }) => (
                  <pattern
                    key={model}
                    id={silencePatternId(model)}
                    patternUnits="userSpaceOnUse"
                    width={5}
                    height={5}
                    patternTransform="rotate(45)"
                  >
                    <rect
                      width={5}
                      height={5}
                      fill={getModelColor(model)}
                      fillOpacity={0.18}
                    />
                    <rect width={2} height={5} fill={getModelColor(model)} />
                  </pattern>
                ))}
              </defs>
            }
            tickFormatter={(value) => `${Math.round(value)}`}
            yAxisLabel="Avg TTFA ms · lower is better"
            tooltip={
              <TtfaBreakdownTooltip getProviderForModel={getProviderForModel} />
            }
            isMobile={isMobile}
            getProviderForModel={getProviderForModel}
            barLabel={totalBarLabel}
            onHover={trackChartHover}
          />
        ) : (
          <BoxPlot
            data={boxPlotData}
            getModelColor={getModelColor}
            getProviderForModel={getProviderForModel}
            normalizeModelName={normalizeModelName}
            dedicatedModels={dedicatedModels}
            crossRegionModels={crossRegionModels}
            isMobile={isMobile}
          />
        )}
      </Card>
    </div>
  );
};

export default BoxPlotSection;
