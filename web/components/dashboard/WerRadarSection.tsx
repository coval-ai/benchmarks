// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import CustomTimelineTooltip from "@/components/charts/tooltips/TimelineTooltip";
import { TimelineLegend } from "@/components/visualizations/TimelineChart";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors, type ThemeColors } from "@/hooks/useThemeColors";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";
import { useWerDatasetMatrix } from "@/hooks/useWerDatasetMatrix";
import { datasetLabel, isPerturbationDataset } from "@/lib/config/datasets";
import { getModelColor } from "@/lib/utils/colors";
import { normalizeModelName } from "@/lib/utils/formatters";

// "WildASR reverb" → ["Reverb", "WildASR"]: the family drops to a second line
// so ten axis labels stay legible on a phone-width radar.
function splitAxisLabel(label: string): [string, string?] {
  if (label.startsWith("WildASR ")) {
    const rest = label.slice(8);
    return [rest.charAt(0).toUpperCase() + rest.slice(1), "WildASR"];
  }
  const paren = label.match(/^(.*?) \((.+)\)$/);
  if (paren) return [paren[1]!, paren[2]!];
  return [label];
}

const RadarAxisTick: React.FC<{
  payload?: { value: string };
  x?: number;
  y?: number;
  cy?: number;
  textAnchor?: "inherit" | "start" | "middle" | "end";
  colors: ThemeColors;
  isMobile: boolean;
  activeLabel?: string;
}> = ({ payload, x = 0, y = 0, cy, textAnchor, colors, isMobile, activeLabel }) => {
  const [name, family] = splitAxisLabel(payload?.value ?? "");
  const active = payload?.value != null && payload.value === activeLabel;
  // Labels straddling the chart's vertical axis sit flush against the top
  // and bottom vertices: lift the top block one line so its family sublabel
  // stays out of the grid, and drop the bottom block clear of the vertex.
  const middle = textAnchor === "middle" && cy != null;
  const aboveChart = middle && family != null && y < cy!;
  const belowChart = middle && y >= cy!;
  const dy = aboveChart ? -(isMobile ? 11 : 13) : belowChart ? (isMobile ? 8 : 10) : 0;
  return (
    <text
      x={x}
      y={y + dy}
      textAnchor={textAnchor}
      fill={active ? colors.label : colors.axisText}
      fontWeight={active ? 600 : undefined}
      fontSize={isMobile ? 10 : 12}
    >
      <tspan x={x}>{name}</tspan>
      {family && (
        <tspan x={x} dy="1.15em" fillOpacity={0.65} fontSize={isMobile ? 9 : 10}>
          {family}
        </tspan>
      )}
    </text>
  );
};

interface PinnedReading {
  label: string;
  x: number;
  y: number;
  flip: boolean;
}

const formatWer = (value: number) => `${value.toFixed(1)}%`;

const WerRadarSection: React.FC = () => {
  const {
    selectedModels,
    legendModels,
    toggleLegendModel,
    getProviderForModel,
    formatChartLabel,
    availableWerDatasets,
    timeWindow,
    isMobile,
  } = useDashboard();
  const themeColors = useThemeColors();
  const trackChartHover = useChartHoverTracking("wer_radar");

  const axes = useMemo(
    () => [
      ...availableWerDatasets.filter((d) => !isPerturbationDataset(d)),
      ...availableWerDatasets.filter(isPerturbationDataset),
    ],
    [availableWerDatasets]
  );

  const { werByDataset, loading } = useWerDatasetMatrix(
    { benchmark: "STT", window: timeWindow },
    axes
  );

  const effectiveAxes = useMemo(
    () => axes.filter((d) => werByDataset?.has(d)),
    [axes, werByDataset]
  );

  // Only models measured on every axis draw a closed shape; partial coverage
  // would render as a misleading dent.
  const coveredModels = useMemo(
    () =>
      effectiveAxes.length === 0
        ? []
        : legendModels.filter((m) =>
            effectiveAxes.every((d) => werByDataset!.get(d)!.has(m))
          ),
    [legendModels, effectiveAxes, werByDataset]
  );

  const ranked = useMemo(() => {
    const selected = new Set(selectedModels);
    return coveredModels
      .filter((model) => selected.has(model))
      .map((model) => ({
        model,
        mean:
          effectiveAxes.reduce(
            (sum, d) => sum + werByDataset!.get(d)!.get(model)!,
            0
          ) / effectiveAxes.length,
      }))
      .sort((a, b) => a.mean - b.mean);
  }, [coveredModels, selectedModels, effectiveAxes, werByDataset]);
  const plotted = useMemo(() => ranked.map((r) => r.model), [ranked]);

  // Radius is 100 − WER so the outer edge is perfect transcription; the scale
  // starts at the worst plotted value (rounded to a step) to keep gaps readable.
  const { scaleMin, scaleTicks } = useMemo(() => {
    let maxWer = 0;
    plotted.forEach((m) =>
      effectiveAxes.forEach((d) => {
        maxWer = Math.max(maxWer, werByDataset?.get(d)?.get(m) ?? 0);
      })
    );
    const step = maxWer > 20 ? 10 : 5;
    const span = Math.max(2 * step, Math.ceil(maxWer / step) * step);
    return { scaleMin: 100 - span, scaleTicks: span / step + 1 };
  }, [plotted, effectiveAxes, werByDataset]);

  const radarData = useMemo(
    () =>
      effectiveAxes.map((dataset) => {
        const row: Record<string, string | number> = {
          label: datasetLabel(dataset),
        };
        plotted.forEach((m) => {
          row[m] = 100 - werByDataset!.get(dataset)!.get(m)!;
        });
        return row;
      }),
    [effectiveAxes, plotted, werByDataset]
  );

  const werRows = (label: string) => {
    const row = radarData.find((r) => r.label === label);
    if (!row) return [];
    return plotted.map((model) => ({
      dataKey: model,
      name: model,
      color: getModelColor(model),
      value: 100 - (row[model] as number),
    }));
  };

  const chartRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState<PinnedReading | null>(null);
  const [scrub, setScrub] = useState<{ label: string; flip: boolean } | null>(
    null
  );
  const touchRef = useRef<{
    x: number;
    y: number;
    moved: boolean;
    hadPin: boolean;
  } | null>(null);

  // The pin anchors to a vertex of the current data; a window or layout
  // change moves the ground under it.
  useEffect(() => {
    setPinned(null);
    setScrub(null);
  }, [timeWindow, isMobile]);

  // Refreshed axes or model toggles can leave the pin pointing at a reading
  // that no longer exists; holding it would hide the panel while keeping
  // hover readouts suppressed.
  useEffect(() => {
    if (
      pinned &&
      (plotted.length === 0 ||
        !radarData.some((r) => r.label === pinned.label))
    ) {
      setPinned(null);
    }
  }, [pinned, plotted, radarData]);

  useEffect(() => {
    if (pinned === null) return;
    const onDocMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (pinnedRef.current?.contains(target)) return;
      if (chartRef.current?.contains(target)) return;
      setPinned(null);
    };
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [pinned]);

  // Touching or scrolling away dismisses the pinned list on mobile; scrolling
  // the list itself keeps it open.
  useEffect(() => {
    if (!isMobile || pinned === null) return;
    const onScroll = (e: Event) => {
      if (pinnedRef.current?.contains(e.target as Node)) return;
      setPinned(null);
    };
    window.addEventListener("scroll", onScroll, { capture: true, passive: true });
    return () => window.removeEventListener("scroll", onScroll, { capture: true });
  }, [isMobile, pinned]);

  const labelAtPoint = (e: React.PointerEvent<HTMLDivElement>) => {
    const n = radarData.length;
    if (n === 0) return undefined;
    const rect = e.currentTarget.getBoundingClientRect();
    const dx = e.clientX - (rect.left + rect.width / 2);
    const dy = e.clientY - (rect.top + rect.height / 2);
    const deg = (Math.atan2(-dy, dx) * 180) / Math.PI;
    const idx = ((Math.round((90 - deg) / (360 / n)) % n) + n) % n;
    return {
      label: radarData[idx]?.label as string,
      flip: e.clientX > rect.left + rect.width / 2,
    };
  };

  const ready = effectiveAxes.length >= 3 && plotted.length > 0;
  const best = ranked[0];
  const activeAxisLabel = scrub?.label ?? pinned?.label;
  const legendPayload = coveredModels.map((model) => ({
    value: formatChartLabel(model, getProviderForModel(model)),
    color: getModelColor(model),
    dataKey: model,
  }));
  const plottedSet = new Set(plotted);
  const legendHiddenKeys = new Set(
    coveredModels.filter((m) => !plottedSet.has(m))
  );

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8">
        <SectionHeader
          label="Accuracy Across Datasets"
          description={{
            short: "Where each model holds up — and where it breaks",
            detailed:
              "Each axis is one evaluation dataset; the radius is word accuracy (100 − WER), so the outer edge is perfect transcription and dents show where a model degrades — accents, noise, reverb, or spontaneous production speech. The scale starts at the worst plotted value, not zero, to keep differences readable. The chart plots the models selected in Filters; click a legend entry to toggle a model, and click or tap the chart to pin a dataset's exact values.",
          }}
          hint={
            isMobile
              ? "Drag the chart to read a dataset · tap to pin"
              : "Hover the chart to read a dataset, click to pin"
          }
          exportRows={() =>
            plotted.flatMap((model) =>
              effectiveAxes.map((dataset) => ({
                dataset: datasetLabel(dataset),
                model,
                provider: getProviderForModel(model),
                avg_wer_percent: werByDataset?.get(dataset)?.get(model),
              }))
            )
          }
          stat={{
            label: best
              ? `Lowest Avg WER (${normalizeModelName(best.model)})`
              : "Lowest Avg WER",
            value: best ? `${best.mean.toFixed(1)}%` : "—",
          }}
        />
        <div
          ref={chartRef}
          className={`relative h-80 transition-opacity sm:h-96 ${loading ? "opacity-40" : ""} ${isMobile ? "select-none" : ""}`}
          data-export-frame
          onMouseEnter={trackChartHover}
          onPointerDown={
            isMobile
              ? (e) => {
                  touchRef.current = {
                    x: e.clientX,
                    y: e.clientY,
                    moved: false,
                    hadPin: pinned !== null,
                  };
                  if (!pinned) setScrub(labelAtPoint(e) ?? null);
                }
              : undefined
          }
          onPointerMove={
            isMobile
              ? (e) => {
                  const t = touchRef.current;
                  if (!t || (e.pointerType === "mouse" && e.buttons === 0))
                    return;
                  if (
                    !t.moved &&
                    Math.hypot(e.clientX - t.x, e.clientY - t.y) > 8
                  ) {
                    t.moved = true;
                    setPinned(null);
                  }
                  if (t.moved) setScrub(labelAtPoint(e) ?? null);
                }
              : undefined
          }
          onPointerUp={
            isMobile
              ? (e) => {
                  const t = touchRef.current;
                  touchRef.current = null;
                  setScrub(null);
                  if (!t || t.moved) return;
                  if (t.hadPin) {
                    setPinned(null);
                    return;
                  }
                  const at = labelAtPoint(e);
                  const width = chartRef.current?.clientWidth ?? 0;
                  if (!at) return;
                  // Anchor the list to the half of the plot the axis is not in.
                  setPinned({
                    label: at.label,
                    x: at.flip ? 0 : width,
                    y: 8,
                    flip: !at.flip,
                  });
                }
              : undefined
          }
          onPointerCancel={
            isMobile
              ? () => {
                  touchRef.current = null;
                  setScrub(null);
                }
              : undefined
          }
          style={isMobile ? { touchAction: "pan-y" } : undefined}
        >
          {ready ? (
            <ResponsiveContainer width="100%" height="100%" debounce={200}>
              <RadarChart
                data={radarData}
                outerRadius={isMobile ? "62%" : "72%"}
                margin={
                  isMobile
                    ? { top: 20, right: 10, bottom: 20, left: 10 }
                    : { top: 24, right: 24, bottom: 24, left: 24 }
                }
                onClick={(state, e) => {
                  if (isMobile) return;
                  const lbl = state?.activeLabel;
                  const rect = chartRef.current?.getBoundingClientRect();
                  const me = e as unknown as React.MouseEvent;
                  const coord = state?.activeCoordinate ?? {
                    x: me.clientX - (rect?.left ?? 0),
                    y: me.clientY - (rect?.top ?? 0),
                  };
                  if (lbl == null) return;
                  setPinned((cur) => {
                    if (cur) return null;
                    const width = chartRef.current?.clientWidth ?? 0;
                    const height = chartRef.current?.clientHeight ?? 0;
                    const pad = 130;
                    const y =
                      height > pad * 2
                        ? Math.min(Math.max(coord.y ?? 0, pad), height - pad)
                        : coord.y ?? 0;
                    // The radar sits centered with empty card space either
                    // side, so the panel extends outward from the vertex —
                    // away from the polygon, not across it.
                    return {
                      label: String(lbl),
                      x: coord.x ?? 0,
                      y,
                      flip: width > 0 && (coord.x ?? 0) < width / 2,
                    };
                  });
                }}
              >
                <PolarGrid stroke={themeColors.grid} />
                <PolarAngleAxis
                  dataKey="label"
                  tick={
                    <RadarAxisTick
                      colors={themeColors}
                      isMobile={isMobile}
                      activeLabel={activeAxisLabel}
                    />
                  }
                />
                <PolarRadiusAxis
                  angle={90 - 180 / Math.max(1, effectiveAxes.length)}
                  domain={[scaleMin, 100]}
                  tickCount={scaleTicks}
                  axisLine={false}
                  tick={{
                    fill: themeColors.axisText,
                    fillOpacity: 0.75,
                    fontSize: 9,
                  }}
                  tickFormatter={(value: number) =>
                    value === 100 ? "" : `${value}%`
                  }
                />
                <Tooltip
                  active={pinned || isMobile ? false : undefined}
                  offset={72}
                  content={({ active, label }) =>
                    active && label != null ? (
                      <CustomTimelineTooltip
                        active
                        payload={werRows(String(label))}
                        labelText={String(label)}
                        formatValue={formatWer}
                        getProviderForModel={getProviderForModel}
                        compact
                        interactionHint="click to pin"
                      />
                    ) : null
                  }
                />
                {[...plotted].reverse().map((model) => (
                  <Radar
                    key={model}
                    name={formatChartLabel(model, getProviderForModel(model))}
                    // Function form: a string key with a dot (e.g. a "-3.5"
                    // slug) would be parsed as an object path and render empty.
                    dataKey={(row: Record<string, number>) => row[model]}
                    stroke={getModelColor(model)}
                    fill={getModelColor(model)}
                    strokeWidth={
                      model === best?.model || plotted.length <= 2 ? 2.5 : 1.25
                    }
                    strokeOpacity={
                      model === best?.model || plotted.length <= 2 ? 1 : 0.7
                    }
                    fillOpacity={
                      model === best?.model || plotted.length <= 2 ? 0.12 : 0
                    }
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </RadarChart>
            </ResponsiveContainer>
          ) : loading ? (
            <span className="block h-full w-full animate-pulse rounded bg-surface-secondary" />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-text-tertiary">
              No dataset-scoped WER data in this window.
            </div>
          )}
          {scrub && !pinned && (
            <div
              className={`pointer-events-none absolute top-2 z-20 shadow-lg ${
                scrub.flip ? "left-2" : "right-2"
              }`}
            >
              <CustomTimelineTooltip
                active
                payload={werRows(scrub.label)}
                labelText={scrub.label}
                formatValue={formatWer}
                getProviderForModel={getProviderForModel}
                compact
                interactionHint="tap to pin"
              />
            </div>
          )}
          {pinned && werRows(pinned.label).length > 0 && (
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
                    ? "translate(calc(-100% - 72px), -50%)"
                    : "translate(72px, -50%)",
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
                payload={werRows(pinned.label)}
                labelText={pinned.label}
                formatValue={formatWer}
                getProviderForModel={getProviderForModel}
                maxHeight={isMobile ? 106 : undefined}
              />
            </div>
          )}
        </div>
        {effectiveAxes.length >= 3 && (
          <p className="mt-2 text-sm text-text-secondary">
            {best ? (
              <>
                <span className="font-medium text-text-primary">
                  {formatChartLabel(best.model, getProviderForModel(best.model))}
                </span>{" "}
                has the lowest average WER ({best.mean.toFixed(1)}%) across{" "}
                {effectiveAxes.length} datasets
                {plotted.length > 1 && (
                  <>
                    , best on{" "}
                    {
                      effectiveAxes.filter((d) =>
                        plotted.every(
                          (m) =>
                            werByDataset!.get(d)!.get(best.model)! <=
                            werByDataset!.get(d)!.get(m)!
                        )
                      ).length
                    }{" "}
                    of them
                  </>
                )}
                .
              </>
            ) : (
              <>No selected models have WER results on every dataset above.</>
            )}
          </p>
        )}
        {legendPayload.length > 0 && (
          <div data-chart-legend>
            <TimelineLegend
              payload={legendPayload}
              hiddenKeys={legendHiddenKeys}
              onToggle={toggleLegendModel}
            />
          </div>
        )}
      </Card>
    </div>
  );
};

export default WerRadarSection;
