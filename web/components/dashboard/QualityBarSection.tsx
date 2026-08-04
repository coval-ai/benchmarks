// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useCallback, useMemo, useRef, useState } from "react";
import { Server } from "lucide-react";
import { Cell, type LabelProps } from "recharts";
import CustomBarTooltip from "@/components/charts/tooltips/BarTooltip";
import QualityMetricBars from "@/components/charts/QualityMetricBars";
import { normalizeModelName, parseModelKey } from "@/lib/utils/formatters";
import Card from "@/components/shared/Card";
import { useDedicatedInfoTip } from "@/components/shared/DedicatedInferenceInfo";
import SectionHeader from "@/components/shared/SectionHeader";
import WerDatasetSelect from "@/components/dashboard/WerDatasetSelect";
import { datasetLabel } from "@/lib/config/datasets";
import { WER_BREAKDOWN_LABELS, type WerBreakdown } from "@/lib/utils/werBreakdown";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";
import { useActiveTab } from "@/hooks/useActiveTab";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS, type QualityBarMetric } from "@/lib/posthog/events";

// Order choices for the WER bars: the default best-first total, or worst-first
// by one error type so the outliers lead.
const BAR_SORTS: { key: "wer" | keyof WerBreakdown; label: string }[] = [
  { key: "wer", label: "Lowest WER" },
  { key: "substitutions", label: "Most substitutions" },
  { key: "deletions", label: "Most deletions" },
  { key: "insertions", label: "Most insertions" },
];

// The WerDatasetSelect chrome, reused for the chart's other dropdowns.
const SelectControl: React.FC<{
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: React.ReactNode;
}> = ({ label, value, onChange, children }) => (
  <span className="inline-flex items-center gap-2 text-xs text-text-secondary">
    {label}
    <span className="relative inline-flex">
      <select
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-11 max-w-44 appearance-none truncate rounded-lg border border-border-primary bg-surface-elevated pl-2.5 pr-7 text-xs font-medium text-text-primary outline-none transition-colors hover:border-selected-border focus:border-selected-border lg:h-auto lg:py-1.5"
      >
        {children}
      </select>
      <svg
        aria-hidden
        viewBox="0 0 12 12"
        className="pointer-events-none absolute right-2 top-1/2 h-3 w-3 -translate-y-1/2 text-text-tertiary"
      >
        <path
          d="M2.5 4.5 6 8l3.5-3.5"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </span>
  </span>
);

const INSTRUCTION_DESCRIPTION = {
  short: "Instruction adherence (%)",
  detailed:
    "Instruction adherence is the share of multi-turn conversations an LLM judge marks as following the agent's instructions, scored over the whole conversation. Higher is better — a model that stays on task across a full back-and-forth is more reliable than one that drifts after the first turn.",
};

// The per-model quality bar chart. STT/TTS plot WER (lower is better, with the
// dataset-view toggle, dedicated-inference markers and click-to-compare); S2S
// plots instruction adherence (higher is better, lean). Both share the
// QualityMetricBars chassis — only data, labels and the WER-only chrome differ.
const QualityBarSection: React.FC = () => {
  const {
    werDescription,
    werBarDataset,
    werBarServedDataset,
    changeWerBarDataset,
    werBarDataWithColors,
    instructionBarDataWithColors,
    getProviderForModel,
    dedicatedModels,
    isMobile,
    clickedWERBars,
    handleWERBarClick,
    clearWERBars,
    hasActiveFacets,
    werBarLoading,
  } = useDashboard();

  const themeColors = useThemeColors();
  const mode = useActiveTab();
  const isS2S = mode === "s2s";
  const trackChartHover = useChartHoverTracking(
    isS2S ? "instruction_bar" : "wer_bar"
  );
  const chartWrapRef = useRef<HTMLDivElement>(null);
  const {
    iconHandlers: dedicatedIconHandlers,
    overlay: dedicatedOverlay,
    open: dedicatedTipOpen,
  } = useDedicatedInfoTip(chartWrapRef);

  const handleWERBarClickTracked = (
    data: Parameters<typeof handleWERBarClick>[0]
  ) => {
    if (data?.model) {
      capturePostHogEvent(POSTHOG_EVENTS.dashboardWerBarClicked, {
        surface: `${mode}_dashboard`,
        mode,
        model_id: data.model,
        metric: (isS2S ? "instruction" : "wer") satisfies QualityBarMetric,
      });
    }
    handleWERBarClick(data);
  };

  const [barSort, setBarSort] = useState<(typeof BAR_SORTS)[number]["key"]>("wer");

  // "wer" keeps the upstream best-first order; an error type leads with the
  // worst offenders, models without a split trailing.
  const displayBars = useMemo(() => {
    if (barSort === "wer") return werBarDataWithColors;
    return [...werBarDataWithColors].sort(
      (a, b) => (b.breakdown?.[barSort] ?? -1) - (a.breakdown?.[barSort] ?? -1)
    );
  }, [werBarDataWithColors, barSort]);

  // The summary sentence follows the click-selection like the radar's does:
  // one model reads out its split, several compare among themselves, none
  // describes the whole set. Filtering displayBars keeps the sort's leader.
  const summaryPool = useMemo(
    () =>
      clickedWERBars.size > 0
        ? displayBars.filter((item) => clickedWERBars.has(item.model))
        : displayBars,
    [displayBars, clickedWERBars]
  );
  const leadBar = summaryPool[0];

  const selectedBars = useMemo(
    () => werBarDataWithColors.filter((item) => clickedWERBars.has(item.model)),
    [werBarDataWithColors, clickedWERBars]
  );

  const avgWER = useMemo(() => {
    const source = selectedBars.length > 0 ? selectedBars : werBarDataWithColors;
    if (source.length === 0) return 0;
    const sum = source.reduce((acc, item) => acc + (item.averageWER ?? 0), 0);
    return sum / source.length;
  }, [selectedBars, werBarDataWithColors]);

  const avgInstruction = useMemo(() => {
    if (instructionBarDataWithColors.length === 0) return 0;
    const sum = instructionBarDataWithColors.reduce(
      (acc, item) => acc + (item.instructionScore ?? 0),
      0
    );
    return sum / instructionBarDataWithColors.length;
  }, [instructionBarDataWithColors]);

  // WER bar-top labels carry the value plus a dedicated-inference marker.
  // Bars ~28px wide collide, so a too-thin bar keeps its label only when
  // selected; the axis and tooltip still carry the value for the rest.
  const werBarLabel = useCallback(
    ({ x = 0, y = 0, width = 0, value, index = 0 }: LabelProps) => {
      const entry = displayBars[index];
      if (Number(width) < 28 && !(entry && clickedWERBars.has(entry.model)))
        return <g />;
      const cx = Number(x) + Number(width) / 2;
      return (
        <g opacity={entry?.fillOpacity}>
          <text
            x={cx}
            y={Number(y) - 8}
            textAnchor="middle"
            fill={themeColors.label}
            fontSize={12}
          >
            {`${Number(value).toFixed(1)}%`}
          </text>
          {entry && dedicatedModels.has(entry.model) && (
            // The dedicated marker rides the top of the bar, under the value;
            // hover or tap opens the explainer.
            <g
              {...dedicatedIconHandlers}
              role="button"
              tabIndex={0}
              aria-label="About dedicated inference"
              style={{ cursor: "help" }}
            >
              <Server
                x={cx - 6}
                y={Number(y) + 5}
                size={12}
                color={themeColors.label}
                strokeWidth={2.4}
                aria-hidden
              />
              <rect
                x={cx - 12}
                y={Number(y) - 1}
                width={24}
                height={24}
                fill="transparent"
              />
            </g>
          )}
        </g>
      );
    },
    [displayBars, clickedWERBars, themeColors.label, dedicatedModels, dedicatedIconHandlers]
  );

  // Instruction labels are plain: value only, hidden on very thin bars.
  const instructionBarLabel = useCallback(
    ({ x = 0, y = 0, width = 0, value }: LabelProps) => {
      if (Number(width) < 24) return <g />;
      const cx = Number(x) + Number(width) / 2;
      return (
        <text
          x={cx}
          y={Number(y) - 8}
          textAnchor="middle"
          fill={themeColors.label}
          fontSize={12}
        >
          {`${Number(value).toFixed(0)}%`}
        </text>
      );
    },
    [themeColors.label]
  );

  // Labeled from the dataset the rows were actually served from, which lags
  // the select while a switch is in flight — exports and the summary must
  // describe the placeholder data on show, not the pending selection.
  const werScopeLabel = werBarServedDataset
    ? datasetLabel(werBarServedDataset)
    : undefined;

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8">
        {isS2S ? (
          <SectionHeader
            label="Instruction Adherence by Model"
            description={INSTRUCTION_DESCRIPTION}
            exportXLabel="Model"
            exportRows={() =>
              instructionBarDataWithColors.map(({ model, instructionScore }) => ({
                model: parseModelKey(model).model,
                provider: getProviderForModel(model),
                instruction_adherence_percent: instructionScore,
              }))
            }
            stat={{
              label: "Avg Instruction (all models)",
              value:
                instructionBarDataWithColors.length === 0
                  ? "—"
                  : `${avgInstruction.toFixed(0)}%`,
            }}
          />
        ) : (
          <SectionHeader
            label="Accuracy by Model"
            description={werDescription}
            exportNote={werScopeLabel}
            hint="Click bar to compare models"
            exportXLabel="Model"
            exportRows={() =>
              displayBars.map(({ model, averageWER }) => ({
                model: parseModelKey(model).model,
                provider: getProviderForModel(model),
                wer_dataset: werBarServedDataset ?? "all",
                avg_wer_percent: averageWER,
              }))
            }
            stat={{
              label:
                selectedBars.length > 0
                  ? `Avg WER (${selectedBars.length} selected)`
                  : hasActiveFacets
                    ? `Avg WER (${werBarDataWithColors.length} filtered)`
                    : "Avg WER (all models)",
              value:
                werBarDataWithColors.length === 0 ? "—" : `${avgWER.toFixed(1)}%`,
            }}
          />
        )}
        {!isS2S && (
          <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2">
            <WerDatasetSelect
              label="Chart dataset"
              value={werBarDataset}
              onChange={changeWerBarDataset}
            />
            <SelectControl
              label="Sort"
              value={barSort}
              onChange={(v) => setBarSort(v as typeof barSort)}
            >
              {BAR_SORTS.map(({ key, label }) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </SelectControl>
          </div>
        )}
        {!isS2S && selectedBars.length > 0 && (
          <div className="mb-3 flex flex-wrap items-center gap-1.5">
            {selectedBars.map((item) => (
              <span
                key={item.model}
                className="inline-flex items-center gap-1.5 rounded-full border border-border-primary px-2.5 py-1 text-xs text-text-secondary"
              >
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: item.fill }}
                />
                {normalizeModelName(item.model)}
                <span className="tabular-nums font-medium text-text-primary">
                  {item.averageWER.toFixed(1)}%
                </span>
                <button
                  type="button"
                  aria-label={`Deselect ${normalizeModelName(item.model)}`}
                  onClick={() => handleWERBarClickTracked(item)}
                  className="text-text-tertiary transition-colors hover:text-text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40"
                >
                  ✕
                </button>
              </span>
            ))}
            <button
              type="button"
              onClick={clearWERBars}
              className="px-1 text-xs text-text-tertiary transition-colors hover:text-text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40"
            >
              Clear
            </button>
          </div>
        )}
        {isS2S ? (
          <QualityMetricBars
            data={instructionBarDataWithColors}
            valueKey="instructionScore"
            yAxisLabel="Instruction % · higher is better"
            tooltip={
              <CustomBarTooltip
                getProviderForModel={getProviderForModel}
                dataKey="instructionScore"
                valueLabel="Instruction adherence"
                formatValue={(value) => `${value.toFixed(0)}%`}
              />
            }
            isMobile={isMobile}
            getProviderForModel={getProviderForModel}
            barLabel={instructionBarLabel}
            onHover={trackChartHover}
          >
            {instructionBarDataWithColors.map((entry) => (
              <Cell
                key={`instruction-cell-${entry.model}`}
                fill={entry.fill}
                fillOpacity={entry.fillOpacity}
                aria-label={`${normalizeModelName(entry.model)}: ${entry.instructionScore.toFixed(0)}% instruction adherence`}
              />
            ))}
          </QualityMetricBars>
        ) : (
          <QualityMetricBars
            data={displayBars}
            valueKey="averageWER"
            yAxisLabel="WER % · lower is better"
            tooltip={
              <CustomBarTooltip
                getProviderForModel={getProviderForModel}
                dedicatedModels={dedicatedModels}
              />
            }
            isMobile={isMobile}
            getProviderForModel={getProviderForModel}
            barLabel={werBarLabel}
            onBarClick={(bar) => handleWERBarClickTracked(bar.payload)}
            barStyle={{ cursor: "pointer" }}
            loading={werBarLoading}
            onHover={trackChartHover}
            // The dedicated marker sits inside the plot, so its open explainer
            // silences the bar tooltip instead of overlapping it.
            tooltipActive={isMobile || dedicatedTipOpen ? false : undefined}
            wrapRef={chartWrapRef}
            overlay={dedicatedOverlay}
          >
            {displayBars.map((entry) => (
              <Cell
                key={`wer-cell-${entry.model}`}
                fill={entry.fill}
                fillOpacity={entry.fillOpacity}
                stroke={dedicatedModels.has(entry.model) ? themeColors.label : undefined}
                strokeWidth={dedicatedModels.has(entry.model) ? 1.5 : undefined}
                role="button"
                tabIndex={0}
                aria-label={`${normalizeModelName(entry.model)}: ${entry.averageWER.toFixed(1)}% WER${
                  entry.breakdown
                    ? ` (${WER_BREAKDOWN_LABELS.map(
                        ([key, text]) =>
                          `${text} ${entry.breakdown![key].toFixed(1)}%`
                      ).join(", ")})`
                    : ""
                }${clickedWERBars.has(entry.model) ? ", selected" : ""}`}
                onKeyDown={(e: React.KeyboardEvent) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    handleWERBarClickTracked(entry);
                  }
                }}
                onMouseDown={(e: React.MouseEvent) => e.preventDefault()}
              />
            ))}
          </QualityMetricBars>
        )}
        {!isS2S && leadBar && (
          <p className="mt-2 text-sm text-text-secondary">
            <span className="font-medium text-text-primary">
              {getProviderForModel(leadBar.model)}{" "}
              {normalizeModelName(leadBar.model)}
            </span>{" "}
            {summaryPool.length === 1
              ? `has a WER of ${leadBar.averageWER.toFixed(1)}%`
              : barSort !== "wer" && leadBar.breakdown
                ? `has the ${BAR_SORTS.find((s) => s.key === barSort)!.label.toLowerCase()} (${leadBar.breakdown[barSort].toFixed(1)}% of its ${leadBar.averageWER.toFixed(1)}% WER) of ${summaryPool.length}${clickedWERBars.size > 0 ? " selected" : ""} models`
                : `has the lowest WER (${leadBar.averageWER.toFixed(1)}%) of ${summaryPool.length}${clickedWERBars.size > 0 ? " selected" : ""} models`}{" "}
            {werScopeLabel ? `on ${werScopeLabel}` : "pooled across datasets"}
            {(barSort === "wer" || summaryPool.length === 1) &&
              leadBar.breakdown && (
                <>
                  {" — "}
                  {WER_BREAKDOWN_LABELS.map(
                    ([key, text]) =>
                      `${text.toLowerCase()} ${leadBar.breakdown![key].toFixed(1)}%`
                  ).join(", ")}
                </>
              )}
            .
          </p>
        )}
      </Card>
    </div>
  );
};

export default QualityBarSection;
