// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useCallback, useMemo, useRef } from "react";
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
import { WER_BREAKDOWN_LABELS } from "@/lib/utils/werBreakdown";
import { useDashboard } from "@/contexts/DashboardContext";
import { useThemeColors } from "@/hooks/useThemeColors";
import { useActiveTab } from "@/hooks/useActiveTab";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS, type QualityBarMetric } from "@/lib/posthog/events";

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
      const entry = werBarDataWithColors[index];
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
    [werBarDataWithColors, clickedWERBars, themeColors.label, dedicatedModels, dedicatedIconHandlers]
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

  // The select carries its own explainer, so the header only names the scope.
  const werScopeLabel = werBarDataset ? datasetLabel(werBarDataset) : undefined;

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
              werBarDataWithColors.map(({ model, averageWER }) => ({
                model: parseModelKey(model).model,
                provider: getProviderForModel(model),
                wer_view: werBarView,
                wer_dataset: activeWerView?.dataset ?? "all",
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
          <WerDatasetSelect
            className="mb-4"
            label="Chart dataset"
            value={werBarDataset}
            onChange={changeWerBarDataset}
          />
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
            data={werBarDataWithColors}
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
            {werBarDataWithColors.map((entry) => (
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
                          `${text} ${entry.breakdown![key].toFixed(1)}`
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
      </Card>
    </div>
  );
};

export default QualityBarSection;
