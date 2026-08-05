// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo } from "react";
import { getModelColor } from "@/lib/utils/colors";
import { normalizeModelName, parseModelKey } from "@/lib/utils/formatters";
import BoxPlot from "@/components/charts/d3/BoxPlot";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import MetricInfo from "@/components/shared/MetricInfo";
import MetricToggle, { useMetricTab } from "@/components/dashboard/MetricToggle";
import { metricAboutNote } from "@/lib/config/metrics";
import { useDashboard } from "@/contexts/DashboardContext";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";

const BoxPlotSection: React.FC = () => {
  const {
    boxPlotDescription: description,
    latencyLabel,
    getBoxPlotData,
    getProviderForModel,
    dedicatedModels,
    crossRegionModels,
    isMobile,
    activeMetric,
  } = useDashboard();
  const trackChartHover = useChartHoverTracking("box_plot");
  const metricTab = useMetricTab();

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

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8" onMouseEnter={trackChartHover}>
        <SectionHeader
          label="Latency Variation"
          description={description}
          note={metricAboutNote(activeMetric)}
          exportNote={metricTab}
          exportXLabel="Ranked by median latency (fastest first)"
          exportRows={() =>
            boxPlotData.data.map(({ model, quartiles, stats }) => ({
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
            avgIqrMs === undefined
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

        <BoxPlot
          data={boxPlotData}
          getModelColor={getModelColor}
          getProviderForModel={getProviderForModel}
          normalizeModelName={normalizeModelName}
          dedicatedModels={dedicatedModels}
          crossRegionModels={crossRegionModels}
          isMobile={isMobile}
        />
      </Card>
    </div>
  );
};

export default BoxPlotSection;
