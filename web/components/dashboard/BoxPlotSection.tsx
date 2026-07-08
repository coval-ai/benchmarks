// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo } from "react";
import { getModelColor } from "@/lib/utils/colors";
import { normalizeModelName } from "@/lib/utils/formatters";
import BoxPlot from "@/components/charts/d3/BoxPlot";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import MetricInfo from "@/components/shared/MetricInfo";
import MetricToggle from "@/components/dashboard/MetricToggle";
import { useDashboard } from "@/contexts/DashboardContext";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";

const BoxPlotSection: React.FC = () => {
  const {
    boxPlotDescription: description,
    latencyLabel,
    getBoxPlotData,
    getAvgLatencyMs,
    getProviderForModel,
    isMobile,
    activeMetric,
  } = useDashboard();
  const trackChartHover = useChartHoverTracking("box_plot");

  const boxPlotData = useMemo(
    () => getBoxPlotData(activeMetric),
    [getBoxPlotData, activeMetric]
  );
  // Run-weighted average latency across all selected models.
  const avgLatency = getAvgLatencyMs(activeMetric);

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8" onMouseEnter={trackChartHover}>
        <SectionHeader
          label="Latency Variation"
          description={description}
          stat={{
            label: (
              <MetricInfo metric={activeMetric} align="right">{`Average ${latencyLabel}`}</MetricInfo>
            ),
            value: `${avgLatency.toFixed(0)} ms`,
          }}
        />

        <MetricToggle />

        <BoxPlot
          data={boxPlotData}
          getModelColor={getModelColor}
          getProviderForModel={getProviderForModel}
          normalizeModelName={normalizeModelName}
          isMobile={isMobile}
        />
      </Card>
    </div>
  );
};

export default BoxPlotSection;
