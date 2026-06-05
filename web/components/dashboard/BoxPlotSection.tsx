// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo } from "react";
import { getModelColor } from "@/lib/utils/colors";
import { normalizeModelName } from "@/lib/utils/formatters";
import { median } from "@/lib/utils/median";
import BoxPlot from "@/components/charts/d3/BoxPlot";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import { useDashboard } from "@/contexts/DashboardContext";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";

const BoxPlotSection: React.FC = () => {
  const {
    boxPlotDescription: description,
    latencyLabel,
    getBoxPlotData,
    getProviderForModel,
    isMobile,
  } = useDashboard();
  const trackChartHover = useChartHoverTracking("box_plot");

  const boxPlotData = getBoxPlotData();
  // Median across the selected models' per-model medians.
  const medianLatency = useMemo(
    () => median(boxPlotData.data.map((modelData) => modelData.quartiles.median)),
    [boxPlotData]
  );

  return (
    <div className="mb-4">
      <Card onMouseEnter={trackChartHover}>
        <SectionHeader
          label="Latency Variation"
          description={description}
          stat={{
            label: `Median ${latencyLabel}`,
            value: `${medianLatency.toFixed(0)} ms`,
          }}
        />

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
