// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import HeatmapPlot from "@/components/charts/d3/HeatmapPlot";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import MetricToggle from "@/components/dashboard/MetricToggle";
import { useDashboard } from "@/contexts/DashboardContext";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";

const HeatmapSection: React.FC = () => {
  const { heatmapDisplayData: data, formatChartLabel, getProviderForModel, isMobile } =
    useDashboard();
  const trackChartHover = useChartHoverTracking("heatmap");

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8" onMouseEnter={trackChartHover}>
        <SectionHeader
          label="Model Performance Heatmap"
          description={{
            short: "Comprehensive model performance comparison",
            detailed: "Click column headers to sort by metric",
          }}
          expandable={false}
        />

        <MetricToggle />

        {/* The mobile scale transform (useDashboardState) targets
            .heatmap-container, so only the plot lives inside it — keeping the
            header at full size. */}
        <div className="heatmap-container">
          <HeatmapPlot
            data={data}
            formatChartLabel={formatChartLabel}
            getProviderForModel={getProviderForModel}
            isMobile={isMobile}
          />
        </div>
      </Card>
    </div>
  );
};

export default HeatmapSection;
