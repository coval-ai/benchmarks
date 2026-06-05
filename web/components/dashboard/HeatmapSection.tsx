// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import HeatmapPlot from "@/components/charts/d3/HeatmapPlot";
import Card from "@/components/shared/Card";
import { useDashboard } from "@/contexts/DashboardContext";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";

const HeatmapSection: React.FC = () => {
  const { heatmapDisplayData: data, formatChartLabel, getProviderForModel, isMobile } =
    useDashboard();
  const trackChartHover = useChartHoverTracking("heatmap");

  return (
    <div className="mb-4">
      <Card onMouseEnter={trackChartHover}>
        {/* Inner unpadded wrapper: the mobile scale transform targets
            .heatmap-container, so it must not include the Card chrome */}
        <div className="heatmap-container">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h2 className="text-[0.9rem] font-light text-text-secondary mb-2">
                Model Performance Heatmap
              </h2>
              <p className="text-text-secondary mb-4">
                Comprehensive model performance comparison &bull; Click column
                headers to sort by metric
              </p>
            </div>
          </div>

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
