"use client";

import React from "react";
import HeatmapPlot from "@/components/charts/d3/HeatmapPlot";
import { useDashboard } from "@/contexts/DashboardContext";

const HeatmapSection: React.FC = () => {
  const { heatmapDisplayData: data, formatChartLabel, getProviderForModel, isMobile } =
    useDashboard();

  return (
    <div className="mb-16">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-2xl font-light mb-2">
            Model Performance Heatmap
          </h2>
          <p className="text-text-secondary mb-4">
            Comprehensive model performance comparison &bull; Click column
            headers to sort by metric
          </p>
        </div>
      </div>

      <div
        className={`heatmap-container ${
          isMobile
            ? ""
            : "border border-border-secondary rounded-lg bg-surface-secondary p-4"
        }`}
      >
        <HeatmapPlot
          data={data}
          formatChartLabel={formatChartLabel}
          getProviderForModel={getProviderForModel}
          isMobile={isMobile}
        />
      </div>
    </div>
  );
};

export default HeatmapSection;
