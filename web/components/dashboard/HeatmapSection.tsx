// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import ModelComparisonTable from "@/components/dashboard/ModelComparisonTable";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import MetricToggle from "@/components/dashboard/MetricToggle";
import { useDashboard } from "@/contexts/DashboardContext";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";

const HeatmapSection: React.FC = () => {
  const { heatmapDisplayData: data, getProviderForModel } = useDashboard();
  const trackChartHover = useChartHoverTracking("heatmap");

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8" onMouseEnter={trackChartHover}>
        <SectionHeader
          label="Model Comparison"
          description={{
            short: "How the models stack up",
            detailed:
              "Latency percentiles come straight from the measured runs — drag the slider to move from the fastest run (p0) through the median to the slowest (p100). Click a column to sort.",
          }}
          expandable={false}
        />

        <MetricToggle />

        <ModelComparisonTable data={data} getProviderForModel={getProviderForModel} />
      </Card>
    </div>
  );
};

export default HeatmapSection;
