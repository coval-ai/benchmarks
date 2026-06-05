// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo } from "react";
import { getModelColor } from "@/lib/utils/colors";
import { normalizeModelName } from "@/lib/utils/formatters";
import { median } from "@/lib/utils/median";
import ViolinPlot from "@/components/charts/d3/ViolinPlot";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import { useDashboard } from "@/contexts/DashboardContext";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";

const ViolinSection: React.FC = () => {
  const {
    violinDescription: description,
    latencyLabel,
    getViolinData,
    getProviderForModel,
    isMobile,
  } = useDashboard();
  const trackChartHover = useChartHoverTracking("violin");

  const violinData = getViolinData();
  const medianLatency = useMemo(() => {
    const values: number[] = [];
    for (const modelData of violinData.data) {
      for (const value of modelData.values) {
        values.push(value);
      }
    }
    return median(values);
  }, [violinData]);

  return (
    <div className="mb-4">
      <Card padding="p-5 lg:p-8" onMouseEnter={trackChartHover}>
        <SectionHeader
          label="Latency Variation"
          description={description}
          stat={{
            label: `Median ${latencyLabel}`,
            value: `${medianLatency.toFixed(0)} ms`,
          }}
        />

        <ViolinPlot
          data={violinData}
          getModelColor={getModelColor}
          getProviderForModel={getProviderForModel}
          normalizeModelName={normalizeModelName}
          isMobile={isMobile}
        />
      </Card>
    </div>
  );
};

export default ViolinSection;
