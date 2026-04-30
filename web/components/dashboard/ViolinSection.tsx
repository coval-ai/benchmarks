// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { getModelColor } from "@/lib/utils/colors";
import { normalizeModelName } from "@/lib/utils/formatters";
import ViolinPlot from "@/components/charts/d3/ViolinPlot";
import ExpandableDescription from "@/components/shared/ExpandableDescription";
import { useDashboard } from "@/contexts/DashboardContext";

const ViolinSection: React.FC = () => {
  const {
    violinDescription: description,
    getViolinData,
    getProviderForModel,
    isMobile,
    sidebarCollapsed,
    chartRefreshKey,
  } = useDashboard();

  return (
    <div className="mb-16">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-2xl font-light mb-2">Latency Variation</h2>
          <ExpandableDescription description={description} />
        </div>
      </div>

      <div
        className={`${
          isMobile
            ? ""
            : "border border-border-secondary rounded-lg bg-surface-secondary p-4"
        }`}
      >
        <ViolinPlot
          data={getViolinData()}
          getModelColor={getModelColor}
          getProviderForModel={getProviderForModel}
          normalizeModelName={normalizeModelName}
          isMobile={isMobile}
          sidebarCollapsed={sidebarCollapsed}
          key={`violin-${chartRefreshKey}-${sidebarCollapsed}`}
        />
      </div>
    </div>
  );
};

export default ViolinSection;
