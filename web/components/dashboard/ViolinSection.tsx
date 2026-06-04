// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo } from "react";
import { getModelColor } from "@/lib/utils/colors";
import { normalizeModelName } from "@/lib/utils/formatters";
import ViolinPlot from "@/components/charts/d3/ViolinPlot";
import SectionHeader from "@/components/shared/SectionHeader";
import { useDashboard } from "@/contexts/DashboardContext";

const ViolinSection: React.FC = () => {
  const {
    violinDescription: description,
    latencyLabel,
    getViolinData,
    getProviderForModel,
    isMobile,
  } = useDashboard();

  const violinData = getViolinData();
  const medianLatency = useMemo(() => {
    const values: number[] = [];
    for (const modelData of violinData.data) {
      values.push(...modelData.values);
    }
    if (values.length === 0) return 0;
    values.sort((a, b) => a - b);
    const mid = Math.floor(values.length / 2);
    return values.length % 2 === 0
      ? ((values[mid - 1] ?? 0) + (values[mid] ?? 0)) / 2
      : (values[mid] ?? 0);
  }, [violinData]);

  return (
    <div className="mb-4">
      <div
        className={`${
          isMobile
            ? ""
            : "relative z-[2] border border-border-secondary rounded-lg bg-white p-8"
        }`}
      >
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
      </div>
    </div>
  );
};

export default ViolinSection;
