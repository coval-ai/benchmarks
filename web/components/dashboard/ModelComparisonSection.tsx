// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useState } from "react";
import ModelComparisonTable, {
  DEFAULT_PERCENTILE_IDX,
  PERCENTILES,
} from "@/components/dashboard/ModelComparisonTable";
import Card from "@/components/shared/Card";
import SectionHeader from "@/components/shared/SectionHeader";
import MetricToggle from "@/components/dashboard/MetricToggle";
import { datasetLabel } from "@/lib/config/datasets";
import { useDashboard } from "@/contexts/DashboardContext";
import { useChartHoverTracking } from "@/hooks/useChartHoverTracking";

const ModelComparisonSection: React.FC = () => {
  const {
    heatmapDisplayData: data,
    getProviderForModel,
    activeMetric,
    werDataset,
    werDatasetLoading,
  } = useDashboard();
  const trackChartHover = useChartHoverTracking("heatmap");
  const [percentileIdx, setPercentileIdx] = useState(DEFAULT_PERCENTILE_IDX);
  const percentile = (PERCENTILES[percentileIdx] ?? PERCENTILES[DEFAULT_PERCENTILE_IDX])!.key;

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
          exportRows={() =>
            data.map(({ model, latency, avgWER, werStdDev, sampleCount }) => ({
              model,
              provider: getProviderForModel(model),
              metric: activeMetric,
              [`latency_${percentile}_ms`]: latency[percentile],
              ...(avgWER !== undefined
                ? { avg_wer_percent: avgWER, wer_std_dev: werStdDev }
                : {}),
              ...(werDataset ? { wer_dataset: werDataset } : {}),
              runs: sampleCount,
            }))
          }
          exportImage={false}
        />

        <MetricToggle />

        <ModelComparisonTable
          data={data}
          getProviderForModel={getProviderForModel}
          percentileIdx={percentileIdx}
          onPercentileChange={setPercentileIdx}
          werLabel={werDataset ? datasetLabel(werDataset) : undefined}
          werLoading={werDatasetLoading}
        />
      </Card>
    </div>
  );
};

export default ModelComparisonSection;
