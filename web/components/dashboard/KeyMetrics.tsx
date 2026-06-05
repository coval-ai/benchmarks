// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import Card from "@/components/shared/Card";
import { useDashboard } from "@/contexts/DashboardContext";

export interface KeyMetricData {
  label: string;
  displayValue: string;
  subtitle?: {
    name?: string;
    detail?: string;
  };
}

const KeyMetrics: React.FC = () => {
  const {
    primaryKeyMetric: primary,
    secondaryKeyMetric: secondary,
    modelsComparedMetric,
    providersMetric,
  } = useDashboard();

  const metrics: KeyMetricData[] = [
    primary,
    secondary,
    modelsComparedMetric,
    providersMetric,
  ];

  return (
    <div className="grid grid-cols-4 gap-4 mb-4 w-full">
      {metrics.map((metric, index) => (
        <Card key={index} className="text-left min-w-0">
          <div className="text-[0.9rem] font-light text-text-secondary mb-2">
            {metric.label}
          </div>
          <div className="font-mono text-5xl font-bold mb-4 break-words leading-tight">
            {metric.displayValue}
          </div>
          {metric.subtitle && (
            <div className="text-text-secondary flex items-baseline gap-2">
              {metric.subtitle.name && (
                <span className="font-medium">{metric.subtitle.name}</span>
              )}
              {metric.subtitle.detail && (
                <span className="text-sm text-text-tertiary">
                  {metric.subtitle.detail}
                </span>
              )}
            </div>
          )}
        </Card>
      ))}
    </div>
  );
};

export default KeyMetrics;
