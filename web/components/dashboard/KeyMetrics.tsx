// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
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
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-[0.8rem] mb-[0.8rem] w-full">
      {metrics.map((metric, index) => (
        <div
          key={index}
          className="text-left border border-border-secondary rounded-lg bg-white p-5 lg:p-8 min-w-0"
        >
          <div className="text-[0.9rem] font-light text-text-secondary mb-2">
            {metric.label}
          </div>
          <div className="font-mono text-3xl sm:text-4xl lg:text-5xl font-bold mb-4 break-words leading-tight">
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
        </div>
      ))}
    </div>
  );
};

export default KeyMetrics;
