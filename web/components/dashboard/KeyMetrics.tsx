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
  const { primaryKeyMetric: primary, secondaryKeyMetric: secondary } = useDashboard();

  return (
    <div className="grid grid-cols-2 gap-16 mb-16 max-w-4xl mx-auto">
      <div className="text-center">
        <div className="text-text-secondary mb-2">{primary.label}</div>
        <div className="text-5xl font-light mb-2">{primary.displayValue}</div>
        {primary.subtitle && (
          <div className="text-text-secondary">
            {primary.subtitle.name && <div>{primary.subtitle.name}</div>}
            {primary.subtitle.detail && (
              <div className="text-sm text-text-tertiary">
                {primary.subtitle.detail}
              </div>
            )}
          </div>
        )}
      </div>
      <div className="text-center">
        <div className="text-text-secondary mb-2">{secondary.label}</div>
        <div className="text-5xl font-light mb-2">
          {secondary.displayValue}
        </div>
        {secondary.subtitle && (
          <div className="text-text-secondary">
            {secondary.subtitle.name && <div>{secondary.subtitle.name}</div>}
            {secondary.subtitle.detail && (
              <div className="text-sm text-text-tertiary">
                {secondary.subtitle.detail}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default KeyMetrics;
