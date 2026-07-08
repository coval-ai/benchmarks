// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import {
  methodologyChanges,
  methodologyChangeTs,
  type MethodologyMetricKey,
} from "@/lib/config/methodologyChanges";
import { useDashboard } from "@/contexts/DashboardContext";
import { formatDate } from "@/lib/utils/formatters";

interface MethodologyNoteProps {
  metric: MethodologyMetricKey;
}

/**
 * Methodology-change note for charts without a time axis. Shown only while the
 * selected data window spans the change, i.e. while the chart aggregates
 * non-comparable values.
 */
const MethodologyNote: React.FC<MethodologyNoteProps> = ({ metric }) => {
  const { getCurrentTimeWindow } = useDashboard();
  const [windowStart, windowEnd] = getCurrentTimeWindow();

  const visibleChanges = methodologyChanges
    .map((change) => ({ change, ts: methodologyChangeTs(change) }))
    .filter(
      ({ change, ts }) =>
        (!change.metrics || change.metrics.includes(metric)) &&
        ts >= windowStart &&
        ts <= windowEnd
    );

  if (visibleChanges.length === 0) return null;

  return (
    <div className="mb-4 space-y-2">
      {visibleChanges.map(({ change, ts }) => (
        <p
          key={`${change.date}-${change.title}`}
          className="border-l-2 border-[#f59e0b] pl-3 text-sm font-light leading-snug text-text-tertiary"
        >
          <span className="text-text-secondary">
            Methodology change · {formatDate(ts)}:
          </span>{" "}
          {change.detail}
        </p>
      ))}
    </div>
  );
};

export default MethodologyNote;
