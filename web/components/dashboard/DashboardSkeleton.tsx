// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";

// A single shimmering placeholder bar. Rendered inline so that the surrounding
// text element's font strut governs the line-box height — this keeps every
// skeleton row the exact same height as its real (text-filled) counterpart.
const Bar: React.FC<{ className?: string }> = ({ className = "" }) => (
  <span
    className={`inline-block max-w-full animate-pulse rounded bg-surface-secondary align-middle ${className}`}
  />
);

// Mirrors a single KeyMetrics card. The wrapper and text-element classes match
// KeyMetrics.tsx exactly (p-8 padding, mb-2/mb-4 spacing, label/value/subtitle
// font sizes), so the card height equals the loaded card. The metric grid uses
// the default `align-items: stretch`, so all four match the tallest card.
const MetricSkeleton: React.FC = () => (
  <div className="text-left border border-border-secondary rounded-lg bg-white p-8 min-w-0">
    <div className="text-[0.9rem] font-light text-text-secondary mb-2">
      <Bar className="h-[0.7em] w-24" />
    </div>
    <div className="font-mono text-5xl font-bold mb-4 break-words leading-tight">
      <Bar className="h-[0.7em] w-32" />
    </div>
    <div className="text-text-secondary flex items-baseline gap-2">
      <span className="font-medium">
        <Bar className="h-[0.7em] w-20" />
      </span>
    </div>
  </div>
);

// Mirrors the first chart card (TimelineChart). The wrapper (p-8 border card)
// and SectionHeader structure match, and the h-96 chart area is reproduced
// verbatim so the overall card height equals the loaded chart.
const ChartSkeleton: React.FC = () => (
  <div className="mb-4">
    <div className="w-full p-8 relative z-[2] border border-border-secondary rounded-lg bg-white">
      {/* SectionHeader mirror */}
      <div className="flex justify-between items-start gap-8 mb-4">
        <div className="w-3/4 min-w-0">
          <div className="text-[0.72rem] font-light text-text-secondary mb-2">
            <Bar className="h-[0.7em] w-28" />
          </div>
          <div className="text-2xl font-medium text-text-primary mb-3">
            <Bar className="h-[0.7em] w-1/2" />
          </div>
          <div className="text-sm leading-snug">
            <Bar className="h-[0.7em] w-full" />
            <Bar className="mt-1 h-[0.7em] w-2/3" />
          </div>
        </div>
        <div className="text-right min-w-0">
          <div className="text-[0.72rem] font-light text-text-secondary mb-2">
            <Bar className="h-[0.7em] w-16" />
          </div>
          <div className="font-mono text-[2.4rem] font-bold break-words">
            <Bar className="h-[0.7em] w-24" />
          </div>
        </div>
      </div>

      {/* Chart area mirror — same h-96 as TimelineChart */}
      <div className="h-96">
        <Bar className="block h-full w-full" />
      </div>
    </div>
  </div>
);

// Loading placeholder for the dashboard: the four key-metric cards plus the
// first chart card, each matching the height of its loaded counterpart.
const DashboardSkeleton: React.FC = () => (
  <>
    <div className="grid grid-cols-4 gap-4 mb-4 w-full">
      {["a", "b", "c", "d"].map((k) => (
        <MetricSkeleton key={k} />
      ))}
    </div>
    <ChartSkeleton />
  </>
);

export default DashboardSkeleton;
