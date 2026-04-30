// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { normalizeModelName } from "@/lib/utils/formatters";
import ExpandableDescription from "@/components/shared/ExpandableDescription";
import { useDashboard } from "@/contexts/DashboardContext";

const PerformanceRankings: React.FC = () => {
  const { getSTTRankingData } = useDashboard();
  const rankingData = getSTTRankingData();

  return (
    <div className="mb-16">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-2xl font-light mb-2">Performance Rankings</h2>
          <ExpandableDescription
            description={{
              short: "Latency Delta",
              detailed:
                "We measure how far behind each model falls at every individual test run, then calculate the 25th, 50th, and 75th percentiles of those deltas. Models ranked higher offer more reliable performance across various test cases.",
            }}
          />
        </div>
      </div>
      <div className="overflow-x-auto">
        <div className="min-w-full">
          <div className="grid grid-cols-6 gap-2 md:gap-6 px-2 md:px-4 py-3 text-[10px] md:text-xs font-medium text-text-secondary uppercase tracking-wider border-b border-border-secondary">
            <div>POS.</div>
            <div>MODEL</div>
            <div>PROVIDER</div>
            <div className="text-right">P25</div>
            <div className="text-right">P50</div>
            <div className="text-right">P75</div>
          </div>

          {rankingData.map((row) => (
            <div
              key={row.model}
              className={`grid grid-cols-6 gap-2 md:gap-4 px-2 md:px-4 py-3 text-xs md:text-sm border-b border-border-secondary hover:bg-hover-bg transition-colors ${
                row.isFirst ? "text-green-600 dark:text-green-400" : "text-text-primary"
              }`}
            >
              <div className="font-bold text-base md:text-lg">
                {row.position}
              </div>
              <div className="font-medium text-[11px] md:text-sm">
                {normalizeModelName(row.model)}
              </div>
              <div className="text-text-secondary text-[10px] md:text-sm">
                {row.provider}
              </div>
              <div className="text-right font-mono text-[10px] md:text-sm">
                {row.isFirst
                  ? "\u2014"
                  : `+${(row.p25Delta / 1000).toFixed(3)}s`}
              </div>
              <div className="text-right font-mono text-[10px] md:text-sm">
                {row.isFirst
                  ? "\u2014"
                  : `+${(row.p50Delta / 1000).toFixed(3)}s`}
              </div>
              <div className="text-right font-mono text-[10px] md:text-sm">
                {row.isFirst
                  ? "\u2014"
                  : `+${(row.p75Delta / 1000).toFixed(3)}s`}
              </div>
            </div>
          ))}

          {rankingData.length === 0 && (
            <div className="px-4 py-8 text-center text-text-tertiary">
              No STT models selected
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PerformanceRankings;
