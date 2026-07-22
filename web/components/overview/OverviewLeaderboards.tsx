// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo } from "react";
import { useAggregatesQuery, useProvidersQuery } from "@/lib/api/queries";
import { buildTagIndex, dedicatedModelKeys } from "@/lib/utils/facets";
import {
  normalizeModelName,
  normalizeSTTProviderName,
  normalizeTTSProviderName,
  toModelKey,
} from "@/lib/utils/formatters";
import type { ModelStatEntry } from "@/lib/api/client";
import { useTimeWindow } from "@/hooks/useTimeWindow";
import { WINDOW_LABELS, type TimeWindow } from "@/lib/config/timeWindows";
import TimeWindowToggle from "@/components/shared/TimeWindowToggle";
import { CymaticLoader } from "@/components/shared/CymaticLoader";
import LeaderboardCard, { type LeaderboardRow } from "./LeaderboardCard";

const TOP_N = 5;

function toRows(
  stats: ModelStatEntry[] | undefined,
  metricType: string,
  rankField: "p50" | "avg_value",
  normalizeProvider: (p: string) => string,
  formatValue: (value: number) => string,
  exclude: Set<string>
): LeaderboardRow[] {
  if (!stats) return [];
  return stats
    .filter(
      (s) =>
        s.metric_type === metricType &&
        !exclude.has(toModelKey(s.provider, s.model))
    )
    .sort((a, b) => a[rankField] - b[rankField])
    .slice(0, TOP_N)
    .map((s) => ({
      key: toModelKey(s.provider, s.model),
      model: normalizeModelName(toModelKey(s.provider, s.model)),
      provider: normalizeProvider(s.provider),
      value: formatValue(s[rankField]),
    }));
}

const windowBadge = (window: TimeWindow): string => `Last ${WINDOW_LABELS[window]}`;

const OverviewLeaderboards: React.FC = () => {
  const { timeWindow, changeTimeWindow } = useTimeWindow("overview");

  // Params match what /tts and /stt send, so React Query shares a cache
  // entry with the dashboards whenever the selected windows coincide.
  const ttsQuery = useAggregatesQuery({ benchmark: "TTS", window: timeWindow });
  const sttQuery = useAggregatesQuery({ benchmark: "STT", window: timeWindow });
  const providersQuery = useProvidersQuery();
  const windowDataStale = ttsQuery.isPlaceholderData || sttQuery.isPlaceholderData;

  // These cards rank shared latency, so dedicated-inference endpoints stay
  // off them — same rule as the dashboards' latency timeline.
  const ttsRows = useMemo(
    () =>
      toRows(
        ttsQuery.data?.model_stats,
        "TTFA",
        "p50",
        normalizeTTSProviderName,
        (value) => `${Math.round(value)} ms`,
        dedicatedModelKeys(buildTagIndex("TTS", providersQuery.data))
      ),
    [ttsQuery.data, providersQuery.data]
  );

  const sttRows = useMemo(
    () =>
      toRows(
        sttQuery.data?.model_stats,
        "TTFS",
        "p50",
        normalizeSTTProviderName,
        (value) => `${Math.round(value * 1000)} ms`,
        dedicatedModelKeys(buildTagIndex("STT", providersQuery.data))
      ),
    [sttQuery.data, providersQuery.data]
  );

  return (
    <div>
      <div className="mb-3 flex items-center justify-end gap-3">
        <CymaticLoader
          size={20}
          animated={windowDataStale}
          className={`text-text-primary transition-opacity duration-300 ${
            windowDataStale ? "opacity-100" : "opacity-0"
          }`}
        />
        <TimeWindowToggle
          value={timeWindow}
          onChange={changeTimeWindow}
          loading={windowDataStale}
        />
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Providers metadata drives the dedicated exclusion, so these cards
            wait for it (and surface its failure) rather than momentarily
            ranking a dedicated endpoint as shared. */}
        <LeaderboardCard
          title="Text-to-Speech"
          metricLabel="Time to First Audio"
          windowLabel={windowBadge(ttsQuery.data?.window ?? timeWindow)}
          rows={ttsRows}
          href="/tts"
          loading={ttsQuery.isLoading || providersQuery.isLoading}
          stale={ttsQuery.isPlaceholderData}
          error={ttsQuery.isError || providersQuery.isError}
        />
        <LeaderboardCard
          title="Speech-to-Text"
          metricLabel="Time to Final Segment"
          windowLabel={windowBadge(sttQuery.data?.window ?? timeWindow)}
          rows={sttRows}
          href="/stt"
          loading={sttQuery.isLoading || providersQuery.isLoading}
          stale={sttQuery.isPlaceholderData}
          error={sttQuery.isError || providersQuery.isError}
        />
      </div>
    </div>
  );
};

export default OverviewLeaderboards;
