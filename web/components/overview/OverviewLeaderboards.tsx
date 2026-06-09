// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useCallback, useMemo, useState } from "react";
import { useAggregatesQuery } from "@/lib/api/queries";
import {
  normalizeModelName,
  normalizeSTTProviderName,
  normalizeTTSProviderName,
  toModelKey,
} from "@/lib/utils/formatters";
import type { ModelStatEntry } from "@/lib/api/client";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";
import TimeWindowToggle, {
  type TimeWindow,
} from "@/components/shared/TimeWindowToggle";
import LeaderboardCard, { type LeaderboardRow } from "./LeaderboardCard";

const TOP_N = 5;

// Rank field is chosen to match the /tts and /stt pages exactly, so the
// overview cards never disagree with the full leaderboards: TTS ranks by p50
// of TTFA, STT by avg of WER. All are "lower is better".
function toRows(
  stats: ModelStatEntry[] | undefined,
  metricType: string,
  rankField: "p50" | "avg_value",
  normalizeProvider: (p: string) => string,
  formatValue: (value: number) => string
): LeaderboardRow[] {
  if (!stats) return [];
  return stats
    .filter((s) => s.metric_type === metricType)
    .sort((a, b) => a[rankField] - b[rankField])
    .slice(0, TOP_N)
    .map((s) => ({
      key: toModelKey(s.provider, s.model),
      model: normalizeModelName(toModelKey(s.provider, s.model)),
      provider: normalizeProvider(s.provider),
      value: formatValue(s[rankField]),
    }));
}

const WINDOW_BADGES: Record<TimeWindow, string> = {
  "24h": "Last 1d",
  "7d": "Last 7d",
  "30d": "Last 30d",
};

const OverviewLeaderboards: React.FC = () => {
  const [timeWindow, setTimeWindow] = useState<TimeWindow>("24h");

  const changeTimeWindow = useCallback(
    (next: TimeWindow) => {
      if (next === timeWindow) return;
      capturePostHogEvent(POSTHOG_EVENTS.dashboardTimeWindowChanged, {
        surface: "overview",
        from: timeWindow,
        to: next
      });
      setTimeWindow(next);
    },
    [timeWindow]
  );

  // Same endpoint and params the /tts and /stt pages use, so React Query
  // serves both from one cache entry and the numbers are identical.
  const ttsQuery = useAggregatesQuery({ benchmark: "TTS", window: timeWindow });
  const sttQuery = useAggregatesQuery({ benchmark: "STT", window: timeWindow });

  const ttsRows = useMemo(
    () =>
      toRows(
        ttsQuery.data?.model_stats,
        "TTFA",
        "p50",
        normalizeTTSProviderName,
        (value) => `${Math.round(value)} ms`
      ),
    [ttsQuery.data]
  );

  const sttRows = useMemo(
    () =>
      toRows(
        sttQuery.data?.model_stats,
        "WER",
        "avg_value",
        normalizeSTTProviderName,
        (value) => `${value.toFixed(1)}%`
      ),
    [sttQuery.data]
  );

  return (
    <div>
      <div className="mb-3 flex justify-end">
        <TimeWindowToggle value={timeWindow} onChange={changeTimeWindow} />
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <LeaderboardCard
          title="Text-to-Speech"
          metricLabel="Time to First Audio"
          windowLabel={WINDOW_BADGES[timeWindow]}
          rows={ttsRows}
          href="/tts"
          loading={ttsQuery.isLoading}
          error={ttsQuery.isError}
        />
        <LeaderboardCard
          title="Speech-to-Text"
          metricLabel="Word Error Rate"
          windowLabel={WINDOW_BADGES[timeWindow]}
          rows={sttRows}
          href="/stt"
          loading={sttQuery.isLoading}
          error={sttQuery.isError}
        />
      </div>
    </div>
  );
};

export default OverviewLeaderboards;
