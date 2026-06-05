// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo } from "react";
import { useLeaderboardQuery } from "@/lib/api/queries";
import {
  normalizeModelName,
  normalizeSTTProviderName,
  normalizeTTSProviderName,
  toModelKey,
} from "@/lib/utils/formatters";
import type { LeaderboardEntry } from "@/lib/api/client";
import LeaderboardCard, { type LeaderboardRow } from "./LeaderboardCard";

const TOP_N = 5;

/**
 * Map leaderboard entries (already sorted ascending by avg — lower is better)
 * into display rows, formatting each model's average with `formatValue`.
 */
function toRows(
  entries: LeaderboardEntry[] | undefined,
  normalizeProvider: (p: string) => string,
  formatValue: (avg: number) => string
): LeaderboardRow[] {
  if (!entries) return [];
  return entries.slice(0, TOP_N).map((entry) => ({
    key: toModelKey(entry.provider, entry.model),
    model: normalizeModelName(toModelKey(entry.provider, entry.model)),
    provider: normalizeProvider(entry.provider),
    value: formatValue(entry.avg),
  }));
}

const OverviewLeaderboards: React.FC = () => {
  // TTS ranked by Time to First Audio; STT ranked by Word Error Rate.
  // Both metrics are "lower is better", so rank 1 is the API's first entry.
  const ttsQuery = useLeaderboardQuery({
    metric: "TTFA",
    benchmark: "TTS",
    window: "24h",
  });
  const sttQuery = useLeaderboardQuery({
    metric: "WER",
    benchmark: "STT",
    window: "24h",
  });

  const ttsRows = useMemo(
    () =>
      toRows(
        ttsQuery.data?.entries,
        normalizeTTSProviderName,
        (avg) => `${Math.round(avg)} ms`
      ),
    [ttsQuery.data]
  );

  const sttRows = useMemo(
    () =>
      toRows(
        sttQuery.data?.entries,
        normalizeSTTProviderName,
        (avg) => `${avg.toFixed(1)}%`
      ),
    [sttQuery.data]
  );

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <LeaderboardCard
        title="Text-to-Speech"
        metricLabel="Time to First Audio"
        rows={ttsRows}
        href="/tts"
        loading={ttsQuery.isLoading}
        error={ttsQuery.isError}
      />
      <LeaderboardCard
        title="Speech-to-Text"
        metricLabel="Word Error Rate"
        rows={sttRows}
        href="/stt"
        loading={sttQuery.isLoading}
        error={sttQuery.isError}
      />
    </div>
  );
};

export default OverviewLeaderboards;
