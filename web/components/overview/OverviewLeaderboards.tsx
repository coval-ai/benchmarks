// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useMemo } from "react";
import { useAggregatesQuery, usePricingQuery, useProvidersQuery } from "@/lib/api/queries";
import { buildTagIndex, dedicatedModelKeys } from "@/lib/utils/facets";
import {
  formatUsd,
  normalizeModelName,
  normalizeS2SProviderName,
  normalizeSTTProviderName,
  normalizeTTSProviderName,
  toModelKey,
} from "@/lib/utils/formatters";
import { priceUnitShortLabel } from "@/lib/utils/pricing";
import type { PricingApiResponse } from "@/lib/api/client";
import { S2S_MULTITURN_DATASET } from "@/lib/config/datasets";
import type { ModelStatEntry, ProvidersApiResponse } from "@/lib/api/client";
import { disabledModelKeys } from "@/lib/utils/modelsFromResults";
import { useTimeWindow } from "@/hooks/useTimeWindow";
import { S2S_24H_ENABLED, WINDOW_LABELS, type TimeWindow } from "@/lib/config/timeWindows";
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

// Cheapest normalized list price, ranked ascending. Prices are window-free
// (list rates, not measurements), so the badge carries the unit instead.
function toPriceRows(
  pricing: PricingApiResponse | undefined,
  normalizeProvider: (p: string) => string
): LeaderboardRow[] {
  return (pricing?.entries ?? [])
    .filter((e) => e.normalized_usd != null)
    .sort((a, b) => a.normalized_usd! - b.normalized_usd!)
    .slice(0, TOP_N)
    .map((e) => ({
      key: toModelKey(e.provider, e.model),
      model: normalizeModelName(toModelKey(e.provider, e.model)),
      provider: normalizeProvider(e.provider),
      value: formatUsd(e.normalized_usd!),
    }));
}

// Dedicated-inference endpoints stay off these shared-latency rankings, and
// disabled (retired/pending) models still present in the window's stats too.
const excludedKeys = (
  benchmark: "STT" | "TTS" | "S2S",
  providers: ProvidersApiResponse
): Set<string> =>
  new Set([
    ...dedicatedModelKeys(buildTagIndex(benchmark, providers)),
    ...disabledModelKeys(benchmark, providers),
  ]);

const OverviewLeaderboards: React.FC = () => {
  const { timeWindow, changeTimeWindow } = useTimeWindow("overview");

  // Params match what /tts and /stt send, so React Query shares a cache
  // entry with the dashboards whenever the selected windows coincide.
  const ttsQuery = useAggregatesQuery({ benchmark: "TTS", window: timeWindow });
  const sttQuery = useAggregatesQuery({ benchmark: "STT", window: timeWindow });
  // S2S ingests daily, so it has no 24h window — pin the card to 7d there.
  const s2sWindow: TimeWindow =
    timeWindow === "24h" && !S2S_24H_ENABLED ? "7d" : timeWindow;
  const s2sQuery = useAggregatesQuery({
    benchmark: "S2S",
    window: s2sWindow,
    dataset: S2S_MULTITURN_DATASET,
  });
  const providersQuery = useProvidersQuery();
  const ttsPricing = usePricingQuery("TTS");
  const sttPricing = usePricingQuery("STT");
  const s2sPricing = usePricingQuery("S2S");
  const windowDataStale =
    ttsQuery.isPlaceholderData ||
    sttQuery.isPlaceholderData ||
    s2sQuery.isPlaceholderData;

  // These cards rank shared latency, so dedicated-inference endpoints stay
  // off them — same rule as the dashboards' latency timeline. Rankings are
  // never built without the provider metadata that classifies endpoints.
  const ttsRows = useMemo(
    () =>
      providersQuery.data
        ? toRows(
            ttsQuery.data?.model_stats,
            "TTFA",
            "p50",
            normalizeTTSProviderName,
            (value) => `${Math.round(value)} ms`,
            excludedKeys("TTS", providersQuery.data)
          )
        : [],
    [ttsQuery.data, providersQuery.data]
  );

  const sttRows = useMemo(
    () =>
      providersQuery.data
        ? toRows(
            sttQuery.data?.model_stats,
            "TTFS",
            "p50",
            normalizeSTTProviderName,
            (value) => `${Math.round(value * 1000)} ms`,
            excludedKeys("STT", providersQuery.data)
          )
        : [],
    [sttQuery.data, providersQuery.data]
  );

  const s2sRows = useMemo(
    () =>
      providersQuery.data
        ? toRows(
            s2sQuery.data?.model_stats,
            "V2V",
            "p50",
            normalizeS2SProviderName,
            (value) => `${Math.round(value)} ms`,
            excludedKeys("S2S", providersQuery.data)
          )
        : [],
    [s2sQuery.data, providersQuery.data]
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
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
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
        <LeaderboardCard
          title="Speech-to-Speech"
          metricLabel="Voice-to-Voice Latency"
          windowLabel={windowBadge(s2sQuery.data?.window ?? s2sWindow)}
          rows={s2sRows}
          href="/s2s"
          loading={s2sQuery.isLoading || providersQuery.isLoading}
          stale={s2sQuery.isPlaceholderData}
          error={s2sQuery.isError || providersQuery.isError}
        />
      </div>
      {/* Cheapest by normalized list price. Hidden entirely (never an error
          card) while pricing has nothing to rank — the latency cards above
          are the page's core. */}
      {(toPriceRows(ttsPricing.data, normalizeTTSProviderName).length > 0 ||
        toPriceRows(sttPricing.data, normalizeSTTProviderName).length > 0 ||
        toPriceRows(s2sPricing.data, normalizeS2SProviderName).length > 0) && (
        <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
          <LeaderboardCard
            title="Cheapest Text-to-Speech"
            metricLabel="List price"
            windowLabel={priceUnitShortLabel("tts")}
            rows={toPriceRows(ttsPricing.data, normalizeTTSProviderName)}
            href="/tts"
            loading={ttsPricing.isLoading}
            error={ttsPricing.isError}
          />
          <LeaderboardCard
            title="Cheapest Speech-to-Text"
            metricLabel="List price"
            windowLabel={priceUnitShortLabel("stt")}
            rows={toPriceRows(sttPricing.data, normalizeSTTProviderName)}
            href="/stt"
            loading={sttPricing.isLoading}
            error={sttPricing.isError}
          />
          <LeaderboardCard
            title="Cheapest Speech-to-Speech"
            metricLabel="List price"
            windowLabel={priceUnitShortLabel("s2s")}
            rows={toPriceRows(s2sPricing.data, normalizeS2SProviderName)}
            href="/s2s"
            loading={s2sPricing.isLoading}
            error={s2sPricing.isError}
          />
        </div>
      )}
    </div>
  );
};

export default OverviewLeaderboards;
