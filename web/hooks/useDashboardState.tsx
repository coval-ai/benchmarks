// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import {
  useState,
  useMemo,
  useCallback,
  useDeferredValue,
} from "react";
import { useChartData } from "@/hooks/useChartData";
import { useMobileDetection } from "@/hooks/useMobileDetection";
import { useBarInteraction } from "@/hooks/useBarInteraction";
import { latencyToMs, normalizeModelName, normalizeSTTProviderName, normalizeTTSProviderName, parseModelKey, toModelKey } from "@/lib/utils/formatters";
import { buildModelsByProvider } from "@/lib/utils/modelsFromResults";
import {
  buildFacetGroups,
  buildTagIndex,
  filterModelsByFacets,
  getTagCategories,
  hasAnySelection,
  restrictToModelKeys,
  toggleFacetValue,
  type FacetSelection,
} from "@/lib/utils/facets";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";
import { getModelColor } from "@/lib/utils/colors";
import { metricDescriptions } from "@/lib/config/metrics";
import { useAggregatesQuery, useProvidersQuery } from "@/lib/api/queries";
import { useTimeWindow } from "@/hooks/useTimeWindow";
import type { ModelStats } from "@/types/benchmark.types";
import type { SeriesPoint } from "@/lib/api/client";

export function useDashboardState(page: "tts" | "stt") {
  // STT shows a TTFS / TTFT toggle that drives every latency surface (headline,
  // timeline, box plot, scatter, heatmap) in sync. TTS is single-metric (TTFA),
  // so the toggle is hidden there and the metric stays TTFA.
  const [sttMetric, setSttMetric] = useState<"TTFS" | "TTFT">("TTFS");
  const activeMetric = page === "tts" ? "TTFA" : sttMetric;
  const { timeWindow, changeTimeWindow } = useTimeWindow(
    `${page}_dashboard`,
    page
  );

  const benchmarkParam = page === "tts" ? "TTS" : "STT";

  const aggregatesQuery = useAggregatesQuery({
    benchmark: benchmarkParam,
    window: timeWindow,
  });
  const providersQuery = useProvidersQuery();

  // The charts keep showing the prior window's data while a new one loads,
  // so window-derived rendering must follow the data, not the toggle.
  const dataTimeWindow = aggregatesQuery.data?.window ?? timeWindow;
  const windowDataStale = aggregatesQuery.isPlaceholderData;
  const loadError = aggregatesQuery.isError;

  const modelStats = useMemo<ModelStats[]>(
    () => aggregatesQuery.data?.model_stats ?? [],
    [aggregatesQuery.data]
  );
  const series = useMemo<SeriesPoint[]>(
    () => aggregatesQuery.data?.series ?? [],
    [aggregatesQuery.data]
  );

  const allModelsByProvider = useMemo(
    () => buildModelsByProvider(modelStats, benchmarkParam, providersQuery.data),
    [providersQuery.data, modelStats, benchmarkParam]
  );
  const tagIndex = useMemo(
    () => buildTagIndex(benchmarkParam, providersQuery.data),
    [benchmarkParam, providersQuery.data]
  );
  const tagCategories = useMemo(
    () => getTagCategories(providersQuery.data),
    [providersQuery.data]
  );

  // Facets are driven only by models that actually have data to plot. A
  // catalogue model without stats (e.g. a batch-only or not-yet-benchmarked
  // model) would otherwise surface a chip that counts 1 but charts nothing.
  const dataBackedByProvider = useMemo(() => {
    const withData = new Set(modelStats.map((s) => toModelKey(s.provider, s.model)));
    return restrictToModelKeys(allModelsByProvider, withData);
  }, [allModelsByProvider, modelStats]);

  const [selectedFacets, setSelectedFacets] = useState<FacetSelection>({});
  const modelsByProvider = useMemo(
    () => filterModelsByFacets(dataBackedByProvider, tagIndex, selectedFacets),
    [dataBackedByProvider, tagIndex, selectedFacets]
  );
  const hasActiveFacets = useMemo(
    () => hasAnySelection(selectedFacets),
    [selectedFacets]
  );
  const toggleFacet = useCallback(
    (category: string, value: string) => {
      const removing = (selectedFacets[category] ?? []).includes(value);
      const next = toggleFacetValue(selectedFacets, category, value);
      setSelectedFacets(next);
      capturePostHogEvent(POSTHOG_EVENTS.dashboardFacetChanged, {
        surface: `${page}_dashboard`,
        mode: page,
        action: removing ? "remove" : "add",
        category,
        value,
        active_facet_count: Object.values(next).reduce((n, v) => n + v.length, 0),
      });
    },
    [selectedFacets, page]
  );
  const clearFacets = useCallback(() => {
    if (!hasAnySelection(selectedFacets)) return;
    setSelectedFacets({});
    capturePostHogEvent(POSTHOG_EVENTS.dashboardFacetChanged, {
      surface: `${page}_dashboard`,
      mode: page,
      action: "clear",
    });
  }, [selectedFacets, page]);

  // The facet filter is the only selector now, and its universe is already the
  // data-backed models, so everything still visible after filtering is plotted.
  const selectedModels = useMemo(
    () => Object.values(modelsByProvider).flat(),
    [modelsByProvider]
  );

  const loading = aggregatesQuery.isLoading || providersQuery.isLoading;

  const isMobile = useMobileDetection();
  const { clickedWERBars, handleWERBarClick } =
    useBarInteraction();

  const deferredSelectedModels = useDeferredValue(selectedModels);


  // useChartData hook - pass page as activeTab internally
  const chartData = useChartData({
    activeTab: page,
    modelStats,
    series,
    selectedTTSModels: page === "tts" ? deferredSelectedModels : [],
    selectedSTTModels: page === "stt" ? deferredSelectedModels : [],
    modelsByProvider,
    timeWindow: dataTimeWindow,
  });

  // Calculate metrics
  const { getStat, getHeatmapData } = chartData;

  // Run-weighted average latency across selected models, in display units:
  // Σ(avg·runs) / Σ(runs). Backs the box plot and timeline headlines.
  const getAvgLatencyMs = useCallback(
    (metric: string) => {
      let weightedSum = 0;
      let totalRuns = 0;
      deferredSelectedModels.forEach((model) => {
        const stat = getStat(model, metric);
        if (
          stat &&
          typeof stat.avg_value === "number" &&
          typeof stat.sample_count === "number"
        ) {
          weightedSum += stat.avg_value * stat.sample_count;
          totalRuns += stat.sample_count;
        }
      });
      if (totalRuns === 0) return 0;
      return latencyToMs(weightedSum / totalRuns, page);
    },
    [getStat, deferredSelectedModels, page]
  );

  // Latency KPI headline: the fastest model (lowest median) and its median, in
  // display units.
  const fastestPrimary = useMemo(() => {
    let lowestP50 = Infinity;
    let fastestModel = "";
    let fastestProvider = "";
    deferredSelectedModels.forEach((model) => {
      const stat = getStat(model, activeMetric);
      if (stat && typeof stat.p50 === "number" && stat.p50 < lowestP50) {
        lowestP50 = stat.p50;
        fastestModel = model;
        fastestProvider = parseModelKey(model).provider;
      }
    });
    return {
      fastestModel,
      fastestProvider,
      fastestMs: lowestP50 === Infinity ? 0 : latencyToMs(lowestP50, page),
    };
  }, [getStat, deferredSelectedModels, activeMetric, page]);

  const keyMetrics = useMemo(() => {
    let lowestSecondary = Infinity;
    let lowestWERModel = "";
    let lowestWERProvider = "";

    deferredSelectedModels.forEach((model) => {
      const werStat = getStat(model, "WER");
      if (werStat && werStat.avg_value < lowestSecondary) {
        lowestSecondary = werStat.avg_value;
        lowestWERModel = model;
        lowestWERProvider = parseModelKey(model).provider;
      }
    });

    return {
      avgSecondary: lowestSecondary !== Infinity ? lowestSecondary : 0,
      lowestWERModel,
      lowestWERProvider,
    };
  }, [getStat, deferredSelectedModels]);

  const { avgSecondary, lowestWERModel, lowestWERProvider } = keyMetrics;

  // Get computed data
  const werBarData = chartData.getWERBarData();

  const werBarDataWithColors = useMemo(() => {
    const hasSelection = clickedWERBars.size > 0;
    return werBarData.map((item) => ({
      ...item,
      fill: getModelColor(item.model),
      fillOpacity: hasSelection && !clickedWERBars.has(item.model) ? 0.25 : 1,
    }));
  }, [werBarData, clickedWERBars]);

  // Derived display values
  const latencyLabel = activeMetric;
  const pageTitle = page === "tts" ? "Text to Speech Model Comparisons" : "Speech to Text Model Comparisons";
  const pageSubtitle = page === "tts"
    ? "Compare performance metrics between different Text-to-Speech models for voice agent applications."
    : "Compare performance metrics between different Speech-to-Text models for voice agent applications.";
  const sidebarTitle = "Models to Compare";
  const benchmarkTitle = page === "tts"
    ? "Text to Speech Voice AI Benchmarks"
    : "Speech to Text Voice AI Benchmarks";
  const mobileSheetTitle = page === "tts"
    ? "Text-to-Speech Models"
    : "Speech-to-Text Models";
  const normalizeProviderName = page === "stt"
    ? normalizeSTTProviderName
    : normalizeTTSProviderName;

  const facetGroups = useMemo(
    () =>
      buildFacetGroups(
        dataBackedByProvider,
        tagIndex,
        selectedFacets,
        tagCategories,
        normalizeProviderName
      ),
    [dataBackedByProvider, tagIndex, selectedFacets, tagCategories, normalizeProviderName]
  );

  const boxPlotDescription = {
    short: `Distribution of ${latencyLabel} values across all runs`,
    detailed:
      "Narrow distributions indicate reliable, predictable response times, while wide distributions show erratic performance that may frustrate users despite good average speeds. A model with moderate median latency and tight distribution often provides superior user experience compared to a faster median model with high variability.",
  };

  const werDescription = page === "tts"
    ? {
        short: metricDescriptions.wer.short,
        detailed: metricDescriptions.wer.detailed,
      }
    : {
        short: "Word Error Rate (%)",
        detailed:
          "In voice AI applications, transcription accuracy directly impacts the performance of downstream tasks. Even small transcription errors can lead to misinterpretations, frustrating experiences, or incorrect system responses. We evaluate against test audio that includes diverse speakers, accents, and real-world audio conditions. Click a bar to highlight it for comparison.",
      };

  const heatmapDisplayData = useMemo(
    () => getHeatmapData(activeMetric),
    [getHeatmapData, activeMetric]
  );

  // Pre-computed key metrics for display
  const primaryKeyMetric = {
    label: `Lowest Median ${activeMetric}`,
    displayValue: `${fastestPrimary.fastestMs.toFixed(0)} ms`,
    subtitle: fastestPrimary.fastestModel
      ? {
          name: normalizeModelName(fastestPrimary.fastestModel),
          detail: fastestPrimary.fastestProvider
            ? normalizeProviderName(fastestPrimary.fastestProvider)
            : undefined,
        }
      : undefined,
  };

  const secondaryKeyMetric = {
    label: `${deferredSelectedModels.length > 1 ? "Lowest" : "Average"} Word Error Rate`,
    displayValue: `${avgSecondary.toFixed(1)}%`,
    subtitle: lowestWERModel
      ? {
          name: normalizeModelName(lowestWERModel),
          detail: lowestWERProvider
            ? normalizeProviderName(lowestWERProvider)
            : undefined,
        }
      : undefined,
  };

  return {
    // Page identity
    latencyLabel,

    // Display strings
    pageTitle,
    pageSubtitle,
    sidebarTitle,
    benchmarkTitle,
    mobileSheetTitle,
    normalizeProviderName,

    // Section descriptions
    boxPlotDescription,
    werDescription,

    // Metric toggle (STT: TTFS/TTFT; TTS: always TTFA)
    sttMetric,
    setSttMetric,
    activeMetric,

    // Key metrics
    primaryKeyMetric,
    secondaryKeyMetric,
    getAvgLatencyMs,

    // Data loading
    loading,
    loadError,

    // Model state
    selectedModels,
    modelsByProvider,

    // Faceted filters
    facetGroups,
    toggleFacet,
    clearFacets,
    hasActiveFacets,

    // UI state
    isMobile,
    timeWindow,
    dataTimeWindow,
    windowDataStale,

    // Actions
    changeTimeWindow,

    // Chart data functions
    formatChartLabel: chartData.formatChartLabel,
    getProviderForModel: chartData.getProviderForModel,
    getWindowedTimelineData: chartData.getWindowedTimelineData,
    getTimelineData: chartData.getTimelineData,
    getCurrentTimeWindow: chartData.getCurrentTimeWindow,
    getTimelineTicks: chartData.getTimelineTicks,
    getModelsWithTimelineData: chartData.getModelsWithTimelineData,
    getBoxPlotData: chartData.getBoxPlotData,

    // Computed chart data
    getScatterData: chartData.getScatterData,
    heatmapDisplayData,
    werBarDataWithColors,

    // Bar interaction
    handleWERBarClick,
  };
}
