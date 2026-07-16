// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import {
  useState,
  useMemo,
  useCallback,
  useDeferredValue,
  useEffect,
} from "react";
import { useChartData } from "@/hooks/useChartData";
import { useMobileDetection } from "@/hooks/useMobileDetection";
import { useBarInteraction } from "@/hooks/useBarInteraction";
import { latencyToMs, normalizeModelName, normalizeProviderNameForTab, parseModelKey, toModelKey } from "@/lib/utils/formatters";
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
import { WER_BAR_VIEWS, type WerBarView } from "@/lib/config/datasets";
import { useAggregatesQuery, useProvidersQuery } from "@/lib/api/queries";
import { useDatasetScopedWer } from "@/hooks/useDatasetScopedWer";
import { useTimeWindow } from "@/hooks/useTimeWindow";
import type { BarDataPoint, ModelStats } from "@/types/benchmark.types";
import type { SeriesPoint } from "@/lib/api/client";

export function useDashboardState(page: "tts" | "stt" | "s2s") {
  // STT shows a TTFS / TTFT toggle that drives every latency surface (headline,
  // timeline, box plot, scatter, heatmap) in sync. TTS is single-metric (TTFA),
  // so the toggle is hidden there and the metric stays TTFA.
  const [sttMetric, setSttMetric] = useState<"TTFS" | "TTFT">("TTFS");
  const activeMetric =
    page === "s2s" ? "V2V" : page === "tts" ? "TTFA" : sttMetric;
  const { timeWindow, changeTimeWindow } = useTimeWindow(
    `${page}_dashboard`,
    page
  );

  const benchmarkParam =
    page === "s2s" ? "S2S" : page === "tts" ? "TTS" : "STT";

  const aggregatesQuery = useAggregatesQuery({
    benchmark: benchmarkParam,
    window: timeWindow,
  });
  const providersQuery = useProvidersQuery();

  // STT only: pin the WER column to one dataset (null = pooled across all).
  const [werDataset, setWerDataset] = useState<string | null>(null);
  const activeWerDataset = page === "stt" ? werDataset : null;
  const changeWerDataset = useCallback(
    (dataset: string | null) => {
      setWerDataset(dataset);
      capturePostHogEvent(POSTHOG_EVENTS.dashboardWerDatasetChanged, {
        surface: `${page}_dashboard`,
        mode: page,
        dataset: dataset ?? "all",
      });
    },
    [page]
  );

  const { werByModel: werDatasetStats, loading: werDatasetLoading } =
    useDatasetScopedWer(
      { benchmark: benchmarkParam, window: timeWindow },
      activeWerDataset
    );
  const availableWerDatasets = useMemo(
    () => aggregatesQuery.data?.datasets ?? [],
    [aggregatesQuery.data]
  );

  // STT only: the accuracy bar chart switches between the pooled WER
  // (cumulative) and the easy/hard single-dataset views.
  const [werBarView, setWerBarView] = useState<WerBarView>("cumulative");
  const activeWerBarView = page === "stt" ? werBarView : "cumulative";
  const werBarDatasetId =
    WER_BAR_VIEWS.find((v) => v.key === activeWerBarView)?.dataset ?? null;
  const changeWerBarView = useCallback(
    (view: WerBarView) => {
      setWerBarView(view);
      capturePostHogEvent(POSTHOG_EVENTS.dashboardWerBarViewChanged, {
        surface: `${page}_dashboard`,
        mode: page,
        view,
      });
    },
    [page]
  );
  const { werByModel: werBarDatasetStats, loading: werBarLoading } =
    useDatasetScopedWer(
      { benchmark: benchmarkParam, window: timeWindow },
      werBarDatasetId
    );

  useEffect(() => {
    if (availableWerDatasets.length === 0) return;
    if (werDataset && !availableWerDatasets.includes(werDataset)) {
      setWerDataset(null);
    }
    if (werBarDatasetId && !availableWerDatasets.includes(werBarDatasetId)) {
      setWerBarView("cumulative");
    }
  }, [availableWerDatasets, werDataset, werBarDatasetId]);

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
  const { clickedWERBars, handleWERBarClick, clearWERBars } =
    useBarInteraction();

  const deferredSelectedModels = useDeferredValue(selectedModels);


  // useChartData hook - pass page as activeTab internally
  const chartData = useChartData({
    activeTab: page,
    modelStats,
    series,
    selectedTTSModels: page === "tts" ? deferredSelectedModels : [],
    selectedSTTModels: page === "stt" ? deferredSelectedModels : [],
    selectedS2SModels: page === "s2s" ? deferredSelectedModels : [],
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
  const cumulativeWerBarData = chartData.getWERBarData();
  const werBarData = useMemo<BarDataPoint[]>(() => {
    if (!werBarDatasetStats) return cumulativeWerBarData;
    return deferredSelectedModels
      .map((model) => {
        const hit = werBarDatasetStats.get(model);
        return hit
          ? { model, averageWER: hit.avg_value, provider: hit.provider }
          : null;
      })
      .filter((b): b is BarDataPoint => b !== null)
      .sort((a, b) => a.averageWER - b.averageWER);
  }, [werBarDatasetStats, cumulativeWerBarData, deferredSelectedModels]);

  const availableWerBarViews = useMemo(() => {
    if (page !== "stt") return [];
    const available = new Set(availableWerDatasets);
    return WER_BAR_VIEWS.filter(
      (v) => v.dataset === null || available.has(v.dataset)
    );
  }, [page, availableWerDatasets]);

  const werBarDataWithColors = useMemo(() => {
    const hasSelection = werBarData.some((item) => clickedWERBars.has(item.model));
    return werBarData.map((item) => ({
      ...item,
      fill: getModelColor(item.model),
      fillOpacity: hasSelection && !clickedWERBars.has(item.model) ? 0.25 : 1,
    }));
  }, [werBarData, clickedWERBars]);

  // Derived display values
  const latencyLabel = activeMetric;
  const modalityName = { stt: "Speech-to-Text", tts: "Text-to-Speech", s2s: "Speech-to-Speech" }[page];
  const modalitySpaced = { stt: "Speech to Text", tts: "Text to Speech", s2s: "Speech to Speech" }[page];
  const pageTitle = `${modalitySpaced} Model Comparisons`;
  const pageSubtitle = `Compare performance metrics between different ${modalityName} models for voice agent applications.`;
  const sidebarTitle = "Models to Compare";
  const benchmarkTitle = `${modalitySpaced} Voice AI Benchmarks`;
  const mobileSheetTitle = `${modalityName} Models`;
  const normalizeProviderName = useCallback(
    (providerName: string) => normalizeProviderNameForTab(providerName, page),
    [page]
  );

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
          "In voice AI applications, transcription accuracy directly impacts the performance of downstream tasks. Even small transcription errors can lead to misinterpretations, frustrating experiences, or incorrect system responses.",
      };

  const heatmapDisplayData = useMemo(() => {
    const rows = getHeatmapData(activeMetric);
    if (!werDatasetStats) return rows;
    return rows.map((row) => {
      const wer = werDatasetStats.get(row.model);
      return {
        ...row,
        avgWER: wer?.avg_value,
        werStdDev: wer?.stddev_value,
        sampleCount: wer?.sample_count ?? 0,
      };
    });
  }, [getHeatmapData, activeMetric, werDatasetStats]);

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

  // S2S has no WER, so it renders a single KeyMetric tile (no secondary).
  const secondaryKeyMetric =
    page === "s2s"
      ? undefined
      : {
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

    // WER dataset pin (STT comparison card)
    werDataset: activeWerDataset,
    changeWerDataset,
    availableWerDatasets,
    werDatasetLoading,

    // WER bar chart view (STT accuracy card)
    werBarView: activeWerBarView,
    changeWerBarView,
    availableWerBarViews,
    werBarLoading,

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
    clickedWERBars,
    handleWERBarClick,
    clearWERBars,
  };
}
