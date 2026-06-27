// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import {
  useState,
  useEffect,
  useMemo,
  useCallback,
  useDeferredValue,
} from "react";
import { useChartData } from "@/hooks/useChartData";
import { useMobileDetection } from "@/hooks/useMobileDetection";
import { useBarInteraction } from "@/hooks/useBarInteraction";
import { latencyToMs, normalizeModelName, normalizeSTTProviderName, normalizeTTSProviderName, parseModelKey, toModelKey } from "@/lib/utils/formatters";
import { buildModelsByProvider, pruneSelection } from "@/lib/utils/modelsFromResults";
import {
  buildFacetGroups,
  buildTagIndex,
  filterModelsByFacets,
  toggleFacetValue,
  type FacetSelection,
} from "@/lib/utils/facets";
import { getModelColor } from "@/lib/utils/colors";
import { metricDescriptions } from "@/lib/config/metrics";
import { useAggregatesQuery, useProvidersQuery } from "@/lib/api/queries";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";
import { useTimeWindow } from "@/hooks/useTimeWindow";
import type { ModelStats } from "@/types/benchmark.types";
import type { SeriesPoint } from "@/lib/api/client";

export function useDashboardState(page: "tts" | "stt") {
  // State declarations
  const [selectedModels, setSelectedModels] = useState<string[]>([]);

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

  // Full catalogue (minus disabled) drives the facet options; the facet
  // selection then narrows it to what the sidebar and charts actually show.
  const allModelsByProvider = useMemo(
    () => buildModelsByProvider(modelStats, benchmarkParam, providersQuery.data),
    [providersQuery.data, modelStats, benchmarkParam]
  );
  const tagIndex = useMemo(
    () => buildTagIndex(benchmarkParam, providersQuery.data),
    [benchmarkParam, providersQuery.data]
  );
  const [selectedFacets, setSelectedFacets] = useState<FacetSelection>({});
  const modelsByProvider = useMemo(
    () => filterModelsByFacets(allModelsByProvider, tagIndex, selectedFacets),
    [allModelsByProvider, tagIndex, selectedFacets]
  );
  const hasActiveFacets = useMemo(
    () => Object.values(selectedFacets).some((v) => v.length > 0),
    [selectedFacets]
  );
  const toggleFacet = useCallback((category: string, value: string) => {
    setSelectedFacets((prev) => toggleFacetValue(prev, category, value));
  }, []);
  const clearFacets = useCallback(() => setSelectedFacets({}), []);

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

  // Event handlers
  const toggleModelSelection = useCallback(
    (model: string) => {
      const willBeSelected = !selectedModels.includes(model);
      const nextSelected = willBeSelected
        ? [...selectedModels, model]
        : selectedModels.filter((m) => m !== model);
      capturePostHogEvent(POSTHOG_EVENTS.dashboardModelSelectionChanged, {
        surface: `${page}_dashboard`,
        mode: page,
        action: willBeSelected ? "add" : "remove",
        model_id: model,
        selected_model_ids: nextSelected,
        selected_model_count: nextSelected.length,
        is_comparison: nextSelected.length >= 2
      });
      setSelectedModels(nextSelected);
    },
    [selectedModels, page]
  );

  // Select all of a provider's models unless they're all already selected.
  const toggleProviderSelection = useCallback(
    (provider: string) => {
      const providerModels = modelsByProvider[provider] ?? [];
      if (providerModels.length === 0) return;
      const willBeSelected = !providerModels.every((m) =>
        selectedModels.includes(m)
      );
      const nextSelected = willBeSelected
        ? [...new Set([...selectedModels, ...providerModels])]
        : selectedModels.filter((m) => !providerModels.includes(m));
      capturePostHogEvent(POSTHOG_EVENTS.dashboardModelSelectionChanged, {
        surface: `${page}_dashboard`,
        mode: page,
        action: willBeSelected ? "add" : "remove",
        provider,
        selected_model_ids: nextSelected,
        selected_model_count: nextSelected.length,
        is_comparison: nextSelected.length >= 2
      });
      setSelectedModels(nextSelected);
    },
    [selectedModels, modelsByProvider, page]
  );

  // Heatmap scaling for mobile. Skipped while loading (only the skeleton is
  // in the DOM); re-runs when loading completes so the first paint of the
  // heatmap gets scaled without waiting for a resize event.
  useEffect(() => {
    if (loading) return;

    const scaleHeatmapForMobile = () => {
      const mobile = window.innerWidth < 768;

      if (mobile) {
        setTimeout(() => {
          const heatmapContainer = document.querySelector(
            ".heatmap-container"
          ) as HTMLElement;
          const heatmapSvg = document.querySelector(
            ".heatmap-container svg"
          ) as SVGElement;

          if (heatmapContainer && heatmapSvg) {
            // Reset any prior scaling so measurements reflect natural layout
            heatmapContainer.style.transform = "";
            heatmapContainer.style.width = "";
            heatmapContainer.style.height = "";
            const availableWidth =
              heatmapContainer.getBoundingClientRect().width;
            const heatmapWidth = heatmapSvg.getBoundingClientRect().width;
            const heatmapHeight = heatmapSvg.getBoundingClientRect().height;
            const scaleFactor = availableWidth / heatmapWidth;

            if (scaleFactor < 1) {
              heatmapContainer.style.transform = `scale(${scaleFactor})`;
              heatmapContainer.style.transformOrigin = "top left";
              heatmapContainer.style.width = `${100 / scaleFactor}%`;
              // A scaled element keeps its natural layout height, leaving
              // white space below. Collapse the layout box to the scaled
              // height so standard card padding applies beneath the map.
              heatmapContainer.style.height = `${heatmapHeight * scaleFactor}px`;
            }
          }
        }, 100);
      } else {
        const heatmapContainer = document.querySelector(
          ".heatmap-container"
        ) as HTMLElement;
        if (heatmapContainer) {
          heatmapContainer.style.transform = "";
          heatmapContainer.style.transformOrigin = "";
          heatmapContainer.style.width = "";
          heatmapContainer.style.height = "";
        }
      }
    };

    scaleHeatmapForMobile();
    window.addEventListener("resize", scaleHeatmapForMobile);
    return () => window.removeEventListener("resize", scaleHeatmapForMobile);
  }, [page, loading]);

  // Auto-select models that have aggregate stats once they load. Catalogue-only
  // models (in modelsByProvider but absent from modelStats) stay unselected —
  // they have nothing to plot.
  useEffect(() => {
    if (modelStats.length === 0 || selectedModels.length > 0) return;
    const visible = new Set(Object.values(modelsByProvider).flat());
    const withStats = [
      ...new Set(modelStats.map((s) => toModelKey(s.provider, s.model))),
    ].filter((key) => visible.has(key));
    if (withStats.length > 0) setSelectedModels(withStats);
  }, [modelStats, modelsByProvider, selectedModels.length]);

  // Keep the selection a subset of the visible models. Once provider metadata
  // loads, a model that results alone had surfaced may be filtered out of
  // modelsByProvider; drop it so it stops being plotted while absent from the
  // sidebar. Removal-only, so it never fights a manual selection.
  useEffect(() => {
    setSelectedModels((prev) => {
      const next = pruneSelection(prev, modelsByProvider);
      return next.length === prev.length ? prev : next;
    });
  }, [modelsByProvider]);

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
    () => buildFacetGroups(allModelsByProvider, tagIndex, selectedFacets, normalizeProviderName),
    [allModelsByProvider, tagIndex, selectedFacets, normalizeProviderName]
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
  const primaryKeyMetric = (() => {
    const multiple = deferredSelectedModels.length > 1;
    return {
      label: `Lowest Median ${activeMetric}`,
      displayValue: `${fastestPrimary.fastestMs.toFixed(0)} ms`,
      subtitle:
        multiple && fastestPrimary.fastestModel
          ? {
              name: normalizeModelName(fastestPrimary.fastestModel),
              detail: fastestPrimary.fastestProvider
                ? normalizeProviderName(fastestPrimary.fastestProvider)
                : undefined,
            }
          : undefined,
    };
  })();

  const secondaryKeyMetric = {
    label: `${deferredSelectedModels.length > 1 ? "Lowest" : "Average"} Word Error Rate`,
    displayValue: `${avgSecondary.toFixed(1)}%`,
    subtitle:
      deferredSelectedModels.length > 1 && lowestWERModel
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
    toggleModelSelection,
    toggleProviderSelection,
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
