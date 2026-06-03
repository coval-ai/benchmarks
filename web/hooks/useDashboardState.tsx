// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import type { BenchmarkData } from "@/types/benchmark.types";
import { TWENTY_FOUR_HOURS_MS } from "@/lib/config/constants";
import { useChartData } from "@/hooks/useChartData";
import { useMobileDetection } from "@/hooks/useMobileDetection";
import { useBarInteraction } from "@/hooks/useBarInteraction";
import { useTimelineWindow } from "@/hooks/useTimelineWindow";
import { normalizeModelName, normalizeSTTProviderName, normalizeTTSProviderName, parseModelKey } from "@/lib/utils/formatters";
import { buildModelsByProviderFromResults, pruneSelection } from "@/lib/utils/modelsFromResults";
import { metricDescriptions } from "@/lib/config/metrics";
import { useResultsQuery, useProvidersQuery } from "@/lib/api/queries";
import { computeModelStats, type Result } from "@/lib/aggregates";

function adaptResult(row: Result): BenchmarkData {
  return {
    provider: row.provider,
    model: row.model,
    voice: row.voice ?? "",
    benchmark: row.benchmark,
    metric_type: row.metric_type,
    metric_value: row.metric_value,
    metric_units: row.metric_units,
    audio_filename: row.audio_filename ?? "",
    timestamp: row.created_at,
    scheduled_at: (row as { scheduled_at?: string }).scheduled_at ?? row.created_at,
    status: row.status,
    transcript: "",
  };
}

export function useDashboardState(page: "tts" | "stt") {
  // State declarations
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [expandedProviders, setExpandedProviders] = useState<{[key: string]: boolean;}>({});
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false);
  const [chartRefreshKey] = useState(0);

  const benchmarkParam = page === "tts" ? "TTS" : "STT";

  const resultsQuery = useResultsQuery({
    benchmark: benchmarkParam,
    window: "24h",
    include_failed: false,
    limit: 100000,
  });
  const providersQuery = useProvidersQuery();

  const resultRows = useMemo<readonly Result[]>(
    () => resultsQuery.data?.results ?? [],
    [resultsQuery.data]
  );

  const rawData = useMemo<BenchmarkData[]>(
    () => resultRows.map(adaptResult),
    [resultRows]
  );

  const modelStats = useMemo(
    () => computeModelStats(resultRows),
    [resultRows]
  );

  const { ttsModelsByProvider, sttModelsByProvider } = useMemo(
    () => ({
      ttsModelsByProvider: buildModelsByProviderFromResults(
        resultRows,
        "TTS",
        providersQuery.data
      ),
      sttModelsByProvider: buildModelsByProviderFromResults(
        resultRows,
        "STT",
        providersQuery.data
      ),
    }),
    [providersQuery.data, resultRows]
  );

  const loading = resultsQuery.isLoading || providersQuery.isLoading;

  const isMobile = useMobileDetection();
  const { clickedWERBars, handleWERBarClick } =
    useBarInteraction();

  const modelsByProvider = page === "tts" ? ttsModelsByProvider : sttModelsByProvider;

  // Anchor the timeline window to the latest plotted bucket, not Date.now().
  // Charts plot scheduled_at so every result from one benchmark run shares a
  // tick; fall back to created_at during API rollout or for legacy rows.
  const latestTimestamp = useMemo<number>(() => {
    if (rawData.length === 0) return Date.now();
    let max = 0;
    for (const item of rawData) {
      const t = new Date(item.scheduled_at || item.timestamp).getTime();
      if (Number.isNaN(t)) continue;
      if (t > max) max = t;
    }
    return max > 0 ? max : Date.now();
  }, [rawData]);

  const [timelineWindowEndTemp, setTimelineWindowEndTemp] =
    useState<number>(latestTimestamp);

  // Anchor once on first successful fetch, then leave the user's window alone.
  // Re-anchoring on every refetch yanks the chart out from under the user mid-drag.
  const hasAnchoredRef = useRef(false);
  useEffect(() => {
    if (hasAnchoredRef.current) return;
    if (rawData.length === 0) return;
    setTimelineWindowEndTemp(latestTimestamp);
    hasAnchoredRef.current = true;
  }, [rawData.length, latestTimestamp]);

  // useChartData hook - pass page as activeTab internally
  const chartData = useChartData({
    activeTab: page,
    rawData,
    modelStats,
    selectedTTSModels: page === "tts" ? selectedModels : [],
    selectedSTTModels: page === "stt" ? selectedModels : [],
    timelineWindowEnd: timelineWindowEndTemp,
    modelsByProvider: page === "tts" ? ttsModelsByProvider : sttModelsByProvider,
  });

  // useTimelineWindow hook
  const {
    timelineWindowEnd,
    isDragging,
    handleMouseDown,
  } = useTimelineWindow({
    initialEnd: timelineWindowEndTemp,
    getTimelineData: chartData.getTimelineData,
    visibleWindowMs: TWENTY_FOUR_HOURS_MS,
  });

  // Sync the timeline window end
  useEffect(() => {
    setTimelineWindowEndTemp(timelineWindowEnd);
  }, [timelineWindowEnd]);

  // Event handlers
  const toggleProvider = useCallback(
    (provider: string) => {
      setExpandedProviders((prev) => ({
        ...prev,
        [provider]: !prev[provider],
      }));
    },
    []
  );

  const toggleModelSelection = useCallback(
    (model: string) => {
      setSelectedModels((prev) =>
        prev.includes(model)
          ? prev.filter((m) => m !== model)
          : [...prev, model]
      );
    },
    []
  );

  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => !prev);
    // Wait for CSS transition to complete (300ms), then trigger resize
    setTimeout(() => {
      window.dispatchEvent(new Event("resize"));
    }, 300);
  }, []);

  // Heatmap scaling for mobile
  useEffect(() => {
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
            const screenWidth = window.innerWidth;
            const padding = 32;
            const availableWidth = screenWidth - padding;
            const heatmapWidth = heatmapSvg.getBoundingClientRect().width;
            const scaleFactor = availableWidth / heatmapWidth;

            if (scaleFactor < 1) {
              heatmapContainer.style.transform = `scale(${scaleFactor})`;
              heatmapContainer.style.transformOrigin = "top left";
              heatmapContainer.style.width = `${100 / scaleFactor}%`;
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
        }
      }
    };

    scaleHeatmapForMobile();
    window.addEventListener("resize", scaleHeatmapForMobile);
    return () => window.removeEventListener("resize", scaleHeatmapForMobile);
  }, [page]);

  // Auto-select all models when data loads (disabled models already filtered out by useMemo above).
  useEffect(() => {
    if (
      Object.keys(modelsByProvider).length > 0 &&
      selectedModels.length === 0
    ) {
      const allModels: string[] = Object.values(modelsByProvider).flat();
      setSelectedModels(allModels);

      const expanded: Record<string, boolean> = {};
      Object.keys(modelsByProvider).forEach((provider) => {
        expanded[provider] = true;
      });
      setExpandedProviders(expanded);
    }
  }, [modelsByProvider, selectedModels.length]);

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
  const currentData = chartData.getCurrentData();
  const primaryMetric = page === "tts" ? "TTFA" : "TTFT";
  const secondaryMetric = "WER";

  const primaryData = currentData.filter(
    (item) => item.metric_type === primaryMetric
  );
  const secondaryData = currentData.filter(
    (item) => item.metric_type === secondaryMetric
  );

  let avgPrimary = 0;
  let avgSecondary = 0;
  let fastestLatencyModel = "";
  let lowestWERModel = "";
  let fastestLatencyProvider = "";
  let lowestWERProvider = "";

  if (selectedModels.length === 0) {
    avgPrimary = 0;
    avgSecondary = 0;
  } else if (page === "stt") {
    const rankingData = chartData.getSTTRankingData();

    if (rankingData.length > 0) {
      const fastestModel = rankingData[0];
      if (fastestModel) {
        fastestLatencyModel = fastestModel.model;
        fastestLatencyProvider = normalizeSTTProviderName(fastestModel.provider);
        avgPrimary = fastestModel.latencyMs;
      }
    }

    const sttSecondaryData = currentData.filter(
      (item) => item.metric_type === secondaryMetric
    );
    if (selectedModels.length === 1) {
      if (sttSecondaryData.length > 0) {
        avgSecondary =
          sttSecondaryData.reduce(
            (sum, item) => sum + (item.metric_value ?? 0),
            0
          ) / sttSecondaryData.length;
      }
    } else {
      const modelMetrics: {
        [key: string]: { avgSecondary: number; provider: string };
      } = {};

      selectedModels.forEach((model) => {
        const { model: modelSlug, provider: modelProvider } = parseModelKey(model);
        const modelSecondaryData = sttSecondaryData.filter(
          (item) => item.model === modelSlug && item.provider === modelProvider
        );

        let modelAvgSecondary = Infinity;
        if (modelSecondaryData.length > 0) {
          modelAvgSecondary =
            modelSecondaryData.reduce(
              (sum, item) => sum + (item.metric_value ?? 0),
              0
            ) / modelSecondaryData.length;
        }

        const provider = modelSecondaryData[0]?.provider ?? "Unknown";
        modelMetrics[model] = {
          avgSecondary: modelAvgSecondary,
          provider: provider,
        };
      });

      let lowestSecondary = Infinity;
      Object.entries(modelMetrics).forEach(([model, metrics]) => {
        if (
          metrics.avgSecondary < lowestSecondary &&
          metrics.avgSecondary !== Infinity
        ) {
          lowestSecondary = metrics.avgSecondary;
          lowestWERModel = model;
          lowestWERProvider = metrics.provider;
        }
      });

      avgSecondary = lowestSecondary !== Infinity ? lowestSecondary : 0;
    }
  } else {
    // TTS logic
    if (selectedModels.length === 1) {
      if (primaryData.length > 0) {
        const primaryValues = primaryData.map((item) => item.metric_value ?? 0);
        const sortedPrimary = primaryValues.sort((a, b) => a - b);
        const medianIndex = Math.floor(sortedPrimary.length / 2);
        avgPrimary =
          sortedPrimary.length % 2 === 0
            ? ((sortedPrimary[medianIndex - 1] ?? 0) + (sortedPrimary[medianIndex] ?? 0)) / 2
            : (sortedPrimary[medianIndex] ?? 0);
      }

      if (secondaryData.length > 0) {
        avgSecondary =
          secondaryData.reduce(
            (sum, item) => sum + (item.metric_value ?? 0),
            0
          ) / secondaryData.length;
      }
    } else {
      const modelMetrics: {
        [key: string]: {
          medianPrimary: number;
          avgSecondary: number;
          provider: string;
        };
      } = {};

      selectedModels.forEach((model) => {
        const { model: modelSlug, provider: modelProvider } = parseModelKey(model);
        const modelPrimaryData = primaryData.filter(
          (item) => item.model === modelSlug && item.provider === modelProvider
        );
        const modelSecondaryData = secondaryData.filter(
          (item) => item.model === modelSlug && item.provider === modelProvider
        );

        let modelMedianPrimary = Infinity;
        let modelAvgSecondary = Infinity;

        if (modelPrimaryData.length > 0) {
          const primaryValues = modelPrimaryData.map(
            (item) => item.metric_value ?? 0
          );
          const sortedPrimary = primaryValues.sort((a, b) => a - b);
          const medianIndex = Math.floor(sortedPrimary.length / 2);
          modelMedianPrimary =
            sortedPrimary.length % 2 === 0
              ? ((sortedPrimary[medianIndex - 1] ?? 0) + (sortedPrimary[medianIndex] ?? 0)) / 2
              : (sortedPrimary[medianIndex] ?? 0);
        }

        if (modelSecondaryData.length > 0) {
          modelAvgSecondary =
            modelSecondaryData.reduce(
              (sum, item) => sum + (item.metric_value ?? 0),
              0
            ) / modelSecondaryData.length;
        }

        const provider =
          (modelPrimaryData[0] ?? modelSecondaryData[0])?.provider ?? "Unknown";

        modelMetrics[model] = {
          medianPrimary: modelMedianPrimary,
          avgSecondary: modelAvgSecondary,
          provider: provider,
        };
      });

      let fastestPrimary = Infinity;
      let lowestSecondary = Infinity;

      Object.entries(modelMetrics).forEach(([model, metrics]) => {
        if (
          metrics.medianPrimary < fastestPrimary &&
          metrics.medianPrimary !== Infinity
        ) {
          fastestPrimary = metrics.medianPrimary;
          fastestLatencyModel = model;
          fastestLatencyProvider = normalizeTTSProviderName(metrics.provider);
        }
        if (
          metrics.avgSecondary < lowestSecondary &&
          metrics.avgSecondary !== Infinity
        ) {
          lowestSecondary = metrics.avgSecondary;
          lowestWERModel = model;
          lowestWERProvider = normalizeTTSProviderName(metrics.provider);
        }
      });

      avgPrimary = fastestPrimary !== Infinity ? fastestPrimary : 0;
      avgSecondary = lowestSecondary !== Infinity ? lowestSecondary : 0;
    }
  }

  // Get computed data
  const scatterDataResult = chartData.getScatterData();
  const scatterData = scatterDataResult.points;
  const scatterP99X = scatterDataResult.p99X;
  const scatterOutlierCount = scatterDataResult.outlierCount;
  const heatmapData = chartData.getModelHeatmapData();
  const werBarData = chartData.getWERBarData();

  const werBarDataWithColors = useMemo(() => {
    return werBarData.map((item) => ({
      ...item,
      fill: clickedWERBars.has(item.model)
        ? "#FF851B"
        : "rgba(255, 255, 255, 0.12)",
    }));
  }, [werBarData, clickedWERBars]);

  // Derived display values
  const latencyLabel = page === "tts" ? "TTFA" : "TTFT";
  const pageTitle = page === "tts" ? "Text to Speech Model Comparisons" : "Speech to Text Model Comparisons";
  const pageSubtitle = page === "tts"
    ? "Compare performance metrics between different Text-to-Speech models for voice agent applications."
    : "Compare performance metrics between different Speech-to-Text models for voice agent applications.";
  const sidebarTitle = page === "tts"
    ? "Select TTS Models to Compare"
    : "Select STT Models to Compare";
  const mobileSheetTitle = page === "tts"
    ? "Text-to-Speech Models"
    : "Speech-to-Text Models";
  const normalizeProviderName = page === "stt"
    ? normalizeSTTProviderName
    : normalizeTTSProviderName;

  const violinDescription = {
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
        short: "Word Error Rate (%) \u2022 Click bar to compare models",
        detailed:
          "In voice AI applications, transcription accuracy directly impacts the performance of downstream tasks. Even small transcription errors can lead to misinterpretations, frustrating experiences, or incorrect system responses. We evaluate against test audio that includes diverse speakers, accents, and real-world audio conditions.",
      };

  const heatmapDisplayData = page === "tts"
    ? chartData.getTTSHeatmapData()
    : heatmapData;

  // Pre-computed key metrics for display
  const primaryKeyMetric = (() => {
    const latencyFullLabel =
      page === "tts" ? "Time to First Audio" : "Time to First Token";
    const label = `${selectedModels.length > 1 ? "Fastest" : "Median"} ${latencyFullLabel}`;
    return {
      label,
      displayValue: `${avgPrimary.toFixed(0)} ms`,
      subtitle:
        selectedModels.length > 1 && fastestLatencyModel
          ? {
              name: normalizeModelName(fastestLatencyModel),
              detail: fastestLatencyProvider
                ? normalizeProviderName(fastestLatencyProvider)
                : undefined,
            }
          : undefined,
    };
  })();

  const secondaryKeyMetric = {
    label: `${selectedModels.length > 1 ? "Lowest" : "Average"} Word Error Rate`,
    displayValue: `${avgSecondary.toFixed(1)}%`,
    subtitle:
      selectedModels.length > 1 && lowestWERModel
        ? {
            name: normalizeModelName(lowestWERModel),
            detail: lowestWERProvider
              ? normalizeProviderName(lowestWERProvider)
              : undefined,
          }
        : undefined,
  };

  const modelsComparedMetric = {
    label: "Models Compared",
    displayValue: `${selectedModels.length}`,
  };

  const providersMetric = {
    label: "Providers",
    displayValue: `${
      new Set(selectedModels.map((model) => parseModelKey(model).provider)).size
    }`,
  };

  return {
    // Page identity
    latencyLabel,

    // Display strings
    pageTitle,
    pageSubtitle,
    sidebarTitle,
    mobileSheetTitle,
    normalizeProviderName,

    // Section descriptions
    violinDescription,
    werDescription,

    // Key metrics
    primaryKeyMetric,
    secondaryKeyMetric,
    modelsComparedMetric,
    providersMetric,

    // Data loading
    loading,
    rawData,

    // Model state
    selectedModels,
    modelsByProvider,
    expandedProviders,

    // UI state
    sidebarCollapsed,
    mobileSheetOpen,
    setMobileSheetOpen,
    isMobile,
    chartRefreshKey,

    // Actions
    toggleProvider,
    toggleModelSelection,
    toggleSidebar,

    // Timeline
    isDragging,
    handleMouseDown,

    // Chart data functions
    formatChartLabel: chartData.formatChartLabel,
    getProviderForModel: chartData.getProviderForModel,
    getWindowedTimelineData: chartData.getWindowedTimelineData,
    getCurrentTimeWindow: chartData.getCurrentTimeWindow,
    getTimelineTicks: chartData.getTimelineTicks,
    getWindowedGapData: chartData.getWindowedGapData,
    getModelsWithTimelineData: chartData.getModelsWithTimelineData,
    getModelsWithGapData: chartData.getModelsWithGapData,
    getViolinData: chartData.getViolinData,
    getSTTRankingData: chartData.getSTTRankingData,

    // Computed chart data
    scatterData,
    scatterP99X,
    scatterOutlierCount,
    heatmapDisplayData,
    werBarDataWithColors,

    // Bar interaction
    handleWERBarClick,
  };
}
