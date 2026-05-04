"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import type {
  BarDataPoint,
  BenchmarkData,
  ModelsByProvider,
} from "@/types/benchmark.types";
import { TWENTY_FOUR_HOURS_MS } from "@/lib/config/constants";
import { useChartData } from "@/hooks/useChartData";
import { useMobileDetection } from "@/hooks/useMobileDetection";
import { useBarInteraction } from "@/hooks/useBarInteraction";
import { useTimelineWindow } from "@/hooks/useTimelineWindow";
import { normalizeModelName, normalizeSTTProviderName } from "@/lib/utils/formatters";
import { metricDescriptions } from "@/lib/config/metrics";
import { useResultsQuery, useProvidersQuery } from "@/lib/api/queries";
import type { ModelInfo } from "@/lib/api/client";
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
    status: row.status,
    transcript: "",
  };
}

export function useDashboardState(page: "tts" | "stt") {
  // State declarations
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [expandedProviders, setExpandedProviders] = useState<Record<string, boolean>>({});
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false);
  const [chartRefreshKey] = useState(0);

  const benchmarkParam = page === "tts" ? "TTS" : "STT";

  const resultsQuery = useResultsQuery({
    benchmark: benchmarkParam,
    window: "24h",
    include_failed: false,
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

  const { ttsModelsByProvider, sttModelsByProvider } = useMemo(() => {
    const tts: ModelsByProvider = {};
    const stt: ModelsByProvider = {};
    const providers = providersQuery.data;
    if (providers) {
      for (const p of providers.tts) {
        tts[p.provider] = p.models
          .filter((m: ModelInfo) => !m.disabled)
          .map((m: ModelInfo) => m.model);
      }
      for (const p of providers.stt) {
        stt[p.provider] = p.models
          .filter((m: ModelInfo) => !m.disabled)
          .map((m: ModelInfo) => m.model);
      }
    }
    return { ttsModelsByProvider: tts, sttModelsByProvider: stt };
  }, [providersQuery.data]);

  const loading = resultsQuery.isLoading || providersQuery.isLoading;

  const isMobile = useMobileDetection();
  const { clickedWERBars, handleWERBarClick } =
    useBarInteraction();

  const modelsByProvider = page === "tts" ? ttsModelsByProvider : sttModelsByProvider;

  // Anchor the timeline window to the latest data timestamp, not Date.now().
  // The data we plot is from a benchmark run that may have finished hours
  // before the user opens this page; using Date.now() pushes all rendered
  // points to the far-left edge of a window that ends "now".
  const latestTimestamp = useMemo<number>(() => {
    if (rawData.length === 0) return Date.now();
    let max = 0;
    for (const item of rawData) {
      const t = new Date(item.timestamp).getTime();
      if (t > max) max = t;
    }
    return max > 0 ? max : Date.now();
  }, [rawData]);

  const [timelineWindowEndTemp, setTimelineWindowEndTemp] =
    useState<number>(latestTimestamp);

  // Re-anchor whenever the data's latest timestamp changes (i.e. on first
  // load after the fetch resolves, and on subsequent refetches).
  useEffect(() => {
    setTimelineWindowEndTemp(latestTimestamp);
  }, [latestTimestamp]);

  // useChartData hook - pass page as activeTab internally
  const chartData = useChartData({
    activeTab: page,
    rawData,
    modelStats,
    selectedTTSModels: page === "tts" ? selectedModels : [],
    selectedSTTModels: page === "stt" ? selectedModels : [],
    timelineWindowEnd: timelineWindowEndTemp,
  });

  // useTimelineWindow hook
  const {
    timelineWindowEnd,
    isDragging,
    chartRef,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
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
      setExpandedProviders((prev: Record<string, boolean>) => ({
        ...prev,
        [provider]: !prev[provider],
      }));
    },
    []
  );

  const toggleModelSelection = useCallback(
    (model: string) => {
      setSelectedModels((prev: string[]) =>
        prev.includes(model)
          ? prev.filter((m: string) => m !== model)
          : [...prev, model]
      );
    },
    []
  );

  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev: boolean) => !prev);
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

  // Mouse drag event listeners
  useEffect(() => {
    if (isDragging) {
      document.addEventListener(
        "mousemove",
        handleMouseMove as (e: Event) => void
      );
      document.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      document.removeEventListener(
        "mousemove",
        handleMouseMove as (e: Event) => void
      );
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Auto-select all models when data loads (disabled models already filtered out by useMemo above).
  useEffect(() => {
    if (
      Object.keys(modelsByProvider).length > 0 &&
      selectedModels.length === 0
    ) {
      const allModels = Object.values(modelsByProvider).flat();
      setSelectedModels(allModels);

      const expanded: Record<string, boolean> = {};
      Object.keys(modelsByProvider).forEach((provider) => {
        expanded[provider] = true;
      });
      setExpandedProviders(expanded);
    }
  }, [modelsByProvider, selectedModels.length]);

  // Calculate metrics
  const currentData = chartData.getCurrentData();
  const primaryMetric = page === "tts" ? "TTFA" : "TTFT";
  const secondaryMetric = "WER";

  const primaryData = currentData.filter(
    (item: BenchmarkData) => item.metric_type === primaryMetric
  );
  const secondaryData = currentData.filter(
    (item: BenchmarkData) => item.metric_type === secondaryMetric
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
        fastestLatencyProvider = fastestModel.provider;
      }
    }

    const sttSecondaryData = currentData.filter(
      (item: BenchmarkData) => item.metric_type === secondaryMetric
    );
    if (selectedModels.length === 1) {
      if (sttSecondaryData.length > 0) {
        avgSecondary =
          sttSecondaryData.reduce(
            (sum: number, item: BenchmarkData) => sum + (item.metric_value ?? 0),
            0
          ) / sttSecondaryData.length;
      }
    } else {
      const modelMetrics: {
        [key: string]: { avgSecondary: number; provider: string };
      } = {};

      selectedModels.forEach((model: string) => {
        const modelSecondaryData = sttSecondaryData.filter(
          (item: BenchmarkData) => item.model === model
        );

        let modelAvgSecondary = Infinity;
        if (modelSecondaryData.length > 0) {
          modelAvgSecondary =
            modelSecondaryData.reduce(
              (sum: number, item: BenchmarkData) =>
                sum + (item.metric_value ?? 0),
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
        const primaryValues = primaryData.map(
          (item: BenchmarkData) => item.metric_value ?? 0
        );
        const sortedPrimary = primaryValues.sort((a: number, b: number) => a - b);
        const medianIndex = Math.floor(sortedPrimary.length / 2);
        avgPrimary =
          sortedPrimary.length % 2 === 0
            ? ((sortedPrimary[medianIndex - 1] ?? 0) + (sortedPrimary[medianIndex] ?? 0)) / 2
            : (sortedPrimary[medianIndex] ?? 0);
      }

      if (secondaryData.length > 0) {
        avgSecondary =
          secondaryData.reduce(
            (sum: number, item: BenchmarkData) =>
              sum + (item.metric_value ?? 0),
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

      selectedModels.forEach((model: string) => {
        const modelPrimaryData = primaryData.filter(
          (item: BenchmarkData) => item.model === model
        );
        const modelSecondaryData = secondaryData.filter(
          (item: BenchmarkData) => item.model === model
        );

        let modelMedianPrimary = Infinity;
        let modelAvgSecondary = Infinity;

        if (modelPrimaryData.length > 0) {
          const primaryValues = modelPrimaryData.map(
            (item: BenchmarkData) => item.metric_value ?? 0
          );
          const sortedPrimary = primaryValues.sort((a: number, b: number) => a - b);
          const medianIndex = Math.floor(sortedPrimary.length / 2);
          modelMedianPrimary =
            sortedPrimary.length % 2 === 0
              ? ((sortedPrimary[medianIndex - 1] ?? 0) + (sortedPrimary[medianIndex] ?? 0)) / 2
              : (sortedPrimary[medianIndex] ?? 0);
        }

        if (modelSecondaryData.length > 0) {
          modelAvgSecondary =
            modelSecondaryData.reduce(
              (sum: number, item: BenchmarkData) =>
                sum + (item.metric_value ?? 0),
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
          fastestLatencyProvider = metrics.provider;
        }
        if (
          metrics.avgSecondary < lowestSecondary &&
          metrics.avgSecondary !== Infinity
        ) {
          lowestSecondary = metrics.avgSecondary;
          lowestWERModel = model;
          lowestWERProvider = metrics.provider;
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
    return werBarData.map((item: BarDataPoint) => ({
      ...item,
      fill: clickedWERBars.has(item.model)
        ? "#FF851B"
        : "rgba(255, 255, 255, 0.12)",
    }));
  }, [werBarData, clickedWERBars]);

  // Derived display values
  const latencyLabel = page === "tts" ? "TTFA" : "TTFT";
  const pageTitle = page === "tts" ? "TTS Model Comparison" : "STT Model Comparison";
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
    : (name: string) => name;

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
          "In voice AI applications, transcription accuracy directly impacts the performance of downstream tasks. Even small transcription errors can lead to misinterpretations, frustrating experiences, or incorrect system responses. We evaluate using the open-source Mozilla's Common Voice test dataset in English, which includes diverse speakers, accents, and real-world audio conditions.",
      };

  const heatmapDisplayData = page === "tts"
    ? chartData.getTTSHeatmapData()
    : heatmapData;

  // Pre-computed key metrics for display
  const primaryKeyMetric = (() => {
    const label = `${selectedModels.length > 1 ? "Fastest" : "Median"} ${latencyLabel}`;
    if (page === "stt" && selectedModels.length > 0 && fastestLatencyModel) {
      return {
        label,
        displayValue: normalizeModelName(fastestLatencyModel),
        subtitle: fastestLatencyProvider
          ? { detail: fastestLatencyProvider }
          : undefined,
      };
    }
    return {
      label,
      displayValue: avgPrimary.toFixed(0),
      subtitle:
        selectedModels.length > 1 && fastestLatencyModel
          ? {
              name: normalizeModelName(fastestLatencyModel),
              detail: fastestLatencyProvider,
            }
          : undefined,
    };
  })();

  const secondaryKeyMetric = {
    label: `${selectedModels.length > 1 ? "Lowest" : "Average"} WER`,
    displayValue: `${avgSecondary.toFixed(1)}%`,
    subtitle:
      selectedModels.length > 1 && lowestWERModel
        ? {
            name: normalizeModelName(lowestWERModel),
            detail: normalizeSTTProviderName(lowestWERProvider),
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
    mobileSheetTitle,
    normalizeProviderName,

    // Section descriptions
    violinDescription,
    werDescription,

    // Key metrics
    primaryKeyMetric,
    secondaryKeyMetric,

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
    chartRef,
    isDragging,
    handleMouseDown,

    // Chart data functions
    formatChartLabel: chartData.formatChartLabel,
    getProviderForModel: chartData.getProviderForModel,
    getWindowedTimelineData: chartData.getWindowedTimelineData,
    getCurrentTimeWindow: chartData.getCurrentTimeWindow,
    getWindowedGapData: chartData.getWindowedGapData,
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
