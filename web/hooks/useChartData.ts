// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useCallback } from "react";
import type {
  BenchmarkData,
  ModelStats,
  TimelineDataPoint,
  ScatterDataPoint,
  ScatterDataResult,
  ViolinPlotData,
  ViolinDataPoint,
  ModelHeatmapData,
  BarDataPoint
} from "@/types/benchmark.types";
import {
  calculateQuartiles,
  calculateStats,
  calculateKernelDensity
} from "@/lib/utils/statistics";
import { normalizeModelName, normalizeSTTProviderName } from "@/lib/utils/formatters";
import { TWENTY_FOUR_HOURS_MS } from "@/lib/config/constants";
import { to15MinuteBucket } from "@/lib/utils/time";

// Parent-run statuses that should contribute to chart aggregates. Mirrors the
// filter used in `lib/aggregates.ts` — keep in sync. The result-row status is
// always 'success' because the API filters server-side; this gate is on the
// PARENT-RUN status (uppercase enum) which is denormalized into ResultOut.
const INCLUDED_STATUSES = new Set<string>(["SUCCEEDED", "PARTIAL"]);

interface UseChartDataParams {
  activeTab: "tts" | "stt";
  rawData: BenchmarkData[];
  modelStats: ModelStats[];
  selectedTTSModels: string[];
  selectedSTTModels: string[];
  timelineWindowEnd: number;
}

export function useChartData({
  activeTab,
  rawData,
  modelStats,
  selectedTTSModels,
  selectedSTTModels,
  timelineWindowEnd
}: UseChartDataParams) {
  // Helper: find a stat row for a given model and metric
  const getStat = useCallback(
    (model: string, metricType: string): ModelStats | undefined => {
      return modelStats.find(
        (s) => s.model === model && s.metric_type === metricType
      );
    },
    [modelStats]
  );

  // Helper function to format chart labels
  const formatChartLabel = useCallback(
    (model: string, provider: string): string => {
      return `${provider} ${normalizeModelName(model)}`;
    },
    []
  );

  // Helper function to get provider for a model
  const getProviderForModel = useCallback(
    (model: string): string => {
      // Try modelStats first (cheaper), then fall back to rawData
      const stat = modelStats.find((s) => s.model === model);
      const rawProvider = stat?.provider ?? rawData.find((d) => d.model === model)?.provider ?? "Unknown";

      if (activeTab === "stt") {
        return normalizeSTTProviderName(rawProvider);
      }

      return rawProvider;
    },
    [modelStats, rawData, activeTab]
  );

  // ─── Functions that still need raw data (distributions, individual points) ───

  const getCurrentData = useCallback(() => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
    const metricTypes = activeTab === "tts" ? ["TTFA", "WER"] : ["TTFT", "WER"];

    if (selectedModels.length === 0) {
      return [];
    }

    return rawData.filter(
      (item) =>
        selectedModels.includes(item.model) &&
        metricTypes.includes(item.metric_type) &&
        item.metric_value !== null &&
        item.metric_value !== undefined
    );
  }, [rawData, activeTab, selectedTTSModels, selectedSTTModels]);

  const getTimelineData = useCallback((): TimelineDataPoint[] => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
    const primaryMetric = activeTab === "tts" ? "TTFA" : "TTFT";

    if (selectedModels.length === 0) {
      return [];
    }

    const timelineData = rawData
      .filter(
        (item) =>
          selectedModels.includes(item.model) &&
          item.metric_type === primaryMetric &&
          INCLUDED_STATUSES.has(item.status)
      )
      .map((item) => ({
        timestamp: to15MinuteBucket(new Date(item.timestamp).getTime()),
        value:
          activeTab === "tts"
            ? item.metric_value ?? 0
            : (item.metric_value ?? 0) * 1000,
        model: item.model,
        benchmark: item.benchmark
      }))
      .sort((a, b) => a.timestamp - b.timestamp);

    const timestampGroups: { [key: number]: TimelineDataPoint } = {};
    const modelValueAccumulator: { [key: number]: { [key: string]: { total: number; count: number } } } = {};

    timelineData.forEach((item) => {
      if (!timestampGroups[item.timestamp]) {
        timestampGroups[item.timestamp] = {
          timestamp: item.timestamp,
          timestampLabel: new Date(item.timestamp).toISOString()
        };
        modelValueAccumulator[item.timestamp] = {};
      }
      const bucketAcc = modelValueAccumulator[item.timestamp];
      if (bucketAcc) {
        if (!bucketAcc[item.model]) {
          bucketAcc[item.model] = { total: 0, count: 0 };
        }
        const modelAcc = bucketAcc[item.model];
        if (modelAcc) {
          modelAcc.total += item.value;
          modelAcc.count += 1;
        }
      }
    });

    Object.entries(modelValueAccumulator).forEach(([bucketTimestamp, bucketModelStats]) => {
      const bucket = Number(bucketTimestamp);
      Object.entries(bucketModelStats).forEach(([model, stats]) => {
        const group = timestampGroups[bucket];
        if (group) {
          group[`${model}_value`] = stats.total / stats.count;
        }
      });
    });

    return Object.values(timestampGroups);
  }, [rawData, activeTab, selectedTTSModels, selectedSTTModels]);

  // Violin plots need raw values for KDE — cannot use pre-aggregated stats
  const getViolinData = useCallback((): ViolinPlotData => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
    const primaryMetric = activeTab === "tts" ? "TTFA" : "TTFT";

    if (selectedModels.length === 0) {
      return {
        data: [],
        globalMin: 0,
        globalMax: 0,
        metricType: primaryMetric
      };
    }

    const modelGroups: { [key: string]: BenchmarkData[] } = {};

    rawData.forEach((item) => {
      if (!selectedModels.includes(item.model)) return;
      if (item.metric_type !== primaryMetric) return;
      if (!INCLUDED_STATUSES.has(item.status)) return;
      if (item.metric_value === null || item.metric_value === undefined) return;

      if (!modelGroups[item.model]) {
        modelGroups[item.model] = [];
      }
      // noUncheckedIndexedAccess: guard after the initialization above
      const modelGroup = modelGroups[item.model];
      if (modelGroup) {
        modelGroup.push(item);
      }
    });

    const violinData: ViolinDataPoint[] = [];
    let globalMin = Infinity;
    let globalMax = -Infinity;

    Object.entries(modelGroups).forEach(([model, items]) => {
      const values = items.map((item) =>
        activeTab === "tts"
          ? item.metric_value ?? 0
          : (item.metric_value ?? 0) * 1000
      );

      if (values.length === 0) return;

      const quartiles = calculateQuartiles(values);
      const stats = calculateStats(values);
      const density = calculateKernelDensity(values);

      globalMin = Math.min(globalMin, ...values);
      globalMax = Math.max(globalMax, ...values);

      const provider = items[0]?.provider ?? "Unknown";

      violinData.push({
        model,
        provider,
        values,
        density,
        quartiles,
        stats
      });
    });

    if (violinData.length === 0) {
      globalMin = 0;
      globalMax = 0;
    }

    let maxUpperWhisker = 0;
    let totalOutliers = 0;

    violinData.forEach((modelData) => {
      const { q3, max } = modelData.quartiles;
      const iqr = q3 - modelData.quartiles.q1;
      const upperWhisker = Math.min(max, q3 + 1.5 * iqr);
      maxUpperWhisker = Math.max(maxUpperWhisker, upperWhisker);

      totalOutliers += modelData.quartiles.outliers.filter(
        (outlier) => outlier > upperWhisker
      ).length;
    });

    if (violinData.length === 0) {
      maxUpperWhisker = 0;
    }

    const sortedViolinData = violinData.sort(
      (a, b) => a.quartiles.median - b.quartiles.median
    );

    return {
      data: sortedViolinData,
      globalMin,
      globalMax: maxUpperWhisker,
      trueGlobalMax: globalMax,
      outlierCount: totalOutliers,
      cappedAt: maxUpperWhisker,
      metricType: primaryMetric
    };
  }, [rawData, activeTab, selectedTTSModels, selectedSTTModels]);

  // Scatter needs paired per-run values — cannot use pre-aggregated stats
  const getScatterData = useCallback((): ScatterDataResult => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
    const xMetric = activeTab === "tts" ? "TTFA" : "TTFT";
    const yMetric = "WER";

    if (selectedModels.length === 0) {
      return { points: [], p99X: 0, outlierCount: 0 };
    }

    const benchmarkGroups: {
      [key: string]: { [key: string]: BenchmarkData[] };
    } = {};

    rawData.forEach((item) => {
      if (!selectedModels.includes(item.model)) return;
      if (item.metric_type !== xMetric && item.metric_type !== yMetric) return;

      const key = `${item.benchmark}_${item.model}_${item.timestamp}`;
      if (!benchmarkGroups[key]) {
        benchmarkGroups[key] = {};
      }
      if (!benchmarkGroups[key][item.metric_type]) {
        benchmarkGroups[key][item.metric_type] = [];
      }
      // noUncheckedIndexedAccess: guard after initialization
      const metricGroup = benchmarkGroups[key][item.metric_type];
      if (metricGroup) {
        metricGroup.push(item);
      }
    });

    const scatterPoints: ScatterDataPoint[] = [];

    Object.values(benchmarkGroups).forEach((group) => {
      const xData = group[xMetric]?.[0];
      const yData = group[yMetric]?.[0];

      if (xData && yData) {
        scatterPoints.push({
          x:
            activeTab === "tts"
              ? xData.metric_value ?? 0
              : (xData.metric_value ?? 0) * 1000,
          y: yData.metric_value ?? 0,
          model: xData.model,
          benchmark: xData.benchmark,
          provider: xData.provider
        });
      }
    });

    const xValues = scatterPoints.map((point) => point.x);
    const sortedX = xValues.sort((a, b) => a - b);
    const p99Index = Math.floor(sortedX.length * 0.99);
    const p99X = sortedX.length > 0 ? (sortedX[p99Index] ?? 0) : 0;
    const outlierCount = xValues.filter((val) => val > p99X).length;

    return { points: scatterPoints, p99X, outlierCount };
  }, [rawData, activeTab, selectedTTSModels, selectedSTTModels]);

  // Gap data needs per-timestamp comparisons — cannot use pre-aggregated stats
  const getGapData = useCallback((): TimelineDataPoint[] => {
    if (activeTab !== "stt" || selectedSTTModels.length === 0) {
      return [];
    }

    const primaryMetric = "TTFT";

    const ttftData = rawData
      .filter(
        (item) =>
          selectedSTTModels.includes(item.model) &&
          item.metric_type === primaryMetric &&
          INCLUDED_STATUSES.has(item.status) &&
          item.metric_value !== null &&
          item.metric_value !== undefined
      )
      .map((item) => ({
        timestamp: to15MinuteBucket(new Date(item.timestamp).getTime()),
        value: (item.metric_value as number) * 1000,
        model: item.model,
        benchmark: item.benchmark
      }))
      .sort((a, b) => a.timestamp - b.timestamp);

    const timestampGroups: {
      [key: number]: {
        timestamp: number;
        timestampLabel: string;
        models: { [key: string]: { total: number; count: number } };
        fastest: number;
      };
    } = {};

    ttftData.forEach((item) => {
      if (!timestampGroups[item.timestamp]) {
        timestampGroups[item.timestamp] = {
          timestamp: item.timestamp,
          timestampLabel: new Date(item.timestamp).toISOString(),
          models: {},
          fastest: Infinity
        };
      }

      const group = timestampGroups[item.timestamp];
      if (group) {
        if (!group.models[item.model]) {
          group.models[item.model] = { total: 0, count: 0 };
        }
        const modelStats = group.models[item.model];
        if (modelStats) {
          modelStats.total += item.value;
          modelStats.count += 1;
        }
      }
    });

    const gapData: TimelineDataPoint[] = [];

    Object.values(timestampGroups).forEach((group) => {
      const dataPoint: TimelineDataPoint = {
        timestamp: group.timestamp,
        timestampLabel: group.timestampLabel
      };

      const avgValueByModel: { [key: string]: number } = {};
      Object.entries(group.models).forEach(([model, stats]) => {
        avgValueByModel[model] = stats.total / stats.count;
      });

      const averagedValues = Object.values(avgValueByModel);
      group.fastest = averagedValues.length > 0 ? Math.min(...averagedValues) : 0;

      selectedSTTModels.forEach((model) => {
        const modelAvg = avgValueByModel[model];
        if (modelAvg !== undefined) {
          const gap = modelAvg - group.fastest;
          dataPoint[`${model}_gap`] = gap;
        }
      });

      gapData.push(dataPoint);
    });

    return gapData;
  }, [rawData, activeTab, selectedSTTModels]);

  // ─── Functions using SQL-aggregated modelStats ───

  // STT heatmap still needs raw data for per-timestamp delta calculation
  const getModelHeatmapData = useCallback((): ModelHeatmapData[] => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
    const latencyMetric = activeTab === "tts" ? "TTFA" : "TTFT";

    if (selectedModels.length === 0) {
      return [];
    }

    // Delta calculation requires raw per-timestamp data
    // Group latency data by timestamp across all selected models
    const timestampGroups: {
      [key: number]: { [key: string]: { total: number; count: number } };
    } = {};

    rawData.forEach((item) => {
      if (!selectedModels.includes(item.model)) return;
      if (item.metric_type !== latencyMetric) return;
      if (!INCLUDED_STATUSES.has(item.status)) return;
      if (item.metric_value === null || item.metric_value === undefined) return;

      const timestamp = to15MinuteBucket(new Date(item.timestamp).getTime());
      if (!timestampGroups[timestamp]) {
        timestampGroups[timestamp] = {};
      }
      const latencyValue =
        activeTab === "tts" ? item.metric_value : item.metric_value * 1000;
      const bucket = timestampGroups[timestamp];
      if (bucket) {
        if (!bucket[item.model]) {
          bucket[item.model] = { total: 0, count: 0 };
        }
        const modelAcc = bucket[item.model];
        if (modelAcc) {
          modelAcc.total += latencyValue;
          modelAcc.count += 1;
        }
      }
    });

    const averagedTimestampGroups: { [key: number]: { [key: string]: number } } = {};
    Object.entries(timestampGroups).forEach(([bucketTimestamp, perModelStats]) => {
      const bucket = Number(bucketTimestamp);
      averagedTimestampGroups[bucket] = {};
      Object.entries(perModelStats).forEach(([model, stats]) => {
        const avg = averagedTimestampGroups[bucket];
        if (avg) {
          avg[model] = stats.total / stats.count;
        }
      });
    });

    const heatmapData: ModelHeatmapData[] = [];

    selectedModels.forEach((model) => {
      const werStat = getStat(model, "WER");
      const latencyStat = getStat(model, latencyMetric);
      const rtfStat = getStat(model, "RTF");

      // Need both latency and WER data
      if (!werStat || !latencyStat) return;

      // Calculate latency deltas relative to fastest at each timestamp
      const latencyDeltas: number[] = [];
      Object.values(averagedTimestampGroups).forEach((modelValues) => {
        const modelVal = modelValues[model];
        if (modelVal !== undefined) {
          const allVals = Object.values(modelValues);
          const fastest = allVals.length > 0 ? Math.min(...allVals) : 0;
          const delta = modelVal - fastest;
          latencyDeltas.push(delta);
        }
      });

      // Compute delta percentiles in JS (these are relative, can't be pre-aggregated)
      const sorted = [...latencyDeltas].sort((a, b) => a - b);
      const n = sorted.length;
      let p25 = 0, p50 = 0, p75 = 0;
      if (n > 0) {
        const getPercentile = (pct: number): number => {
          const index = (pct / 100) * (n - 1);
          const lower = Math.floor(index);
          const upper = Math.ceil(index);
          if (lower === upper) return sorted[lower] ?? 0;
          const weight = index - lower;
          return (sorted[lower] ?? 0) * (1 - weight) + (sorted[upper] ?? 0) * weight;
        };
        p25 = getPercentile(25);
        p50 = getPercentile(50);
        p75 = getPercentile(75);
      }

      heatmapData.push({
        model,
        latencyP25: p25,
        latencyP50: p50,
        latencyP75: p75,
        latencyIQR: p75 - p25,
        avgWER: werStat.avg_value,
        werStdDev: werStat.stddev_value,
        avgRTF: rtfStat?.avg_value ?? 0
      });
    });

    return heatmapData.sort((a, b) => a.model.localeCompare(b.model));
  }, [
    rawData,
    activeTab,
    selectedTTSModels,
    selectedSTTModels,
    getStat
  ]);

  // TTS heatmap uses absolute percentiles — fully from SQL stats
  const getTTSHeatmapData = useCallback((): ModelHeatmapData[] => {
    if (activeTab !== "tts" || selectedTTSModels.length === 0) {
      return [];
    }

    const heatmapData: ModelHeatmapData[] = [];

    selectedTTSModels.forEach((model) => {
      const latencyStat = getStat(model, "TTFA");
      const werStat = getStat(model, "WER");
      const rtfStat = getStat(model, "RTF");

      if (!latencyStat || !werStat) return;

      heatmapData.push({
        model,
        latencyP25: latencyStat.p25,
        latencyP50: latencyStat.p50,
        latencyP75: latencyStat.p75,
        latencyIQR: latencyStat.p75 - latencyStat.p25,
        avgWER: werStat.avg_value,
        werStdDev: werStat.stddev_value,
        avgRTF: rtfStat?.avg_value ?? 0
      });
    });

    return heatmapData.sort((a, b) => a.model.localeCompare(b.model));
  }, [activeTab, selectedTTSModels, getStat]);

  const getWERBarData = useCallback((): BarDataPoint[] => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;

    if (selectedModels.length === 0) {
      return [];
    }

    return selectedModels
      .map((model) => {
        const werStat = getStat(model, "WER");
        if (!werStat) return null;

        return {
          model,
          averageWER: werStat.avg_value,
          provider: werStat.provider
        };
      })
      .filter((item): item is BarDataPoint => item !== null)
      .sort((a, b) => a.averageWER - b.averageWER);
  }, [activeTab, selectedTTSModels, selectedSTTModels, getStat]);

  // ─── Timeline window functions ───

  const getFullTimeRange = useCallback((): [number, number] => {
    const allTimelineData = getTimelineData().map((item) => item.timestamp);

    if (allTimelineData.length === 0)
      return [Date.now() - TWENTY_FOUR_HOURS_MS, Date.now()];

    return [Math.min(...allTimelineData), Math.max(...allTimelineData)];
  }, [getTimelineData]);

  const getCurrentTimeWindow = useCallback((): [number, number] => {
    const windowStart = timelineWindowEnd - TWENTY_FOUR_HOURS_MS;
    return [windowStart, timelineWindowEnd];
  }, [timelineWindowEnd]);

  const getWindowedTimelineData = useCallback((): TimelineDataPoint[] => {
    const [windowStart, windowEnd] = getCurrentTimeWindow();
    const fullData = getTimelineData();

    const buffer = 30 * 60 * 1000;
    const extendedStart = windowStart - buffer;
    const extendedEnd = windowEnd + buffer;

    return fullData.filter(
      (item) => item.timestamp >= extendedStart && item.timestamp <= extendedEnd
    );
  }, [getCurrentTimeWindow, getTimelineData]);

  const getWindowedGapData = useCallback((): TimelineDataPoint[] => {
    const [windowStart, windowEnd] = getCurrentTimeWindow();
    const fullData = getGapData();

    const buffer = 30 * 60 * 1000;
    const extendedStart = windowStart - buffer;
    const extendedEnd = windowEnd + buffer;

    return fullData.filter(
      (item) => item.timestamp >= extendedStart && item.timestamp <= extendedEnd
    );
  }, [getCurrentTimeWindow, getGapData]);

  const getSTTRankingData = useCallback(() => {
    if (activeTab !== "stt" || selectedSTTModels.length === 0) {
      return [];
    }

    const heatmapData = activeTab === "stt" ? getModelHeatmapData() : [];

    if (heatmapData.length === 0) {
      return [];
    }

    const fastestModel = [...heatmapData].sort(
      (a, b) => a.latencyP50 - b.latencyP50
    )[0];

    if (!fastestModel) return [];

    const rankingData = heatmapData.map((modelData) => ({
      model: modelData.model,
      provider: getProviderForModel(modelData.model),
      p25Delta: modelData.latencyP25 - fastestModel.latencyP25,
      p50Delta: modelData.latencyP50 - fastestModel.latencyP50,
      p75Delta: modelData.latencyP75 - fastestModel.latencyP75
    }));

    rankingData.sort((a, b) => a.p50Delta - b.p50Delta);

    return rankingData.map((stat, index) => ({
      position: index + 1,
      model: stat.model,
      provider: stat.provider,
      p25Delta: stat.p25Delta,
      p50Delta: stat.p50Delta,
      p75Delta: stat.p75Delta,
      isFirst: index === 0
    }));
  }, [activeTab, selectedSTTModels, getModelHeatmapData, getProviderForModel]);

  return {
    formatChartLabel,
    getProviderForModel,
    getCurrentData,
    getTimelineData,
    getViolinData,
    getScatterData,
    getModelHeatmapData,
    getTTSHeatmapData,
    getWERBarData,
    getGapData,
    getFullTimeRange,
    getCurrentTimeWindow,
    getWindowedTimelineData,
    getWindowedGapData,
    getSTTRankingData
  };
}
