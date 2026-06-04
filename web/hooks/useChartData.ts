// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useCallback, useMemo } from "react";
import type {
  BenchmarkData,
  ModelsByProvider,
  ModelStats,
  TimelineDataPoint,
  ScatterDataPoint,
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
import { normalizeModelName, normalizeSTTProviderName, normalizeTTSProviderName, toModelKey, parseModelKey } from "@/lib/utils/formatters";
import { TWENTY_FOUR_HOURS_MS } from "@/lib/config/constants";

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
  modelsByProvider: ModelsByProvider;
}

export function useChartData({
  activeTab,
  rawData,
  modelStats,
  selectedTTSModels,
  selectedSTTModels,
  timelineWindowEnd,
  modelsByProvider
}: UseChartDataParams) {
  // Helper: find a stat row for a given model key and metric.
  // Accepts composite "provider:model" keys or bare model slugs.
  const getStat = useCallback(
    (modelKey: string, metricType: string): ModelStats | undefined => {
      const { provider, model } = parseModelKey(modelKey);
      return modelStats.find(
        (s) =>
          s.model === model &&
          (provider === "" || s.provider === provider) &&
          s.metric_type === metricType
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

  // Helper function to get provider for a model.
  // Accepts composite "provider:model" keys — extracts the provider directly.
  const getProviderForModel = useCallback(
    (modelKey: string): string => {
      const { provider, model } = parseModelKey(modelKey);
      if (provider) {
        return activeTab === "stt"
          ? normalizeSTTProviderName(provider)
          : normalizeTTSProviderName(provider);
      }
      // Fallback for bare slugs: search modelStats, then rawData, then providers config
      const stat = modelStats.find((s) => s.model === model);
      const rawFromData = stat?.provider ?? rawData.find((d) => d.model === model)?.provider;
      const rawProvider =
        rawFromData ??
        Object.entries(modelsByProvider).find(([, models]) =>
          models.includes(modelKey)
        )?.[0] ??
        "Unknown";
      return activeTab === "stt"
        ? normalizeSTTProviderName(rawProvider)
        : normalizeTTSProviderName(rawProvider);
    },
    [modelStats, rawData, activeTab, modelsByProvider]
  );

  // ─── Functions that still need raw data (distributions, individual points) ───

  const getCurrentData = useCallback(() => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
    const metricTypes = activeTab === "tts" ? ["TTFA", "WER"] : ["TTFT", "WER"];

    if (selectedModels.length === 0) {
      return [];
    }

    const selectedModelKeys = new Set(selectedModels);
    return rawData.filter(
      (item) =>
        selectedModelKeys.has(toModelKey(item.provider, item.model)) &&
        metricTypes.includes(item.metric_type) &&
        item.metric_value !== null &&
        item.metric_value !== undefined
    );
  }, [rawData, activeTab, selectedTTSModels, selectedSTTModels]);

  // Memoized result so repeated callers (getFullTimeRange, getWindowedTimelineData,
  // useTimelineWindow) share a single computation per (rawData, tab, selection).
  // Per-model timeline series (timestamp → averaged value) — the heavy pass,
  // computed once per dataset so selection only merges precomputed series.
  const timelineByModel = useMemo<Record<string, Record<number, number>>>(() => {
    const primaryMetric = activeTab === "tts" ? "TTFA" : "TTFT";

    const acc: Record<
      string,
      { [key: number]: { total: number; count: number } }
    > = {};

    rawData.forEach((item) => {
      if (item.metric_type !== primaryMetric) return;
      if (!INCLUDED_STATUSES.has(item.status)) return;

      const modelKey = toModelKey(item.provider, item.model);
      const timestamp = new Date(item.scheduled_at).getTime();
      const value =
        activeTab === "tts"
          ? item.metric_value ?? 0
          : (item.metric_value ?? 0) * 1000;

      if (!acc[modelKey]) {
        acc[modelKey] = {};
      }
      const modelAcc = acc[modelKey];
      if (!modelAcc[timestamp]) {
        modelAcc[timestamp] = { total: 0, count: 0 };
      }
      const bucket = modelAcc[timestamp];
      if (bucket) {
        bucket.total += value;
        bucket.count += 1;
      }
    });

    const byModel: Record<string, Record<number, number>> = {};
    Object.entries(acc).forEach(([model, tsMap]) => {
      const series: Record<number, number> = {};
      Object.entries(tsMap).forEach(([timestamp, stats]) => {
        series[Number(timestamp)] = stats.total / stats.count;
      });
      byModel[model] = series;
    });

    return byModel;
  }, [rawData, activeTab]);

  // Timeline rows: merge the selected models' precomputed series into one row
  // per timestamp (ascending), each with a `${model}_value` column.
  const timelineData = useMemo<TimelineDataPoint[]>(() => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;

    if (selectedModels.length === 0) {
      return [];
    }

    const timestampSet = new Set<number>();
    selectedModels.forEach((model) => {
      const tsMap = timelineByModel[model];
      if (tsMap) {
        Object.keys(tsMap).forEach((ts) => timestampSet.add(Number(ts)));
      }
    });
    const timestamps = [...timestampSet].sort((a, b) => a - b);

    return timestamps.map((timestamp) => {
      const row: TimelineDataPoint = {
        timestamp,
        timestampLabel: new Date(timestamp).toISOString()
      };
      selectedModels.forEach((model) => {
        const value = timelineByModel[model]?.[timestamp];
        if (value !== undefined) {
          row[`${model}_value`] = value;
        }
      });
      return row;
    });
  }, [timelineByModel, activeTab, selectedTTSModels, selectedSTTModels]);

  const getTimelineData = useCallback(
    (): TimelineDataPoint[] => timelineData,
    [timelineData]
  );

  // Per-model violin pieces (values, KDE, quartiles, stats) — the heavy work.
  // Computed once per dataset (independent of selection) so toggling a model
  // only reassembles the precomputed pieces instead of re-scanning rawData.
  const violinByModel = useMemo<Record<string, ViolinDataPoint>>(() => {
    const primaryMetric = activeTab === "tts" ? "TTFA" : "TTFT";
    const modelGroups: { [key: string]: BenchmarkData[] } = {};

    rawData.forEach((item) => {
      if (item.metric_type !== primaryMetric) return;
      if (!INCLUDED_STATUSES.has(item.status)) return;
      if (item.metric_value === null || item.metric_value === undefined) return;

      const itemKey = toModelKey(item.provider, item.model);
      if (!modelGroups[itemKey]) {
        modelGroups[itemKey] = [];
      }
      const modelGroup = modelGroups[itemKey];
      if (modelGroup) {
        modelGroup.push(item);
      }
    });

    const byModel: Record<string, ViolinDataPoint> = {};
    Object.entries(modelGroups).forEach(([model, items]) => {
      const values = items.map((item) =>
        activeTab === "tts"
          ? item.metric_value ?? 0
          : (item.metric_value ?? 0) * 1000
      );

      if (values.length === 0) return;

      byModel[model] = {
        model,
        provider: items[0]?.provider ?? "Unknown",
        values,
        density: calculateKernelDensity(values),
        quartiles: calculateQuartiles(values),
        stats: calculateStats(values)
      };
    });

    return byModel;
  }, [rawData, activeTab]);

  // Violin plot data: gather the selected models' precomputed pieces and
  // recompute only the pooled summary (axis bounds, whisker cap, outliers).
  const violinDataMemo = useMemo<ViolinPlotData>(() => {
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

    const violinData: ViolinDataPoint[] = [];
    let globalMin = Infinity;
    let globalMax = -Infinity;

    selectedModels.forEach((model) => {
      const entry = violinByModel[model];
      if (!entry) return;
      violinData.push(entry);
      globalMin = Math.min(globalMin, ...entry.values);
      globalMax = Math.max(globalMax, ...entry.values);
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

    const sortedViolinData = [...violinData].sort(
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
  }, [violinByModel, activeTab, selectedTTSModels, selectedSTTModels]);

  const getViolinData = useCallback(
    (): ViolinPlotData => violinDataMemo,
    [violinDataMemo]
  );

  // Pair latency with WER per run, then average per model — pairing restricts
  // the average to runs that produced both metrics
  const scatterDataMemo = useMemo<ScatterDataPoint[]>(() => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
    const xMetric = activeTab === "tts" ? "TTFA" : "TTFT";
    const yMetric = "WER";

    if (selectedModels.length === 0) {
      return [];
    }

    const benchmarkGroups: {
      [key: string]: { [key: string]: BenchmarkData[] };
    } = {};

    const selectedModelKeys = new Set(selectedModels);
    rawData.forEach((item) => {
      if (!selectedModelKeys.has(toModelKey(item.provider, item.model))) return;
      if (item.metric_type !== xMetric && item.metric_type !== yMetric) return;
      if (!INCLUDED_STATUSES.has(item.status)) return;
      if (item.metric_value === null || item.metric_value === undefined) return;

      const key = `${item.run_id}\x00${item.provider}\x00${item.model}\x00${item.voice}\x00${item.audio_filename}`;
      if (!benchmarkGroups[key]) {
        benchmarkGroups[key] = {};
      }
      if (!benchmarkGroups[key][item.metric_type]) {
        benchmarkGroups[key][item.metric_type] = [];
      }
      const metricGroup = benchmarkGroups[key][item.metric_type];
      if (metricGroup) {
        metricGroup.push(item);
      }
    });

    // One averaged point per model: sum paired run values, divide by run count
    const modelAggregates: {
      [model: string]: {
        xSum: number;
        ySum: number;
        count: number;
        benchmark: string;
        provider: string;
      };
    } = {};

    Object.values(benchmarkGroups).forEach((group) => {
      const xData = group[xMetric]?.[0];
      const yData = group[yMetric]?.[0];

      if (xData && yData) {
        const model = toModelKey(xData.provider, xData.model);
        const x =
          activeTab === "tts"
            ? xData.metric_value ?? 0
            : (xData.metric_value ?? 0) * 1000;
        const y = yData.metric_value ?? 0;

        const agg = modelAggregates[model];
        if (agg) {
          agg.xSum += x;
          agg.ySum += y;
          agg.count += 1;
        } else {
          modelAggregates[model] = {
            xSum: x,
            ySum: y,
            count: 1,
            benchmark: xData.benchmark,
            provider:
              activeTab === "stt"
                ? normalizeSTTProviderName(xData.provider)
                : normalizeTTSProviderName(xData.provider)
          };
        }
      }
    });

    return Object.entries(modelAggregates).map(([model, agg]) => ({
      x: agg.xSum / agg.count,
      y: agg.ySum / agg.count,
      model,
      benchmark: agg.benchmark,
      provider: agg.provider,
      count: agg.count
    }));
  }, [rawData, activeTab, selectedTTSModels, selectedSTTModels]);

  const getScatterData = useCallback(
    (): ScatterDataPoint[] => scatterDataMemo,
    [scatterDataMemo]
  );

  // ─── Functions using SQL-aggregated modelStats ───

  // STT heatmap still needs raw data for per-timestamp delta calculation
  const modelHeatmapDataMemo = useMemo<ModelHeatmapData[]>(() => {
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

    const selectedModelKeys = new Set(selectedModels);
    rawData.forEach((item) => {
      const itemKey = toModelKey(item.provider, item.model);
      if (!selectedModelKeys.has(itemKey)) return;
      if (item.metric_type !== latencyMetric) return;
      if (!INCLUDED_STATUSES.has(item.status)) return;
      if (item.metric_value === null || item.metric_value === undefined) return;

      const timestamp = new Date(item.scheduled_at).getTime();
      if (!timestampGroups[timestamp]) {
        timestampGroups[timestamp] = {};
      }
      const latencyValue =
        activeTab === "tts" ? item.metric_value : item.metric_value * 1000;
      const bucket = timestampGroups[timestamp];
      if (bucket) {
        if (!bucket[itemKey]) {
          bucket[itemKey] = { total: 0, count: 0 };
        }
        const modelAcc = bucket[itemKey];
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
        werStdDev: werStat.stddev_value
      });
    });

    return heatmapData.sort(
      (a, b) =>
        normalizeModelName(a.model).localeCompare(normalizeModelName(b.model)) ||
        a.model.localeCompare(b.model)
    );
  }, [
    rawData,
    activeTab,
    selectedTTSModels,
    selectedSTTModels,
    getStat
  ]);

  const getModelHeatmapData = useCallback(
    (): ModelHeatmapData[] => modelHeatmapDataMemo,
    [modelHeatmapDataMemo]
  );

  // TTS heatmap uses absolute percentiles — fully from SQL stats
  const ttsHeatmapDataMemo = useMemo<ModelHeatmapData[]>(() => {
    if (activeTab !== "tts" || selectedTTSModels.length === 0) {
      return [];
    }

    const heatmapData: ModelHeatmapData[] = [];

    selectedTTSModels.forEach((model) => {
      const latencyStat = getStat(model, "TTFA");
      const werStat = getStat(model, "WER");

      if (!latencyStat || !werStat) return;

      heatmapData.push({
        model,
        latencyP25: latencyStat.p25,
        latencyP50: latencyStat.p50,
        latencyP75: latencyStat.p75,
        latencyIQR: latencyStat.p75 - latencyStat.p25,
        avgWER: werStat.avg_value,
        werStdDev: werStat.stddev_value
      });
    });

    return heatmapData.sort(
      (a, b) =>
        normalizeModelName(a.model).localeCompare(normalizeModelName(b.model)) ||
        a.model.localeCompare(b.model)
    );
  }, [activeTab, selectedTTSModels, getStat]);

  const getTTSHeatmapData = useCallback(
    (): ModelHeatmapData[] => ttsHeatmapDataMemo,
    [ttsHeatmapDataMemo]
  );

  const werBarDataMemo = useMemo<BarDataPoint[]>(() => {
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

  const getWERBarData = useCallback(
    (): BarDataPoint[] => werBarDataMemo,
    [werBarDataMemo]
  );

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

  // Pinned tick positions for the X axis. Letting Recharts auto-pick ticks
  // gives uneven spacing that shifts as the window pans. Six ticks (every
  // 4 hours, snapped to whole-hour boundaries inside the window) stays
  // readable without crowding.
  const getTimelineTicks = useCallback((): number[] => {
    const [windowStart, windowEnd] = [
      timelineWindowEnd - TWENTY_FOUR_HOURS_MS,
      timelineWindowEnd,
    ];
    const FOUR_HOURS_MS = 4 * 60 * 60 * 1000;
    const firstTick =
      Math.ceil(windowStart / FOUR_HOURS_MS) * FOUR_HOURS_MS;
    const ticks: number[] = [];
    for (let t = firstTick; t <= windowEnd; t += FOUR_HOURS_MS) {
      ticks.push(t);
    }
    return ticks;
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

  /** Models that have at least one plotted point in the current timeline window. */
  const getModelsWithTimelineData = useCallback((): string[] => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
    const [windowStart, windowEnd] = getCurrentTimeWindow();
    const windowed = getTimelineData().filter(
      (point) => point.timestamp >= windowStart && point.timestamp <= windowEnd
    );
    return selectedModels.filter((model) =>
      windowed.some(
        (point) =>
          point[`${model}_value`] !== undefined &&
          point[`${model}_value`] !== null
      )
    );
  }, [
    activeTab,
    selectedTTSModels,
    selectedSTTModels,
    getCurrentTimeWindow,
    getTimelineData,
  ]);

  return {
    formatChartLabel,
    getProviderForModel,
    getStat,
    getCurrentData,
    getTimelineData,
    getViolinData,
    getScatterData,
    getModelHeatmapData,
    getTTSHeatmapData,
    getWERBarData,
    getFullTimeRange,
    getCurrentTimeWindow,
    getTimelineTicks,
    getWindowedTimelineData,
    getModelsWithTimelineData
  };
}
