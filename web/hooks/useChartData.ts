// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useCallback, useMemo } from "react";
import type {
  ModelsByProvider,
  ModelStats,
  TimelineDataPoint,
  ScatterDataPoint,
  BoxPlotData,
  BoxPlotDataPoint,
  ModelHeatmapData,
  BarDataPoint
} from "@/types/benchmark.types";
import type { SeriesPoint } from "@/lib/api/client";
import { latencyToMs, normalizeModelName, normalizeSTTProviderName, normalizeTTSProviderName, toModelKey, parseModelKey } from "@/lib/utils/formatters";
import { WINDOW_MS, type TimeWindow } from "@/lib/config/timeWindows";

interface UseChartDataParams {
  activeTab: "tts" | "stt";
  modelStats: ModelStats[];
  series: SeriesPoint[];
  selectedTTSModels: string[];
  selectedSTTModels: string[];
  modelsByProvider: ModelsByProvider;
  timeWindow: TimeWindow;
}

export function useChartData({
  activeTab,
  modelStats,
  series,
  selectedTTSModels,
  selectedSTTModels,
  modelsByProvider,
  timeWindow
}: UseChartDataParams) {
  const toDisplayUnits = useCallback(
    (value: number): number => latencyToMs(value, activeTab),
    [activeTab]
  );

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
      // Fallback for bare slugs: search modelStats, then providers config
      const rawProvider =
        modelStats.find((s) => s.model === model)?.provider ??
        Object.entries(modelsByProvider).find(([, models]) =>
          models.includes(modelKey)
        )?.[0] ??
        "Unknown";
      return activeTab === "stt"
        ? normalizeSTTProviderName(rawProvider)
        : normalizeTTSProviderName(rawProvider);
    },
    [modelStats, activeTab, modelsByProvider]
  );

  // Merge a per-model series map into one row per timestamp (ascending), each
  // with a `${model}_value` column. Shared by every metric's timeline.
  const buildTimelineRows = useCallback(
    (
      byModel: Record<string, Record<number, number>>,
      models: string[]
    ): TimelineDataPoint[] => {
      if (models.length === 0) {
        return [];
      }

      const timestampSet = new Set<number>();
      models.forEach((model) => {
        const tsMap = byModel[model];
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
        models.forEach((model) => {
          const value = byModel[model]?.[timestamp];
          if (value !== undefined) {
            row[`${model}_value`] = value;
          }
        });
        return row;
      });
    },
    []
  );

  // Per-metric, per-model timeline series (scheduled_at bucket → avg) built in
  // one pass. Indexed by metric_type so any metric can read its own series.
  const timelineByMetricModel = useMemo<
    Record<string, Record<string, Record<number, number>>>
  >(() => {
    const out: Record<string, Record<string, Record<number, number>>> = {};
    series.forEach((point) => {
      const byModel = (out[point.metric_type] ??= {});
      const modelKey = toModelKey(point.provider, point.model);
      const modelSeries = (byModel[modelKey] ??= {});
      modelSeries[new Date(point.scheduled_at).getTime()] = toDisplayUnits(
        point.avg_value
      );
    });
    return out;
  }, [series, toDisplayUnits]);

  // Primary latency metric series (TTFT for STT, TTFA for TTS). Also feeds the
  // STT heatmap deltas and the gap chart.
  const timelineByModel = useMemo<Record<string, Record<number, number>>>(() => {
    const primaryMetric = activeTab === "tts" ? "TTFA" : "TTFT";
    return timelineByMetricModel[primaryMetric] ?? {};
  }, [timelineByMetricModel, activeTab]);

  // Timeline rows for a given metric. Models follow the active tab (STT models
  // for TTFS/TTFT, TTS models for TTFA).
  const getTimelineData = useCallback(
    (metric: string): TimelineDataPoint[] => {
      const models =
        activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
      return buildTimelineRows(timelineByMetricModel[metric] ?? {}, models);
    },
    [
      buildTimelineRows,
      timelineByMetricModel,
      activeTab,
      selectedTTSModels,
      selectedSTTModels
    ]
  );

  // Per-model box-plot pieces from the SQL stats: box = p25..p75, whiskers
  // clamped to 1.5x IQR beyond the box (against the true data extremes).
  const boxByModel = useMemo<Record<string, BoxPlotDataPoint>>(() => {
    const primaryMetric = activeTab === "tts" ? "TTFA" : "TTFT";

    const byModel: Record<string, BoxPlotDataPoint> = {};
    modelStats.forEach((stat) => {
      if (stat.metric_type !== primaryMetric) return;

      const q1 = toDisplayUnits(stat.p25);
      const median = toDisplayUnits(stat.p50);
      const q3 = toDisplayUnits(stat.p75);
      const iqr = q3 - q1;

      byModel[toModelKey(stat.provider, stat.model)] = {
        model: toModelKey(stat.provider, stat.model),
        provider: stat.provider,
        quartiles: {
          min: Math.max(toDisplayUnits(stat.min_value), q1 - 1.5 * iqr),
          q1,
          median,
          q3,
          max: Math.min(toDisplayUnits(stat.max_value), q3 + 1.5 * iqr)
        },
        stats: {
          mean: toDisplayUnits(stat.avg_value),
          std: toDisplayUnits(stat.stddev_value),
          count: stat.sample_count
        }
      };
    });

    return byModel;
  }, [modelStats, activeTab, toDisplayUnits]);

  // Box plot data: the selected models' pieces, sorted by median, with the
  // pooled axis bounds.
  const boxPlotDataMemo = useMemo<BoxPlotData>(() => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
    const primaryMetric = activeTab === "tts" ? "TTFA" : "TTFT";

    const data: BoxPlotDataPoint[] = [];
    let globalMin = Infinity;
    let globalMax = -Infinity;

    selectedModels.forEach((model) => {
      const entry = boxByModel[model];
      if (!entry) return;
      data.push(entry);
      globalMin = Math.min(globalMin, entry.quartiles.min);
      globalMax = Math.max(globalMax, entry.quartiles.max);
    });

    if (data.length === 0) {
      globalMin = 0;
      globalMax = 0;
    }

    data.sort((a, b) => a.quartiles.median - b.quartiles.median);

    return {
      data,
      globalMin,
      globalMax,
      metricType: primaryMetric
    };
  }, [boxByModel, activeTab, selectedTTSModels, selectedSTTModels]);

  const getBoxPlotData = useCallback(
    (): BoxPlotData => boxPlotDataMemo,
    [boxPlotDataMemo]
  );

  // One (avg latency, avg WER) point per model. The averages are independent
  // per metric — not restricted to runs that produced both.
  const scatterByModel = useMemo<Record<string, ScatterDataPoint>>(() => {
    const xMetric = activeTab === "tts" ? "TTFA" : "TTFT";

    const byModel: Record<string, ScatterDataPoint> = {};
    modelStats.forEach((stat) => {
      if (stat.metric_type !== xMetric) return;

      const modelKey = toModelKey(stat.provider, stat.model);
      const werStat = getStat(modelKey, "WER");
      if (!werStat) return;

      byModel[modelKey] = {
        x: toDisplayUnits(stat.avg_value),
        y: werStat.avg_value,
        model: modelKey,
        benchmark: activeTab === "tts" ? "TTS" : "STT",
        provider:
          activeTab === "stt"
            ? normalizeSTTProviderName(stat.provider)
            : normalizeTTSProviderName(stat.provider),
        count: stat.sample_count
      };
    });
    return byModel;
  }, [modelStats, activeTab, getStat, toDisplayUnits]);

  const scatterDataMemo = useMemo<ScatterDataPoint[]>(() => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;

    const points: ScatterDataPoint[] = [];
    selectedModels.forEach((model) => {
      const point = scatterByModel[model];
      if (point) {
        points.push(point);
      }
    });
    return points;
  }, [scatterByModel, activeTab, selectedTTSModels, selectedSTTModels]);

  const getScatterData = useCallback(
    (): ScatterDataPoint[] => scatterDataMemo,
    [scatterDataMemo]
  );

  // STT heatmap: latency deltas relative to the fastest selected model at
  // each timestamp, percentiled per model. Deltas depend on the selection, so
  // they are computed here from the per-model series.
  const modelHeatmapDataMemo = useMemo<ModelHeatmapData[]>(() => {
    const selectedModels =
      activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
    const latencyMetric = activeTab === "tts" ? "TTFA" : "TTFT";

    if (selectedModels.length === 0) {
      return [];
    }

    // Per-timestamp averages for the selected models.
    const averagedTimestampGroups: { [key: number]: { [key: string]: number } } = {};
    selectedModels.forEach((model) => {
      const tsMap = timelineByModel[model];
      if (!tsMap) return;
      Object.entries(tsMap).forEach(([timestamp, avg]) => {
        const bucket = Number(timestamp);
        if (!averagedTimestampGroups[bucket]) {
          averagedTimestampGroups[bucket] = {};
        }
        const group = averagedTimestampGroups[bucket];
        if (group) {
          group[model] = avg;
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

      // Delta percentiles (relative to the selection, so computed here)
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
    timelineByModel,
    activeTab,
    selectedTTSModels,
    selectedSTTModels,
    getStat
  ]);

  const getModelHeatmapData = useCallback(
    (): ModelHeatmapData[] => modelHeatmapDataMemo,
    [modelHeatmapDataMemo]
  );

  // TTS heatmap uses absolute percentiles — straight from the SQL stats
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

  // Anchor the timeline window to the latest plotted bucket, not Date.now().
  // Charts plot scheduled_at so every result from one benchmark run shares a
  // tick.
  const timelineWindowEnd = useMemo<number>(() => {
    if (series.length === 0) return Date.now();
    let max = 0;
    for (const point of series) {
      const t = new Date(point.scheduled_at).getTime();
      if (Number.isNaN(t)) continue;
      if (t > max) max = t;
    }
    return max > 0 ? max : Date.now();
  }, [series]);

  const getCurrentTimeWindow = useCallback((): [number, number] => {
    const windowStart = timelineWindowEnd - WINDOW_MS[timeWindow];
    return [windowStart, timelineWindowEnd];
  }, [timelineWindowEnd, timeWindow]);

  // Pinned ticks — Recharts' auto-picked ticks space unevenly. Wider windows
  // tick at local midnights so date labels sit on day starts in the viewer's
  // timezone.
  const getTimelineTicks = useCallback((): number[] => {
    const [windowStart, windowEnd] = getCurrentTimeWindow();
    if (timeWindow === "24h") {
      const FOUR_HOURS_MS = 4 * 60 * 60 * 1000;
      const firstTick =
        Math.ceil(windowStart / FOUR_HOURS_MS) * FOUR_HOURS_MS;
      const ticks: number[] = [];
      for (let t = firstTick; t <= windowEnd; t += FOUR_HOURS_MS) {
        ticks.push(t);
      }
      return ticks;
    }
    const stepDays = timeWindow === "7d" ? 1 : 5;
    const cursor = new Date(windowStart);
    cursor.setHours(0, 0, 0, 0);
    if (cursor.getTime() < windowStart) {
      cursor.setDate(cursor.getDate() + 1);
    }
    const ticks: number[] = [];
    while (cursor.getTime() <= windowEnd) {
      ticks.push(cursor.getTime());
      cursor.setDate(cursor.getDate() + stepDays);
    }
    return ticks;
  }, [getCurrentTimeWindow, timeWindow]);

  const getWindowedTimelineData = useCallback(
    (metric: string): TimelineDataPoint[] => {
      const [windowStart, windowEnd] = getCurrentTimeWindow();
      return getTimelineData(metric).filter(
        (item) => item.timestamp >= windowStart && item.timestamp <= windowEnd
      );
    },
    [getCurrentTimeWindow, getTimelineData]
  );

  /** Models that have at least one plotted point in the current timeline window. */
  const getModelsWithTimelineData = useCallback(
    (metric: string): string[] => {
      const selectedModels =
        activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
      const [windowStart, windowEnd] = getCurrentTimeWindow();
      const windowed = getTimelineData(metric).filter(
        (point) => point.timestamp >= windowStart && point.timestamp <= windowEnd
      );
      return selectedModels.filter((model) =>
        windowed.some(
          (point) =>
            point[`${model}_value`] !== undefined &&
            point[`${model}_value`] !== null
        )
      );
    },
    [
      activeTab,
      selectedTTSModels,
      selectedSTTModels,
      getCurrentTimeWindow,
      getTimelineData
    ]
  );

  return {
    formatChartLabel,
    getProviderForModel,
    getStat,
    getTimelineData,
    getBoxPlotData,
    getScatterData,
    getModelHeatmapData,
    getTTSHeatmapData,
    getWERBarData,
    getCurrentTimeWindow,
    getTimelineTicks,
    getWindowedTimelineData,
    getModelsWithTimelineData
  };
}
