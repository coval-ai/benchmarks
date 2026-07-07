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

  // Per-metric, per-model timeline series (scheduled_at bucket → value) built in
  // one pass. Latency metrics plot the bucket median (p50); WER the average.
  // The generated SeriesPoint type follows the deployed API, which may still be
  // the pre-rollup shape (avg_value only) — fall back to it so web and API can
  // deploy in either order. Drop the fallback once the rollup API is live.
  const timelineByMetricModel = useMemo<
    Record<string, Record<string, Record<number, number>>>
  >(() => {
    const out: Record<string, Record<string, Record<number, number>>> = {};
    series.forEach((point) => {
      const byModel = (out[point.metric_type] ??= {});
      const modelKey = toModelKey(point.provider, point.model);
      const modelSeries = (byModel[modelKey] ??= {});
      const p = point as SeriesPoint & {
        avg_value?: number;
        value_sum?: number;
        p50?: number;
      };
      const value =
        p.metric_type === "WER"
          ? p.value_sum !== undefined
            ? p.value_sum / p.sample_count
            : p.avg_value
          : (p.p50 ?? p.avg_value);
      if (value === undefined) return;
      modelSeries[new Date(point.scheduled_at).getTime()] =
        toDisplayUnits(value);
    });
    return out;
  }, [series, toDisplayUnits]);

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

  // Per-metric, per-model box-plot pieces from the SQL stats, built in one pass
  // and indexed by metric_type so any latency metric can read its own pieces:
  // box = p25..p75, whiskers clamped to 1.5x IQR beyond the box (against the
  // true data extremes).
  const boxByMetricModel = useMemo<
    Record<string, Record<string, BoxPlotDataPoint>>
  >(() => {
    const out: Record<string, Record<string, BoxPlotDataPoint>> = {};
    modelStats.forEach((stat) => {
      if (
        stat.metric_type !== "TTFS" &&
        stat.metric_type !== "TTFT" &&
        stat.metric_type !== "TTFA"
      ) {
        return;
      }

      const q1 = toDisplayUnits(stat.p25);
      const median = toDisplayUnits(stat.p50);
      const q3 = toDisplayUnits(stat.p75);
      const iqr = q3 - q1;

      const byModel = (out[stat.metric_type] ??= {});
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

    return out;
  }, [modelStats, toDisplayUnits]);

  // Box plot data for a given metric: the selected models' pieces, sorted by
  // median, with the pooled axis bounds.
  const getBoxPlotData = useCallback(
    (metric: string): BoxPlotData => {
      const selectedModels =
        activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
      const byModel = boxByMetricModel[metric] ?? {};

      const data: BoxPlotDataPoint[] = [];
      let globalMin = Infinity;
      let globalMax = -Infinity;

      selectedModels.forEach((model) => {
        const entry = byModel[model];
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
        metricType: metric
      };
    },
    [boxByMetricModel, activeTab, selectedTTSModels, selectedSTTModels]
  );

  // Per-metric, per-model scatter point (avg latency, avg WER) built in one
  // pass, indexed by metric_type so any latency metric can read its own points.
  // Averages are independent per metric — not restricted to runs producing both.
  const scatterByMetricModel = useMemo<
    Record<string, Record<string, ScatterDataPoint>>
  >(() => {
    const out: Record<string, Record<string, ScatterDataPoint>> = {};
    modelStats.forEach((stat) => {
      // Only latency metrics belong on the scatter's x-axis; skip WER/RTF/etc.
      if (
        stat.metric_type !== "TTFS" &&
        stat.metric_type !== "TTFT" &&
        stat.metric_type !== "TTFA"
      ) {
        return;
      }
      const modelKey = toModelKey(stat.provider, stat.model);
      const werStat = getStat(modelKey, "WER");
      if (!werStat) return;
      const byModel = (out[stat.metric_type] ??= {});
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
    return out;
  }, [modelStats, activeTab, getStat, toDisplayUnits]);

  const getScatterData = useCallback(
    (metric: string): ScatterDataPoint[] => {
      const selectedModels =
        activeTab === "tts" ? selectedTTSModels : selectedSTTModels;
      const byModel = scatterByMetricModel[metric] ?? {};
      return selectedModels
        .map((model) => byModel[model])
        .filter((point): point is ScatterDataPoint => point !== undefined);
    },
    [scatterByMetricModel, activeTab, selectedTTSModels, selectedSTTModels]
  );

  // Comparison rows for a given latency metric: the full latency percentile
  // ladder straight from the SQL stats, plus avg WER and sample counts.
  const getHeatmapData = useCallback(
    (metric: string): ModelHeatmapData[] => {
      const selectedModels =
        activeTab === "tts" ? selectedTTSModels : selectedSTTModels;

      const heatmapData: ModelHeatmapData[] = [];

      selectedModels.forEach((model) => {
        const latencyStat = getStat(model, metric);
        const werStat = getStat(model, "WER");

        if (!latencyStat || !werStat) return;

        const latency = {
          p0: toDisplayUnits(latencyStat.min_value),
          p25: toDisplayUnits(latencyStat.p25),
          p50: toDisplayUnits(latencyStat.p50),
          p75: toDisplayUnits(latencyStat.p75),
          p90: toDisplayUnits(latencyStat.p90),
          p95: toDisplayUnits(latencyStat.p95),
          p99: toDisplayUnits(latencyStat.p99),
          p100: toDisplayUnits(latencyStat.max_value)
        };
        // The schema types every stat as a number, but guard against a
        // response with missing/non-finite fields leaking NaN into the table.
        if (
          ![...Object.values(latency), werStat.avg_value, werStat.stddev_value]
            .every(Number.isFinite)
        ) {
          return;
        }

        heatmapData.push({
          model,
          latency,
          avgWER: werStat.avg_value,
          werStdDev: werStat.stddev_value,
          sampleCount: latencyStat.sample_count
        });
      });

      return heatmapData.sort(
        (a, b) =>
          normalizeModelName(a.model).localeCompare(
            normalizeModelName(b.model)
          ) || a.model.localeCompare(b.model)
      );
    },
    [activeTab, selectedTTSModels, selectedSTTModels, getStat, toDisplayUnits]
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
    getHeatmapData,
    getWERBarData,
    getCurrentTimeWindow,
    getTimelineTicks,
    getWindowedTimelineData,
    getModelsWithTimelineData
  };
}
