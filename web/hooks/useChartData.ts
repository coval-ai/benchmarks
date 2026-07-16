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
  BarDataPoint,
  LatencyPercentile
} from "@/types/benchmark.types";
import type { SeriesPoint } from "@/lib/api/client";
import { latencyToMs, normalizeModelName, normalizeProviderNameForTab, toModelKey, parseModelKey } from "@/lib/utils/formatters";
import { WINDOW_MS, type TimeWindow } from "@/lib/config/timeWindows";

// Latency metrics share every chart's number machinery (box plot, scatter,
// comparison table). Membership gates the builders so a new one (V2V for S2S)
// is never silently skipped.
const LATENCY_METRICS: readonly string[] = ["TTFS", "TTFT", "TTFA", "V2V"];

interface UseChartDataParams {
  activeTab: "tts" | "stt" | "s2s";
  modelStats: ModelStats[];
  series: SeriesPoint[];
  selectedTTSModels: string[];
  selectedSTTModels: string[];
  selectedS2SModels: string[];
  modelsByProvider: ModelsByProvider;
  timeWindow: TimeWindow;
}

export function useChartData({
  activeTab,
  modelStats,
  series,
  selectedTTSModels,
  selectedSTTModels,
  selectedS2SModels,
  modelsByProvider,
  timeWindow
}: UseChartDataParams) {
  // The active tab's selected models, resolved once so the builders below don't
  // each re-derive the tab→list mapping.
  const selectedModels =
    activeTab === "tts"
      ? selectedTTSModels
      : activeTab === "s2s"
        ? selectedS2SModels
        : selectedSTTModels;
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
        return normalizeProviderNameForTab(provider, activeTab);
      }
      // Fallback for bare slugs: search modelStats, then providers config
      const rawProvider =
        modelStats.find((s) => s.model === model)?.provider ??
        Object.entries(modelsByProvider).find(([, models]) =>
          models.includes(modelKey)
        )?.[0] ??
        "Unknown";
      return normalizeProviderNameForTab(rawProvider, activeTab);
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
      return buildTimelineRows(
        timelineByMetricModel[metric] ?? {},
        selectedModels
      );
    },
    [buildTimelineRows, timelineByMetricModel, selectedModels]
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
      if (!LATENCY_METRICS.includes(stat.metric_type)) {
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
          count: stat.sample_count,
          min: toDisplayUnits(stat.min_value),
          max: toDisplayUnits(stat.max_value),
          p95: toDisplayUnits(stat.p95)
        }
      };
    });

    return out;
  }, [modelStats, toDisplayUnits]);

  // Box plot data for a given metric: the selected models' pieces, sorted by
  // median, with the pooled axis bounds.
  const getBoxPlotData = useCallback(
    (metric: string): BoxPlotData => {
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
    [boxByMetricModel, selectedModels]
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
      if (!LATENCY_METRICS.includes(stat.metric_type)) {
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
        provider: normalizeProviderNameForTab(stat.provider, activeTab),
        count: stat.sample_count
      };
    });
    return out;
  }, [modelStats, activeTab, getStat, toDisplayUnits]);

  const getScatterData = useCallback(
    (metric: string): ScatterDataPoint[] => {
      const byModel = scatterByMetricModel[metric] ?? {};
      return selectedModels
        .map((model) => byModel[model])
        .filter((point): point is ScatterDataPoint => point !== undefined);
    },
    [scatterByMetricModel, selectedModels]
  );

  // Comparison rows for a given latency metric: the full latency percentile
  // ladder straight from the SQL stats, plus avg WER and sample counts.
  const getHeatmapData = useCallback(
    (metric: string): ModelHeatmapData[] => {
      const heatmapData: ModelHeatmapData[] = [];

      selectedModels.forEach((model) => {
        const latencyStat = getStat(model, metric);
        // S2S is latency-only; the table hides the WER column for such rows.
        const werStat = activeTab === "s2s" ? undefined : getStat(model, "WER");

        // The schema types every stat as a number, but guard the raw fields
        // (before unit conversion, which would coerce null to 0) against a
        // response with missing/non-finite values leaking into the table.
        let latency: Record<LatencyPercentile, number> | undefined;
        let latencySampleCount: number | undefined;
        if (
          latencyStat &&
          [
            latencyStat.min_value,
            latencyStat.p25,
            latencyStat.p50,
            latencyStat.p75,
            latencyStat.p90,
            latencyStat.p95,
            latencyStat.p99,
            latencyStat.max_value,
            latencyStat.sample_count
          ].every(Number.isFinite)
        ) {
          latency = {
            p0: toDisplayUnits(latencyStat.min_value),
            p25: toDisplayUnits(latencyStat.p25),
            p50: toDisplayUnits(latencyStat.p50),
            p75: toDisplayUnits(latencyStat.p75),
            p90: toDisplayUnits(latencyStat.p90),
            p95: toDisplayUnits(latencyStat.p95),
            p99: toDisplayUnits(latencyStat.p99),
            p100: toDisplayUnits(latencyStat.max_value)
          };
          latencySampleCount = latencyStat.sample_count;
        }

        // S2S needs a latency stat (no WER to fall back on). Other tabs anchor
        // on WER: a model measured for WER but not the active latency metric
        // (e.g. TTFT but not TTFS) stays in the table with N/A latency.
        if (activeTab === "s2s") {
          if (!latency) return;
        } else if (
          !werStat ||
          !Number.isFinite(werStat.avg_value) ||
          !Number.isFinite(werStat.stddev_value)
        ) {
          return;
        }

        heatmapData.push({
          model,
          ...(latency ? { latency } : {}),
          ...(werStat
            ? { avgWER: werStat.avg_value, werStdDev: werStat.stddev_value }
            : {}),
          sampleCount: latencySampleCount ?? werStat?.sample_count ?? 0
        });
      });

      return heatmapData.sort(
        (a, b) =>
          normalizeModelName(a.model).localeCompare(
            normalizeModelName(b.model)
          ) || a.model.localeCompare(b.model)
      );
    },
    [selectedModels, getStat, toDisplayUnits, activeTab]
  );

  const werBarDataMemo = useMemo<BarDataPoint[]>(() => {
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
  }, [selectedModels, getStat]);

  const getWERBarData = useCallback(
    (): BarDataPoint[] => werBarDataMemo,
    [werBarDataMemo]
  );

  // ─── Timeline window functions ───

  const currentTimeWindow = useMemo<[number, number]>(() => {
    const windowEnd = Date.now();
    const windowStart = windowEnd - WINDOW_MS[timeWindow];
    return [windowStart, windowEnd];
    // eslint-disable-next-line react-hooks/exhaustive-deps -- series retriggers the now() snapshot on data reload
  }, [timeWindow, series]);

  const getCurrentTimeWindow = useCallback(
    (): [number, number] => currentTimeWindow,
    [currentTimeWindow]
  );

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
    [selectedModels, getCurrentTimeWindow, getTimelineData]
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
