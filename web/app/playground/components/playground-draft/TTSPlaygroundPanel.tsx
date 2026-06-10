"use client";

/**
 * Live TTS panel backed by `/api/playground/tts`.
 * Sample prompt chips are static product copy; benchmark metrics are request-derived.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { createPortal } from "react-dom";
import { Pause, Play, X } from "lucide-react";
import type { TtsModelConfig } from "@/lib/playground/providers";
import { useTTSGeneration } from "@/app/playground/hooks/useTTSGeneration";
import { useAudioPlayback } from "@/app/playground/hooks/useAudioPlayback";
import {
  formatProviderDisplayName,
  getPlaygroundModelVisual,
  getTtsPillLabel,
} from "@/lib/playground/provider-styles";
import { isTypingInteractionTarget } from "@/lib/playground/hotkeys";
import { ModelPill } from "./ModelPill";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS, type PlaygroundRunTrigger } from "@/lib/posthog/events";

/** HARDCODED: demo prompt chips only — not provider data. */
const EXAMPLES = [
  { label: "Make a phone call", text: "Hi, this is Alex calling about the appointment we scheduled for Tuesday." },
  { label: "Narrate a story", text: "The lighthouse keeper had watched the same horizon for thirty years, until the night the foghorn stayed silent." },
  { label: "Leave a voicemail", text: "Hey, it's me—running a few minutes late. Save me a seat and I'll see you soon." }
] as const;

type MetricKey = "ttfa" | "total";

/** All playground TTS metrics shown in benchmark modal (fixed — not user-toggleable). */
const TTS_METRICS_ALL: Record<MetricKey, boolean> = {
  ttfa: true,
  total: true
};

const BENCH_METRIC_COL = "min-w-[4.25rem] w-[4.25rem] sm:min-w-[4.75rem] sm:w-[4.75rem]";
const BENCH_METRICS_GAP = "gap-3 sm:gap-4";
const BENCH_ROW_GAP = "gap-3";
const BENCH_ROW_MINW = "min-w-[min(100%,520px)]";
/** Narrow column for play control (TTS rows). */
const BENCH_PLAY_COL = "flex w-7 shrink-0 justify-center sm:w-8";
/** Invisible header cell: same width as play column. */
const BENCH_PLAY_HEAD_SPACER = "w-7 shrink-0 sm:w-8";
/** Column headers — match playground section labels (sans, 10px caps). */
const BENCH_HEAD_TEXT =
  "font-sans text-[10px] font-medium uppercase tracking-[0.14em] text-text-tertiary";
const BENCH_ROW_PY = "py-2.5 sm:py-3";

type BenchmarkRow = {
  model: TtsModelConfig;
  /** From ``X-TTFA-Ms`` when present; otherwise null (no synthetic fill). */
  ttfaMs: number | null;
  totalMs: number;
  audioUrl: string | null;
};

function sortMetricValue(row: BenchmarkRow, key: MetricKey): number {
  if (key === "ttfa") return row.ttfaMs ?? Number.POSITIVE_INFINITY;
  return row.totalMs;
}

/** Primary column for ranking when all metrics are on — TTFA. */
function ttsRankingMetric(): MetricKey {
  return "ttfa";
}

export function TTSPlaygroundPanel({
  models,
  onBenchmarkOverlayChange
}: {
  models: TtsModelConfig[];
  onBenchmarkOverlayChange?: (open: boolean) => void;
}) {
  const [text, setText] = useState("");
  /**
   * Per-model inclusion for this run (`true` = include in leaderboard). Separate from `m.enabled`
   * on the config object: parent passes **allowlisted/enabled** models only (`getEnabledTtsModels`);
   * `selectedMap` is user choice to compare a subset. If the parent ever passes disabled rows,
   * filter them in the parent — do not rely on this map for allowlisting.
   */
  const [selectedMap, setSelectedMap] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(models.map((m) => [m.id, true]))
  );
  const [benchmarkOpen, setBenchmarkOpen] = useState(false);
  const typingFiredRef = useRef(false);
  const { rows: generationRows, runBenchmark, cancelAll, resetRows, hasInFlight } = useTTSGeneration();
  const { playingModelId, toggle, stop } = useAudioPlayback();

  useEffect(() => {
    setSelectedMap((prev) => {
      const next = { ...prev };
      for (const m of models) {
        if (!(m.id in next)) next[m.id] = true;
      }
      return next;
    });
  }, [models]);

  const activeModels = useMemo(
    () => models.filter((m) => selectedMap[m.id] !== false),
    [models, selectedMap]
  );
  const modelsById = useMemo(
    () => new Map(models.map((model) => [model.id, model])),
    [models]
  );

  const modelsTagCloudOrder = useMemo(
    () =>
      [...models].sort((a, b) =>
        getTtsPillLabel(a).localeCompare(getTtsPillLabel(b), undefined, { sensitivity: "base" })
      ),
    [models]
  );

  const rows = useMemo<BenchmarkRow[]>(() => {
    const sortKey = ttsRankingMetric();
    const completeRows = activeModels
      .map((model) => {
        const row = generationRows[model.id];
        if (!row || row.status !== "success" || row.totalMs == null) {
          return null;
        }
        return {
          model,
          ttfaMs: row.ttfaMs != null ? Math.round(row.ttfaMs) : null,
          totalMs: Math.round(row.totalMs),
          audioUrl: row.audioUrl,
        };
      })
      .filter((row): row is BenchmarkRow => row !== null);

    return [...completeRows].sort((a, b) => {
      const delta = sortMetricValue(a, sortKey) - sortMetricValue(b, sortKey);
      if (delta !== 0) return delta;
      return a.totalMs - b.totalMs;
    });
  }, [activeModels, generationRows]);

  const toggleModel = useCallback((id: string) => {
    if (hasInFlight) return;
    if (!modelsById.has(id)) return;
    const willBeSelected = !selectedMap[id];
    const nextMap = { ...selectedMap, [id]: willBeSelected };
    const selectedIds = models.filter((m) => nextMap[m.id] !== false).map((m) => m.id);
    capturePostHogEvent(POSTHOG_EVENTS.playgroundModelSelectionChanged, {
      surface: "playground",
      mode: "tts",
      action: willBeSelected ? "add" : "remove",
      model_id: id,
      selected_model_ids: selectedIds,
      selected_model_count: selectedIds.length,
      is_comparison: selectedIds.length >= 2
    });
    setSelectedMap(nextMap);
  }, [hasInFlight, modelsById, selectedMap, models]);

  const dialogRef = useRef<HTMLDivElement>(null);
  const doneBtnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!benchmarkOpen) return;
    const restoreTo = document.activeElement as HTMLElement | null;
    doneBtnRef.current?.focus();
    return () => restoreTo?.focus?.();
  }, [benchmarkOpen]);

  const onDialogKeyDown = useCallback((e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      setBenchmarkOpen(false);
      return;
    }
    if (e.key !== "Tab") return;
    const root = dialogRef.current;
    if (!root) return;
    const focusables = root.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (focusables.length === 0) return;
    const first = focusables[0]!;
    const last = focusables[focusables.length - 1]!;
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }, []);

  useEffect(() => {
    if (!benchmarkOpen) stop();
  }, [benchmarkOpen, stop]);

  useEffect(() => {
    onBenchmarkOverlayChange?.(benchmarkOpen);
  }, [benchmarkOpen, onBenchmarkOverlayChange]);

  useEffect(() => {
    return () => onBenchmarkOverlayChange?.(false);
  }, [onBenchmarkOverlayChange]);

  const [portalRoot, setPortalRoot] = useState<HTMLElement | null>(null);
  useEffect(() => {
    setPortalRoot(document.body);
  }, []);

  const canBenchmark = Boolean(text.trim()) && activeModels.length > 0;

  const visibleMetricKeys = useMemo(
    (): MetricKey[] => (["ttfa", "total"] as const).filter((k) => TTS_METRICS_ALL[k]),
    []
  );

  const openBenchmarkRef = useRef<(trigger: PlaygroundRunTrigger) => Promise<void>>(async () => {});
  openBenchmarkRef.current = async (trigger: PlaygroundRunTrigger) => {
    if (!canBenchmark || hasInFlight) return;
    const selectedIds = activeModels.map((model) => model.id);
    capturePostHogEvent(POSTHOG_EVENTS.playgroundTtsBenchmarkPressed, {
      surface: "playground",
      mode: "tts",
      text_length: text.trim().length,
      selected_model_ids: selectedIds,
      selected_model_count: selectedIds.length,
      trigger,
      is_comparison: selectedIds.length >= 2
    });
    const summary = await runBenchmark(text, selectedIds);
    if (!summary) return;
    const ttfas = summary.successes
      .map((s) => s.ttfaMs)
      .filter((v): v is number => v != null);
    const totals = summary.successes.map((s) => s.totalMs);
    capturePostHogEvent(POSTHOG_EVENTS.playgroundBenchmarkCompleted, {
      surface: "playground",
      mode: "tts",
      selected_model_ids: selectedIds,
      selected_model_count: selectedIds.length,
      success_count: summary.successes.length,
      failure_count: summary.failures.length,
      overall_duration_ms: Math.round(summary.overallDurationMs),
      best_ttfa_ms: ttfas.length ? Math.min(...ttfas) : null,
      best_total_ms: totals.length ? Math.min(...totals) : null,
      is_comparison: selectedIds.length >= 2
    });
    setBenchmarkOpen(true);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Enter") return;
      if (benchmarkOpen) return;
      const ae = document.activeElement;
      const cmdEnterFromPrompt =
        ae instanceof HTMLTextAreaElement && (e.ctrlKey || e.metaKey);
      if (!cmdEnterFromPrompt && isTypingInteractionTarget(e.target)) return;
      if (!canBenchmark || hasInFlight) return;
      e.preventDefault();
      void openBenchmarkRef.current("keyboard");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [benchmarkOpen, canBenchmark, hasInFlight]);

  const metricHeaderLabel = (key: MetricKey) => {
    if (key === "ttfa") return "TTFA";
    return "Total latency";
  };

  const renderMetricCell = (key: MetricKey, row: BenchmarkRow) => {
    if (key === "ttfa") {
      return (
        <div key="ttfa" className={`${BENCH_METRIC_COL} text-right`}>
          <p
            className="text-xs font-medium tabular-nums leading-snug text-text-primary"
            title={
              row.ttfaMs == null
                ? "TTFA not reported (server did not send X-TTFA-Ms)."
                : undefined
            }
          >
            {row.ttfaMs == null ? "—" : `${row.ttfaMs}ms`}
          </p>
        </div>
      );
    }
    if (key === "total") {
      return (
        <div key="total" className={`${BENCH_METRIC_COL} text-right`}>
          <p className="text-xs font-medium tabular-nums leading-snug text-text-primary">{row.totalMs}ms</p>
        </div>
      );
    }
  };

  const handleClear = useCallback(() => {
    if (hasInFlight) return;
    setText("");
    cancelAll();
    resetRows();
    stop();
    setBenchmarkOpen(false);
  }, [cancelAll, hasInFlight, resetRows, stop]);

  const handleBenchmarkButtonClick = useCallback(() => {
    if (!canBenchmark || hasInFlight) return;
    void openBenchmarkRef.current("button");
  }, [canBenchmark, hasInFlight]);

  return (
    <section
      id="playground-panel-tts"
      role="tabpanel"
      aria-labelledby="tab-playground-tts"
      className="space-y-6"
    >
      <div className="flex flex-col gap-6 md:flex-row md:items-start md:gap-8">
        <div className="min-w-0 flex-1 space-y-3">
          <textarea
            value={text}
            onChange={(e) => {
              const value = e.target.value;
              if (!typingFiredRef.current && value.trim().length > 0) {
                typingFiredRef.current = true;
                capturePostHogEvent(POSTHOG_EVENTS.playgroundTtsTypingStarted, {
                  surface: "playground",
                  mode: "tts"
                });
              }
              setText(value);
            }}
            placeholder="Write something to say…"
            rows={8}
            className="font-sans w-full resize-y rounded-xl border border-border-primary bg-surface-primary px-4 py-3 text-sm leading-relaxed text-text-primary placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-text-tertiary/30"
          />
          <div className="flex items-center gap-2">
            <button
              type="button"
              aria-label="Clear prompt text, cancel in-flight requests, and close results"
              onClick={handleClear}
              disabled={hasInFlight}
              className="rounded-full border border-border-primary px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:bg-hover-bg hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40"
            >
              Clear
            </button>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:gap-x-3 sm:gap-y-2">
            <span className="font-sans text-[10px] font-medium uppercase tracking-[0.2em] text-text-tertiary">
              Examples
            </span>
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.label}
                  type="button"
                  className="rounded-full border border-border-primary bg-transparent px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-text-tertiary hover:text-text-primary"
                  onClick={() => {
                    capturePostHogEvent(POSTHOG_EVENTS.playgroundExamplePromptUsed, {
                      surface: "playground",
                      mode: "tts",
                      example_id: ex.label
                    });
                    setText(ex.text);
                  }}
                >
                  {ex.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="w-full shrink-0 space-y-3 md:w-[360px]">
          <p className="font-sans text-[10px] font-medium uppercase tracking-[0.28em] text-text-tertiary">
            Models
          </p>
          <div className="flex flex-wrap gap-2">
            {modelsTagCloudOrder.map((m) => (
              <ModelPill
                key={m.id}
                label={getTtsPillLabel(m)}
                providerLabel={formatProviderDisplayName(m.provider)}
                visual={getPlaygroundModelVisual(m)}
                selected={selectedMap[m.id] !== false}
                disabled={hasInFlight}
                onToggle={() => toggleModel(m.id)}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-4 border-t border-border-primary pt-6 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-2">
          <p className="font-sans text-[10px] font-medium uppercase tracking-[0.2em] text-text-tertiary">
            metrics
          </p>
          <div className="flex flex-wrap gap-2" aria-label="Metrics included in benchmark">
            {(["TTFA", "Total latency"] as const).map((label) => (
              <span
                key={label}
                className="rounded-full border border-border-primary px-3 py-1.5 text-xs font-medium text-text-primary dark:border-white/20 dark:text-white"
              >
                {label}
              </span>
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          {Object.keys(generationRows).length > 0 && !hasInFlight ? (
            <button
              type="button"
              onClick={() => setBenchmarkOpen(true)}
              className="inline-flex items-center justify-center gap-2 self-stretch rounded-full border border-border-primary px-5 py-2.5 text-sm font-medium text-text-primary transition-colors hover:bg-hover-bg focus-visible:ring-2 focus-visible:ring-text-tertiary/30 sm:self-auto"
            >
              View results
            </button>
          ) : null}
          <button
            type="button"
            disabled={!canBenchmark || hasInFlight}
            onClick={handleBenchmarkButtonClick}
            title={
              canBenchmark
                ? "Enter runs benchmark when focus is outside fields, or ⌘/Ctrl+Enter from the text box"
                : undefined
            }
            className="inline-flex items-center justify-center gap-2 self-stretch rounded-full border border-border-primary px-5 py-2.5 text-sm font-medium text-text-primary transition-colors hover:bg-hover-bg disabled:cursor-not-allowed disabled:opacity-40 sm:self-auto"
          >
            <Play className="size-4 shrink-0" aria-hidden />
            Benchmark
          </button>
        </div>
      </div>

      {benchmarkOpen && portalRoot
        ? createPortal(
            <div
              className="fixed inset-0 isolate z-[100] flex items-center justify-center bg-black/80 p-4 backdrop-blur-[3px]"
              role="presentation"
              data-playground-benchmark-overlay
              onMouseDown={(e) => {
                if (e.target === e.currentTarget) setBenchmarkOpen(false);
              }}
            >
              <div
                ref={dialogRef}
                role="dialog"
                aria-modal="true"
                aria-labelledby="tts-benchmark-title"
                className="playground-modal-base relative flex max-h-[min(88vh,640px)] w-full max-w-md flex-col overflow-hidden rounded-2xl shadow-2xl sm:max-w-xl"
                onMouseDown={(e) => e.stopPropagation()}
                onKeyDown={onDialogKeyDown}
              >
                <div className="flex shrink-0 items-start justify-between gap-3 border-b playground-modal-row-divider px-4 pb-0 pt-4 sm:px-5 sm:pt-5">
                  <div className="min-w-0 pr-2 pb-3 sm:pb-4">
                    <h2 id="tts-benchmark-title" className="text-base font-semibold tracking-tight text-text-primary">
                      Benchmark results
                    </h2>
                  </div>
                  <button
                    type="button"
                    aria-label="Close benchmark results"
                    className="mb-3 flex size-9 shrink-0 items-center justify-center rounded-full border playground-modal-border-subtle text-text-tertiary transition-colors hover:bg-hover-bg hover:text-text-primary sm:mb-4 dark:text-zinc-400 dark:hover:bg-white/5 dark:hover:text-white"
                    onClick={() => setBenchmarkOpen(false)}
                  >
                    <X className="size-4" />
                  </button>
                </div>
                <div className="min-h-0 flex-1 overflow-x-auto overflow-y-auto px-4 pb-2 pt-0 sm:px-5 sm:pb-2">
                  <div
                    className={`sticky top-0 z-10 flex ${BENCH_ROW_MINW} min-h-[3.75rem] items-center ${BENCH_ROW_GAP} border-b playground-modal-row-divider playground-modal-table-head ${BENCH_ROW_PY}`}
                  >
                    <div className={BENCH_PLAY_HEAD_SPACER} aria-hidden />
                    <div className="min-w-0 flex-1 self-center">
                      <p className={`${BENCH_HEAD_TEXT} text-left`}>Model</p>
                    </div>
                    <div className={`flex shrink-0 items-end ${BENCH_METRICS_GAP}`}>
                      {visibleMetricKeys.map((k) => (
                        <div key={k} className={`${BENCH_METRIC_COL} text-right`}>
                          <p className={BENCH_HEAD_TEXT}>{metricHeaderLabel(k)}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                  {rows.length === 0 ? (
                    process.env.NODE_ENV === "development" ? (
                      <div className="py-4 text-xs text-text-secondary space-y-1">
                        <p className="font-semibold">Dev: all models failed</p>
                        {activeModels.map((model) => {
                          const row = generationRows[model.id];
                          if (row?.status !== "error" || !row.error) return null;
                          return (
                            <p key={model.id}>
                              <span className="font-medium">{model.label}</span> — {row.error.code}: {row.error.error}
                            </p>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="py-4 text-sm text-text-secondary">
                        Audio generation failed. Please try again.
                      </div>
                    )
                  ) : rows.map((row) => {
                    const playing = playingModelId === row.model.id;
                    return (
                      <div
                        key={row.model.id}
                        className={`flex ${BENCH_ROW_MINW} items-start ${BENCH_ROW_GAP} border-b playground-modal-row-divider ${BENCH_ROW_PY} last:border-b-0`}
                      >
                        <div className={`${BENCH_PLAY_COL} pt-0.5`}>
                          <button
                            type="button"
                            className="text-text-tertiary transition-colors hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-text-tertiary/30 dark:text-zinc-500 dark:hover:text-zinc-200"
                            aria-label={playing ? `Pause ${row.model.label}` : `Play ${row.model.label}`}
                            aria-pressed={playing}
                            onClick={() => {
                              if (!playing && row.audioUrl) {
                                capturePostHogEvent(POSTHOG_EVENTS.playgroundResultPlayed, {
                                  surface: "playground",
                                  mode: "tts",
                                  model_id: row.model.id
                                });
                              }
                              toggle(row.model.id, row.audioUrl);
                            }}
                          >
                            {playing ? (
                              <Pause className="size-3" aria-hidden strokeWidth={2.25} />
                            ) : (
                              <Play className="size-3" aria-hidden strokeWidth={2.25} />
                            )}
                          </button>
                        </div>
                        <div className="min-w-0 flex-1 space-y-0.5 pt-0.5">
                          <span className="text-sm font-semibold leading-snug text-text-primary dark:text-white">
                            {row.model.label}
                          </span>
                          <p className="text-xs leading-snug text-text-secondary">
                            {formatProviderDisplayName(row.model.provider)}
                          </p>
                        </div>
                        <div className={`flex shrink-0 items-start ${BENCH_METRICS_GAP} pt-0.5`}>
                          {visibleMetricKeys.map((k) => renderMetricCell(k, row))}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="shrink-0 border-t playground-modal-row-divider px-4 py-3 sm:px-5 sm:py-4">
                  <button
                    ref={doneBtnRef}
                    type="button"
                    title="Close (Enter)"
                    className="w-full rounded-lg py-2.5 text-xs font-medium text-text-secondary transition-colors hover:bg-hover-bg hover:text-text-primary sm:text-sm dark:text-zinc-400 dark:hover:bg-white/5 dark:hover:text-white"
                    onClick={() => setBenchmarkOpen(false)}
                  >
                    Done
                  </button>
                </div>
              </div>
            </div>,
            portalRoot
          )
        : null}
    </section>
  );
}
