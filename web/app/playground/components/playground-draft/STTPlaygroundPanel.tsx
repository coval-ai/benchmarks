"use client";

import {
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import { createPortal } from "react-dom";
import { ChevronLeft, ChevronRight, Mic, X } from "lucide-react";
import type { SttModelConfig } from "@/lib/playground/providers";
import {
  formatProviderDisplayName,
  getPlaygroundModelVisual,
  getSttPillLabel
} from "@/lib/playground/provider-styles";
import { isTypingInteractionTarget } from "@/lib/playground/hotkeys";
import { ModelPill } from "./ModelPill";
import { SttTrianglePulseCanvas } from "./SttTrianglePulseCanvas";
import { useSTTBenchmark, type STTBenchmarkCompletionSummary } from "@/app/playground/hooks/useSTTBenchmark";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";

type SttMetricKey = "ttfa" | "audioToFinal";

const STT_METRICS_ALL: Record<SttMetricKey, boolean> = {
  ttfa: true,
  audioToFinal: true
};

const BENCH_METRIC_COL = "min-w-[4.25rem] w-[4.25rem] sm:min-w-[4.75rem] sm:w-[4.75rem]";
const BENCH_METRICS_GAP = "gap-3 sm:gap-4";
const BENCH_ROW_GAP = "gap-3";
const BENCH_ROW_MINW = "min-w-[min(100%,520px)]";
const BENCH_HEAD_TEXT =
  "font-sans text-[10px] font-medium uppercase tracking-[0.14em] text-text-tertiary";
const BENCH_ROW_PY = "py-2.5 sm:py-3";

type LeaderRow = {
  model: SttModelConfig;
  ttfaMs: number | null;
  audioToFinalMs: number;
  transcript: string;
  error?: string;
  pending: boolean;
};

function sortMetricValue(row: LeaderRow, key: SttMetricKey): number {
  if (key === "ttfa") return row.ttfaMs ?? Number.POSITIVE_INFINITY;
  return row.audioToFinalMs;
}

/** Left/right fades + chevron visibility from scroll extents; `activityKey` should change when strip content changes. */
function useCarouselScrollHints(
  ref: RefObject<HTMLDivElement | null>,
  activityKey: string
): { left: boolean; right: boolean } {
  const [hints, setHints] = useState({ left: false, right: false });

  useEffect(() => {
    const el = ref.current;
    if (!el || !activityKey) {
      setHints({ left: false, right: false });
      return;
    }

    const measure = () => {
      requestAnimationFrame(() => {
        const node = ref.current;
        if (!node) return;
        const { scrollLeft, scrollWidth, clientWidth } = node;
        const maxScroll = scrollWidth - clientWidth;
        const eps = 8;
        if (maxScroll <= eps) {
          setHints({ left: false, right: false });
          return;
        }
        setHints({
          left: scrollLeft > eps,
          right: scrollLeft < maxScroll - eps
        });
      });
    };

    measure();
    el.addEventListener("scroll", measure, { passive: true });
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    const inner = el.firstElementChild as HTMLElement | null;
    if (inner) ro.observe(inner);

    return () => {
      el.removeEventListener("scroll", measure);
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `ref` from useRef is stable; only activityKey should rebind.
  }, [activityKey]);

  return hints;
}

function sttRankingMetric(): SttMetricKey {
  return "ttfa";
}

export function STTPlaygroundPanel({
  models,
  onBenchmarkOverlayChange
}: {
  models: SttModelConfig[];
  onBenchmarkOverlayChange?: (open: boolean) => void;
}) {
  const modelsById = useMemo(
    () => new Map(models.map((model) => [model.id, model])),
    [models]
  );
  const handleBenchmarkComplete = useCallback((summary: STTBenchmarkCompletionSummary) => {
    const ttfas = summary.successes
      .map((s) => s.ttfaMs)
      .filter((v): v is number => v != null);
    const finals = summary.successes.map((s) => s.audioToFinalMs);
    capturePostHogEvent(POSTHOG_EVENTS.playgroundBenchmarkCompleted, {
      surface: "playground",
      mode: "stt",
      selected_model_ids: summary.modelIds,
      selected_model_count: summary.modelIds.length,
      success_count: summary.successes.length,
      failure_count: summary.failures.length,
      audio_duration_ms: Math.round(summary.audioDurationMs),
      best_ttfa_ms: ttfas.length ? Math.min(...ttfas) : null,
      best_audio_to_final_ms: finals.length ? Math.min(...finals) : null,
      is_comparison: summary.modelIds.length >= 2
    });
  }, []);
  const { phase, results, errors, sessionError, start, stop, reset } = useSTTBenchmark({
    onComplete: handleBenchmarkComplete
  });
  /**
   * User picks which **enabled** models participate in compare. `selectedMap` does not gate the
   * server allowlist — parent should pass configs consistent with `/v1/providers` + runner matrix;
   * `visibleModels` below filters `m.enabled`, so disabled catalogue rows never appear in UI.
   */
  const [selectedMap, setSelectedMap] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(models.map((m) => [m.id, true]))
  );
  const [leaderOpen, setLeaderOpen] = useState(false);
  const [portalRoot, setPortalRoot] = useState<HTMLElement | null>(null);
  useEffect(() => {
    setPortalRoot(document.body);
  }, []);

  useEffect(() => {
    onBenchmarkOverlayChange?.(leaderOpen);
  }, [leaderOpen, onBenchmarkOverlayChange]);

  useEffect(() => {
    return () => onBenchmarkOverlayChange?.(false);
  }, [onBenchmarkOverlayChange]);

  useEffect(() => {
    setSelectedMap((prev) => {
      const next = { ...prev };
      for (const m of models) {
        if (!(m.id in next)) next[m.id] = true;
      }
      return next;
    });
  }, [models]);

  const visibleModels = useMemo(() => models.filter((m) => m.enabled), [models]);

  const visibleModelsTagCloudOrder = useMemo(
    () =>
      [...visibleModels].sort((a, b) =>
        getSttPillLabel(a).localeCompare(getSttPillLabel(b), undefined, { sensitivity: "base" })
      ),
    [visibleModels]
  );

  const activeModels = useMemo(
    () => visibleModels.filter((m) => selectedMap[m.id] !== false),
    [visibleModels, selectedMap]
  );

  const carouselViewportRef = useRef<HTMLDivElement>(null);
  const carouselTrackRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const transcriptBrowsedFiredRef = useRef(false);

  const carouselActivityKey =
    activeModels.length > 0 ? activeModels.map((m) => m.id).join(":") : "";

  const carouselHints = useCarouselScrollHints(carouselViewportRef, carouselActivityKey);

  const scrollCarouselStep = useCallback((dir: -1 | 1) => {
    const vp = carouselViewportRef.current;
    if (!vp) return;
    if (!transcriptBrowsedFiredRef.current) {
      transcriptBrowsedFiredRef.current = true;
      capturePostHogEvent(POSTHOG_EVENTS.sttTranscriptBrowsed, {
        surface: "playground",
        mode: "stt",
        method: "arrow"
      });
    }
    const first = vp.querySelector('[role="listitem"]') as HTMLElement | null;
    const step = first ? first.offsetWidth + 16 : 256;
    vp.scrollBy({ left: dir * step, behavior: "smooth" });
  }, []);

  useEffect(() => {
    if (!carouselActivityKey) return;
    const vp = carouselViewportRef.current;
    if (!vp) return;
    const onScroll = () => {
      if (transcriptBrowsedFiredRef.current) return;
      transcriptBrowsedFiredRef.current = true;
      capturePostHogEvent(POSTHOG_EVENTS.sttTranscriptBrowsed, {
        surface: "playground",
        mode: "stt",
        method: "scroll"
      });
    };
    vp.addEventListener("scroll", onScroll, { passive: true });
    return () => vp.removeEventListener("scroll", onScroll);
  }, [carouselActivityKey]);

  useEffect(() => {
    if (!carouselActivityKey) return;
    const vp = carouselViewportRef.current;
    const trackMaybe = carouselTrackRef.current;
    if (!vp || !trackMaybe) return;
    const trackEl = trackMaybe;

    const reduceMq =
      typeof window !== "undefined" ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
    const prefersReduce = () => reduceMq?.matches === true;

    let touching = false;
    let lastX = 0;
    let pullAccum = 0;

    const maxSL = () => Math.max(0, vp.scrollWidth - vp.clientWidth);

    function springRelease() {
      const reduced = prefersReduce();
      pullAccum = 0;
      trackEl.style.transition = reduced ? "none" : "transform 0.42s cubic-bezier(0.33, 1.22, 0.52, 1)";
      trackEl.style.transform = "translateX(0)";
      if (!reduced) {
        window.setTimeout(() => {
          trackEl.style.transition = "";
        }, 440);
      } else {
        trackEl.style.transition = "";
      }
    }

    const onStart = (e: TouchEvent) => {
      if (!e.touches[0]) return;
      touching = true;
      lastX = e.touches[0].pageX;
      pullAccum = 0;
      trackEl.style.transition = "none";
      trackEl.style.transform = "translateX(0)";
    };

    const onMove = (e: TouchEvent) => {
      if (!touching || !e.touches[0]) return;
      const x = e.touches[0].pageX;
      const delta = x - lastX;
      lastX = x;

      const sl = vp.scrollLeft;
      const m = maxSL();
      if (m < 12 || prefersReduce()) return;

      const atLeft = sl <= 2;
      const atRight = m >= 12 && sl >= m - 2;

      if (atLeft && delta > 0) {
        pullAccum = Math.min(pullAccum + delta * 0.42, 30);
      } else if (atRight && delta < 0) {
        pullAccum = Math.max(pullAccum + delta * 0.42, -30);
      } else {
        pullAccum = 0;
      }

      trackEl.style.transform =
        pullAccum !== 0 ? `translateX(${pullAccum}px)` : "translateX(0)";
    };

    const onEnd = () => {
      if (!touching) return;
      touching = false;
      springRelease();
    };

    vp.addEventListener("touchstart", onStart, { passive: true });
    vp.addEventListener("touchmove", onMove, { passive: true });
    vp.addEventListener("touchend", onEnd);
    vp.addEventListener("touchcancel", onEnd);

    return () => {
      vp.removeEventListener("touchstart", onStart);
      vp.removeEventListener("touchmove", onMove);
      vp.removeEventListener("touchend", onEnd);
      vp.removeEventListener("touchcancel", onEnd);
      trackEl.style.transition = "";
      trackEl.style.transform = "";
    };
  }, [carouselActivityKey]);

  const leaderRows = useMemo<LeaderRow[]>(() => {
    const rankingMetric = sttRankingMetric();
    const rows: LeaderRow[] = activeModels.map((model) => {
      const result = results.get(model.id);
      const error = errors.get(model.id);
      return {
        model,
        ttfaMs: result?.ttfaMs ?? null,
        audioToFinalMs: result?.audioToFinalMs ?? 0,
        transcript: result?.transcript ?? "",
        error,
        pending: !result && !error,
      };
    });
    return [...rows].sort((a, b) => {
      // Completed (success) rows first, then pending (transcribing), then errors.
      const aRank = a.error ? 2 : a.pending ? 1 : 0;
      const bRank = b.error ? 2 : b.pending ? 1 : 0;
      if (aRank !== bRank) return aRank - bRank;
      const byMetric = sortMetricValue(a, rankingMetric) - sortMetricValue(b, rankingMetric);
      if (byMetric !== 0) return byMetric;
      return a.audioToFinalMs - b.audioToFinalMs;
    });
  }, [activeModels, results, errors]);

  const toggleModel = useCallback((id: string) => {
    if (!modelsById.has(id)) return;
    if (phase !== "idle") return;
    const willBeSelected = !selectedMap[id];
    const nextMap = { ...selectedMap, [id]: willBeSelected };
    const selectedIds = visibleModels.filter((m) => nextMap[m.id] !== false).map((m) => m.id);
    capturePostHogEvent(POSTHOG_EVENTS.playgroundModelSelectionChanged, {
      surface: "playground",
      mode: "stt",
      action: willBeSelected ? "add" : "remove",
      model_id: id,
      selected_model_ids: selectedIds,
      selected_model_count: selectedIds.length,
      is_comparison: selectedIds.length >= 2
    });
    setSelectedMap(nextMap);
  }, [modelsById, phase, selectedMap, visibleModels]);

  useEffect(() => {
    if (!leaderOpen) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    closeButtonRef.current?.focus({ preventScroll: true });
    return () => {
      previouslyFocused?.focus?.({ preventScroll: true });
    };
  }, [leaderOpen]);

  useEffect(() => {
    if (!leaderOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setLeaderOpen(false);
        return;
      }
      if (e.key === "Enter") {
        e.preventDefault();
        setLeaderOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [leaderOpen]);

  const canStartOrRetakeRecording = activeModels.length > 0;
  /** True while the current recording was started by holding Enter (so keyup only stops that session). */
  const enterHoldArmRef = useRef(false);

  useEffect(() => {
    if ((phase === "submitting" || phase === "complete") && activeModels.length > 0) {
      setLeaderOpen(true);
    }
  }, [phase, activeModels.length]);

  const handleRecord = useCallback(() => {
    if (phase === "recording") {
      enterHoldArmRef.current = false;
      stop();
      return;
    }
    if (phase === "complete") {
      if (!canStartOrRetakeRecording) return;
      enterHoldArmRef.current = false;
      setLeaderOpen(false);
      reset();
      return;
    }
    if (phase === "submitting") return;
    if (phase === "error") {
      enterHoldArmRef.current = false;
      reset();
      setLeaderOpen(false);
      return;
    }
    if (!canStartOrRetakeRecording) return;
    enterHoldArmRef.current = false;
    const selectedIds = activeModels.map((m) => m.id);
    capturePostHogEvent(POSTHOG_EVENTS.playgroundSttRecordPressed, {
      surface: "playground",
      mode: "stt",
      selected_model_ids: selectedIds,
      selected_model_count: selectedIds.length,
      trigger: "button",
      is_comparison: selectedIds.length >= 2
    });
    void start(selectedIds);
  }, [phase, activeModels, canStartOrRetakeRecording, start, stop, reset]);

  const phaseRef = useRef(phase);
  phaseRef.current = phase;
  const leaderOpenRef = useRef(leaderOpen);
  leaderOpenRef.current = leaderOpen;
  const activeModelsLenRef = useRef(activeModels.length);
  activeModelsLenRef.current = activeModels.length;
  const activeModelsRef = useRef(activeModels);
  activeModelsRef.current = activeModels;

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Enter") return;
      if (e.repeat) return;
      if (leaderOpenRef.current) return;
      if (isTypingInteractionTarget(e.target)) return;
      const p = phaseRef.current;
      if (p === "recording" || p === "submitting") return;
      if (activeModelsLenRef.current === 0) return;
      e.preventDefault();
      if (p === "complete") setLeaderOpen(false);
      enterHoldArmRef.current = true;
      const selectedIds = activeModelsRef.current.map((model) => model.id);
      capturePostHogEvent(POSTHOG_EVENTS.playgroundSttRecordPressed, {
        surface: "playground",
        mode: "stt",
        selected_model_ids: selectedIds,
        selected_model_count: selectedIds.length,
        trigger: "keyboard",
        is_comparison: selectedIds.length >= 2
      });
      void start(selectedIds);
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key !== "Enter") return;
      if (!enterHoldArmRef.current) return;
      enterHoldArmRef.current = false;
      if (phaseRef.current !== "recording") return;
      stop();
    };

    const onBlur = () => {
      if (!enterHoldArmRef.current) return;
      enterHoldArmRef.current = false;
      if (phaseRef.current === "recording") stop();
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
    };
  }, [start, stop]);

  const statusLabel =
    phase === "idle"
      ? "Ready to record"
      : phase === "recording"
        ? "Listening…"
        : phase === "submitting"
          ? "Transcribing…"
          : phase === "error"
            ? (sessionError ?? "Error")
            : "Recording complete";

  const visibleMetricKeys = useMemo(
    (): SttMetricKey[] => (["ttfa", "audioToFinal"] as const).filter((k) => STT_METRICS_ALL[k]),
    []
  );

  const metricHeaderLabel = (key: SttMetricKey) => {
    if (key === "ttfa") return "First word";
    return "Final";
  };

  const renderMetricCell = (key: SttMetricKey, row: LeaderRow) => {
    if (key === "ttfa") {
      return (
        <div key="ttfa" className={`${BENCH_METRIC_COL} text-right`}>
          <p className="text-xs font-medium tabular-nums leading-snug text-text-primary">
            {row.ttfaMs !== null && row.ttfaMs > 0 ? `${row.ttfaMs}ms` : "—"}
          </p>
        </div>
      );
    }
    if (key === "audioToFinal") {
      return (
        <div key="audioToFinal" className={`${BENCH_METRIC_COL} text-right`}>
          <p className="text-xs font-medium tabular-nums leading-snug text-text-primary">
            {row.audioToFinalMs > 0 ? `${row.audioToFinalMs}ms` : "—"}
          </p>
        </div>
      );
    }
  };

  return (
    <section
      id="playground-panel-stt"
      role="tabpanel"
      aria-labelledby="tab-playground-stt"
      className="space-y-8"
    >
      <div className="flex flex-col gap-8 md:flex-row md:items-start md:justify-between">
        <div className="flex min-w-0 flex-1 flex-col items-center space-y-4">
          <div className="playground-viz-ring">
            <SttTrianglePulseCanvas
              recording={phase === "recording"}
              className="pointer-events-none absolute inset-0 size-full rounded-full"
            />
          </div>
          <p className="text-center text-sm text-text-secondary">{statusLabel}</p>
        </div>

        <div className="w-full min-w-0 shrink-0 space-y-3 md:w-[360px] md:pt-2">
          <p className="font-sans text-[10px] font-medium uppercase tracking-[0.28em] text-text-tertiary">
            Models
          </p>
          <div className="flex flex-wrap gap-2">
            {visibleModelsTagCloudOrder.map((m) => (
              <ModelPill
                key={m.id}
                label={getSttPillLabel(m)}
                providerLabel={formatProviderDisplayName(m.provider)}
                visual={getPlaygroundModelVisual(m)}
                selected={selectedMap[m.id] !== false}
                disabled={phase !== "idle"}
                onToggle={() => toggleModel(m.id)}
              />
            ))}
          </div>
        </div>
      </div>

      {activeModels.length > 0 ? (
        <div className="relative">
          {carouselHints.left ? (
            <>
              <div
                className="playground-stt-carousel-fade playground-stt-carousel-fade--left"
                aria-hidden
              />
              <button
                type="button"
                className="absolute left-0.5 top-1/2 z-[3] flex size-10 -translate-y-1/2 items-center justify-center rounded-full bg-surface-secondary/90 text-text-primary shadow-md ring-1 ring-border-primary backdrop-blur-[1px] transition-opacity hover:opacity-95 dark:bg-black/55 dark:text-white/90 dark:ring-white/12"
                aria-label="Swipe or scroll transcripts to the left"
                onClick={() => scrollCarouselStep(-1)}
              >
                <ChevronLeft className="size-6 shrink-0 opacity-90 drop-shadow-[0_1px_2px_rgba(0,0,0,0.45)] dark:opacity-100" aria-hidden />
              </button>
            </>
          ) : null}
          {carouselHints.right ? (
            <>
              <div
                className="playground-stt-carousel-fade playground-stt-carousel-fade--right"
                aria-hidden
              />
              <button
                type="button"
                className="absolute right-0.5 top-1/2 z-[3] flex size-10 -translate-y-1/2 items-center justify-center rounded-full bg-surface-secondary/90 text-text-primary shadow-md ring-1 ring-border-primary backdrop-blur-[1px] transition-opacity hover:opacity-95 dark:bg-black/55 dark:text-white/90 dark:ring-white/12"
                aria-label="Swipe or scroll transcripts to the right for more models"
                onClick={() => scrollCarouselStep(1)}
              >
                <ChevronRight className="size-6 shrink-0 opacity-90 drop-shadow-[0_1px_2px_rgba(0,0,0,0.45)] dark:opacity-100" aria-hidden />
              </button>
            </>
          ) : null}
        <div
          ref={carouselViewportRef}
          className="playground-stt-carousel-viewport scrollbar-hide overflow-x-auto snap-x snap-mandatory pb-2"
          aria-label="Transcripts by model"
          role="list"
        >
          <div ref={carouselTrackRef} className="flex min-w-full w-max gap-4 will-change-transform">
          {activeModels.map((m, panelIndex) => {
            const v = getPlaygroundModelVisual(m);
            const result = results.get(m.id);
            const err = errors.get(m.id);
            const audioToFinalMs = result?.audioToFinalMs ?? 0;
            const bodyText =
              phase === "submitting"
                ? "Transcribing…"
                : phase === "complete"
                  ? (result?.transcript ?? err ?? "")
                  : phase === "recording"
                    ? "Listening…"
                    : "";
            const cornerIsTiming = phase === "complete" || phase === "submitting";
            const cornerContent =
              cornerIsTiming && audioToFinalMs > 0 ? `${audioToFinalMs}ms` : phase === "complete" && err ? "—" : "—";
            return (
              <div
                key={m.id}
                role="listitem"
                className="flex min-h-[160px] w-[240px] min-w-[240px] max-w-[240px] shrink-0 snap-start flex-col rounded-xl border border-border-primary bg-surface-secondary p-3"
              >
                <div className="mb-2 flex flex-row items-center justify-between gap-2">
                  <div className="flex min-w-0 flex-row items-center gap-1.5">
                    <div
                      className={`playground-stt-dot shrink-0 ${
                        phase === "recording" ? "playground-stt-dot--recording" : ""
                      }`}
                      style={{
                        background: v.dot,
                        animationDelay: `${panelIndex * 0.15}s`
                      }}
                      aria-hidden
                    />
                    <span className="truncate text-xs font-semibold leading-none text-text-primary">
                      {getSttPillLabel(m)}
                    </span>
                  </div>
                  <span
                    className={`shrink-0 tabular-nums ${
                      cornerIsTiming ? "playground-mono text-[11px] text-text-tertiary" : "text-[13px] text-[color:var(--color-border-secondary)]"
                    }`}
                  >
                    {cornerContent}
                  </span>
                </div>
                <p
                  className={`playground-mono flex flex-1 text-[12px] leading-[1.6] ${
                    phase === "idle" ? "" : "text-text-primary"
                  }`}
                  aria-live={phase === "recording" ? "polite" : "off"}
                >
                  {phase === "idle" ? null : bodyText}
                </p>
              </div>
            );
          })}
          </div>
        </div>
        </div>
      ) : null}

      <div className="flex flex-col gap-4 border-t border-border-primary pt-6 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-2">
          <p className="font-sans text-[10px] font-medium uppercase tracking-[0.2em] text-text-tertiary">
            metrics
          </p>
          <div className="flex flex-wrap gap-2" aria-label="Metrics included in results">
            {(["First word", "Final"] as const).map((label) => (
              <span
                key={label}
                className="rounded-full border border-border-primary px-3 py-1.5 text-xs font-medium text-text-primary dark:border-white/20 dark:text-white"
              >
                {label}
              </span>
            ))}
          </div>
          <p className="font-sans text-[10px] text-text-tertiary">
            Hold Enter to record, release to stop (when focus isn&apos;t in a control) · Ranking by first recognized word after upload (not live TTFT)
          </p>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          {phase === "complete" && leaderRows.length > 0 ? (
            <button
              type="button"
              onClick={() => setLeaderOpen(true)}
              className="inline-flex items-center justify-center gap-2 self-stretch rounded-full border border-border-primary px-5 py-2.5 text-sm font-medium text-text-primary transition-colors hover:bg-hover-bg focus-visible:ring-2 focus-visible:ring-text-tertiary/30 sm:self-auto"
            >
              View results
            </button>
          ) : null}
          <button
            type="button"
            onClick={handleRecord}
            disabled={
              ((phase === "idle" || phase === "complete" || phase === "error") &&
                !canStartOrRetakeRecording) ||
              phase === "submitting"
            }
            title={
              (phase === "idle" || phase === "complete") && !canStartOrRetakeRecording
                ? "Turn on at least one model to record a comparison."
                : "Hold Enter to record, release to stop when focus is outside buttons and fields"
            }
            className={`inline-flex items-center justify-center gap-2 self-stretch rounded-full border px-5 py-2.5 text-sm font-medium transition-colors sm:self-auto ${
              ((phase === "idle" || phase === "complete" || phase === "error") &&
                !canStartOrRetakeRecording) ||
              phase === "submitting"
                ? "cursor-not-allowed border-border-primary text-text-tertiary opacity-60"
                : "border-border-primary text-text-primary hover:bg-hover-bg"
            }`}
          >
            {phase === "recording" ? (
              <span className="relative flex size-2.5 shrink-0 items-center justify-center">
                <span className="absolute size-2.5 animate-pulse rounded-full bg-red-500" />
              </span>
            ) : (
              <Mic className="size-4 shrink-0 text-text-secondary" aria-hidden />
            )}
            {phase === "recording"
              ? "Stop"
              : phase === "submitting"
                ? "Transcribing…"
                : phase === "complete"
                  ? "New take"
                  : "Record"}
          </button>
        </div>
      </div>

      {leaderOpen && leaderRows.length > 0 && portalRoot
        ? createPortal(
            <div
              className="fixed inset-0 isolate z-[100] flex items-center justify-center bg-black/80 p-4 backdrop-blur-[3px]"
              role="presentation"
              data-playground-benchmark-overlay
              onMouseDown={(e) => {
                if (e.target === e.currentTarget) setLeaderOpen(false);
              }}
            >
              <div
                role="dialog"
                aria-modal="true"
                aria-labelledby="stt-results-title"
                className="playground-modal-base relative flex max-h-[min(88vh,640px)] w-full max-w-md flex-col overflow-hidden rounded-2xl shadow-2xl sm:max-w-xl"
                onMouseDown={(e) => e.stopPropagation()}
              >
                <div className="flex shrink-0 items-start justify-between gap-3 border-b playground-modal-row-divider px-4 pb-0 pt-4 sm:px-5 sm:pt-5">
                  <div className="min-w-0 pr-2 pb-3 sm:pb-4">
                    <h2 id="stt-results-title" className="text-base font-semibold tracking-tight text-text-primary">
                      Benchmark results
                    </h2>
                  </div>
                  <button
                    ref={closeButtonRef}
                    type="button"
                    aria-label="Close results"
                    className="mb-3 flex size-9 shrink-0 items-center justify-center rounded-full border playground-modal-border-subtle text-text-tertiary transition-colors hover:bg-hover-bg hover:text-text-primary sm:mb-4 dark:text-zinc-400 dark:hover:bg-white/5 dark:hover:text-white"
                    onClick={() => setLeaderOpen(false)}
                  >
                    <X className="size-4" />
                  </button>
                </div>

                <div className="min-h-0 flex-1 overflow-x-auto overflow-y-auto px-4 pb-2 pt-0 sm:px-5 sm:pb-2">
                  <div
                    className={`sticky top-0 z-10 flex ${BENCH_ROW_MINW} min-h-[3.75rem] items-center ${BENCH_ROW_GAP} border-b playground-modal-row-divider playground-modal-table-head ${BENCH_ROW_PY}`}
                  >
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
                  {leaderRows.map((row) => (
                    <div
                      key={row.model.id}
                      className={`flex ${BENCH_ROW_MINW} items-start ${BENCH_ROW_GAP} border-b playground-modal-row-divider ${BENCH_ROW_PY} last:border-b-0`}
                    >
                      <div className="min-w-0 flex-1 space-y-0.5 pt-0.5">
                        <span className="text-sm font-semibold leading-snug text-text-primary dark:text-white">
                          {getSttPillLabel(row.model)}
                        </span>
                        <p className="text-xs leading-snug text-text-secondary">
                          {formatProviderDisplayName(row.model.provider)}
                        </p>
                      </div>
                      <div className={`flex shrink-0 items-start ${BENCH_METRICS_GAP} pt-0.5`}>
                        {row.pending ? (
                          <p className="text-right text-xs text-text-tertiary">Transcribing…</p>
                        ) : row.error ? (
                          <p className="max-w-[12rem] text-right text-xs text-red-400">{row.error}</p>
                        ) : (
                          visibleMetricKeys.map((k) => renderMetricCell(k, row))
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="shrink-0 border-t playground-modal-row-divider px-4 py-3 sm:px-5 sm:py-4">
                  <button
                    type="button"
                    title="Close (Enter)"
                    className="w-full rounded-lg py-2.5 text-xs font-medium text-text-secondary transition-colors hover:bg-hover-bg hover:text-text-primary sm:text-sm dark:text-zinc-400 dark:hover:bg-white/5 dark:hover:text-white"
                    onClick={() => setLeaderOpen(false)}
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
