// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { type RefObject, useEffect, useMemo, useRef, useState } from "react";
import { useSequencedPlayback, type PlaybackTrack } from "@/hooks/useSequencedPlayback";
import { getModelColor } from "@/lib/utils/colors";
import { toModelKey } from "@/lib/utils/formatters";

export interface SampleOutputItem {
  provider: string;
  model: string;
  url: string;
}

// Left/right chevron/fade visibility from scroll extents; remeasures on scroll,
// resize, and when the item set changes.
function useScrollHints(
  ref: RefObject<HTMLDivElement | null>,
  activityKey: string
): { left: boolean; right: boolean } {
  const [hints, setHints] = useState({ left: false, right: false });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const { scrollLeft, scrollWidth, clientWidth } = el;
      const maxScroll = scrollWidth - clientWidth;
      const eps = 8;
      if (maxScroll <= eps) {
        setHints({ left: false, right: false });
        return;
      }
      setHints({ left: scrollLeft > eps, right: scrollLeft < maxScroll - eps });
    };
    measure();
    el.addEventListener("scroll", measure, { passive: true });
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => {
      el.removeEventListener("scroll", measure);
      ro.disconnect();
    };
  }, [ref, activityKey]);
  return hints;
}

export function SampleOutputs({
  items,
  normalizeProvider,
  onPlay,
  playRequest,
}: {
  items: SampleOutputItem[];
  normalizeProvider: (provider: string) => string;
  onPlay?: (provider: string) => void;
  playRequest?: { provider: string; nonce: number } | null;
}) {
  const tracks = useMemo<PlaybackTrack[]>(
    () => items.map((i) => ({ key: i.provider, url: i.url })),
    [items]
  );
  const viewportRef = useRef<HTMLDivElement>(null);
  const paneRefs = useRef<(HTMLDivElement | null)[]>([]);

  const { audioRef, activeIndex, isPlaying, toggle, playFrom, handleEnded } = useSequencedPlayback(
    tracks,
    (track) => onPlay?.(track.key)
  );

  // A timeline-tooltip click plays a specific provider. Wait until that tick's
  // items have loaded (provider present) before playing, then mark the request
  // consumed — so the async manifest swap doesn't play the previous tick.
  const consumedNonce = useRef<number | null>(null);
  useEffect(() => {
    if (!playRequest || consumedNonce.current === playRequest.nonce) return;
    const idx = items.findIndex((i) => i.provider === playRequest.provider);
    if (idx < 0) return;
    consumedNonce.current = playRequest.nonce;
    playFrom(idx);
  }, [playRequest, items, playFrom]);

  // Auto-scroll the active pane to center while a sequence is playing.
  useEffect(() => {
    if (!isPlaying) return;
    const vp = viewportRef.current;
    const pane = paneRefs.current[activeIndex];
    if (!vp || !pane) return;
    const target = pane.offsetLeft - (vp.clientWidth - pane.clientWidth) / 2;
    vp.scrollTo({ left: Math.max(0, target), behavior: "smooth" });
  }, [activeIndex, isPlaying]);

  const hints = useScrollHints(viewportRef, `${items.length}:${items.map((i) => i.provider).join(",")}`);

  const step = (dir: -1 | 1) => {
    const vp = viewportRef.current;
    if (!vp) return;
    const first = vp.querySelector<HTMLElement>('[role="listitem"]');
    vp.scrollBy({ left: dir * (first ? first.offsetWidth + 12 : 240), behavior: "smooth" });
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] font-medium uppercase tracking-[0.28em] text-text-tertiary">
          Responses
        </p>
        <button
          type="button"
          onClick={toggle}
          className="flex items-center gap-1.5 rounded-full bg-text-primary px-3 py-1 text-[11px] font-medium text-surface-primary transition-opacity hover:opacity-90"
          aria-label={isPlaying ? "Pause" : "Play all responses"}
        >
          {isPlaying ? <Pause className="size-3" /> : <Play className="size-3" />}
          <span>{isPlaying ? "Pause" : "Play all"}</span>
        </button>
      </div>

      <div className="relative">
        {hints.left ? (
          <button
            type="button"
            onClick={() => step(-1)}
            aria-label="Scroll responses left"
            className="absolute left-0.5 top-1/2 z-[3] flex size-8 -translate-y-1/2 items-center justify-center rounded-full bg-surface-secondary/90 text-text-primary shadow-md ring-1 ring-border-primary backdrop-blur-[1px]"
          >
            <ChevronLeft className="size-5" />
          </button>
        ) : null}
        {hints.right ? (
          <button
            type="button"
            onClick={() => step(1)}
            aria-label="Scroll responses right"
            className="absolute right-0.5 top-1/2 z-[3] flex size-8 -translate-y-1/2 items-center justify-center rounded-full bg-surface-secondary/90 text-text-primary shadow-md ring-1 ring-border-primary backdrop-blur-[1px]"
          >
            <ChevronRight className="size-5" />
          </button>
        ) : null}

        <div
          ref={viewportRef}
          className="scrollbar-hide flex snap-x snap-mandatory gap-3 overflow-x-auto pb-1"
          role="list"
          aria-label="Provider responses"
        >
          {items.map((item, i) => {
            const active = i === activeIndex;
            const playingThis = active && isPlaying;
            return (
              <div
                key={item.provider}
                ref={(el) => {
                  paneRefs.current[i] = el;
                }}
                role="listitem"
                className={`flex w-[180px] min-w-[180px] shrink-0 snap-start flex-col gap-2 rounded-xl border p-3 transition-colors ${
                  active
                    ? "border-text-primary/40 bg-surface-secondary"
                    : "border-border-primary bg-surface-secondary/60"
                }`}
              >
                <div className="flex items-center gap-1.5">
                  <span
                    className="size-2 shrink-0 rounded-full"
                    style={{ backgroundColor: getModelColor(toModelKey(item.provider, item.model)) }}
                  />
                  <span className="truncate text-xs font-medium text-text-primary">
                    {normalizeProvider(item.provider)}
                  </span>
                </div>
                <span className="truncate font-mono text-[11px] text-text-tertiary">
                  {item.model}
                </span>
                <button
                  type="button"
                  onClick={() => (playingThis ? toggle() : playFrom(i))}
                  className="mt-auto flex items-center gap-1 self-start rounded-full border border-border-primary px-2 py-0.5 text-[11px] text-text-secondary transition-colors hover:text-text-primary"
                  aria-label={playingThis ? `Pause ${item.provider}` : `Play ${item.provider}`}
                >
                  {playingThis ? <Pause className="size-3" /> : <Play className="size-3" />}
                  <span>{playingThis ? "Playing" : "Play"}</span>
                </button>
              </div>
            );
          })}
        </div>
      </div>

      <audio ref={audioRef} onEnded={handleEnded} hidden />
    </div>
  );
}
