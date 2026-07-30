// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { type RefObject, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useSequencedPlayback,
  type PlaybackCoordinator,
  type PlaybackTrack,
} from "@/hooks/useSequencedPlayback";
import { getModelColor } from "@/lib/utils/colors";
import { toModelKey } from "@/lib/utils/formatters";
import type { S2SPlayTrigger, S2SSeekMethod } from "@/lib/posthog/events";
import type { S2STurn } from "@/lib/audioSamples/s2sFeed";
import { ConversationTurns } from "./ConversationTurns";

export interface SampleOutputItem {
  provider: string;
  model: string;
  url: string;
  // Multi-turn (v2) only: this provider's own conversation transcript.
  turns?: S2STurn[];
}

export interface SampleSeekInfo {
  method: S2SSeekMethod;
  toSeconds: number;
  // "turn" method only: which transcript turn was clicked.
  turnIndex?: number;
  turnRole?: "caller" | "agent";
}

export interface SampleListenInfo {
  trigger: S2SPlayTrigger;
  listenPct: number;
  durationSeconds: number;
  completed: boolean;
}

// Elapsed position as m:ss. The existing formatTime helpers render wall-clock
// timestamps, which is a different thing.
function elapsed(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
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
  onSeeked,
  onPlaybackEnded,
  playRequest,
  coordinator,
}: {
  items: SampleOutputItem[];
  normalizeProvider: (provider: string) => string;
  onPlay?: (provider: string, trigger: S2SPlayTrigger) => void;
  onSeeked?: (provider: string, seek: SampleSeekInfo) => void;
  onPlaybackEnded?: (provider: string, listen: SampleListenInfo) => void;
  playRequest?: { provider: string; nonce: number } | null;
  coordinator?: PlaybackCoordinator;
}) {
  const tracks = useMemo<PlaybackTrack[]>(
    () => items.map((i) => ({ key: i.provider, url: i.url })),
    [items]
  );
  // Multi-turn manifests carry per-provider turns; widen the panes and show the
  // conversation. All items in a manifest share a shape, so a single flag holds.
  const conversation = items.some((i) => (i.turns?.length ?? 0) > 0);
  const viewportRef = useRef<HTMLDivElement>(null);

  // How the next playFrom was initiated: the Play buttons and the timeline
  // effect each set this just before calling it. Panes never auto-advance
  // (the <audio> stops on end), so no third path exists.
  const triggerRef = useRef<S2SPlayTrigger>("button");

  // One listen session per activation, flushed from cleanup paths that would
  // otherwise capture a stale callback prop.
  const callbacksRef = useRef({ onPlaybackEnded });
  useEffect(() => {
    callbacksRef.current = { onPlaybackEnded };
  });
  const listenRef = useRef<{
    provider: string;
    trigger: S2SPlayTrigger;
    maxTime: number;
    duration: number;
    completed: boolean;
  } | null>(null);

  const flushListen = useCallback(() => {
    const session = listenRef.current;
    listenRef.current = null;
    // duration 0 means metadata never loaded (or play() was rejected):
    // nothing was heard, so there is no listen to report.
    if (!session || session.duration <= 0) return;
    callbacksRef.current.onPlaybackEnded?.(session.provider, {
      trigger: session.trigger,
      listenPct: session.completed
        ? 100
        : Math.min(100, Math.round((session.maxTime / session.duration) * 100)),
      durationSeconds: Math.round(session.duration),
      completed: session.completed,
    });
  }, []);

  const { audioRef, activeIndex, isPlaying, toggle, playFrom, stop, currentTime, duration, seek } =
    useSequencedPlayback(
      tracks,
      (track) => {
        flushListen();
        listenRef.current = {
          provider: track.key,
          trigger: triggerRef.current,
          maxTime: 0,
          duration: 0,
          completed: false,
        };
        onPlay?.(track.key, triggerRef.current);
      },
      coordinator
    );

  // Mirror playhead progress into the session; maxTime survives backward seeks.
  useEffect(() => {
    const session = listenRef.current;
    if (!session) return;
    if (duration > 0) session.duration = duration;
    if (currentTime > session.maxTime) session.maxTime = currentTime;
  }, [currentTime, duration]);

  // Pause, stop (including a track-list swap under playback), and play()
  // rejection all land here; a pane-to-pane switch flushes in onActivate above.
  useEffect(() => {
    if (!isPlaying) flushListen();
  }, [isPlaying, flushListen]);

  useEffect(() => () => flushListen(), [flushListen]);

  // The slider's onChange fires for every step of a drag; report one seek per
  // burst, at its final position. Each input captures its own emit closure so
  // a burst that outlives a tick swap still reports the manifest it scrubbed.
  const seekBurstRef = useRef<{
    provider: string;
    emit: () => void;
    timer: ReturnType<typeof setTimeout>;
  } | null>(null);
  const flushSeekBurst = useCallback(() => {
    const burst = seekBurstRef.current;
    if (!burst) return;
    seekBurstRef.current = null;
    clearTimeout(burst.timer);
    burst.emit();
  }, []);
  const reportSliderSeek = useCallback(
    (provider: string, emit: () => void) => {
      const burst = seekBurstRef.current;
      // Another pane's pending seek is a separate gesture: emit it now rather
      // than coalescing across providers.
      if (burst && burst.provider !== provider) flushSeekBurst();
      else if (burst) clearTimeout(burst.timer);
      seekBurstRef.current = {
        provider,
        emit,
        timer: setTimeout(() => {
          seekBurstRef.current = null;
          emit();
        }, 1000),
      };
    },
    [flushSeekBurst]
  );
  useEffect(() => () => flushSeekBurst(), [flushSeekBurst]);

  // A timeline-tooltip click plays a specific provider. Wait until that tick's
  // items have loaded (provider present) before playing, then mark the request
  // consumed — so the async manifest swap doesn't play the previous tick.
  const consumedNonce = useRef<number | null>(null);
  useEffect(() => {
    if (!playRequest || consumedNonce.current === playRequest.nonce) return;
    const idx = items.findIndex((i) => i.provider === playRequest.provider);
    if (idx < 0) return;
    consumedNonce.current = playRequest.nonce;
    triggerRef.current = "timeline";
    playFrom(idx);
    viewportRef.current?.children.item(idx)?.scrollIntoView({
      behavior: "smooth",
      block: "center",
      inline: "center",
    });
  }, [playRequest, items, playFrom]);

  const hints = useScrollHints(viewportRef, `${items.length}:${items.map((i) => i.provider).join(",")}`);

  const step = (dir: -1 | 1) => {
    const vp = viewportRef.current;
    if (!vp) return;
    const first = vp.querySelector<HTMLElement>('[role="listitem"]');
    vp.scrollBy({ left: dir * (first ? first.offsetWidth + 12 : 240), behavior: "smooth" });
  };

  return (
    <div className="space-y-2">
      <p className="font-mono text-[10px] font-medium uppercase tracking-[0.28em] text-text-tertiary">
        {conversation ? "Conversation" : "Responses"}
      </p>

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
            const color = getModelColor(toModelKey(item.provider, item.model));
            const turns = item.turns ?? [];
            return (
              <div
                key={item.provider}
                role="listitem"
                className={`flex w-[300px] min-w-[300px] shrink-0 snap-start flex-col gap-2 rounded-xl border p-3 transition-colors ${
                  active
                    ? "border-text-primary/40 bg-surface-secondary"
                    : "border-border-primary bg-surface-secondary/60"
                }`}
              >
                <div className="flex items-center gap-1.5">
                  <span
                    className="size-2 shrink-0 rounded-full"
                    style={{ backgroundColor: color }}
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
                  onClick={() => {
                    triggerRef.current = "button";
                    if (playingThis) toggle();
                    else playFrom(i);
                  }}
                  className={`flex items-center gap-1 self-start rounded-full border border-border-primary px-2 py-0.5 text-[11px] text-text-secondary transition-colors hover:text-text-primary ${
                    turns.length ? "" : "mt-auto"
                  }`}
                  aria-label={playingThis ? `Pause ${item.provider}` : `Play ${item.provider}`}
                >
                  {playingThis ? <Pause className="size-3" /> : <Play className="size-3" />}
                  <span>{playingThis ? "Playing" : "Play"}</span>
                </button>
                {/* One <audio> is shared, so the playhead belongs to the active
                    pane alone; other panes keep their turns static. */}
                {active && duration > 0 ? (
                  <div className="flex items-center gap-1.5">
                    <input
                      type="range"
                      min={0}
                      max={duration}
                      step={0.1}
                      value={Math.min(currentTime, duration)}
                      onChange={(e) => {
                        const target = Number(e.target.value);
                        seek(target);
                        reportSliderSeek(item.provider, () =>
                          onSeeked?.(item.provider, {
                            method: "slider",
                            toSeconds: target,
                          })
                        );
                      }}
                      aria-label={`Seek ${normalizeProvider(item.provider)} recording`}
                      className="h-1 w-full cursor-pointer"
                      style={{ accentColor: color }}
                    />
                    <span className="shrink-0 font-mono text-[9px] tabular-nums text-text-tertiary">
                      {elapsed(currentTime)}/{elapsed(duration)}
                    </span>
                  </div>
                ) : null}
                {turns.length ? (
                  <ConversationTurns
                    turns={turns}
                    accentColor={color}
                    currentTime={active ? currentTime : 0}
                    onSeek={
                      active
                        ? (seconds, turn) => {
                            seek(seconds);
                            onSeeked?.(item.provider, {
                              method: "turn",
                              toSeconds: seconds,
                              turnIndex: turn.index,
                              turnRole: turn.role,
                            });
                          }
                        : undefined
                    }
                  />
                ) : null}
              </div>
            );
          })}
        </div>
      </div>

      <audio
        ref={audioRef}
        onEnded={() => {
          // Mark completion before stop() flushes: the last timeupdate can
          // land short of the full duration.
          const session = listenRef.current;
          if (session) {
            session.completed = true;
            if (session.duration > 0) session.maxTime = session.duration;
          }
          stop();
        }}
        hidden
      />
    </div>
  );
}
