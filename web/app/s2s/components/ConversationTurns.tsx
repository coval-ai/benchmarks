// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useCallback, useEffect, useRef } from "react";
import type { S2STurn } from "@/lib/audioSamples/s2sFeed";

// The turn holding the playhead, or -1. The latest turn started at or before
// the playhead wins; deliberately ignoring end_offset keeps the highlight put
// during the silence between utterances instead of flickering off, and covers
// turns whose end Coval omitted. Offsets are optional, so a turn without a
// start is never active and a conversation with none never highlights.
export function activeTurnIndex(turns: S2STurn[], currentTime: number): number {
  for (let i = turns.length - 1; i >= 0; i--) {
    const start = turns[i]?.start_offset;
    if (start !== undefined && start <= currentTime) return i;
  }
  return -1;
}

// Compact per-provider conversation: Caller (the persona) and Agent turns.
// Presentational + provider-agnostic — `accentColor` ties the agent's turns to
// its pane/dot color, so this can be reused verbatim if STT/TTS ever surface
// transcripts. The caller passes the color from the shared getModelColor map.
// When `onSeek` is given, a turn carrying an offset jumps the playhead to it.
export function ConversationTurns({
  turns,
  agentLabel,
  accentColor,
  currentTime = 0,
  onSeek,
  seeked,
}: {
  turns: S2STurn[];
  // Model id from the pane header, so each agent turn stays attributed to the
  // model being compared. The caller side is always Coval's simulated caller.
  agentLabel: string;
  accentColor: string;
  currentTime?: number;
  onSeek?: (seconds: number, turn: { index: number; role: "caller" | "agent" }) => void;
  // Slider scrubs, so the list can glide to the turn being scrubbed to. The
  // list never follows playback on its own — only user gestures move it.
  seeked?: { time: number; nonce: number };
}) {
  const active = onSeek ? activeTurnIndex(turns, currentTime) : -1;
  // Only fade the others once something is genuinely highlighted. Without
  // offsets nothing ever is, and dimming every turn would just look broken.
  const fadeInactive = active >= 0;
  const listRef = useRef<HTMLDivElement>(null);
  // One retargetable glide owns all list motion (the SectionHeader anchor
  // pattern): each frame closes 15% of the gap, so a new target mid-flight
  // bends the animation instead of restarting it, and rapid target changes
  // can never fight each other. User wheel/touch input cancels it outright.
  const targetRef = useRef<number | null>(null);
  const rafRef = useRef(0);
  const glideTo = useCallback((top: number) => {
    const el = listRef.current;
    if (!el) return;
    targetRef.current = Math.max(0, Math.min(top, el.scrollHeight - el.clientHeight));
    if (rafRef.current) return;
    const step = () => {
      const list = listRef.current;
      const target = targetRef.current;
      if (!list || target == null) {
        rafRef.current = 0;
        return;
      }
      const delta = target - list.scrollTop;
      if (Math.abs(delta) < 1) {
        list.scrollTop = target;
        targetRef.current = null;
        rafRef.current = 0;
        return;
      }
      list.scrollTop += delta * 0.15;
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
  }, []);
  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    const cancel = () => {
      targetRef.current = null;
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    };
    el.addEventListener("wheel", cancel, { passive: true });
    el.addEventListener("touchmove", cancel, { passive: true });
    return () => {
      cancel();
      el.removeEventListener("wheel", cancel);
      el.removeEventListener("touchmove", cancel);
    };
  }, []);
  const turnTop = useCallback((index: number) => {
    const el = listRef.current;
    const turn = el?.children.item(index) as HTMLElement | null;
    return el && turn ? turn.offsetTop - el.offsetTop : 0;
  }, []);
  const resting = active < 0;
  useEffect(() => {
    if (resting) glideTo(0);
  }, [resting, glideTo]);
  const seenSeek = useRef(seeked?.nonce ?? 0);
  useEffect(() => {
    if (!seeked || seeked.nonce === seenSeek.current) return;
    seenSeek.current = seeked.nonce;
    const idx = activeTurnIndex(turns, seeked.time);
    if (idx >= 0) glideTo(turnTop(idx));
  }, [seeked, turns, glideTo, turnTop]);
  return (
    <div ref={listRef} className="mt-1 max-h-56 space-y-2 overflow-y-auto pr-1">
      {turns.map((t, i) => {
        const agent = t.role === "assistant";
        const seekable = onSeek !== undefined && t.start_offset !== undefined;
        const isActive = i === active;
        const body = (
          <>
            <span className="font-mono text-[9px] uppercase tracking-wider text-text-tertiary">
              {agent ? agentLabel : "Caller (simulated)"}
            </span>
            <p className={`text-[11px] leading-snug ${agent ? "text-text-primary" : "text-text-secondary"}`}>
              {t.content}
            </p>
          </>
        );
        const className = `border-l-2 pl-2 transition-opacity ${
          agent ? "" : "border-border-primary"
        } ${fadeInactive && !isActive ? "opacity-60" : ""}`;
        const style = agent ? { borderColor: accentColor } : undefined;

        if (!seekable) {
          return (
            <div key={t.index} className={className} style={style}>
              {body}
            </div>
          );
        }
        return (
          <button
            key={t.index}
            type="button"
            onClick={() => {
              glideTo(turnTop(i));
              onSeek(t.start_offset as number, {
                index: t.index,
                role: agent ? "agent" : "caller",
              });
            }}
            aria-current={isActive ? "true" : undefined}
            className={`${className} block w-full cursor-pointer text-left hover:opacity-100`}
            style={style}
          >
            {body}
          </button>
        );
      })}
    </div>
  );
}
