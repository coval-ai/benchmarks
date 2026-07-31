// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useEffect, useRef } from "react";
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
  accentColor,
  currentTime = 0,
  onSeek,
}: {
  turns: S2STurn[];
  accentColor: string;
  currentTime?: number;
  onSeek?: (seconds: number, turn: { index: number; role: "caller" | "agent" }) => void;
}) {
  const active = onSeek ? activeTurnIndex(turns, currentTime) : -1;
  // Only fade the others once something is genuinely highlighted. Without
  // offsets nothing ever is, and dimming every turn would just look broken.
  const fadeInactive = active >= 0;
  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    if (active < 0) {
      el.scrollTo({ top: 0 });
      return;
    }
    const turn = el.children.item(active) as HTMLElement | null;
    if (turn) {
      el.scrollTo({ top: Math.max(0, turn.offsetTop - el.offsetTop - 8), behavior: "smooth" });
    }
  }, [active]);
  return (
    <div ref={listRef} className="mt-1 max-h-56 space-y-2 overflow-y-auto pr-1">
      {turns.map((t, i) => {
        const agent = t.role === "assistant";
        const seekable = onSeek !== undefined && t.start_offset !== undefined;
        const isActive = i === active;
        const body = (
          <>
            <span className="font-mono text-[9px] uppercase tracking-wider text-text-tertiary">
              {agent ? "Agent" : "Caller"}
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
            onClick={() =>
              onSeek(t.start_offset as number, {
                index: t.index,
                role: agent ? "agent" : "caller",
              })
            }
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
