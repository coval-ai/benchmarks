// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import type { S2STurn } from "@/lib/audioSamples/s2sFeed";

// Compact per-provider conversation: Caller (the persona) and Agent turns.
// Presentational + provider-agnostic — `accentColor` ties the agent's turns to
// its pane/dot color, so this can be reused verbatim if STT/TTS ever surface
// transcripts. The caller passes the color from the shared getModelColor map.
export function ConversationTurns({
  turns,
  accentColor,
}: {
  turns: S2STurn[];
  accentColor: string;
}) {
  return (
    <div className="mt-1 max-h-56 space-y-2 overflow-y-auto pr-1">
      {turns.map((t) => {
        const agent = t.role === "assistant";
        return (
          <div
            key={t.index}
            className={`border-l-2 pl-2 ${agent ? "" : "border-border-primary"}`}
            style={agent ? { borderColor: accentColor } : undefined}
          >
            <span className="font-mono text-[9px] uppercase tracking-wider text-text-tertiary">
              {agent ? "Agent" : "Caller"}
            </span>
            <p className="text-[11px] leading-snug text-text-primary">{t.content}</p>
          </div>
        );
      })}
    </div>
  );
}
