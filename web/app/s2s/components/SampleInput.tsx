// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { Pause, Play } from "lucide-react";
import { useMemo, useState } from "react";
import {
  useSequencedPlayback,
  type PlaybackCoordinator,
} from "@/hooks/useSequencedPlayback";

// The shared side of a tick: the one utterance every provider heard. Rendered
// once above the per-provider outputs. Input audio is optional — the sampler
// emits a null clip URL when the transcript can't be resolved to a dataset clip.
export function SampleInput({
  transcript,
  inputAudioUrl,
  coordinator,
}: {
  transcript: string | null;
  inputAudioUrl: string | null;
  coordinator?: PlaybackCoordinator;
}) {
  const tracks = useMemo(
    () => (inputAudioUrl ? [{ key: "input", url: inputAudioUrl }] : []),
    [inputAudioUrl]
  );
  const { audioRef, isPlaying, toggle, handleEnded } = useSequencedPlayback(
    tracks,
    undefined,
    coordinator
  );
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <p className="font-mono text-[10px] font-medium uppercase tracking-[0.28em] text-text-tertiary">
          Prompt
        </p>
        {inputAudioUrl ? (
          <button
            type="button"
            onClick={toggle}
            className="flex items-center gap-1 rounded-full border border-border-primary bg-surface-secondary px-2 py-0.5 text-[11px] text-text-secondary transition-colors hover:text-text-primary"
            aria-label={isPlaying ? "Pause input clip" : "Play input clip"}
          >
            {isPlaying ? <Pause className="size-3" /> : <Play className="size-3" />}
            <span>Input clip</span>
          </button>
        ) : null}
      </div>
      {transcript ? (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className={`block w-full text-left text-sm text-text-primary ${expanded ? "" : "line-clamp-2"}`}
          title={expanded ? "Collapse" : "Expand"}
        >
          &ldquo;{transcript}&rdquo;
        </button>
      ) : (
        <p className="text-sm italic text-text-tertiary">Transcript unavailable</p>
      )}
      <audio ref={audioRef} onEnded={handleEnded} hidden />
    </div>
  );
}
