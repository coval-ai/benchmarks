// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface PlaybackTrack {
  key: string;
  url: string;
}

export interface SequencedPlayback {
  audioRef: React.RefObject<HTMLAudioElement | null>;
  activeIndex: number;
  isPlaying: boolean;
  toggle: () => void;
  playFrom: (index: number) => void;
  handleEnded: () => void;
  stop: () => void;
}

// A playlist over one <audio> element: one click plays the active track, and
// each track's end auto-advances to the next until the list is exhausted or
// the user stops. onActivate fires the moment a track becomes active (for
// auto-scroll + analytics), not when its play() promise resolves.
export function useSequencedPlayback(
  tracks: PlaybackTrack[],
  onActivate?: (track: PlaybackTrack, index: number) => void
): SequencedPlayback {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (tracks.length > 0 && activeIndex >= tracks.length) setActiveIndex(0);
  }, [tracks.length, activeIndex]);

  const playFrom = useCallback(
    (index: number) => {
      const audio = audioRef.current;
      const track = tracks[index];
      if (!audio || !track) return;
      setActiveIndex(index);
      onActivate?.(track, index);
      audio.src = track.url;
      audio.currentTime = 0;
      setIsPlaying(true);
      void audio.play().catch(() => setIsPlaying(false));
    },
    [tracks, onActivate]
  );

  const toggle = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
      return;
    }
    // Resume a paused track; otherwise (re)start the active one.
    if (audio.src && !audio.ended) {
      setIsPlaying(true);
      void audio.play().catch(() => setIsPlaying(false));
    } else {
      playFrom(activeIndex);
    }
  }, [isPlaying, activeIndex, playFrom]);

  const handleEnded = useCallback(() => {
    const next = activeIndex + 1;
    if (next < tracks.length) playFrom(next);
    else setIsPlaying(false);
  }, [activeIndex, tracks.length, playFrom]);

  const stop = useCallback(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }
    setIsPlaying(false);
  }, []);

  useEffect(
    () => () => {
      const audio = audioRef.current;
      if (audio) audio.pause();
    },
    []
  );

  return { audioRef, activeIndex, isPlaying, toggle, playFrom, handleEnded, stop };
}
