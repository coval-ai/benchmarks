// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export interface PlaybackTrack {
  key: string;
  url: string;
}

// Stable identity of what's loaded into the <audio> element, so we can tell a
// genuine track swap (key or url changed) from a same-index re-render.
const trackId = (t: PlaybackTrack): string => JSON.stringify([t.key, t.url]);

// Makes sibling players mutually exclusive: a player claims it before playing,
// which pauses whoever held it. Share one per group; omit to run independent.
export interface PlaybackCoordinator {
  claim: (pause: () => void) => void;
}

export function usePlaybackCoordinator(): PlaybackCoordinator {
  const currentPause = useRef<(() => void) | null>(null);
  return useMemo(
    () => ({
      claim: (pause: () => void) => {
        if (currentPause.current && currentPause.current !== pause) {
          currentPause.current();
        }
        currentPause.current = pause;
      },
    }),
    []
  );
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
  onActivate?: (track: PlaybackTrack, index: number) => void,
  coordinator?: PlaybackCoordinator
): SequencedPlayback {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Bumped by every operation that supersedes playback (new play, pause, stop,
  // unmount, track swap) so a late play() rejection can't clear a newer state.
  const genRef = useRef(0);
  // Identity currently loaded into the element, or null when nothing is loaded.
  const playingIdRef = useRef<string | null>(null);

  const requestPlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const gen = ++genRef.current;
    setIsPlaying(true);
    void audio.play().catch(() => {
      if (genRef.current === gen) setIsPlaying(false);
    });
  }, []);

  const pauseSelf = useCallback(() => {
    genRef.current++;
    audioRef.current?.pause();
    setIsPlaying(false);
  }, []);

  const stop = useCallback(() => {
    genRef.current++;
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }
    playingIdRef.current = null;
    setIsPlaying(false);
  }, []);

  const playFrom = useCallback(
    (index: number) => {
      const audio = audioRef.current;
      const track = tracks[index];
      if (!audio || !track) return;
      coordinator?.claim(pauseSelf);
      setActiveIndex(index);
      playingIdRef.current = trackId(track);
      onActivate?.(track, index);
      audio.src = track.url;
      audio.currentTime = 0;
      requestPlay();
    },
    [tracks, onActivate, coordinator, pauseSelf, requestPlay]
  );

  const toggle = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      pauseSelf();
      return;
    }
    // Resume only if the loaded clip is still the active track; after a track
    // swap the element holds a stale src, so (re)load via playFrom instead.
    const active = tracks[activeIndex];
    if (audio.src && !audio.ended && active && playingIdRef.current === trackId(active)) {
      coordinator?.claim(pauseSelf);
      requestPlay();
    } else {
      playFrom(activeIndex);
    }
  }, [isPlaying, activeIndex, tracks, playFrom, coordinator, pauseSelf, requestPlay]);

  const handleEnded = useCallback(() => {
    const next = activeIndex + 1;
    if (next < tracks.length) playFrom(next);
    else setIsPlaying(false);
  }, [activeIndex, tracks.length, playFrom]);

  // Stop and rebase when the list mutates under us: empty list, active index now
  // out of range, or the active slot's identity changed — otherwise the old clip
  // keeps playing after a tick/manifest swap.
  useEffect(() => {
    if (tracks.length === 0) {
      if (playingIdRef.current !== null) stop();
      if (activeIndex !== 0) setActiveIndex(0);
      return;
    }
    const active = tracks[activeIndex];
    if (!active) {
      stop();
      setActiveIndex(0);
      return;
    }
    if (playingIdRef.current !== null && playingIdRef.current !== trackId(active)) {
      stop();
    }
  }, [tracks, activeIndex, stop]);

  useEffect(
    () => () => {
      genRef.current++;
      audioRef.current?.pause();
    },
    []
  );

  return { audioRef, activeIndex, isPlaying, toggle, playFrom, handleEnded, stop };
}
