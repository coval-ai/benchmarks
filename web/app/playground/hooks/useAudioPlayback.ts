"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export function useAudioPlayback() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingModelId, setPlayingModelId] = useState<string | null>(null);

  const stop = useCallback(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }
    setPlayingModelId(null);
  }, []);

  const toggle = useCallback((modelId: string, audioUrl: string | null) => {
    if (!audioUrl) return;
    const existing = audioRef.current;
    if (existing && playingModelId === modelId) {
      existing.pause();
      setPlayingModelId(null);
      return;
    }

    if (existing) {
      existing.pause();
      existing.src = "";
    }

    const audio = new Audio(audioUrl);
    audioRef.current = audio;
    setPlayingModelId(modelId);
    void audio.play().catch(() => {
      if (audioRef.current === audio) setPlayingModelId(null);
    });
    audio.onended = () => {
      if (audioRef.current === audio) setPlayingModelId(null);
    };
  }, [playingModelId]);

  useEffect(() => () => stop(), [stop]);

  return { playingModelId, toggle, stop };
}

