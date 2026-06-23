"use client";

import { useEffect, useRef, useState } from "react";

type Props = {
  src: string | null; // null until a battle is generated -> the player renders disabled
  label: string;
  isActive: boolean; // true when this is the side currently allowed to play
  onActivate: () => void; // claim single-playback focus
};

function fmt(t: number): string {
  if (!Number.isFinite(t)) return "0:00";
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function AudioPlayer({ src, label, isActive, onActivate }: Props) {
  const ref = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const disabled = !src;

  // Single-playback: when another side claims focus, pause this one (keep its
  // position so play resumes rather than restarts).
  useEffect(() => {
    if (!isActive) ref.current?.pause();
  }, [isActive]);

  // Pause on unmount so audio never outlives the component.
  useEffect(() => {
    const el = ref.current;
    return () => {
      el?.pause();
    };
  }, []);

  const toggle = () => {
    const el = ref.current;
    if (!el || disabled) return;
    if (playing) {
      el.pause();
      return;
    }
    onActivate();
    void el.play();
  };

  return (
    <div className="flex items-center justify-between gap-3">
      {src && (
        <audio
          ref={ref}
          src={src}
          preload="metadata"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => {
            setPlaying(false);
            setProgress(0);
            if (ref.current) ref.current.currentTime = 0; // rearm for replay from the start
          }}
          onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
          onTimeUpdate={(e) =>
            setProgress(
              e.currentTarget.duration ? e.currentTarget.currentTime / e.currentTarget.duration : 0,
            )
          }
        />
      )}
      <button
        type="button"
        onClick={toggle}
        disabled={disabled}
        aria-label={disabled ? `${label} — generate to enable` : `${playing ? "Pause" : "Play"} ${label}`}
        className="shrink-0 rounded-full border border-border-primary bg-surface-primary px-4 py-2 font-mono text-sm text-text-primary hover:bg-hover-bg disabled:cursor-not-allowed disabled:opacity-40"
      >
        {playing ? "❚❚" : "▶"}
      </button>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-secondary">
        <div
          className="h-full rounded-full bg-text-primary"
          style={{ width: `${Math.round(progress * 100)}%` }}
        />
      </div>
      <span className="shrink-0 font-mono text-xs text-text-tertiary">{src ? fmt(duration) : "—"}</span>
    </div>
  );
}
