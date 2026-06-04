"use client";

import { useEffect, useRef } from "react";

const N = 44;
const W = 200;
const H = 200;
const cy = H / 2;

/** Slower evolution than original 0.025 per frame. */
const T_STEP = 0.0085;
/** Follow targets while recording (softer motion). */
const AMP_LERP_RECORD = 0.07;
/** Relax toward idle when recording stops. */
const AMP_LERP_IDLE = 0.16;
const IDLE_SETTLE_EPS = 0.008;

function envelope(norm: number): number {
  const tallPeak = Math.exp(-Math.pow((norm - 0.28) * 4.5, 2));
  const shortPeak = Math.exp(-Math.pow((norm - 0.7) * 5.0, 2)) * 0.65;
  return Math.max(tallPeak, shortPeak);
}

function idleAmplitude(norm: number): number {
  return 0.05 + envelope(norm) * 0.2;
}

function fillBar(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  barW: number,
  barFullH: number,
  opacity: number,
  lightMode: boolean
) {
  ctx.fillStyle = lightMode
    ? `rgba(0, 0, 0, ${opacity})`
    : `rgba(255, 255, 255, ${opacity})`;
  const r = 2;
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barFullH, r);
    ctx.fill();
  } else {
    ctx.fillRect(x, y, barW, barFullH);
  }
}

/** `recording` | `idleDecay` | `paintOnly` (resize: no physics step). */
type SimMode = "recording" | "idleDecay" | "paintOnly";

function resolvedDiskRgb(diskProbe: HTMLElement): string {
  const rgb = getComputedStyle(diskProbe).backgroundColor;
  if (rgb && rgb !== "rgba(0, 0, 0, 0)" && rgb !== "transparent") {
    return rgb;
  }
  return "rgb(32, 32, 35)";
}

/**
 * Phases use `i * k - t * ω` so activity sweeps toward **lower indices** → reads as motion to the left.
 */
function recordNoise(i: number, t: number): number {
  return Math.abs(
    Math.sin(i * 0.42 - t * 0.52) * 0.55 +
      Math.sin(i * 0.24 - t * 0.74) * 0.3 +
      Math.sin(i * 0.58 - t * 0.34) * 0.15
  );
}

type Props = {
  className?: string;
  recording: boolean;
};

/**
 * Triangle pulse: animates only while `recording` is true; decays when stopped.
 */
export function SttTrianglePulseCanvas({ className, recording }: Props) {
  // Light-only theme (coval.ai cream); bars always use the light treatment.
  const lightBarsRef = useRef(true);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recordingRef = useRef(recording);
  recordingRef.current = recording;

  const kickLoopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const diskProbe = document.createElement("span");
    diskProbe.setAttribute("aria-hidden", "true");
    diskProbe.style.cssText =
      "position:absolute;overflow:hidden;width:0;height:0;opacity:0;pointer-events:none;background-color:var(--playground-stt-disk)";
    parent.insertBefore(diskProbe, parent.firstChild);

    const amps = new Float32Array(N);
    let t = 0;
    let rafId = 0;
    let cancelled = false;
    let loopRunning = false;

    const reduceMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const paintAmplitudes = () => {
      const slotW = W / N;
      const barW = slotW * 0.54;
      ctx.fillStyle = resolvedDiskRgb(diskProbe);
      ctx.fillRect(0, 0, W, H);
      const lightBars = lightBarsRef.current;
      for (let i = 0; i < N; i++) {
        const barH = amps[i]! * H * 0.42;
        const x = (i / N) * W + slotW * 0.23;
        const opacity = 0.4 + 0.6 * amps[i]!;
        fillBar(ctx, x, cy - barH, barW, barH * 2, opacity, lightBars);
      }
    };

    const simulateAmplitudes = (sim: SimMode) => {
      if (reduceMotion) {
        for (let i = 0; i < N; i++) {
          const norm = N > 1 ? i / (N - 1) : 0;
          amps[i] = 0.08 + envelope(norm) * 0.55;
        }
        return;
      }

      if (sim === "paintOnly") return;

      if (sim === "recording") {
        t += T_STEP;
        for (let i = 0; i < N; i++) {
          const norm = N > 1 ? i / (N - 1) : 0;
          const env = envelope(norm);
          const noise = recordNoise(i, t);
          const target = 0.06 + env * 0.9 * noise;
          amps[i] = amps[i]! + (target - amps[i]!) * AMP_LERP_RECORD;
        }
      } else if (sim === "idleDecay") {
        for (let i = 0; i < N; i++) {
          const norm = N > 1 ? i / (N - 1) : 0;
          const idle = idleAmplitude(norm);
          amps[i] = amps[i]! + (idle - amps[i]!) * AMP_LERP_IDLE;
        }
      }
    };

    const layoutAndDraw = (sim: SimMode) => {
      if (cancelled) return;

      const rect = parent.getBoundingClientRect();
      const side = Math.max(1, Math.min(rect.width, rect.height));
      const dpr = Math.min(window.devicePixelRatio ?? 1, 2);
      const bw = Math.floor(side * dpr);
      const bh = Math.floor(side * dpr);
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const ox = (rect.width - side) / 2;
      const oy = (rect.height - side) / 2;
      ctx.save();
      ctx.translate(ox, oy);
      ctx.scale(side / W, side / H);

      simulateAmplitudes(sim);
      paintAmplitudes();

      ctx.restore();
    };

    const ampsNeedFrames = (): boolean => {
      for (let i = 0; i < N; i++) {
        const norm = N > 1 ? i / (N - 1) : 0;
        if (Math.abs(amps[i]! - idleAmplitude(norm)) > IDLE_SETTLE_EPS) {
          return true;
        }
      }
      return false;
    };

    const scheduleLoop = () => {
      if (cancelled || reduceMotion || loopRunning) return;
      loopRunning = true;

      const step = () => {
        if (cancelled) {
          loopRunning = false;
          return;
        }
        if (reduceMotion) {
          layoutAndDraw("idleDecay");
          loopRunning = false;
          return;
        }

        const rec = recordingRef.current;
        layoutAndDraw(rec ? "recording" : "idleDecay");

        const needNext = rec || ampsNeedFrames();
        if (needNext && !cancelled) {
          rafId = requestAnimationFrame(step);
        } else {
          loopRunning = false;
          rafId = 0;
        }
      };

      rafId = requestAnimationFrame(step);
    };

    kickLoopRef.current = () => {
      if (reduceMotion) {
        layoutAndDraw(recordingRef.current ? "recording" : "idleDecay");
      } else {
        scheduleLoop();
      }
    };

    if (reduceMotion) {
      layoutAndDraw("idleDecay");
    } else {
      scheduleLoop();
    }

    const ro = new ResizeObserver(() => {
      if (cancelled) return;
      layoutAndDraw("paintOnly");
    });
    ro.observe(parent);

    return () => {
      cancelled = true;
      kickLoopRef.current = null;
      cancelAnimationFrame(rafId);
      ro.disconnect();
      diskProbe.remove();
      loopRunning = false;
    };
  }, []);

  useEffect(() => {
    recordingRef.current = recording;
    kickLoopRef.current?.();
  }, [recording]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      width={W}
      height={H}
      aria-hidden
    />
  );
}
