"use client";

import { type RefObject, useEffect, useRef } from "react";
import { useTheme } from "next-themes";

export type CymaticsFamily = "neutral" | "blue" | "purple" | "green" | "redOrange";

/** Brand line colors per family (guidelines: dark tone on light, mid tone on ink). */
const LINE_RGB: Record<CymaticsFamily, { light: string; dark: string }> = {
  neutral: { light: "15, 12, 10", dark: "249, 250, 248" },
  blue: { light: "26, 44, 54", dark: "198, 220, 250" },
  purple: { light: "64, 20, 27", dark: "218, 196, 225" },
  green: { light: "18, 50, 43", dark: "223, 234, 194" },
  redOrange: { light: "50, 31, 22", dark: "245, 206, 162" }
};

/** Map an arbitrary model hex to the nearest brand family; low-saturation → neutral. */
export function cymaticsFamilyFromHex(hex: string): CymaticsFamily {
  const h = hex.replace("#", "");
  if (!/^[0-9a-fA-F]{6}$/.test(h)) return "neutral";
  const r = parseInt(h.slice(0, 2), 16) / 255;
  const g = parseInt(h.slice(2, 4), 16) / 255;
  const b = parseInt(h.slice(4, 6), 16) / 255;
  const max = Math.max(r, g, b);
  const d = max - Math.min(r, g, b);
  if (d < 0.08) return "neutral";
  let hue =
    max === r ? 60 * (((g - b) / d) % 6) : max === g ? 60 * ((b - r) / d + 2) : 60 * ((r - g) / d + 4);
  if (hue < 0) hue += 360;
  if (hue < 70 || hue >= 335) return "redOrange";
  if (hue < 190) return "green";
  if (hue < 255) return "blue";
  return "purple";
}

const W = 280;
const GRID = 80;
const P = GRID + 1;

/**
 * Curated plate designs [n, m, sign, radial] — Chladni plates (radial=0, all sign +1 so no hard
 * diagonal frames) alternating with wavy radial rings (radial=1). Ordered coarse → fine.
 */
const DESIGNS = [
  [1, 3, 1, 0],
  [0, 3, 1, 1],
  [1, 5, 1, 0],
  [0, 4, -1, 1],
  [3, 5, 1, 0],
  [0, 5, 1, 1],
  [1, 7, 1, 0],
  [0, 6, -1, 1],
  [2, 8, 1, 0],
  [0, 7, 1, 1],
  [1, 9, 1, 0],
  [0, 8, -1, 1],
  [3, 9, 1, 0],
  [0, 9, 1, 1],
  [5, 9, 1, 0],
  [0, 10, -1, 1]
] as const;
const IDLE_DESIGN = 3;
/** Swap rotation within the pitch band — variety without leaving the voice's design family. */
const NEIGHBOR_STEPS = [0, 1, -1];
const WEIGHT_LERP_REC = 0.085;
const WEIGHT_LERP_IDLE = 0.045;
const LEVELS = [0, 0.24, -0.24, 0.52, -0.52];
const LEVEL_ALPHA = [0.8, 0.38, 0.38, 0.18, 0.18];
/** Pitch (F0) search range for autocorrelation — covers low male through high female voice. */
const PITCH_MIN = 70;
const PITCH_MAX = 400;
/** Autocorrelation window (samples). */
const PITCH_WIN = 1024;
/** Pitch range mapped across the design library — low voice → coarse plates, high → fine. */
const TONE_LO = 90;
const TONE_HI = 320;
/** Voice gate (signal RMS) above the adaptive noise floor — ambient noise never drives the plate. */
const GATE = 0.02;

function resolvedDiskRgb(diskProbe: HTMLElement): string {
  const rgb = getComputedStyle(diskProbe).backgroundColor;
  if (rgb && rgb !== "rgba(0, 0, 0, 0)" && rgb !== "transparent") return rgb;
  return "rgb(32, 32, 35)";
}

type Props = {
  className?: string;
  recording: boolean;
  analyser: AnalyserNode | null;
  family: CymaticsFamily;
  /** Geist Mono readout target — receives "‒42 dB · 210 Hz" style text while recording. */
  readoutRef?: RefObject<HTMLElement | null>;
};

/**
 * Chladni-plate cymatics, choreographed: speech ticks a sequencer that dissolves and resolves
 * whole plate designs near the voice's pitch anchor (low voice → coarse plates, high → fine),
 * while gated amplitude drives contrast, ripple, cycle speed, and a slow rotation.
 */
export function SttCymaticsCanvas({ className, recording, analyser, family, readoutRef }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recordingRef = useRef(recording);
  recordingRef.current = recording;
  const analyserRef = useRef(analyser);
  analyserRef.current = analyser;
  const familyRef = useRef(family);
  familyRef.current = family;
  const { resolvedTheme } = useTheme();
  const darkRef = useRef(resolvedTheme === "dark");
  darkRef.current = resolvedTheme === "dark";

  const kickLoopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const parent = canvas?.parentElement;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !parent || !ctx) return;

    const diskProbe = document.createElement("span");
    diskProbe.setAttribute("aria-hidden", "true");
    diskProbe.style.cssText =
      "position:absolute;overflow:hidden;width:0;height:0;opacity:0;pointer-events:none;background-color:var(--playground-stt-disk)";
    parent.insertBefore(diskProbe, parent.firstChild);

    const weights = new Float32Array(DESIGNS.length);
    const targets = new Float32Array(DESIGNS.length);
    weights[IDLE_DESIGN] = targets[IDLE_DESIGN] = 1;
    const field = new Float32Array(P * P);
    const radius = new Float32Array(P * P);
    for (let y = 0; y < P; y++) {
      for (let x = 0; x < P; x++) {
        radius[y * P + x] = Math.hypot((x / GRID) * 2 - 1, (y / GRID) * 2 - 1);
      }
    }
    const cosA = new Float32Array(P);
    const cosB = new Float32Array(P);
    let timeData: Uint8Array<ArrayBuffer> | null = null;
    let wave: Float32Array | null = null;
    let amp = 0;
    let ripplePhase = 0;
    let pitchHz = 0;
    let noiseFloor = 0.02;
    let level = 0;
    let designIdx = IDLE_DESIGN;
    let seqK = 0;
    let resumePending = true;
    let swapAnchor = -1;
    let shiftFrames = 0;
    let unvoicedFrames = 99;
    let speechFrames = 0;
    let silenceFrames = 99;
    let rawEnv = 0;
    let voiceActive = false;
    let activity = 0;
    let rot = 0;
    let rotSpeed = 0;
    let frame = 0;
    let rafId = 0;
    let cancelled = false;
    let loopRunning = false;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const readAudio = () => {
      const an = recordingRef.current ? analyserRef.current : null;
      let ampTarget = 0;
      if (an) {
        if (!timeData || timeData.length !== an.fftSize) {
          timeData = new Uint8Array(an.fftSize);
          wave = new Float32Array(an.fftSize);
        }
        an.getByteTimeDomainData(timeData);
        // True signal RMS — honest loudness for the dB readout and the voice gate.
        let sq = 0;
        for (let i = 0; i < timeData.length; i++) {
          const v = (timeData[i]! - 128) / 128;
          wave![i] = v;
          sq += v * v;
        }
        level = Math.sqrt(sq / timeData.length);
        // Adaptive noise floor: falls fast, creeps up slowly — steady wind/room tone gets absorbed.
        noiseFloor += (level - noiseFloor) * (level < noiseFloor ? 0.3 : 0.006);
        const voiced = Math.max(0, level - noiseFloor - GATE);
        ampTarget = Math.min(1, voiced * 7);
        // Voice-activity hysteresis on a RAW envelope (never gated, unlike `amp`), so waking
        // and sleeping behave identically before and after activation. Dips between syllables
        // decay the wake counter instead of resetting it — choppy speech still wakes the plate;
        // only a genuine sentence pause calms it and arms the next design swap.
        rawEnv += (ampTarget - rawEnv) * (ampTarget > rawEnv ? 0.3 : 0.08);
        if (ampTarget > 0.12) {
          speechFrames++;
          silenceFrames = 0;
          if (speechFrames > 4) voiceActive = true;
        } else {
          speechFrames = Math.max(0, speechFrames - 1);
          if (rawEnv < 0.06) {
            silenceFrames++;
            if (silenceFrames === 40) resumePending = true;
            if (silenceFrames > 50) voiceActive = false;
          }
        }
        if (!voiceActive) ampTarget = 0;
        let pitchConfident = false;
        if ((voiceActive || ampTarget > 0.08) && wave) {
          // Autocorrelation pitch (F0) — tracks the actual voice fundamental, so a low chest
          // voice and a high girly voice read ~110 Hz vs ~250 Hz (a spectral centroid doesn't).
          // Runs whenever active; the confidence check below holds the last pitch through
          // consonants and gaps instead of dropping it.
          const sr = an.context.sampleRate;
          const lagMin = Math.floor(sr / PITCH_MAX);
          const lagMax = Math.min(Math.ceil(sr / PITCH_MIN), wave.length - PITCH_WIN);
          let norm0 = 0;
          for (let i = 0; i < PITCH_WIN; i += 2) norm0 += wave[i]! * wave[i]!;
          let best = 0;
          let bestLag = 0;
          for (let lag = lagMin; lag <= lagMax; lag += 2) {
            let s = 0;
            for (let i = 0; i < PITCH_WIN; i += 2) s += wave[i]! * wave[i + lag]!;
            if (s > best) {
              best = s;
              bestLag = lag;
            }
          }
          if (bestLag > 0 && best > norm0 * 0.3) {
            pitchConfident = true;
            const hz = sr / bestLag;
            pitchHz = pitchHz === 0 ? hz : pitchHz + (hz - pitchHz) * 0.2;
          }
        }
        unvoicedFrames = pitchConfident ? 0 : unvoicedFrames + 1;
        // Design swaps fire on (a) resuming speech after a beat of silence, or (b) a sustained
        // pitch-register shift mid-speech. Register jumps snap EXACTLY to the pitch anchor —
        // pitch → pattern is a pure function, so swapping girly → manly → girly returns to the
        // same plates, like a real cymatic plate following the tone. Neighbor rotation only
        // adds variety when resuming in the same register.
        if (voiceActive && pitchHz > 0) {
          const t = Math.min(1, Math.max(0, Math.log(pitchHz / TONE_LO) / Math.log(TONE_HI / TONE_LO)));
          const curAnchor = Math.round(t * (DESIGNS.length - 1));
          const shifted = swapAnchor >= 0 && Math.abs(curAnchor - swapAnchor) >= 2;
          shiftFrames = shifted ? shiftFrames + 1 : 0;
          if (resumePending || shiftFrames > 15) {
            const clampIdx = (i: number) => Math.max(0, Math.min(DESIGNS.length - 1, i));
            let next = curAnchor;
            if (resumePending && curAnchor === swapAnchor) {
              seqK++;
              next = clampIdx(curAnchor + NEIGHBOR_STEPS[seqK % NEIGHBOR_STEPS.length]!);
              if (next === designIdx) {
                next = clampIdx(curAnchor + NEIGHBOR_STEPS[(seqK + 1) % NEIGHBOR_STEPS.length]!);
              }
            }
            designIdx = next;
            swapAnchor = curAnchor;
            resumePending = false;
            shiftFrames = 0;
            targets.fill(0);
            targets[designIdx] = 1;
          }
        }
      } else {
        // Idle "getting ready": breathe a whisper of the neighboring design in and out.
        targets.fill(0);
        targets[IDLE_DESIGN] = 1;
        targets[IDLE_DESIGN + 2] = 0.05 + 0.03 * Math.sin(frame * 0.004);
        designIdx = IDLE_DESIGN;
        resumePending = true;
        swapAnchor = -1;
        shiftFrames = 0;
        unvoicedFrames = 99;
        speechFrames = 0;
        silenceFrames = 99;
        rawEnv = 0;
        voiceActive = false;
        pitchHz = 0;
        level = 0;
      }
      amp += (ampTarget - amp) * (ampTarget > amp ? 0.14 : 0.05);
      activity += ((voiceActive ? 1 : 0) - activity) * (voiceActive ? 0.06 : 0.12);
      ripplePhase += 0.012 + activity * (0.05 + amp * 0.12);
      rotSpeed += (0.00015 + activity * (0.00165 + amp * 0.005) - rotSpeed) * 0.05;
      rot += rotSpeed;
      const lerp = recordingRef.current ? WEIGHT_LERP_REC : WEIGHT_LERP_IDLE;
      for (let i = 0; i < DESIGNS.length; i++) {
        weights[i]! += (targets[i]! - weights[i]!) * lerp;
      }
    };

    const computeField = () => {
      field.fill(0);
      let sumW = 0;
      for (let k = 0; k < DESIGNS.length; k++) {
        const w = weights[k]!;
        if (w < 0.012) continue;
        sumW += w;
        const [n, m, sign, radial] = DESIGNS[k]!;
        if (radial) {
          for (let y = 0; y < P; y++) {
            const py = (y / GRID) * 2 - 1;
            const row = y * P;
            for (let x = 0; x < P; x++) {
              const px = (x / GRID) * 2 - 1;
              const r = radius[row + x]!;
              const theta = Math.atan2(py, px);
              field[row + x]! +=
                w * Math.cos(Math.PI * (m * r + sign * 0.42 * Math.cos(4 * theta)));
            }
          }
          continue;
        }
        for (let i = 0; i < P; i++) {
          const u = (i / GRID) * Math.PI;
          cosA[i] = Math.cos(n * u);
          cosB[i] = Math.cos(m * u);
        }
        for (let y = 0; y < P; y++) {
          const cy = cosB[y]!;
          const cy2 = sign * cosA[y]!;
          const row = y * P;
          for (let x = 0; x < P; x++) {
            field[row + x]! += w * (cosA[x]! * cy + cosB[x]! * cy2);
          }
        }
      }
      const rippleAmp = amp * 0.35 + 0.018 * (1 - activity);
      const norm = 1 / (2 * sumW + rippleAmp + 1e-6);
      for (let i = 0; i < P * P; i++) {
        const r = radius[i]!;
        const ripple = rippleAmp > 0.01 ? rippleAmp * Math.cos(r * 14 - ripplePhase) * (1 - Math.min(1, r)) : 0;
        field[i] = (field[i]! + ripple) * norm;
      }
    };

    const paint = () => {
      const rect = parent.getBoundingClientRect();
      const side = Math.max(1, Math.min(rect.width, rect.height));
      const dpr = Math.min(window.devicePixelRatio ?? 1, 2);
      const bw = Math.floor(side * dpr);
      if (canvas.width !== bw || canvas.height !== bw) {
        canvas.width = bw;
        canvas.height = bw;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const s = side / W;
      ctx.save();
      ctx.translate((rect.width - side) / 2, (rect.height - side) / 2);
      ctx.scale(s, s);

      ctx.fillStyle = resolvedDiskRgb(diskProbe);
      ctx.fillRect(0, 0, W, W);

      const rgb = LINE_RGB[familyRef.current][darkRef.current ? "dark" : "light"];
      const c = W / 2;

      const env = 0.55 + 0.45 * Math.min(1, amp * 1.7 + activity * 0.12);
      const cell = W / GRID;
      ctx.translate(c, c);
      ctx.rotate(rot);
      ctx.translate(-c, -c);
      ctx.lineWidth = 1.1 / s;
      ctx.lineJoin = "round";
      for (let li = 0; li < LEVELS.length; li++) {
        const L = LEVELS[li]!;
        ctx.strokeStyle = `rgba(${rgb}, ${LEVEL_ALPHA[li]! * env})`;
        ctx.beginPath();
        for (let y = 0; y < GRID; y++) {
          const row = y * P;
          for (let x = 0; x < GRID; x++) {
            if (radius[row + x]! > 1.02 && radius[row + x + P + 1]! > 1.02) continue;
            const a = field[row + x]! - L;
            const b = field[row + x + 1]! - L;
            const cc = field[row + x + P + 1]! - L;
            const d = field[row + x + P]! - L;
            const idx = (a > 0 ? 8 : 0) | (b > 0 ? 4 : 0) | (cc > 0 ? 2 : 0) | (d > 0 ? 1 : 0);
            if (idx === 0 || idx === 15) continue;
            const x0 = x * cell;
            const y0 = y * cell;
            const top: [number, number] = [x0 + (a / (a - b)) * cell, y0];
            const right: [number, number] = [x0 + cell, y0 + (b / (b - cc)) * cell];
            const bottom: [number, number] = [x0 + (d / (d - cc)) * cell, y0 + cell];
            const left: [number, number] = [x0, y0 + (a / (a - d)) * cell];
            const seg = (p1: [number, number], p2: [number, number]) => {
              ctx.moveTo(p1[0], p1[1]);
              ctx.lineTo(p2[0], p2[1]);
            };
            switch (idx) {
              case 1:
              case 14:
                seg(left, bottom);
                break;
              case 2:
              case 13:
                seg(bottom, right);
                break;
              case 3:
              case 12:
                seg(left, right);
                break;
              case 4:
              case 11:
                seg(top, right);
                break;
              case 5:
                seg(top, right);
                seg(left, bottom);
                break;
              case 6:
              case 9:
                seg(top, bottom);
                break;
              case 7:
              case 8:
                seg(left, top);
                break;
              case 10:
                seg(left, top);
                seg(bottom, right);
                break;
            }
          }
        }
        ctx.stroke();
      }
      ctx.restore();
    };

    const updateReadout = () => {
      const el = readoutRef?.current;
      if (!el || frame % 9 !== 0) return;
      if (recordingRef.current && analyserRef.current) {
        // dB above the adaptive room noise floor — silence reads 0, voice reads positive.
        const db = Math.max(0, Math.round(20 * Math.log10(level / Math.max(noiseFloor, 1e-4))));
        const speaking = pitchHz > 0 && (voiceActive || unvoicedFrames < 40);
        el.textContent = `+${db} dB · ${speaking ? Math.round(pitchHz / 5) * 5 : "—"} Hz`;
      } else {
        el.textContent = "— dB · — Hz";
      }
    };

    const drawFrame = () => {
      frame++;
      readAudio();
      computeField();
      paint();
      updateReadout();
    };

    const scheduleLoop = () => {
      if (cancelled || reduceMotion || loopRunning) return;
      loopRunning = true;
      const step = () => {
        if (cancelled) {
          loopRunning = false;
          return;
        }
        // Idle runs at half rate — the low-energy drift doesn't need 60fps.
        if (recordingRef.current || frame % 2 === 0) drawFrame();
        else frame++;
        rafId = requestAnimationFrame(step);
      };
      rafId = requestAnimationFrame(step);
    };

    kickLoopRef.current = () => {
      if (reduceMotion) drawFrame();
      else scheduleLoop();
    };
    kickLoopRef.current();

    const ro = new ResizeObserver(() => {
      if (!cancelled) paint();
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
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refs carry live props; loop is kicked below.
  }, []);

  useEffect(() => {
    kickLoopRef.current?.();
  }, [recording, analyser, family, resolvedTheme]);

  return <canvas ref={canvasRef} className={className} width={W} height={W} aria-hidden />;
}
