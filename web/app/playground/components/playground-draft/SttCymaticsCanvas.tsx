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
const GRID = 96;
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
/**
 * Coval symbol — verbatim path from the brand site's coval-symbol.svg (482×482 viewBox).
 * Rasterized once into a signed-distance field so the contour renderer draws the mark itself:
 * silence resolves the plate into the logo, speech dissolves it back into Chladni designs.
 */
const LOGO_PATH =
  "M118.185 7.71157C199.45 -2.57057 281.688 -2.57048 362.953 7.71157C420.672 15.0146 466.124 60.4662 473.427 118.185C483.709 199.45 483.709 281.688 473.427 362.953C466.123 420.671 420.671 466.124 362.953 473.427C281.688 483.709 199.45 483.709 118.185 473.427C60.4672 466.124 15.0153 420.671 7.71147 362.953C-2.57058 281.688 -2.5704 199.45 7.71147 118.185C15.0144 60.4667 60.4668 15.0151 118.185 7.71157ZM240.572 404.537C236.053 404.537 232.708 405.538 229.701 406.789C229.339 406.944 228.973 407.102 228.603 407.265C227.722 407.655 226.855 408.043 225.994 408.442C222.067 410.263 217.719 412.454 213.068 414.944C200.691 421.572 187.442 429.488 175.276 436.591L173.182 437.813C172.203 438.383 171.241 438.961 170.279 439.517C169.857 439.76 169.421 439.989 168.988 440.235L155.055 448.277C211.95 453.442 269.194 453.442 326.089 448.277L312.348 440.343C311.85 440.06 311.349 439.796 310.865 439.517C309.732 438.862 308.592 438.187 307.436 437.513L305.857 436.591C293.692 429.489 280.444 421.571 268.07 414.944C263.341 412.412 258.925 410.19 254.946 408.351C254.17 407.991 253.385 407.646 252.592 407.293C252.197 407.119 251.806 406.949 251.421 406.784C248.419 405.536 245.081 404.537 240.572 404.537ZM108.169 281.256C104.361 276.724 97.2366 277.241 94.1223 282.275L83.5509 299.343C78.0887 308.827 72.5221 318.189 67.971 326.072C60.7477 338.583 53.737 353.487 52.1647 368.669C50.6801 383.013 53.9485 398.047 68.5199 412.618C83.0912 427.189 98.1206 430.458 112.464 428.974C127.647 427.402 142.554 420.391 155.067 413.167C162.963 408.608 172.345 403.031 181.846 397.559L181.761 397.565L198.841 386.999C203.874 383.886 204.394 376.762 199.865 372.953L189.367 364.13C162.76 342.607 138.501 318.341 116.98 291.732L108.169 281.256ZM386.982 282.275C383.868 277.241 376.744 276.724 372.936 281.256L364.86 290.86C342.961 318.065 318.198 342.832 291.001 364.741L281.239 372.953C276.708 376.761 277.224 383.885 282.258 386.999L299.286 397.559C308.791 403.032 318.179 408.607 326.077 413.167C338.589 420.391 353.492 427.402 368.674 428.974C383.018 430.458 398.052 427.19 412.624 412.618C427.195 398.047 430.458 383.012 428.973 368.669C427.401 353.487 420.396 338.583 413.173 326.072C408.607 318.164 403.021 308.767 397.542 299.253L386.982 282.275ZM265.263 134.807C250.73 123.54 230.408 123.539 215.875 134.807C185.557 158.315 158.32 185.551 134.812 215.869C123.544 230.403 123.545 250.724 134.812 265.258C158.32 295.576 185.557 322.813 215.875 346.32C230.409 357.59 250.729 357.589 265.263 346.32C295.582 322.812 322.818 295.576 346.326 265.258C357.594 250.724 357.595 230.403 346.326 215.869C322.818 185.552 295.581 158.314 265.263 134.807ZM32.8554 154.97C27.6819 211.913 27.6826 269.208 32.8554 326.151L39.3692 314.872C40.1347 313.487 40.8805 312.143 41.6216 310.86C49.4015 297.385 58.6446 282.167 66.194 268.07C69.7696 261.393 72.7369 255.343 74.847 250.255C75.8718 247.545 76.6464 244.502 76.6466 240.561C76.6466 236.694 75.9043 233.69 74.9092 231.019C72.7959 225.902 69.8031 219.802 66.194 213.063C58.6444 198.965 49.4015 183.748 41.6216 170.273C40.8164 168.878 39.9987 167.418 39.1655 165.904L32.8554 154.97ZM436.602 175.344C429.496 187.516 421.577 200.678 414.944 213.063C412.873 216.93 411.006 220.589 409.387 223.968C407.304 228.828 404.548 233.099 404.548 240.606C404.548 245.084 405.537 248.407 406.772 251.392C406.959 251.829 407.146 252.274 407.344 252.722C407.654 253.418 407.961 254.106 408.278 254.788C410.13 258.808 412.378 263.279 414.944 268.07C422.494 282.168 431.737 297.384 439.517 310.86C439.924 311.566 440.317 312.301 440.733 313.039L448.277 326.1C453.442 269.216 453.443 211.984 448.283 155.101L436.602 175.344ZM112.464 52.1592C98.1207 50.6751 83.0912 53.9435 68.5199 68.5144C53.949 83.0853 50.681 98.1152 52.1647 112.458C53.7362 127.642 60.7471 142.549 67.971 155.061C72.5227 162.945 78.0881 172.294 83.5509 181.778L94.1223 198.858C97.2364 203.892 104.36 204.409 108.169 199.876L116.98 189.401C138.502 162.79 162.769 138.521 189.379 116.997L199.865 108.186C204.397 104.377 203.875 97.2477 198.841 94.1337L181.761 83.5679C172.261 78.0984 162.963 72.5242 155.067 67.9654C142.554 60.7414 127.647 53.7306 112.464 52.1592ZM412.624 68.5144C398.052 53.943 383.018 50.6747 368.674 52.1592C353.492 53.7308 338.589 60.742 326.077 67.9654C318.182 72.5237 308.798 78.0976 299.298 83.5679L282.258 94.1337C277.224 97.2478 276.707 104.378 281.239 108.186L291.018 116.397C318.191 138.288 342.93 163.038 364.815 190.216L372.936 199.876C376.744 204.409 383.868 203.892 386.982 198.858L397.548 181.778C403.026 172.264 408.608 162.968 413.173 155.061C420.396 142.549 427.402 127.641 428.973 112.458C430.457 98.1155 427.194 83.0849 412.624 68.5144ZM326.19 32.8555C269.253 27.6811 211.964 27.6744 155.027 32.8442L168.179 40.4389C168.89 40.8406 169.597 41.2222 170.279 41.616C171.785 42.486 173.305 43.3908 174.851 44.2929L175.344 44.5758C187.492 51.6687 200.713 59.572 213.068 66.1884C218.135 68.9018 222.839 71.269 227.035 73.1776C227.275 73.2888 227.518 73.3953 227.759 73.5058C228.611 73.8877 229.448 74.2346 230.255 74.5754C233.133 75.7285 236.349 76.6354 240.617 76.6354C244.347 76.6352 247.275 75.9473 249.876 75.0055C255.042 72.8877 261.225 69.854 268.07 66.1884C282.168 58.6387 297.389 49.3963 310.865 41.616C312.572 40.6307 314.369 39.6122 316.253 38.5883L326.19 32.8555Z";
const LOGO_VIEW = 482;
/** Mark width as a fraction of the plate; chaos keeps the ring around it alive. */
const LOGO_SPAN = 0.62;
/** SDF softness (grid cells) — tight, so all contour levels collapse into crisp mark edges. */
const LOGO_SOFT = 2;
/**
 * Field plateau inside/outside the ink, kept well under the first contour level (0.24): the
 * mark contributes ONLY its own outline — nothing ever draws inside the shapes.
 */
const LOGO_CLAMP = 0.1;
/**
 * Idle ritual, in rAF ticks (~1/60s): chaos → the mark resolves out of it → holds crisp →
 * dissolves back. One appearance per cycle, phased so first paint lands on the resolved mark.
 */
const LOGO_CYCLE = 840;
const LOGO_CYCLE_OFFSET = 504;

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
 * while gated amplitude drives contrast, ripple, cycle speed, and a slow rotation. At idle the
 * plate performs a slow ritual: out of drifting cymatic chaos the Coval mark resolves, holds
 * crisp and axis-aligned, then dissolves back — and speech suspends it for the voice's geometry.
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
    // Rasterize the mark once, then chamfer-transform into a signed distance field whose zero
    // contour traces the mark's exact edges; the clamp keeps every other contour level away,
    // so the mark renders as pure outline — nothing draws inside its shapes.
    const buildLogoField = () => {
      const off = document.createElement("canvas");
      off.width = off.height = P;
      const octx = off.getContext("2d", { willReadFrequently: true });
      if (!octx) return null;
      const span = P * LOGO_SPAN;
      const sc = span / LOGO_VIEW;
      octx.setTransform(sc, 0, 0, sc, (P - span) / 2, (P - span) / 2);
      octx.fill(new Path2D(LOGO_PATH), "evenodd");
      const img = octx.getImageData(0, 0, P, P).data;
      const dOut = new Float32Array(P * P);
      const dIn = new Float32Array(P * P);
      for (let i = 0; i < P * P; i++) {
        const inside = img[i * 4 + 3]! > 127;
        dOut[i] = inside ? 0 : 1e6;
        dIn[i] = inside ? 1e6 : 0;
      }
      const chamfer = (d: Float32Array) => {
        for (let y = 0; y < P; y++) {
          for (let x = 0; x < P; x++) {
            const i = y * P + x;
            if (x > 0) d[i] = Math.min(d[i]!, d[i - 1]! + 3);
            if (y > 0) {
              d[i] = Math.min(d[i]!, d[i - P]! + 3);
              if (x > 0) d[i] = Math.min(d[i]!, d[i - P - 1]! + 4);
              if (x < GRID) d[i] = Math.min(d[i]!, d[i - P + 1]! + 4);
            }
          }
        }
        for (let y = P - 1; y >= 0; y--) {
          for (let x = P - 1; x >= 0; x--) {
            const i = y * P + x;
            if (x < GRID) d[i] = Math.min(d[i]!, d[i + 1]! + 3);
            if (y < GRID) {
              d[i] = Math.min(d[i]!, d[i + P]! + 3);
              if (x < GRID) d[i] = Math.min(d[i]!, d[i + P + 1]! + 4);
              if (x > 0) d[i] = Math.min(d[i]!, d[i + P - 1]! + 4);
            }
          }
        }
      };
      chamfer(dOut);
      chamfer(dIn);
      const base = new Float32Array(P * P);
      const mask = new Float32Array(P * P);
      for (let i = 0; i < P * P; i++) {
        let sd = (dOut[i]! > 0 ? dOut[i]! - 1.5 : 1.5 - dIn[i]!) / 3;
        // Subpixel edge from the rasterizer's antialiased alpha — kills marching-square wiggle.
        if (Math.abs(sd) <= 1.4) sd = 0.5 - img[i * 4 + 3]! / 255;
        base[i] = Math.max(-LOGO_CLAMP, Math.min(LOGO_CLAMP, Math.tanh(sd / LOGO_SOFT)));
        // Blend mask: pure mark in the center, chaos keeps swirling in the outer ring.
        const m = Math.min(1, Math.max(0, (radius[i]! - 0.5) / 0.45));
        mask[i] = 1 - m * m * 0.35;
      }
      return { base, mask };
    };
    const logo = buildLogoField();
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
    let logoW = 0.95;
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
        // Idle flow: the under-texture drifts through the whole design library on a slow
        // crossfade, so the field behind the resolved mark never sits still.
        targets.fill(0);
        const drift = frame * 0.0022;
        const di = Math.floor(drift) % DESIGNS.length;
        const fr = drift - Math.floor(drift);
        targets[di] = 1 - fr;
        targets[(di + 1) % DESIGNS.length] = fr;
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
      // The idle ritual: out of symmetric cymatic chaos the mark resolves, holds crisp, then
      // dissolves back — vibrations forming the design, once per cycle. Speech suspends it.
      const ss = (a: number, b: number, x: number) => {
        const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
        return t * t * (3 - 2 * t);
      };
      const cyc = ((frame + LOGO_CYCLE_OFFSET) % LOGO_CYCLE) / LOGO_CYCLE;
      const logoT = (1 - activity) * 0.95 * ss(0.32, 0.5, cyc) * (1 - ss(0.86, 0.98, cyc));
      logoW += (logoT - logoW) * 0.06;
      ripplePhase += 0.012 + activity * (0.05 + amp * 0.12);
      // The mark never spins (brand rule): base drift fades with the logo blend, and while
      // resolved the rotation eases onto the nearest quarter turn of its D4 symmetry.
      rotSpeed += ((1 - logoW) * 0.0004 + activity * (0.00165 + amp * 0.005) - rotSpeed) * 0.05;
      rot += rotSpeed;
      if (logoW > 0.5) {
        const q = Math.PI / 2;
        rot += (Math.round(rot / q) * q - rot) * (logoW - 0.5) * 0.06;
      }
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
      const rippleAmp = amp * 0.35 + 0.012 * (1 - activity);
      const plateNorm = 1 / (2 * sumW + 1e-6);
      const post = 1 / (1 + rippleAmp);
      const lw = logo ? logoW : 0;
      for (let i = 0; i < P * P; i++) {
        const r = radius[i]!;
        let v = field[i]! * plateNorm;
        if (lw > 0.003) v += (logo!.base[i]! - v) * lw * logo!.mask[i]!;
        const ripple = rippleAmp > 0.01 ? rippleAmp * Math.cos(r * 14 - ripplePhase) * (1 - Math.min(1, r)) : 0;
        field[i] = (v + ripple) * post;
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
