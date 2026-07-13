"use client";

import { type RefObject, useEffect, useRef } from "react";

const MODES = [
  [3, 1],
  [5, 1],
  [5, 2],
  [7, 2],
  [8, 3],
  [9, 4],
  [10, 4]
] as const;
const DEFAULT_MODE = 1;
const PITCH_MIN = 70;
const PITCH_MAX = 400;
const PITCH_WINDOW = 1024;
const TONE_LO = 90;
const TONE_HI = 320;
const REST_VSTR = 0.02;

type Props = {
  className?: string;
  recording: boolean;
  analyser: AnalyserNode | null;
  readoutRef?: RefObject<HTMLElement | null>;
};

type Particle = {
  x: number;
  y: number;
  h: number;
};

export function SttCymaticsCanvas({ className, recording, analyser, readoutRef }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recordingRef = useRef(recording);
  recordingRef.current = recording;
  const analyserRef = useRef(analyser);
  analyserRef.current = analyser;
  const kickLoopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const parent = canvas?.parentElement;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !parent || !ctx) return;

    const diskProbe = document.createElement("span");
    const particleProbe = document.createElement("span");
    diskProbe.setAttribute("aria-hidden", "true");
    particleProbe.setAttribute("aria-hidden", "true");
    diskProbe.style.cssText =
      "position:absolute;width:0;height:0;opacity:0;pointer-events:none;background:var(--playground-stt-disk)";
    particleProbe.style.cssText =
      "position:absolute;width:0;height:0;opacity:0;pointer-events:none;color:var(--color-text-secondary)";
    parent.prepend(diskProbe, particleProbe);

    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    let reduceMotion = motionQuery.matches;
    const particles: Particle[] = [];
    let width = 0;
    let height = 0;
    let cx = 0;
    let cy = 0;
    let radius = 0;
    let dotRadius = 1;
    let vStr = REST_VSTR;
    let kick = 0;
    let modeIndex = DEFAULT_MODE;
    let timeData: Uint8Array<ArrayBuffer> | null = null;
    let wave: Float32Array | null = null;
    const lagScores = new Float32Array(512);
    const pitchHist = new Float32Array(5);
    let pitchHistLen = 0;
    let pendingIdx = -1;
    let pendingCount = 0;
    let swapCooldown = 0;
    let settleFrames = 0;
    let level = 0;
    let noiseFloor = 0.012;
    let envelope = 0;
    let presence = 0;
    let silenceRun = 99;
    let pitchHz = 0;
    let pitchAge = 99;
    let frame = 0;
    let rafId = 0;
    let readoutTimer = 0;
    let cancelled = false;
    let loopRunning = false;
    let inView = true;
    let diskColor = "";
    let particleColor = "";

    const spawnParticle = (particle: Particle) => {
      const angle = Math.random() * Math.PI * 2;
      const spread = Math.sqrt(Math.random()) * radius;
      particle.x = cx + Math.cos(angle) * spread;
      particle.y = cy + Math.sin(angle) * spread;
      particle.h = Math.random() * Math.PI * 2;
    };

    const clampToPlate = (particle: Particle) => {
      const dx = particle.x - cx;
      const dy = particle.y - cy;
      const distance = Math.hypot(dx, dy);
      if (distance > radius) {
        const scale = radius / distance;
        particle.x = cx + dx * scale;
        particle.y = cy + dy * scale;
      }
    };

    const stepRest = (particle: Particle) => {
      particle.h += (Math.random() - 0.5) * 0.4;
      particle.x += Math.cos(particle.h) * 0.6;
      particle.y += Math.sin(particle.h) * 0.6;
      const dx = particle.x - cx;
      const dy = particle.y - cy;
      if (Math.hypot(dx, dy) > radius * 0.98) {
        particle.h = Math.atan2(-dy, -dx) + (Math.random() - 0.5) * 0.9;
        clampToPlate(particle);
      }
    };

    const stepVibration = (particle: Particle) => {
      const span = 2 * radius;
      const nx = (particle.x - (cx - radius)) / span;
      const ny = (particle.y - (cy - radius)) / span;
      const mode = MODES[modeIndex]!;
      const m = mode[0];
      const n = mode[1];
      const pi = Math.PI;
      const snx = Math.sin(pi * n * nx);
      const cnx = Math.cos(pi * n * nx);
      const smx = Math.sin(pi * m * nx);
      const cmx = Math.cos(pi * m * nx);
      const sny = Math.sin(pi * n * ny);
      const cny = Math.cos(pi * n * ny);
      const smy = Math.sin(pi * m * ny);
      const cmy = Math.cos(pi * m * ny);
      const value = snx * smy + smx * sny;
      const dfdx = pi * (n * cnx * smy + m * cmx * sny);
      const dfdy = pi * (m * snx * cmy + n * smx * cny);
      const amplitude = Math.max(0.002, vStr * Math.abs(value));
      const drive = Math.min(1.6, vStr * 50);
      const driftScale = Math.min(0.006, 0.00016 * drive * Math.abs(value) * Math.hypot(dfdx, dfdy));
      const gradMag = Math.hypot(dfdx, dfdy) || 1;
      const sign = value > 0 ? 1 : -1;
      particle.x += (Math.random() * 2 - 1) * amplitude * span - (sign * dfdx / gradMag) * driftScale * span;
      particle.y += (Math.random() * 2 - 1) * amplitude * span - (sign * dfdy / gradMag) * driftScale * span;
      clampToPlate(particle);
    };

    const rebuildParticles = () => {
      const side = Math.min(width, height);
      const count = Math.min(4200, Math.max(1200, Math.round(3600 * (side / 560) ** 2)));
      while (particles.length < count) {
        const particle = { x: 0, y: 0, h: 0 };
        spawnParticle(particle);
        particles.push(particle);
      }
      if (particles.length > count) particles.length = count;
    };

    const readAudio = () => {
      const node = recordingRef.current ? analyserRef.current : null;
      if (!node) {
        level = 0;
        envelope += (0 - envelope) * 0.08;
        presence += (0 - presence) * 0.1;
        silenceRun = 99;
        pitchHz = 0;
        pitchAge = 99;
        pitchHistLen = 0;
        pendingIdx = -1;
        pendingCount = 0;
        swapCooldown = 0;
        settleFrames = 0;
        modeIndex = DEFAULT_MODE;
        kick = 0;
        vStr = REST_VSTR;
        return;
      }

      if (!timeData || timeData.length !== node.fftSize) {
        timeData = new Uint8Array(node.fftSize);
        wave = new Float32Array(node.fftSize);
      }
      node.getByteTimeDomainData(timeData);
      let sum = 0;
      for (let i = 0; i < timeData.length; i++) {
        const sample = (timeData[i]! - 128) / 128;
        wave![i] = sample;
        sum += sample * sample;
      }
      level = Math.sqrt(sum / timeData.length);
      const rise = envelope > 0.15 ? 0.0002 : 0.002;
      noiseFloor += (level - noiseFloor) * (level < noiseFloor ? 0.2 : rise);
      const audible = Math.max(0, level - noiseFloor - 0.008);
      const targetEnvelope = Math.min(1, audible * 9);
      envelope += (targetEnvelope - envelope) * (targetEnvelope > envelope ? 0.22 : 0.06);
      if (targetEnvelope > 0.08) {
        silenceRun = 0;
        presence += (1 - presence) * 0.12;
      } else {
        silenceRun++;
        if (silenceRun > 45) presence += (0 - presence) * 0.03;
      }

      if (wave && frame % 3 === 0 && envelope > 0.06) {
        const sampleRate = node.context.sampleRate;
        const lagMin = Math.floor(sampleRate / PITCH_MAX);
        const lagMax = Math.min(Math.ceil(sampleRate / PITCH_MIN), wave.length - PITCH_WINDOW);
        let baseEnergy = 0;
        for (let i = 0; i < PITCH_WINDOW; i += 2) baseEnergy += wave[i]! * wave[i]!;
        let bestScore = 0;
        for (let lag = lagMin; lag <= lagMax; lag += 2) {
          let score = 0;
          for (let i = 0; i < PITCH_WINDOW; i += 2) score += wave[i]! * wave[i + lag]!;
          lagScores[(lag - lagMin) >> 1] = score;
          if (score > bestScore) bestScore = score;
        }
        let chosenLag = 0;
        if (bestScore > baseEnergy * 0.3) {
          const cutoff = bestScore * 0.9;
          for (let lag = lagMin; lag <= lagMax; lag += 2) {
            if (lagScores[(lag - lagMin) >> 1]! >= cutoff) {
              chosenLag = lag;
              break;
            }
          }
        }
        if (chosenLag) {
          pitchHist[pitchHistLen % 5] = sampleRate / chosenLag;
          pitchHistLen++;
          const window = Array.from(pitchHist.slice(0, Math.min(pitchHistLen, 5))).sort(
            (a, b) => a - b
          );
          const median = window[window.length >> 1]!;
          pitchHz = pitchHz ? pitchHz + (median - pitchHz) * 0.15 : median;
          pitchAge = 0;
          const t = Math.min(
            1,
            Math.max(0, Math.log(pitchHz / TONE_LO) / Math.log(TONE_HI / TONE_LO))
          );
          const target = t * (MODES.length - 1);
          const idx = Math.round(target);
          if (idx === pendingIdx) {
            pendingCount++;
          } else {
            pendingIdx = idx;
            pendingCount = 0;
          }
          if (swapCooldown > 0) swapCooldown--;
          if (
            pendingIdx !== modeIndex &&
            pendingCount >= 12 &&
            swapCooldown === 0 &&
            Math.abs(target - modeIndex) > 0.75
          ) {
            modeIndex = pendingIdx;
            kick = 0.02;
            swapCooldown = 20;
            settleFrames = 110;
          }
        } else {
          pitchAge++;
        }
      } else {
        pitchAge++;
      }

      kick *= 0.95;
      vStr = 0.006 + envelope * 0.022 + kick;
      if (settleFrames > 0) {
        settleFrames--;
        vStr = Math.max(vStr, 0.022);
      }
    };

    const resolveColors = () => {
      const disk = getComputedStyle(diskProbe).backgroundColor;
      const ink = getComputedStyle(particleProbe).color;
      diskColor = disk && disk !== "rgba(0, 0, 0, 0)" ? disk : "rgb(245, 245, 242)";
      particleColor = ink || "rgb(15, 12, 10)";
    };

    const paint = (animate: boolean) => {
      if (!width || !height) return;
      if (!diskColor || !particleColor) resolveColors();
      ctx.fillStyle = diskColor;
      ctx.fillRect(0, 0, width, height);
      const vibing = recordingRef.current && analyserRef.current;
      ctx.fillStyle = particleColor;
      ctx.beginPath();
      for (const particle of particles) {
        if (animate) {
          if (vibing && Math.random() < presence) stepVibration(particle);
          else stepRest(particle);
        }
        ctx.moveTo(particle.x + dotRadius, particle.y);
        ctx.arc(particle.x, particle.y, dotRadius, 0, Math.PI * 2);
      }
      ctx.fill();
    };

    const resize = () => {
      const rect = parent.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cx = width / 2;
      cy = height / 2;
      radius = Math.min(width, height) * 0.488;
      dotRadius = Math.min(1.6, Math.max(0.9, Math.min(width, height) * 0.0035));
      particles.length = 0;
      rebuildParticles();
      if (reduceMotion) {
        for (let i = 0; i < 90; i++) particles.forEach(stepVibration);
      }
      paint(false);
    };

    const updateReadout = (force = false) => {
      const output = readoutRef?.current;
      if (!output || (!force && frame % 9 !== 0)) return;
      if (recordingRef.current && analyserRef.current) {
        const db = Math.max(0, Math.round(20 * Math.log10(level / Math.max(noiseFloor, 0.0001))));
        output.textContent = `+${db} dB · ${pitchHz > 0 && pitchAge < 35 ? Math.round(pitchHz / 5) * 5 : "—"} Hz`;
      } else {
        output.textContent = "— dB · — Hz";
      }
    };

    const drawFrame = () => {
      frame++;
      readAudio();
      paint(true);
      updateReadout();
    };

    const scheduleLoop = () => {
      if (cancelled || reduceMotion || loopRunning) return;
      loopRunning = true;
      const step = () => {
        if (cancelled || reduceMotion) {
          loopRunning = false;
          return;
        }
        if (inView) drawFrame();
        rafId = requestAnimationFrame(step);
      };
      rafId = requestAnimationFrame(step);
    };

    const syncReadoutTimer = () => {
      const shouldRun = reduceMotion && recordingRef.current && analyserRef.current;
      if (shouldRun && !readoutTimer) {
        readoutTimer = window.setInterval(() => {
          frame += 3;
          readAudio();
          updateReadout(true);
        }, 400);
      } else if (!shouldRun && readoutTimer) {
        window.clearInterval(readoutTimer);
        readoutTimer = 0;
        updateReadout(true);
      }
    };

    const handleMotionPreference = (event: MediaQueryListEvent) => {
      reduceMotion = event.matches;
      if (reduceMotion) {
        cancelAnimationFrame(rafId);
        loopRunning = false;
        paint(false);
      } else {
        scheduleLoop();
      }
      syncReadoutTimer();
    };
    motionQuery.addEventListener("change", handleMotionPreference);

    const syncTheme = () => {
      resolveColors();
      paint(false);
    };
    const themeObserver = new MutationObserver(syncTheme);
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"]
    });

    const resizeObserver = new ResizeObserver(() => {
      resize();
    });
    resizeObserver.observe(parent);

    const intersectionObserver = new IntersectionObserver((entries) => {
      inView = entries.at(-1)?.isIntersecting ?? true;
    });
    intersectionObserver.observe(parent);

    resize();

    kickLoopRef.current = () => {
      if (reduceMotion) {
        paint(false);
        syncReadoutTimer();
      } else {
        scheduleLoop();
      }
    };
    kickLoopRef.current();
    syncTheme();

    return () => {
      cancelled = true;
      kickLoopRef.current = null;
      cancelAnimationFrame(rafId);
      if (readoutTimer) window.clearInterval(readoutTimer);
      motionQuery.removeEventListener("change", handleMotionPreference);
      themeObserver.disconnect();
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      diskProbe.remove();
      particleProbe.remove();
      loopRunning = false;
    };
  }, [readoutRef]);

  useEffect(() => {
    kickLoopRef.current?.();
  }, [recording, analyser]);

  return <canvas ref={canvasRef} className={className} aria-hidden />;
}
