"use client";

import { type RefObject, useEffect, useRef } from "react";

const W = 280;
const PARTICLE_COUNT = 3600;
const PITCH_MIN = 70;
const PITCH_MAX = 400;
const PITCH_WINDOW = 1024;
const MODES = [
  [1, 2],
  [1, 3],
  [1, 4],
  [1, 5],
  [1, 6],
  [2, 3],
  [2, 4],
  [2, 5],
  [2, 6],
  [2, 7],
  [3, 4],
  [3, 5],
  [3, 6],
  [3, 7],
  [3, 8],
  [4, 5],
  [4, 6],
  [4, 7],
  [4, 8],
  [4, 9]
] as const;
const IDLE_MODES = [3, 0, 2, 5, 7, 10, 13, 16, 19] as const;
const IDLE_MODE_FRAMES = 720;

type Props = {
  className?: string;
  recording: boolean;
  analyser: AnalyserNode | null;
  readoutRef?: RefObject<HTMLElement | null>;
};

type Particle = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  phase: number;
};

function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function createParticle(random: () => number): Particle {
  const angle = random() * Math.PI * 2;
  const radius = Math.sqrt(random()) * 0.965;
  return {
    x: Math.cos(angle) * radius,
    y: Math.sin(angle) * radius,
    vx: 0,
    vy: 0,
    phase: random() * Math.PI * 2
  };
}

function chladniForce(x: number, y: number, m: number, n: number) {
  const pi = Math.PI;
  const mx = m * pi * x;
  const my = m * pi * y;
  const nx = n * pi * x;
  const ny = n * pi * y;
  const cmx = Math.cos(mx);
  const cmy = Math.cos(my);
  const cnx = Math.cos(nx);
  const cny = Math.cos(ny);
  const field = cnx * cmy - cmx * cny;
  const dx = -n * pi * Math.sin(nx) * cmy + m * pi * Math.sin(mx) * cny;
  const dy = -m * pi * cnx * Math.sin(my) + n * pi * cmx * Math.sin(ny);
  return { x: -field * dx, y: -field * dy };
}

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
      "position:absolute;width:0;height:0;opacity:0;pointer-events:none;color:var(--color-text-primary)";
    parent.prepend(diskProbe, particleProbe);

    const random = seededRandom(406);
    const particles = Array.from({ length: PARTICLE_COUNT }, () => createParticle(random));
    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    let reduceMotion = motionQuery.matches;
    let timeData: Uint8Array<ArrayBuffer> | null = null;
    let wave: Float32Array | null = null;
    let level = 0;
    let noiseFloor = 0.012;
    let envelope = 0;
    let pitchHz = 0;
    let pitchAge = 99;
    let modeIndex: number = IDLE_MODES[0];
    let idleModeIndex = 0;
    let candidateMode: number = modeIndex;
    let candidateFrames = 0;
    let boundsW = 0;
    let boundsH = 0;
    let frame = 0;
    let rafId = 0;
    let readoutTimer = 0;
    let cancelled = false;
    let loopRunning = false;
    let inView = true;
    let diskColor = "";
    let particleColor = "";

    const readAudio = () => {
      const node = recordingRef.current ? analyserRef.current : null;
      if (!node) {
        level = 0;
        envelope += (0 - envelope) * 0.08;
        pitchHz = 0;
        pitchAge = 99;
        candidateMode = modeIndex;
        candidateFrames = 0;
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
      noiseFloor += (level - noiseFloor) * (level < noiseFloor ? 0.2 : 0.002);
      const audible = Math.max(0, level - noiseFloor - 0.008);
      const targetEnvelope = Math.min(1, audible * 9);
      envelope += (targetEnvelope - envelope) * (targetEnvelope > envelope ? 0.22 : 0.06);

      if (wave && frame % 3 === 0 && envelope > 0.06) {
        const sampleRate = node.context.sampleRate;
        const lagMin = Math.floor(sampleRate / PITCH_MAX);
        const lagMax = Math.min(Math.ceil(sampleRate / PITCH_MIN), wave.length - PITCH_WINDOW);
        let baseEnergy = 0;
        for (let i = 0; i < PITCH_WINDOW; i += 2) baseEnergy += wave[i]! * wave[i]!;
        let bestScore = 0;
        let bestLag = 0;
        for (let lag = lagMin; lag <= lagMax; lag += 2) {
          let score = 0;
          for (let i = 0; i < PITCH_WINDOW; i += 2) score += wave[i]! * wave[i + lag]!;
          if (score > bestScore) {
            bestScore = score;
            bestLag = lag;
          }
        }
        if (bestLag && bestScore > baseEnergy * 0.3) {
          const nextPitch = sampleRate / bestLag;
          pitchHz = pitchHz ? pitchHz + (nextPitch - pitchHz) * 0.18 : nextPitch;
          pitchAge = 0;
          const tone = Math.min(
            1,
            Math.max(0, Math.log(pitchHz / 90) / Math.log(320 / 90))
          );
          const nextMode = Math.round(tone * (MODES.length - 1));
          if (candidateMode === nextMode) {
            candidateFrames++;
          } else {
            candidateMode = nextMode;
            candidateFrames = 1;
          }
          if (candidateFrames >= 4) modeIndex = candidateMode;
        } else {
          pitchAge++;
        }
      } else {
        pitchAge++;
      }
    };

    const advanceParticles = (steps = 1) => {
      const active = recordingRef.current && analyserRef.current;
      if (!active && frame > 0 && frame % IDLE_MODE_FRAMES === 0) {
        idleModeIndex = (idleModeIndex + 1) % IDLE_MODES.length;
        modeIndex = IDLE_MODES[idleModeIndex]!;
      }
      const [modeM, modeN] = MODES[modeIndex]!;
      const vibration = active ? 0.00035 + envelope * 0.0024 : 0.0002;
      const modeScale = Math.min(1.4, 5 / modeN);
      const attraction = (active ? 0.0002 + envelope * 0.00016 : 0.00022) * modeScale;
      const damping = active ? 0.91 : 0.9;

      for (let step = 0; step < steps; step++) {
        for (const particle of particles) {
          const force = chladniForce(particle.x, particle.y, modeM, modeN);
          const pulse = Math.sin(frame * 0.035 + particle.phase);
          particle.vx =
            particle.vx * damping + force.x * attraction + (random() - 0.5) * vibration * (1 + pulse * 0.25);
          particle.vy =
            particle.vy * damping + force.y * attraction + (random() - 0.5) * vibration * (1 + pulse * 0.25);
          particle.x += particle.vx;
          particle.y += particle.vy;
          const radius = Math.hypot(particle.x, particle.y);
          if (radius > 0.978) {
            const nx = particle.x / radius;
            const ny = particle.y / radius;
            particle.x = nx * 0.976;
            particle.y = ny * 0.976;
            const outward = particle.vx * nx + particle.vy * ny;
            if (outward > 0) {
              particle.vx -= nx * outward * 1.8;
              particle.vy -= ny * outward * 1.8;
            }
          }
        }
      }
    };

    const resolveColors = () => {
      const disk = getComputedStyle(diskProbe).backgroundColor;
      const ink = getComputedStyle(particleProbe).color;
      diskColor = disk && disk !== "rgba(0, 0, 0, 0)" ? disk : "rgb(245, 245, 242)";
      particleColor = ink || "rgb(15, 12, 10)";
    };

    const paint = () => {
      if (!boundsW || !boundsH) {
        const rect = parent.getBoundingClientRect();
        boundsW = rect.width;
        boundsH = rect.height;
      }
      const side = Math.max(1, Math.min(boundsW, boundsH));
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const pixels = Math.max(1, Math.round(side * dpr));
      if (canvas.width !== pixels || canvas.height !== pixels) {
        canvas.width = pixels;
        canvas.height = pixels;
      }
      if (!diskColor || !particleColor) resolveColors();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, side, side);
      ctx.fillStyle = diskColor;
      ctx.fillRect(0, 0, side, side);
      const center = side / 2;
      const radius = side * 0.488;
      const size = Math.max(0.8, side / 225);
      ctx.fillStyle = particleColor;
      ctx.globalAlpha = 0.62 + envelope * 0.2;
      for (const particle of particles) {
        const x = center + particle.x * radius;
        const y = center + particle.y * radius;
        ctx.fillRect(x - size / 2, y - size / 2, size, size);
      }
      ctx.globalAlpha = 1;
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
      advanceParticles();
      paint();
      updateReadout();
    };

    for (let i = 0; i < 150; i++) advanceParticles();

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
          frame++;
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
        paint();
      } else {
        scheduleLoop();
      }
      syncReadoutTimer();
    };
    motionQuery.addEventListener("change", handleMotionPreference);

    const syncTheme = () => {
      resolveColors();
      paint();
    };
    const themeObserver = new MutationObserver(syncTheme);
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"]
    });

    const resizeObserver = new ResizeObserver((entries) => {
      const rect = entries.at(-1)?.contentRect;
      if (rect?.width) {
        boundsW = rect.width;
        boundsH = rect.height;
      }
      paint();
    });
    resizeObserver.observe(parent);

    const intersectionObserver = new IntersectionObserver((entries) => {
      inView = entries.at(-1)?.isIntersecting ?? true;
    });
    intersectionObserver.observe(parent);

    kickLoopRef.current = () => {
      if (reduceMotion) {
        paint();
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

  return <canvas ref={canvasRef} className={className} width={W} height={W} aria-hidden />;
}
