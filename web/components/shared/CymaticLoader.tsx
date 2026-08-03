// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useEffect, useRef, useState } from "react";

// One beat: cloud → reassemble onto a figure → hold → disperse back to cloud.
const BEAT_MS = 3200;
const PARTICLE_HIDE_DELAY_MS = 950;

// Smoothstep easing for the cloud ↔ figure transitions.
function smoothstep(t: number) {
  const c = t < 0 ? 0 : t > 1 ? 1 : t;
  return c * c * (3 - 2 * c);
}

// How dispersed the interior is over one beat: 1 = diffuse cloud, 0 = settled on
// the figure. Holds the cloud briefly, reassembles, holds the figure, disperses.
function cloudiness(p: number) {
  if (p < 0.15) {
    return 1;
  }
  if (p < 0.45) {
    return 1 - smoothstep((p - 0.15) / 0.3);
  }
  if (p < 0.75) {
    return 0;
  }
  return smoothstep((p - 0.75) / 0.25);
}

// Superellipse exponent of the logo's rounded square (mirrors the static mark's
// 2 / 4.4 corner curve) — used to trace the particle border and to keep every
// particle inside the logo silhouette.
const SQUIRCLE_EXP = 4.4;
const SQUIRCLE_RADIUS = 0.84;

function getPrefersReducedMotion() {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

// A parametric curve segment: t in [0, 1) → unit-coordinate point.
type Segment = (t: number) => readonly [number, number];

function ellipseSegment(rx: number, ry: number, rotation: number): Segment {
  const c = Math.cos(rotation);
  const s = Math.sin(rotation);
  return (t) => {
    const angle = t * Math.PI * 2;
    const lx = rx * Math.cos(angle);
    const ly = ry * Math.sin(angle);
    return [lx * c - ly * s, lx * s + ly * c];
  };
}

// The logo's crossed loops — the mark's interior. The field resolves back onto
// these so border + interior together reform the static Coval logo.
const DEG_45 = Math.PI / 4;
const LOGO_LOOPS: ReadonlyArray<Segment> = [
  ellipseSegment(0.36, 0.8, DEG_45),
  ellipseSegment(0.36, 0.8, -DEG_45),
];

// Distribute `n` points evenly along a shape's segments.
function sampleShape(segments: ReadonlyArray<Segment>, n: number) {
  const points = new Float32Array(n * 2);
  const perSegment = Math.ceil(n / segments.length);
  for (let i = 0; i < n; i += 1) {
    const segment = segments[i % segments.length]!;
    const k = Math.floor(i / segments.length);
    const t = perSegment > 1 ? (k % perSegment) / perSegment : 0;
    const [x, y] = segment(t);
    points[i * 2] = x;
    points[i * 2 + 1] = y;
  }
  return points;
}

// How far the interior figures reach from center (keeps them inside the border).
const INTERIOR_EXTENT = 0.58;

// Classic square-plate Chladni function; its nodal lines (f = 0) are the figures
// on the reference plates. x, y in [0, 1].
function chladniValue(x: number, y: number, n: number, m: number) {
  const PI = Math.PI;
  return (
    Math.cos(n * PI * x) * Math.cos(m * PI * y) -
    Math.cos(m * PI * x) * Math.cos(n * PI * y)
  );
}

// The four reference plates, in order: (1,2), (2,3), (1,3), (3,4).
const CHLADNI_MODES: ReadonlyArray<readonly [number, number]> = [
  [1, 2],
  [2, 3],
  [1, 3],
  [3, 4],
];

// Scan the plate for points on a mode's nodal lines, in [-1, 1] normalized
// coordinates. |f| / |∇f| keeps the traced lines a uniform width regardless of
// how fast the field changes near each line.
function scanNodalLine(n: number, m: number) {
  const RES = 72;
  const DIST = 0.009;
  const eps = 1e-3;
  const points: Array<readonly [number, number]> = [];
  for (let i = 0; i <= RES; i += 1) {
    const x = i / RES;
    for (let j = 0; j <= RES; j += 1) {
      const y = j / RES;
      const f = chladniValue(x, y, n, m);
      const fx =
        (chladniValue(x + eps, y, n, m) - chladniValue(x - eps, y, n, m)) /
        (2 * eps);
      const fy =
        (chladniValue(x, y + eps, n, m) - chladniValue(x, y - eps, n, m)) /
        (2 * eps);
      const grad = Math.hypot(fx, fy) + 1e-6;
      if (Math.abs(f) / grad < DIST) {
        points.push([x * 2 - 1, y * 2 - 1]);
      }
    }
  }
  return points;
}

// Nodal candidates are size-independent, so scan each mode once at module load.
const CHLADNI_CANDIDATES = CHLADNI_MODES.map(([n, m]) => scanNodalLine(n, m));

// Pick `count` well-spread points from a mode's nodal candidates, scaled into
// the interior. A greedy min-distance pass spreads them so the sparse figure
// still reads; a deterministic shuffle keeps it stable across renders.
function selectNodalTargets(
  candidates: ReadonlyArray<readonly [number, number]>,
  count: number,
  scale: number,
  rand: () => number
) {
  const result = new Float32Array(count * 2);
  if (candidates.length === 0) {
    return result;
  }

  const order = candidates.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = order[i]!;
    order[i] = order[j]!;
    order[j] = tmp;
  }

  const chosen: number[] = [];
  const taken = new Uint8Array(candidates.length);
  // Keep the chosen points well apart so they read as a sparse swarm of dots
  // rather than merging into a continuous stroke.
  let minDistance = 0.22;
  while (chosen.length < count && minDistance > 0.05) {
    for (const k of order) {
      if (chosen.length >= count || taken[k]) {
        continue;
      }
      const x = candidates[k]![0] * scale;
      const y = candidates[k]![1] * scale;
      let ok = true;
      for (const c of chosen) {
        const dx = candidates[c]![0] * scale - x;
        const dy = candidates[c]![1] * scale - y;
        if (Math.hypot(dx, dy) < minDistance) {
          ok = false;
          break;
        }
      }
      if (ok) {
        chosen.push(k);
        taken[k] = 1;
      }
    }
    minDistance *= 0.7;
  }
  for (const k of order) {
    if (chosen.length >= count) {
      break;
    }
    if (!taken[k]) {
      chosen.push(k);
      taken[k] = 1;
    }
  }

  for (let i = 0; i < count; i += 1) {
    const k = chosen[i % chosen.length]!;
    result[i * 2] = candidates[k]![0] * scale;
    result[i * 2 + 1] = candidates[k]![1] * scale;
  }
  return result;
}

// The logo's rounded-square perimeter as exactly `n` anchors spaced by arc
// length (not angle) so the beads are evenly distributed around the ring.
function ringPoints(n: number) {
  const exponent = 2 / SQUIRCLE_EXP;
  const point = (theta: number): readonly [number, number] => {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    return [
      Math.sign(c) * Math.abs(c) ** exponent * 0.72,
      Math.sign(s) * Math.abs(s) ** exponent * 0.72,
    ];
  };

  // Walk the curve finely and accumulate arc length.
  const dense = 1600;
  const xs = new Float32Array(dense + 1);
  const ys = new Float32Array(dense + 1);
  const cum = new Float32Array(dense + 1);
  let [prevX, prevY] = point(0);
  xs[0] = prevX;
  ys[0] = prevY;
  for (let i = 1; i <= dense; i += 1) {
    const [x, y] = point((i / dense) * Math.PI * 2);
    cum[i] = cum[i - 1]! + Math.hypot(x - prevX, y - prevY);
    xs[i] = x;
    ys[i] = y;
    prevX = x;
    prevY = y;
  }
  const total = cum[dense]!;

  // Resample at even arc-length steps, interpolating within each segment so the
  // flat edges (where the trig parametrization races through the axis points)
  // stay evenly spaced instead of leaving gaps there.
  const points = new Float32Array(n * 2);
  let j = 0;
  for (let i = 0; i < n; i += 1) {
    const targetLen = (i / n) * total;
    while (j < dense - 1 && cum[j + 1]! < targetLen) {
      j += 1;
    }
    const segLen = cum[j + 1]! - cum[j]! || 1;
    const f = (targetLen - cum[j]!) / segLen;
    points[i * 2] = xs[j]! + (xs[j + 1]! - xs[j]!) * f;
    points[i * 2 + 1] = ys[j]! + (ys[j + 1]! - ys[j]!) * f;
  }
  return points;
}

/**
 * The particle mark that plays while data loads. One shared pool of particles
 * flows between a series of figures — the Coval logo (ring + crossed loops) and
 * the nodal figures of several Chladni plate modes, each traced with a ring
 * outline — dispersing into a diffuse cloud between each and reassembling onto
 * the next. Every settled figure carries a border; a particle on the ring in
 * one figure becomes an inner-line particle in the next, so the outline
 * reshapes organically. When loading finishes (`animated` goes false) the pool
 * settles onto the logo so the mark reforms. Canvas-based so the dot count
 * stays light at any size.
 */
function CymaticLoaderCanvas({
  size,
  animated,
  prefersReducedMotion,
}: {
  size: number;
  animated: boolean;
  prefersReducedMotion: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animatedRef = useRef(animated);

  useEffect(() => {
    animatedRef.current = animated;
  }, [animated]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    // Inherit the surrounding text color so the mark stays theme-aware. Re-read
    // each frame so a live theme switch (the `.dark` class toggling without a
    // remount) recolors the particles instead of leaving them stale.
    let color = getComputedStyle(canvas).color || "currentColor";
    const readColor = () => {
      color = getComputedStyle(canvas).color || color;
    };

    const drawR = size * 0.5;
    const center = size / 2;
    const particleR = Math.max(0.5, Math.min(1.2, size * 0.026));
    // One shared pool of particles. Any particle may land on the ring in one
    // figure and on an inner line in the next. Kept dense enough that figure
    // lines read as thicker-than-one-particle strokes rather than sparse dots.
    const count = Math.round(Math.max(72, size * 2.6) * 1.7);

    // Xorshift32 — deterministic point selection.
    let seed = 0x9e3779b9;
    const rand = () => {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      return (seed >>> 0) / 0x100000000;
    };

    // Each figure is a full-pool layout, every one carrying a ring so the mark
    // always reads with a border when the pool settles onto a figure. Index 0
    // (ring + crossed loops) is the Coval logo — the state the mark resolves to
    // when loading finishes, not a stop in the loading cycle.
    const buildFigure = (
      ringFraction: number,
      inner: (n: number) => Float32Array
    ) => {
      const out = new Float32Array(count * 2);
      const ringCount = Math.round(count * ringFraction);
      const ring = ringCount > 0 ? ringPoints(ringCount) : null;
      const innerPoints = inner(count - ringCount);
      for (let i = 0; i < count; i += 1) {
        if (ring && i < ringCount) {
          out[i * 2] = ring[i * 2]!;
          out[i * 2 + 1] = ring[i * 2 + 1]!;
        } else {
          const j = i - ringCount;
          out[i * 2] = innerPoints[j * 2]!;
          out[i * 2 + 1] = innerPoints[j * 2 + 1]!;
        }
      }
      return out;
    };
    const chladniInner = (modeIndex: number) => (n: number) =>
      selectNodalTargets(
        CHLADNI_CANDIDATES[modeIndex]!,
        n,
        INTERIOR_EXTENT,
        rand
      );
    const figures = [
      buildFigure(0.5, (n) => sampleShape(LOGO_LOOPS, n)), // 0: the Coval logo
      buildFigure(0.42, chladniInner(1)), // (2,3) — with outline
      buildFigure(0.42, chladniInner(2)), // (1,3) — with outline
      buildFigure(0.42, chladniInner(3)), // (3,4) — with outline
    ];
    // The loading swarm cycles through the Chladni figures only; the logo
    // (figures[0]) is reserved for the resting/reform state.
    const cycleFigures = figures.slice(1);

    // A diffuse cloud the whole pool scatters into between figures.
    const cloudTarget = new Float32Array(count * 2);
    for (let i = 0; i < count; i += 1) {
      const angle = rand() * Math.PI * 2;
      const r = Math.sqrt(rand()) * 0.72;
      cloudTarget[i * 2] = Math.cos(angle) * r;
      cloudTarget[i * 2 + 1] = Math.sin(angle) * r;
    }

    const px = new Float32Array(count);
    const py = new Float32Array(count);
    const tx = new Float32Array(count);
    const ty = new Float32Array(count);

    // Blend every target between the current figure and the cloud, so the whole
    // pool disperses and reassembles as `cloud` moves 0 → 1 → 0.
    const setTargets = (figure: Float32Array, cloud: number) => {
      for (let i = 0; i < count; i += 1) {
        const fx = figure[i * 2]!;
        const fy = figure[i * 2 + 1]!;
        tx[i] = fx + (cloudTarget[i * 2]! - fx) * cloud;
        ty[i] = fy + (cloudTarget[i * 2 + 1]! - fy) * cloud;
      }
    };
    // Spawn on the logo so the first frame reads as the mark before it breaks
    // apart into the cloud and the figures.
    setTargets(figures[0]!, 0);
    for (let i = 0; i < count; i += 1) {
      px[i] = tx[i]!;
      py[i] = ty[i]!;
    }

    // Keep a point inside the logo's rounded-square silhouette.
    const confine = (x: number, y: number): [number, number] => {
      const s =
        (Math.abs(x) ** SQUIRCLE_EXP + Math.abs(y) ** SQUIRCLE_EXP) **
          (1 / SQUIRCLE_EXP) /
        SQUIRCLE_RADIUS;
      if (s > 1) {
        return [x / s, y / s];
      }
      return [x, y];
    };

    // A loose spring toward the target plus a steady buzz makes every particle
    // hover and drift like a swarm rather than snapping onto a line.
    const PULL = 0.07;
    const SWARM_BUZZ = 0.012;
    const SUBSTEPS = 2;

    // Buzz rises while dispersed so the cloud looks more agitated than the
    // settled figure; updated each frame from the current cloudiness.
    let buzz = SWARM_BUZZ;

    const step = () => {
      for (let i = 0; i < count; i += 1) {
        let x = px[i]!;
        let y = py[i]!;
        x += (tx[i]! - x) * PULL;
        y += (ty[i]! - y) * PULL;
        x += (Math.random() * 2 - 1) * buzz;
        y += (Math.random() * 2 - 1) * buzz;
        [x, y] = confine(x, y);
        px[i] = x;
        py[i] = y;
      }
    };

    const draw = () => {
      ctx.clearRect(0, 0, size, size);
      ctx.fillStyle = color;
      ctx.globalAlpha = 1;
      ctx.beginPath();
      for (let i = 0; i < count; i += 1) {
        const cx = center + px[i]! * drawR;
        const cy = center + py[i]! * drawR;
        ctx.moveTo(cx + particleR, cy);
        ctx.arc(cx, cy, particleR, 0, Math.PI * 2);
      }
      ctx.fill();
    };

    if (prefersReducedMotion) {
      // Hold the logo as a static particle mark.
      setTargets(figures[0]!, 0);
      for (let k = 0; k < 40; k += 1) {
        step();
      }
      draw();
      return;
    }

    let rafId = 0;
    let startTs = 0;
    const frame = (ts: number) => {
      if (!startTs) {
        startTs = ts;
      }
      const elapsed = ts - startTs;
      readColor();

      if (animatedRef.current) {
        // Cycle: cloud → reassemble onto the next figure → hold → disperse back
        // to cloud, so particles flow organically between outline and interior.
        const p = (elapsed % BEAT_MS) / BEAT_MS;
        const cloud = cloudiness(p);
        const figureIndex = Math.floor(elapsed / BEAT_MS) % cycleFigures.length;
        buzz = SWARM_BUZZ * (1 + cloud * 1.6);
        setTargets(cycleFigures[figureIndex]!, cloud);
      } else {
        // Once loading finishes, settle onto the logo so the mark reforms.
        buzz = SWARM_BUZZ;
        setTargets(figures[0]!, 0);
      }

      for (let s = 0; s < SUBSTEPS; s += 1) {
        step();
      }
      draw();
      rafId = requestAnimationFrame(frame);
    };
    rafId = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafId);
  }, [size, prefersReducedMotion]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="absolute inset-0"
      style={{ width: size, height: size }}
    />
  );
}

/**
 * The Coval mark as a loading indicator. It rests as the static resonance-tile
 * logo, then — while loading (`animated`) — dissolves into a particle mark: a
 * persistent particle border with an interior that morphs between simple line
 * figures. Respects prefers-reduced-motion (renders a single static figure).
 * aria-hidden — callers that need to announce loading supply their own
 * accessible text.
 */
export function CymaticLoader({
  size = 20,
  className = "",
  animated = false,
}: {
  size?: number;
  className?: string;
  animated?: boolean;
}) {
  const [particlesVisible, setParticlesVisible] = useState(false);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(
    getPrefersReducedMotion
  );

  useEffect(() => {
    if (typeof window.matchMedia !== "function") {
      return;
    }

    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updatePreference = () => {
      setPrefersReducedMotion(mediaQuery.matches);
    };

    updatePreference();
    mediaQuery.addEventListener("change", updatePreference);
    return () => {
      mediaQuery.removeEventListener("change", updatePreference);
    };
  }, []);

  useEffect(() => {
    if (animated) {
      setParticlesVisible(true);
      return;
    }

    const hideTimer = window.setTimeout(() => {
      setParticlesVisible(false);
    }, PARTICLE_HIDE_DELAY_MS);

    return () => {
      window.clearTimeout(hideTimer);
    };
  }, [animated]);

  const shouldShowParticles = animated || particlesVisible;

  return (
    <span
      aria-hidden="true"
      style={{ width: size, height: size }}
      className={`relative inline-flex shrink-0 items-center justify-center overflow-hidden ${className}`}
    >
      <svg
        aria-hidden="true"
        viewBox="0 0 86 86"
        className={`absolute inset-[6%] transition-opacity duration-300 ease-out ${
          animated ? "opacity-0" : "opacity-100"
        }`}
        style={{ transitionDelay: animated ? "0ms" : "430ms" }}
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path
          strokeWidth={5.25}
          d="M3.96912 21.2938C5.11437 12.2423 12.2423 5.11437 21.2938 3.96912C35.4581 2.17696 49.7919 2.17696 63.9562 3.96912C73.0077 5.11437 80.1356 12.2423 81.2809 21.2938C83.073 35.4581 83.073 49.7919 81.2809 63.9562C80.1356 73.0077 73.0077 80.1356 63.9562 81.2809C49.7919 83.073 35.4581 83.073 21.2938 81.2809C12.2423 80.1356 5.11437 73.0077 3.96912 63.9562C2.17696 49.7919 2.17696 35.4581 3.96912 21.2938Z"
        />
        <path
          strokeWidth={5.5}
          d="M36.6763 22.3469C40.1011 19.6913 44.8893 19.6913 48.3142 22.3469C53.6736 26.5024 58.4879 31.3167 62.6434 36.6761C65.299 40.101 65.2989 44.8892 62.6434 48.314C58.4879 53.6734 53.6736 58.4877 48.3142 62.6432C44.8893 65.2988 40.1011 65.2988 36.6763 62.6432C31.3169 58.4877 26.5025 53.6734 22.347 48.314C19.6915 44.8892 19.6915 40.101 22.347 36.6761C26.5025 31.3167 31.3169 26.5024 36.6763 22.3469Z"
        />
        <path
          strokeWidth={5.5}
          d="M42.6966 67.7434C34.0423 67.7434 21.6521 84.7845 10.8927 74.1096C0.912984 64.2082 19.3911 51.1549 19.3911 42.5685C19.3911 33.9822 0.133318 21.818 10.8927 11.1431C20.8725 1.24174 34.0423 17.4908 42.6966 17.4908C51.3509 17.4908 63.5982 0.468205 74.3576 11.1431C84.3374 21.0445 65.8593 33.9822 65.8593 42.5685C65.8593 51.1549 85.117 63.4347 74.3576 74.1096C64.3779 84.0109 51.3509 67.7434 42.6966 67.7434Z"
        />
      </svg>

      {shouldShowParticles ? (
        <span
          aria-hidden="true"
          className={`absolute inset-0 transition-opacity ease-out ${
            animated ? "opacity-100" : "opacity-0"
          }`}
          style={{
            transitionDuration: prefersReducedMotion
              ? "0ms"
              : animated
                ? "420ms"
                : "300ms",
            // Hold the swarm at full opacity while it settles back onto the
            // logo figure, then cross-fade to the crisp SVG in sync with its
            // fade-in — the mark reforms instead of evaporating mid-cloud.
            transitionDelay: prefersReducedMotion || animated ? "0ms" : "430ms",
          }}
        >
          <CymaticLoaderCanvas
            key={size}
            size={size}
            animated={animated}
            prefersReducedMotion={prefersReducedMotion}
          />
        </span>
      ) : null}
    </span>
  );
}
