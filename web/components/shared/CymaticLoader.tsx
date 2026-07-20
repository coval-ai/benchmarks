// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useEffect, useState } from "react";

const PARTICLE_COUNT = 144;
const STEP_MS = 700;

interface ParticleTarget {
  x: number;
  y: number;
  opacity: number;
  scale: number;
}

function chladni(x: number, y: number, n: number, m: number) {
  return (
    Math.cos(n * Math.PI * x) * Math.cos(m * Math.PI * y) -
    Math.cos(m * Math.PI * x) * Math.cos(n * Math.PI * y)
  );
}

function seededNoise(x: number, y: number, salt: number) {
  const value = Math.sin((x + 1) * 127.1 + (y + 1) * 311.7 + salt * 74.7);
  return value - Math.floor(value);
}

function makeScatterTargets(salt: number) {
  return Array.from({ length: PARTICLE_COUNT }, (_, index) => {
    const angle = index * 2.399963 + salt * 0.71;
    const radius = Math.sqrt((index + 0.5) / PARTICLE_COUNT) * 39;
    const wobble = 1 + (seededNoise(index, salt, 3) - 0.5) * 0.35;
    return {
      x: 50 + Math.cos(angle) * radius * wobble,
      y: 50 + Math.sin(angle) * radius * wobble,
      opacity: 0.55 + seededNoise(index, salt, 4) * 0.3,
      scale: 0.72 + seededNoise(index, salt, 5) * 0.38,
    };
  });
}

// Beads settle onto the nodal lines of a Chladni plate mode (n, m) — the
// pattern sand forms on a vibrating plate, per the brand's cymatics identity.
function makeChladniTargets(n: number, m: number, salt: number) {
  const candidates: Array<ParticleTarget & { score: number }> = [];
  const grid = 76;

  for (let row = 0; row < grid; row += 1) {
    const y = -1 + (row / (grid - 1)) * 2;
    for (let column = 0; column < grid; column += 1) {
      const x = -1 + (column / (grid - 1)) * 2;
      if (Math.hypot(x, y) > 1.06) continue;

      const distanceFromNode = Math.abs(chladni(x, y, n, m));
      const score = distanceFromNode + seededNoise(column, row, salt) * 0.018;
      const closeness = Math.max(0, 1 - distanceFromNode / 0.22);

      candidates.push({
        x: 50 + x * 40 + (seededNoise(column, row, salt + 10) - 0.5) * 2.2,
        y: 50 - y * 40 + (seededNoise(column, row, salt + 20) - 0.5) * 2.2,
        opacity: 0.58 + closeness * 0.42,
        scale: 0.72 + closeness * 0.46,
        score,
      });
    }
  }

  candidates.sort((a, b) => a.score - b.score);

  const selected: ParticleTarget[] = [];
  let minDistance = 5.1;
  while (selected.length < PARTICLE_COUNT && minDistance >= 1.6) {
    for (const candidate of candidates) {
      if (selected.length >= PARTICLE_COUNT) break;
      if (
        selected.every(
          (target) =>
            Math.hypot(target.x - candidate.x, target.y - candidate.y) >=
            minDistance
        )
      ) {
        selected.push(candidate);
      }
    }
    minDistance -= 0.55;
  }
  return selected.concat(candidates).slice(0, PARTICLE_COUNT);
}

const CHLADNI_23 = makeChladniTargets(2, 3, 11);
const CHLADNI_35 = makeChladniTargets(3, 5, 22);
const CHLADNI_45 = makeChladniTargets(4, 5, 33);
const CHLADNI_53 = makeChladniTargets(5, 3, 44);

const TARGETS = [
  makeScatterTargets(1),
  CHLADNI_23,
  CHLADNI_23,
  makeScatterTargets(2),
  CHLADNI_35,
  CHLADNI_35,
  makeScatterTargets(3),
  CHLADNI_45,
  CHLADNI_45,
  makeScatterTargets(4),
  CHLADNI_53,
  CHLADNI_53,
];

export function CymaticLoader({
  size = 20,
  className = "",
  animated = false,
}: {
  size?: number;
  className?: string;
  animated?: boolean;
}) {
  const [step, setStep] = useState(1);
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;

    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updateReduceMotion = () => setReduceMotion(mediaQuery.matches);
    updateReduceMotion();
    mediaQuery.addEventListener("change", updateReduceMotion);
    return () => mediaQuery.removeEventListener("change", updateReduceMotion);
  }, []);

  useEffect(() => {
    if (!animated || reduceMotion) return;
    setStep(0);
    const interval = window.setInterval(
      () => setStep((current) => (current + 1) % TARGETS.length),
      STEP_MS
    );
    return () => window.clearInterval(interval);
  }, [animated, reduceMotion]);

  const targets = TARGETS[step] ?? CHLADNI_23;
  const dotSize = Math.max(1.2, size * 0.058);

  return (
    <span
      aria-hidden="true"
      style={{ width: size, height: size }}
      className={`relative inline-flex shrink-0 ${className}`}
    >
      {targets.map((target, index) => (
        <span
          // eslint-disable-next-line react/no-array-index-key -- fixed-size positional bead grid; index is the identity
          key={index}
          className="absolute rounded-full bg-current will-change-[transform,opacity]"
          style={{
            left: "50%",
            top: "50%",
            width: dotSize,
            height: dotSize,
            opacity: target.opacity,
            transform: `translate(${((target.x - 50) * size) / 100}px, ${
              ((target.y - 50) * size) / 100
            }px) translate(-50%, -50%) scale(${target.scale})`,
            transitionProperty: reduceMotion ? "none" : "opacity, transform",
            transitionDuration: reduceMotion ? "0ms" : "560ms",
            transitionDelay: reduceMotion ? "0ms" : `${(index % 9) * 8}ms`,
            transitionTimingFunction: "cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
      ))}
    </span>
  );
}
