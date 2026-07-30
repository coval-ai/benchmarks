// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowRight, ChevronDown } from "lucide-react";
import DashboardHeader from "@/components/layout/DashboardHeader";
import DashboardFooter from "@/components/dashboard/DashboardFooter";
import OverviewLeaderboards from "@/components/overview/OverviewLeaderboards";
import AboutMethodology from "@/components/overview/AboutMethodology";
import Card from "@/components/shared/Card";
import { CymaticLoader } from "@/components/shared/CymaticLoader";

function ScrollHint() {
  const [hidden, setHidden] = useState(false);
  useEffect(() => {
    const onScroll = () => setHidden(window.scrollY > 80);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  return (
    <button
      type="button"
      aria-label="Scroll down to methodology"
      onClick={() =>
        document
          .getElementById("about-methodology")
          ?.scrollIntoView({ behavior: "smooth" })
      }
      className={`mx-auto mt-auto flex h-11 w-11 items-center justify-center pt-4 text-text-tertiary transition-opacity duration-300 hover:text-text-primary ${
        hidden ? "pointer-events-none opacity-0" : "opacity-100"
      }`}
    >
      <ChevronDown size={20} aria-hidden className="animate-bounce" />
    </button>
  );
}

// The arena is the one benchmark a visitor participates in rather than reads,
// so it sits above the leaderboards with the cymatic settling on hover.
function ArenaCard() {
  const [animated, setAnimated] = useState(false);

  return (
    <Card padding="p-0" className="overflow-hidden">
      <Link
        href="/overview/arena"
        aria-label="Voice Arena — interactive benchmark"
        onMouseEnter={() => setAnimated(true)}
        onMouseLeave={() => setAnimated(false)}
        onFocus={() => setAnimated(true)}
        onBlur={() => setAnimated(false)}
        className="group flex items-center gap-4 p-4 transition-colors hover:bg-hover-bg focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-text-tertiary/40 sm:gap-5 sm:p-5"
      >
        <span className="flex size-14 shrink-0 items-center justify-center rounded-full border border-border-secondary bg-surface-primary text-text-primary sm:size-16">
          <CymaticLoader size={44} animated={animated} />
        </span>

        <span className="min-w-0 flex-1">
          <span className="block font-mono text-[11px] uppercase tracking-[0.14em] text-text-tertiary">
            Interactive benchmark
          </span>
          <span className="mt-1 block font-sans text-lg font-medium text-text-primary sm:text-xl">
            Voice Arena
          </span>
          <span className="mt-1 block text-pretty text-sm leading-snug text-text-secondary">
            Compare two voices and vote for the one that sounds more natural.
          </span>
        </span>

        <span className="flex min-h-11 shrink-0 items-center gap-2 rounded-full border border-border-secondary bg-surface-primary px-3 font-sans text-sm font-medium text-text-primary sm:px-4">
          <span className="hidden sm:inline">Enter arena</span>
          <ArrowRight
            size={18}
            aria-hidden
            className="transition-transform duration-200 group-hover:translate-x-0.5"
          />
        </span>
      </Link>
    </Card>
  );
}

export function OverviewPageClient() {
  return (
    <div className="flex min-h-screen flex-col overflow-hidden bg-background text-text-primary">
      <DashboardHeader />

      <main className="relative z-10 mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 sm:px-6 pb-10 pt-[84px] md:pt-[96px]">
        <div className="flex min-h-[calc(100vh-84px)] flex-col md:min-h-[calc(100vh-96px)]">
          <h1 className="mx-auto max-w-xl text-balance text-center text-2xl font-medium leading-tight tracking-tight text-text-primary sm:text-3xl md:text-4xl">
            Voice AI benchmarks in real world conditions.
          </h1>

          <p className="mx-auto mt-3 max-w-md text-pretty text-center text-base leading-snug text-text-secondary">
            Measuring the accuracy, latency, and quality of text-to-speech,
            speech-to-text, and speech-to-speech models.
          </p>

          <div className="mt-6 md:mt-8">
            <ArenaCard />
          </div>

          <div className="mt-4 md:mt-6">
            <OverviewLeaderboards />
          </div>

          <ScrollHint />
        </div>

        <AboutMethodology />
      </main>

      <DashboardFooter />
    </div>
  );
}
