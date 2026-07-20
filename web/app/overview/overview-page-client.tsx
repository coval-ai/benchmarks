// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useEffect, useState } from "react";
import { ChevronDown } from "lucide-react";
import DashboardHeader from "@/components/layout/DashboardHeader";
import DashboardFooter from "@/components/dashboard/DashboardFooter";
import OverviewLeaderboards from "@/components/overview/OverviewLeaderboards";
import AboutMethodology from "@/components/overview/AboutMethodology";

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
            Measuring the accuracy, latency, and quality of text-to-speech and
            speech-to-text models.
          </p>

          <div className="mt-6 md:mt-8">
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
