// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import DashboardHeader from "@/components/layout/DashboardHeader";
import DashboardFooter from "@/components/dashboard/DashboardFooter";
import OverviewLeaderboards from "@/components/overview/OverviewLeaderboards";

export function OverviewPageClient() {
  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden bg-background text-text-primary">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-0"
        style={{ backgroundColor: "rgba(120, 72, 40, 0.06)" }}
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-0 opacity-[0.07]"
        style={{
          backgroundImage: "url(/chladni-13-1-6.svg)",
          backgroundSize: "max(120vw, 120vh)",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
        }}
      />
      <DashboardHeader />

      <main className="relative z-10 mx-auto flex min-h-[90vh] w-full max-w-5xl flex-1 flex-col px-6 pb-10 pt-[84px] md:pt-[96px]">
        <h1 className="mx-auto max-w-xl text-balance text-center text-2xl font-medium leading-tight tracking-tight text-text-primary sm:text-3xl md:text-4xl">
          Voice AI Benchmarks in real world conditions.
        </h1>

        <p className="mx-auto mt-3 max-w-md text-pretty text-center text-sm leading-relaxed text-text-secondary md:text-base">
          Measuring the accuracy, latency, and quality of text-to-speech and
          speech-to-text models.
        </p>

        <div className="mt-8">
          <OverviewLeaderboards />
        </div>
      </main>

      <DashboardFooter />
    </div>
  );
}
