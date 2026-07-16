// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import DashboardHeader from "@/components/layout/DashboardHeader";
import DashboardFooter from "@/components/dashboard/DashboardFooter";

export default function Page() {
  return (
    <div className="relative flex min-h-screen flex-col bg-background text-text-primary">
      <DashboardHeader />
      <main className="relative z-10 mx-auto w-full max-w-5xl flex-1 px-4 pb-10 pt-[84px] sm:px-6 md:pt-[96px]">
        <h1 className="text-2xl font-medium tracking-tight sm:text-3xl">Arena monitoring</h1>
        <p className="mt-2 text-sm text-text-secondary">
          Convergence shows whether rating confidence intervals shrink as votes accumulate.
          Co-occurrence shows which model pairs have battled — empty blocks away from the
          diagonal mean groups of models that never face each other.
        </p>

        <h2 className="mt-8 text-lg font-medium">Convergence</h2>
        <iframe
          src="/api/arena/admin/convergence"
          title="Arena convergence"
          className="mt-3 h-[560px] w-full rounded-md border bg-white"
        />

        <h2 className="mt-8 text-lg font-medium">Co-occurrence</h2>
        <iframe
          src="/api/arena/admin/cooccurrence"
          title="Arena co-occurrence"
          className="mt-3 h-[760px] w-full rounded-md border bg-white"
        />
      </main>
      <DashboardFooter />
    </div>
  );
}
