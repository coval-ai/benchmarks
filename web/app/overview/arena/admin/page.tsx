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

        <h2 className="mt-8 text-lg font-medium">Pairing SCALE tuning</h2>
        <p className="mt-2 text-sm text-text-secondary">
          SCALE is the pairing knob: how sharply matchup priority decays with the Elo gap
          between two models. It is a committed constant in the runner
          (runner/src/coval_bench/arena/pairing.py), tuned offline — the simulation below
          does not run live. Loss is penalized cross-entropy per decisive battle: how well
          the fitted board predicted outcomes as votes accumulated (0.693 = coin flip,
          lower is better).
        </p>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full max-w-xl text-sm">
            <thead>
              <tr className="border-b text-left text-text-secondary">
                <th className="py-1.5 pr-4 font-medium">SCALE</th>
                <th className="py-1.5 pr-4 font-medium">300 battles</th>
                <th className="py-1.5 pr-4 font-medium">2000 battles</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b">
                <td className="py-1.5 pr-4">50</td>
                <td className="py-1.5 pr-4">0.744</td>
                <td className="py-1.5 pr-4">0.743</td>
              </tr>
              <tr className="border-b">
                <td className="py-1.5 pr-4">150</td>
                <td className="py-1.5 pr-4">0.816</td>
                <td className="py-1.5 pr-4">0.748</td>
              </tr>
              <tr className="border-b">
                <td className="py-1.5 pr-4">300</td>
                <td className="py-1.5 pr-4">0.838</td>
                <td className="py-1.5 pr-4">0.719</td>
              </tr>
              <tr className="border-b">
                <td className="py-1.5 pr-4">600</td>
                <td className="py-1.5 pr-4">0.856</td>
                <td className="py-1.5 pr-4">0.703</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-sm text-text-secondary">
          Offline simulation, 2026-07-16: real 27-model roster, true strengths seeded from
          the live board, observed 19% tie rate, mean of 3 seeds. Reading: past a few
          hundred votes, SCALE 150 consistently loses to 300/600; at today&apos;s volume the
          short-horizon column is noise (no setting beats coin flip yet, and small SCALE
          partly grades its own homework by re-matching pairs it already knows).
          Re-run with: <code className="font-mono text-xs">uv run coval-bench arena
          tune-scale --scales 50,150,300,600</code>
        </p>
      </main>
      <DashboardFooter />
    </div>
  );
}
