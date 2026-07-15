// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import DashboardHeader from "@/components/layout/DashboardHeader";
import DashboardFooter from "@/components/dashboard/DashboardFooter";
import {
  normalizeModelName,
  normalizeTTSProviderName,
  toModelKey,
} from "@/lib/utils/formatters";
import { useArenaLeaderboardQuery, type ArenaLeaderboardEntry } from "@/lib/arena/leaderboard";

function eloLabel(entry: ArenaLeaderboardEntry): string {
  const elo = Math.round(entry.rating_elo);
  if (entry.ci_half_width == null) return `${elo}`;
  return `${elo} ± ${Math.round(entry.ci_half_width)}`;
}

export function ArenaLeaderboardPage() {
  const { data, isLoading, isError } = useArenaLeaderboardQuery();
  const entries = data?.entries ?? [];
  const isEmpty = !isLoading && !isError && entries.length === 0;

  return (
    <div className="relative flex min-h-screen flex-col bg-background text-text-primary">
      <DashboardHeader />
      <main className="relative z-10 mx-auto w-full max-w-3xl flex-1 px-4 pb-10 pt-[84px] sm:px-6 md:pt-[96px]">
        <h1 className="text-2xl font-medium tracking-tight sm:text-3xl">Voice Arena leaderboard</h1>
        <p className="mt-2 text-sm text-text-secondary">
          Models ranked by Elo rating from blind A/B votes. Greyed rows are
          provisional — the confidence interval is still wide.
        </p>

        {isLoading && <p className="mt-8 text-sm text-text-tertiary">Loading…</p>}
        {isError && (
          <p className="mt-8 text-sm text-text-tertiary">
            Couldn’t load the leaderboard. Try again.
          </p>
        )}
        {isEmpty && (
          <p className="mt-8 text-sm text-text-tertiary">No ratings yet. Vote to populate.</p>
        )}

        {entries.length > 0 && (
          <div className="mt-6 overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-border-primary text-left text-text-tertiary">
                  <th className="py-2 pr-4 font-medium">#</th>
                  <th className="py-2 pr-4 font-medium">Model</th>
                  <th className="py-2 pr-4 font-medium">Provider</th>
                  <th className="py-2 pr-4 text-right font-medium">Elo</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry, i) => {
                  const preliminary = entry.status === "preliminary";
                  return (
                    <tr
                      key={`${entry.provider}/${entry.model}`}
                      className={`border-b border-border-secondary ${
                        preliminary ? "text-text-tertiary" : "text-text-primary"
                      }`}
                    >
                      <td className="py-2.5 pr-4 tabular-nums">{i + 1}</td>
                      <td className="py-2.5 pr-4">
                        {normalizeModelName(toModelKey(entry.provider, entry.model))}
                      </td>
                      <td className="py-2.5 pr-4 text-text-secondary">
                        {normalizeTTSProviderName(entry.provider)}
                      </td>
                      <td className="py-2.5 pr-4 text-right tabular-nums">
                        {eloLabel(entry)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </main>
      <DashboardFooter />
    </div>
  );
}
