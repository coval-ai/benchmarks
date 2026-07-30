// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import Link from "next/link";
import DashboardHeader from "@/components/layout/DashboardHeader";
import DashboardFooter from "@/components/dashboard/DashboardFooter";
import Card from "@/components/shared/Card";
import { CymaticLoader } from "@/components/shared/CymaticLoader";
import {
  normalizeModelName,
  normalizeTTSProviderName,
  toModelKey,
} from "@/lib/utils/formatters";
import { useArenaLeaderboardQuery } from "@/lib/arena/leaderboard";

export function ArenaLeaderboardPage() {
  const { data, isLoading, isError } = useArenaLeaderboardQuery();
  const entries = data?.entries ?? [];
  const isEmpty = !isLoading && !isError && entries.length === 0;

  // Every live rating is currently "preliminary", so a per-row badge fires on all 27 and
  // greying by status drains the whole table to tertiary. Both only carry meaning once the
  // board actually spans tiers; until then the state is stated once, above the table.
  const statuses = new Set(entries.map((e) => e.status));
  const mixedStatus = statuses.size > 1;
  const allProvisional = statuses.size === 1 && statuses.has("preliminary");

  return (
    <div className="relative flex min-h-screen flex-col bg-background text-text-primary">
      <DashboardHeader />
      <main className="relative z-10 mx-auto w-full max-w-3xl flex-1 px-4 pb-10 pt-[84px] sm:px-6 md:pt-[96px]">
        <Link
          href="/overview/arena"
          className="inline-flex min-h-11 items-center text-sm text-text-secondary transition-colors hover:text-text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40"
        >
          ← Arena
        </Link>
        <h1 className="mt-1 text-2xl font-medium tracking-tight sm:text-3xl">Voice Arena leaderboard</h1>
        <p className="mt-2 text-sm text-text-secondary">
          Models ranked by Elo rating from blind A/B votes. The ± figure is the
          confidence interval — the fewer votes a model has, the wider it runs.
          {allProvisional && " Every rating is still provisional at these vote counts."}
          {mixedStatus && " Ratings marked provisional have too few votes to separate them yet."}
        </p>

        {isLoading && (
          <div role="status" className="mt-8 flex justify-center">
            {/* CymaticLoader is aria-hidden, so the spinner alone announces nothing. */}
            <CymaticLoader size={40} animated className="text-text-primary" />
            <span className="sr-only">Loading the leaderboard…</span>
          </div>
        )}
        {(isError || isEmpty) && (
          <p className="mt-8 text-sm text-text-tertiary">
            {isError
              ? "Couldn’t load the leaderboard. Try again."
              : "No ratings yet. Vote to populate."}
          </p>
        )}

        {entries.length > 0 && (
          <Card padding="p-4 sm:p-6 md:p-8" className="mt-6">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-border-primary text-left text-text-tertiary">
                  <th className="w-8 py-2 pr-2 font-medium">#</th>
                  <th className="py-2 pr-4 font-medium">Model</th>
                  <th className="hidden whitespace-nowrap py-2 pr-4 text-right font-medium sm:table-cell">
                    Votes
                  </th>
                  <th className="whitespace-nowrap py-2 text-right font-medium">Elo</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry, i) => {
                  const preliminary = entry.status === "preliminary";
                  return (
                    <tr
                      key={`${entry.provider}/${entry.model}`}
                      className="border-b border-border-secondary last:border-b-0 hover:bg-hover-bg"
                    >
                      <td className="py-3 pr-2 align-top font-mono tabular-nums text-text-tertiary">
                        {i + 1}
                      </td>
                      {/* Provider stacks under the model instead of taking its own column, so the
                          table fits a phone without horizontal scroll. */}
                      <td className="py-3 pr-4 align-top">
                        <div className="font-medium text-text-primary">
                          {normalizeModelName(toModelKey(entry.provider, entry.model))}
                        </div>
                        <div className="text-xs text-text-tertiary">
                          {normalizeTTSProviderName(entry.provider)}
                          {/* The Votes column is hidden on phones, so carry the count here
                              rather than dropping it off the small screen entirely. */}
                          <span className="font-mono sm:hidden">
                            {" · "}
                            {entry.votes_total.toLocaleString()} votes
                          </span>
                          {mixedStatus && preliminary && (
                            <span className="font-mono uppercase"> provisional</span>
                          )}
                        </div>
                      </td>
                      <td className="hidden py-3 pr-4 text-right align-top font-mono text-xs tabular-nums text-text-tertiary sm:table-cell">
                        {entry.votes_total.toLocaleString()}
                      </td>
                      <td className="whitespace-nowrap py-3 text-right align-top">
                        <span className="font-mono text-base tabular-nums text-text-primary">
                          {Math.round(entry.rating_elo)}
                        </span>{" "}
                        {entry.ci_half_width != null && (
                          <span className="font-mono text-xs tabular-nums text-text-tertiary">
                            ± {Math.round(entry.ci_half_width)}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>
        )}
      </main>
      <DashboardFooter />
    </div>
  );
}
