// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import Link from "next/link";

export interface LeaderboardRow {
  /** Stable key — composite `provider:model`. */
  key: string;
  model: string;
  provider: string;
  /** Pre-formatted metric value, e.g. "412 ms" or "4.2%". */
  value: string;
}

interface LeaderboardCardProps {
  /** Card heading, e.g. "Text-to-Speech". */
  title: string;
  /** Right-aligned metric column header, e.g. "TTFA" / "WER". */
  metricLabel: string;
  rows: LeaderboardRow[];
  /** Link to the full dashboard for this benchmark. */
  href: string;
  loading?: boolean;
  error?: boolean;
}

const LeaderboardCard: React.FC<LeaderboardCardProps> = ({
  title,
  metricLabel,
  rows,
  href,
  loading = false,
  error = false,
}) => {
  return (
    <div className="flex flex-col rounded-lg border border-border-secondary bg-white p-6 md:p-8">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h2 className="text-lg font-medium text-text-primary md:text-xl">{title}</h2>
        <span className="text-xs font-light tracking-wider text-text-secondary">
          Last 24h
        </span>
      </div>

      <div className="grid grid-cols-[1.25rem_1fr_auto] items-center gap-x-2 gap-y-4 border-b border-border-secondary pb-2 text-[0.7rem] font-light tracking-wider text-text-secondary">
        <div>#</div>
        <div>Model</div>
        <div className="text-right">{metricLabel}</div>
      </div>

      {loading && (
        <div className="flex flex-col">
          {["a", "b", "c"].map((slot) => (
            <div
              key={slot}
              className="grid grid-cols-[1.25rem_1fr_auto] items-center gap-x-2 gap-y-4 border-b border-border-secondary py-4"
            >
              <div className="h-4 w-4 animate-pulse rounded bg-surface-secondary" />
              <div className="h-4 w-32 animate-pulse rounded bg-surface-secondary" />
              <div className="h-4 w-12 animate-pulse justify-self-end rounded bg-surface-secondary" />
            </div>
          ))}
        </div>
      )}

      {!loading && error && (
        <div className="py-10 text-center text-sm text-text-tertiary">
          Couldn&rsquo;t load the leaderboard.
        </div>
      )}

      {!loading && !error && rows.length === 0 && (
        <div className="py-10 text-center text-sm text-text-tertiary">
          No results yet.
        </div>
      )}

      {!loading && !error &&
        rows.map((row, index) => (
          <div
            key={row.key}
            className="grid grid-cols-[1.25rem_1fr_auto] items-center gap-x-2 gap-y-4 border-b border-border-secondary py-2 text-text-primary"
          >
            <div className="text-base leading-none">
              <span className="font-mono text-sm text-text-tertiary">
                {index + 1}
              </span>
            </div>
            <div className="min-w-0">
              <div className="truncate text-sm font-medium">
                {row.model}
                <span className="ml-1 font-normal text-text-tertiary">
                  {row.provider}
                </span>
              </div>
            </div>
            <div className="justify-self-end text-right font-mono text-lg">
              {row.value}
            </div>
          </div>
        ))}

      {!loading && !error && rows.length > 0 && (
        <Link
          href={href}
          className="mt-6 inline-flex items-center gap-1 self-start rounded-full bg-surface-secondary px-4 py-2 text-sm font-medium text-text-secondary transition-colors hover:bg-surface-elevated hover:text-text-primary"
        >
          View full leaderboard
          <span aria-hidden>&rarr;</span>
        </Link>
      )}
    </div>
  );
};

export default LeaderboardCard;
