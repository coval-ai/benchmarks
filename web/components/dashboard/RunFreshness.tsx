// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useEffect, useState } from "react";
import { useDashboard } from "@/contexts/DashboardContext";
import { useRunsQuery } from "@/lib/api/queries";
import MetricInfo from "@/components/shared/MetricInfo";

function relativeTime(ms: number): string {
  const min = Math.floor(ms / 60_000);
  if (min < 1) return "just now";
  if (min < 60) return `${min} min ago`;
  const hours = Math.floor(min / 60);
  if (hours < 24) return `${hours} hr ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

const RunFreshness: React.FC = () => {
  const { page, latestDataBucket } = useDashboard();
  const { data: runsData } = useRunsQuery();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(id);
  }, []);

  // Reserved height whether or not the chip has data, so the header never
  // shifts as queries resolve (or silently fail).
  if (!latestDataBucket) return <div aria-hidden className="min-h-11 lg:min-h-7" />;

  // Runs are per-dataset with modality-prefixed ids. PARTIAL is the norm with
  // flaky providers — only FAILED matters.
  const runs =
    runsData?.runs.filter(
      (run) => run.dataset_id.startsWith(`${page}-`) && run.status !== "RUNNING"
    ) ?? [];
  const failed = runs[0]?.status === "FAILED";

  // "Updated" = the newest run the charts already include: runs finishing after
  // the served bucket ends are materialized later, and the bucket start alone
  // undersells daily-bucketed boards. Shown as the run's start (its schedule
  // slot — what the chart's newest point is labeled with), not its finish.
  // Chart bucket as fallback if runs fail.
  const included = runs.find(
    (run) =>
      run.status !== "FAILED" &&
      Date.parse(run.finished_at ?? run.started_at) <= latestDataBucket.end
  );
  const updatedAt = included
    ? Date.parse(included.started_at)
    : latestDataBucket.start;

  return (
    <div className="flex min-h-11 items-center lg:min-h-7">
      <MetricInfo
        align="left"
        panelClassName="top-full mt-1.5 w-max"
        content={`Newest ${page.toUpperCase()} data: ${new Date(updatedAt).toUTCString()}`}
      >
        <span className="inline-flex cursor-help items-center gap-1.5 rounded-full border border-border-primary px-2.5 py-1 font-mono text-[11px] text-text-tertiary">
          <span
            aria-hidden
            className={`h-2 w-2 shrink-0 rounded-full ${failed ? "bg-accent-rust" : "bg-accent-teal"}`}
          />
          Updated {relativeTime(now - updatedAt)}
          {failed && <span className="text-text-secondary">· last run failed</span>}
        </span>
      </MetricInfo>
    </div>
  );
};

export default RunFreshness;
