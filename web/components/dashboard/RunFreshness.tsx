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
  const { page } = useDashboard();
  const { data } = useRunsQuery();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(id);
  }, []);

  // Runs are per-dataset; dataset ids are prefixed with their modality, so this
  // scopes freshness to the benchmark type being viewed.
  const runs =
    data?.runs.filter((run) => run.dataset_id.startsWith(`${page}-`)) ?? [];
  // PARTIAL still produced data — with flaky providers nearly every run is
  // PARTIAL, so only FAILED is worth surfacing.
  const lastGood = runs.find(
    (run) => run.status === "SUCCEEDED" || run.status === "PARTIAL"
  );
  const failed = runs.find((run) => run.status !== "RUNNING")?.status === "FAILED";
  // Reserved height whether or not the chip has data, so the header never
  // shifts when the runs query resolves (or silently fails).
  if (!lastGood) return <div aria-hidden className="min-h-11 lg:min-h-7" />;

  const updatedAt = Date.parse(lastGood.finished_at ?? lastGood.started_at);

  return (
    <div className="flex min-h-11 items-center lg:min-h-7">
      <MetricInfo
        align="left"
        panelClassName="top-full mt-1.5 w-max"
        content={`Last ${page.toUpperCase()} run finished ${new Date(updatedAt).toUTCString()}`}
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
