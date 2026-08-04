// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useEffect, useState } from "react";
import { useRunsQuery } from "@/lib/api/queries";
import type { RunOut } from "@/lib/api/client";
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

const DOT_CLASS = {
  SUCCEEDED: "bg-accent-teal",
  PARTIAL: "bg-accent-orange",
  FAILED: "bg-accent-rust",
} as const;

const RunFreshness: React.FC = () => {
  const { data } = useRunsQuery();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(id);
  }, []);

  const completed =
    data?.runs.filter(
      (run): run is RunOut & { status: keyof typeof DOT_CLASS } =>
        run.status !== "RUNNING"
    ) ?? [];
  const lastGood = completed.find(
    (run) => run.status === "SUCCEEDED" || run.status === "PARTIAL"
  );
  const latest = completed[0];
  // Reserved height whether or not the chip has data, so the header never
  // shifts when the runs query resolves (or silently fails).
  if (!lastGood || !latest) return <div aria-hidden className="min-h-11 lg:min-h-7" />;

  const updatedAt = Date.parse(lastGood.finished_at ?? lastGood.started_at);
  const degraded = latest.status !== "SUCCEEDED";

  return (
    <div className="flex min-h-11 items-center lg:min-h-7">
      <MetricInfo
        align="left"
        panelClassName="top-full mt-1.5 w-max"
        content={`Last successful run finished ${new Date(updatedAt).toUTCString()}`}
      >
        <span className="inline-flex cursor-help items-center gap-1.5 rounded-full border border-border-primary px-2.5 py-1 font-mono text-[11px] text-text-tertiary">
          <span
            aria-hidden
            className={`h-2 w-2 shrink-0 rounded-full ${DOT_CLASS[latest.status]}`}
          />
          Updated {relativeTime(now - updatedAt)}
          {degraded && (
            <span className="text-text-secondary">
              · last run {latest.status.toLowerCase()}
            </span>
          )}
        </span>
      </MetricInfo>
    </div>
  );
};

export default RunFreshness;
