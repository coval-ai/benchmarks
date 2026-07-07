// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useEffect, useState } from "react";
import { Check, Link2 } from "lucide-react";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";

interface SectionHeaderProps {
  label: string;
  description: {
    short: string;
    detailed: string;
  };
  stat?: {
    label: string;
    value: string;
  };
  /** Optional hint shown after the "About this benchmark" link, separated by an interpunct. */
  hint?: string;
  /** When false, the detailed text shows inline instead of behind a toggle. */
  expandable?: boolean;
}

const SectionHeader: React.FC<SectionHeaderProps> = ({
  label,
  description,
  stat,
  hint,
  expandable = true,
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const [copied, setCopied] = useState(false);
  const anchorId = label.toLowerCase().replace(/[^a-z0-9]+/g, "-");

  useEffect(() => {
    if (window.location.hash !== `#${anchorId}`) return;
    window.scrollTo(0, 0);
    let stop = false;
    const cancel = () => {
      stop = true;
    };
    window.addEventListener("wheel", cancel, { passive: true });
    window.addEventListener("touchstart", cancel, { passive: true });
    const t0 = performance.now();
    const step = (now: number) => {
      if (stop || now - t0 > 2500) return;
      const delta =
        (document.getElementById(anchorId)?.getBoundingClientRect().top ?? 96) -
        96;
      if (Math.abs(delta) > 0.5) window.scrollTo(0, window.scrollY + delta * 0.15);
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
    return () => {
      stop = true;
      window.removeEventListener("wheel", cancel);
      window.removeEventListener("touchstart", cancel);
    };
  }, [anchorId]);

  const copyLink = () => {
    navigator.clipboard.writeText(
      `${window.location.origin}${window.location.pathname}#${anchorId}`
    );
    window.history.replaceState(null, "", `#${anchorId}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    capturePostHogEvent(POSTHOG_EVENTS.dashboardChartShared, {
      chart: anchorId,
      path: window.location.pathname,
    });
  };

  return (
    <div id={anchorId} className="flex justify-between items-start gap-8 mb-4 scroll-mt-24">
      <div className="w-3/4 min-w-0">
        <h2 className="flex items-center gap-2 text-[0.9rem] font-light text-text-secondary mb-2">
          {label}
          <button
            type="button"
            onClick={copyLink}
            aria-label="Copy link to this chart"
            title="Copy link"
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-border-secondary bg-white text-text-secondary transition-colors hover:bg-surface-toggle-inactive hover:text-text-primary"
          >
            {copied ? <Check size={14} /> : <Link2 size={14} />}
          </button>
        </h2>
        <p className="text-2xl font-medium text-text-primary mb-1">
          {description.short}
        </p>
        {expandable ? (
          <>
            <span className="text-sm font-light text-text-tertiary">
              <button
                type="button"
                onClick={() => setShowDetails((prev) => !prev)}
                aria-expanded={showDetails}
                className="underline decoration-1 underline-offset-2 decoration-text-tertiary/40"
              >
                About this benchmark
              </button>
              {hint && <span> • {hint}</span>}
            </span>
            {showDetails && (
              <p className="mt-2 text-text-tertiary text-sm leading-snug">
                {description.detailed}
              </p>
            )}
          </>
        ) : (
          <p className="text-text-tertiary text-sm leading-snug">
            {description.detailed}
          </p>
        )}
      </div>
      {stat && (
        <div className="text-right min-w-0">
          <div className="text-[0.9rem] font-light text-text-secondary mb-2">
            {stat.label}
          </div>
          <div className="font-mono text-3xl sm:text-[2.4rem] font-bold break-words">{stat.value}</div>
        </div>
      )}
    </div>
  );
};

export default SectionHeader;
