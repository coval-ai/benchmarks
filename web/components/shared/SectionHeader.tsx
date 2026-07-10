// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useEffect, useState } from "react";
import { Check, ImageDown, Link2, Table } from "lucide-react";
import { downloadCSV, downloadChartPNG } from "@/lib/utils/chartExport";
import { useThemeColors } from "@/hooks/useThemeColors";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";

const iconButtonClass =
  "flex h-7 w-7 items-center justify-center rounded-lg border border-border-secondary bg-surface-primary text-text-secondary transition-colors hover:bg-surface-toggle-inactive hover:text-text-primary";

interface SectionHeaderProps {
  label: string;
  description: {
    short: string;
    detailed: string;
  };
  stat?: {
    label: React.ReactNode;
    value: string;
  };
  /** Optional hint shown after the "About this benchmark" link, separated by an interpunct. */
  hint?: string;
  /** When false, the detailed text shows inline instead of behind a toggle. */
  expandable?: boolean;
  /** Rows for the Download Data (CSV) button; must reflect the active filters. */
  exportRows?: () => Record<string, unknown>[];
  /** Set false for sections without a chart SVG to hide Download Image. */
  exportImage?: boolean;
  /** X-axis title drawn under the exported PNG when the SVG lacks one. */
  exportXLabel?: string;
  /** Mutates the exported SVG clone, e.g. to label scatter dots by name. */
  exportAnnotate?: (clone: SVGSVGElement) => void;
}

const SectionHeader: React.FC<SectionHeaderProps> = ({
  label,
  description,
  stat,
  hint,
  expandable = true,
  exportRows,
  exportImage = true,
  exportXLabel,
  exportAnnotate,
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const [copied, setCopied] = useState(false);
  const themeColors = useThemeColors();
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

  const trackShare = (format: "link" | "png" | "csv") =>
    capturePostHogEvent(POSTHOG_EVENTS.dashboardChartShared, {
      chart: anchorId,
      path: window.location.pathname,
      format,
    });

  const copyLink = () => {
    navigator.clipboard.writeText(
      `${window.location.origin}${window.location.pathname}#${anchorId}`
    );
    window.history.replaceState(null, "", `#${anchorId}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    trackShare("link");
  };

  const downloadImage = async () => {
    const root = document.getElementById(anchorId);
    const card = root?.parentElement;
    if (!card) return;
    // The chart is the largest SVG in the card; icon SVGs and a chart still
    // sizing to zero during layout are excluded so we never export those.
    const findSvg = () =>
      Array.from(card.querySelectorAll("svg"))
        .filter((el) => el.clientWidth > 100 && el.clientHeight > 100)
        .sort(
          (a, b) =>
            b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight
        )[0];
    let svg = findSvg();
    // The exporting flag rejects overlapping clicks on the same card, which
    // would otherwise read the temporary stage size as the layout to restore.
    if (!svg || card.dataset.exporting) return;
    card.dataset.exporting = "1";
    // Exports render on a fixed desktop-sized stage so every device produces
    // the same image, taller than the on-screen chart so dense charts have
    // room to place labels. The charts re-render on container resize, which
    // can replace the SVG node, hence the re-query once the stage settles.
    const wrapper = svg.closest<HTMLElement>(
      ".recharts-responsive-container"
    )?.parentElement;
    const priorWidth = card.style.width;
    const priorHeight = wrapper?.style.height ?? "";
    card.style.width = "880px";
    if (wrapper) wrapper.style.height = "420px";
    const settled = () => {
      const el = findSvg();
      return (
        !!el && el.clientWidth >= 640 && (!wrapper || el.clientHeight >= 416)
      );
    };
    for (let i = 0; i < 30 && !settled(); i++) {
      await new Promise(requestAnimationFrame);
    }
    svg = findSvg() ?? svg;
    // The stat label can be a ReactNode (MetricInfo), so its text comes from
    // the DOM — minus the hidden tooltip MetricInfo keeps mounted.
    const statLabel = root?.querySelector("[data-stat-label]")?.cloneNode(true);
    if (statLabel instanceof HTMLElement) {
      statLabel
        .querySelectorAll('[role="tooltip"]')
        .forEach((el) => el.remove());
    }
    const capture = downloadChartPNG(svg, `${anchorId}.png`, {
      label,
      title: description.short,
      xLabel: exportXLabel,
      stat: stat && {
        label:
          statLabel instanceof HTMLElement
            ? (statLabel.textContent?.trim() ?? "")
            : "",
        value: stat.value,
      },
      annotate: exportAnnotate,
      legend: Array.from(
        card.querySelectorAll(".recharts-legend-wrapper li, [data-chart-legend] li")
      ).map((li) => {
        const entry = li.cloneNode(true) as HTMLElement;
        entry.querySelectorAll('[role="tooltip"]').forEach((el) => el.remove());
        return {
          label: entry.textContent?.trim() ?? "",
          color: li.querySelector("span")?.style.backgroundColor ?? "#0f0c0a",
          dimmed: li.hasAttribute("data-dimmed"),
        };
      }),
    }, themeColors);
    // downloadChartPNG measures and clones the SVG synchronously before its
    // first await, so the stage can be struck as soon as it returns — the
    // layout is never left widened if rasterizing stalls or rejects.
    card.style.width = priorWidth;
    if (wrapper) wrapper.style.height = priorHeight;
    delete card.dataset.exporting;
    const ok = await capture.catch(() => false);
    if (ok) trackShare("png");
  };

  const downloadData = () => {
    const rows = exportRows?.() ?? [];
    if (rows.length === 0) return;
    downloadCSV(rows, `${anchorId}.csv`);
    trackShare("csv");
  };

  return (
    <div id={anchorId} className="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-3 sm:gap-8 mb-4 scroll-mt-24">
      <div className="w-full sm:w-3/4 min-w-0">
        <h2 className="flex items-center gap-2 text-[0.9rem] font-light text-text-secondary mb-2">
          {label}
          <button
            type="button"
            onClick={copyLink}
            aria-label="Copy link to this chart"
            title="Copy link"
            className={iconButtonClass}
          >
            {copied ? <Check size={14} /> : <Link2 size={14} />}
          </button>
          {exportRows && exportImage && (
            <button
              type="button"
              onClick={() => void downloadImage()}
              aria-label="Download chart as image"
              title="Download image"
              className={iconButtonClass}
            >
              <ImageDown size={14} />
            </button>
          )}
          {exportRows && (
            <button
              type="button"
              onClick={downloadData}
              aria-label="Download chart data"
              title="Download data"
              className={iconButtonClass}
            >
              <Table size={14} />
            </button>
          )}
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
        <div className="text-left sm:text-right min-w-0">
          <div
            data-stat-label
            className="text-[0.9rem] font-light text-text-secondary mb-2"
          >
            {stat.label}
          </div>
          <div className="font-mono text-3xl sm:text-[2.4rem] font-bold break-words">{stat.value}</div>
        </div>
      )}
    </div>
  );
};

export default SectionHeader;
