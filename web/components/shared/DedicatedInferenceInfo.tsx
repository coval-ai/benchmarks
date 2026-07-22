// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useCallback, useEffect, useMemo, useState, type RefObject } from "react";
import { Server } from "lucide-react";
import MetricInfo from "@/components/shared/MetricInfo";
import {
  DEDICATED_INFERENCE_BLURB,
  DEDICATED_INFERENCE_LABEL,
} from "@/lib/utils/facets";

/** The one explainer body every dedicated-inference popover shows. */
const CONTENT = (
  <>
    <span className="font-semibold">{DEDICATED_INFERENCE_LABEL}.</span>{" "}
    {DEDICATED_INFERENCE_BLURB}
  </>
);

/**
 * Server-icon button that opens the explainer on hover/tap. The panel opens
 * upward (inside the z-[2] Cards only upward overlap paints above later
 * siblings) and becomes a viewport-fixed sheet below sm, where legend/scroll
 * containers would clip an anchored panel.
 */
export const DedicatedInfoIcon: React.FC<{ size?: number; className?: string }> = ({
  size = 13,
  className = "",
}) => (
  <MetricInfo
    content={CONTENT}
    align="left"
    panelClassName="bottom-full mb-1.5 w-60 max-sm:fixed max-sm:inset-x-4 max-sm:bottom-auto max-sm:top-24 max-sm:mb-0 max-sm:w-auto"
  >
    <button
      type="button"
      aria-label="About dedicated inference"
      className={`flex shrink-0 cursor-help items-center justify-center rounded-md text-text-tertiary transition-colors hover:bg-surface-toggle-inactive hover:text-text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40 ${className}`}
    >
      <Server size={size} aria-hidden />
    </button>
  </MetricInfo>
);

/** "Dedicated inference" tooltip row; hover/tap opens the explainer. */
export const DedicatedBadge: React.FC = () => (
  <MetricInfo
    content={CONTENT}
    align="left"
    panelClassName="top-full mt-1.5 w-60 whitespace-normal"
  >
    <button
      type="button"
      className="flex cursor-help items-center gap-1 border-0 bg-transparent p-0 text-[color:var(--color-text-on-tooltip-secondary)]"
      style={{ font: "inherit" }}
    >
      <Server size={10} aria-hidden /> {DEDICATED_INFERENCE_LABEL}
    </button>
  </MetricInfo>
);

/**
 * Explainer for dedicated icons rendered inside scroll or SVG containers,
 * where an anchored panel would clip. The trigger spreads `iconHandlers`; the
 * host renders `overlay` inside the unclipped position:relative ancestor that
 * `containerRef` points at, and calls `dismiss` before repositioning content.
 */
export function useDedicatedInfoTip(containerRef: RefObject<HTMLElement | null>) {
  const [tip, setTip] = useState<{ x: number; yTop: number; pinned: boolean } | null>(
    null
  );
  const dismiss = useCallback(() => setTip(null), []);

  useEffect(() => {
    if (!tip?.pinned) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && dismiss();
    document.addEventListener("click", dismiss);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("click", dismiss);
      document.removeEventListener("keydown", onKey);
    };
  }, [tip, dismiss]);

  // Identity-stable so hosts can safely reference the handlers from redraw
  // effects (BoxPlot's d3 pass) without re-triggering them every render.
  const iconHandlers = useMemo(() => {
    const anchor = (e: React.MouseEvent) => {
      const box = containerRef.current?.getBoundingClientRect();
      const icon = (e.currentTarget as Element).getBoundingClientRect();
      return {
        x: icon.x + icon.width / 2 - (box?.x ?? 0),
        yTop: icon.y - (box?.y ?? 0),
      };
    };
    // Anchors are computed before setTip: React may replay updater functions
    // on a later render, when the event's currentTarget is already null.
    return {
      onMouseEnter: (e: React.MouseEvent) => {
        const a = anchor(e);
        setTip((t) => (t?.pinned ? t : { ...a, pinned: false }));
      },
      onMouseLeave: () => setTip((t) => (t?.pinned ? t : null)),
      onClick: (e: React.MouseEvent) => {
        e.stopPropagation();
        const a = anchor(e);
        setTip((t) => (t?.pinned ? null : { ...a, pinned: true }));
      },
    };
  }, [containerRef]);

  const width = containerRef.current?.clientWidth ?? 0;
  const overlay = tip ? (
    <div
      role="tooltip"
      onClick={(e) => e.stopPropagation()}
      className="absolute z-10 w-60 rounded-lg border border-border-secondary bg-surface-tooltip px-3 py-2 text-left text-xs font-normal leading-snug text-[var(--color-text-on-tooltip)] shadow-md"
      style={{
        left: width ? Math.min(Math.max(tip.x, 122), width - 122) : tip.x,
        top: tip.yTop,
        transform: "translate(-50%, calc(-100% - 8px))",
        pointerEvents: tip.pinned ? "auto" : "none",
      }}
    >
      {CONTENT}
    </div>
  ) : null;

  return { iconHandlers, overlay, dismiss };
}
