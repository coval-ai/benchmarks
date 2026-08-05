// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { Globe } from "lucide-react";
import MetricInfo from "@/components/shared/MetricInfo";
import TooltipBadge from "@/components/shared/TooltipBadge";
import { REGION_BLURB, REGION_LABEL } from "@/lib/utils/facets";

/**
 * The one explainer body every inference-region popover shows. Exported so the
 * clip-safe overlay (`handlersFor`) can anchor the same content from an SVG or
 * scroll container, where an anchored panel would be clipped.
 */
export const REGION_CONTENT = (
  <>
    <span className="font-semibold">{REGION_LABEL}.</span> {REGION_BLURB}
  </>
);

/**
 * Globe-icon button that opens the explainer on hover/tap. Mirrors
 * DedicatedInfoIcon so the two caveat markers read as one system.
 */
export const RegionInfoIcon: React.FC<{ size?: number; className?: string }> = ({
  size = 13,
  className = "",
}) => (
  <MetricInfo
    content={REGION_CONTENT}
    align="left"
    panelClassName="bottom-full mb-1.5 w-60 max-sm:fixed max-sm:inset-x-4 max-sm:bottom-auto max-sm:top-24 max-sm:mb-0 max-sm:w-auto"
  >
    <button
      type="button"
      aria-label="About inference region"
      className={`flex shrink-0 cursor-help items-center justify-center rounded-md text-text-tertiary transition-colors hover:bg-surface-toggle-inactive hover:text-text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40 ${className}`}
    >
      <Globe size={size} aria-hidden />
    </button>
  </MetricInfo>
);

/**
 * Cross-region tooltip row naming the provider's region; hover/tap explains the
 * handicap. Renders nothing without a region, so callers can pass a lookup
 * straight through.
 */
export const RegionBadge: React.FC<{ region?: string }> = ({ region }) =>
  region ? <TooltipBadge icon={Globe} label={region} content={REGION_CONTENT} /> : null;
