// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import type { LucideIcon } from "lucide-react";
import MetricInfo from "@/components/shared/MetricInfo";

/**
 * Caveat row inside a chart tooltip: an icon, a short label, and an explainer
 * that opens on hover/tap. Shared by every per-model caveat badge so they can't
 * drift in size, color, or panel placement.
 */
const TooltipBadge: React.FC<{
  icon: LucideIcon;
  label: string;
  content: React.ReactNode;
}> = ({ icon: Icon, label, content }) => (
  <MetricInfo
    content={content}
    align="left"
    panelClassName="top-full mt-1.5 w-60 whitespace-normal"
  >
    <button
      type="button"
      className="flex cursor-help items-center gap-1 border-0 bg-transparent p-0 text-[color:var(--color-text-on-tooltip-secondary)]"
      style={{ font: "inherit" }}
    >
      <Icon size={10} aria-hidden /> {label}
    </button>
  </MetricInfo>
);

export default TooltipBadge;
