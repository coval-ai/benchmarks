// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useId } from "react";
import { metricDescriptions } from "@/lib/config/metrics";

const alignClasses = {
  left: "left-0",
  center: "left-1/2 -translate-x-1/2",
  right: "right-0",
};

const MetricInfo: React.FC<{
  metric: string;
  align?: keyof typeof alignClasses;
  children: React.ReactNode;
}> = ({ metric, align = "center", children }) => {
  const id = useId();
  const info =
    metricDescriptions[metric.toLowerCase() as keyof typeof metricDescriptions];
  if (!info || !("tooltip" in info)) return <>{children}</>;
  const isElement = React.isValidElement(children);
  return (
    <span
      className="group relative inline-block"
      tabIndex={isElement ? undefined : 0}
      aria-describedby={isElement ? undefined : id}
    >
      {isElement
        ? React.cloneElement(children as React.ReactElement<Record<string, unknown>>, {
            "aria-describedby": id,
          })
        : children}
      <span
        id={id}
        role="tooltip"
        className={`pointer-events-none invisible absolute bottom-full z-50 mb-1.5 w-64 rounded-lg border border-border-secondary bg-surface-tooltip px-3 py-2 text-left text-xs font-normal leading-snug text-[var(--color-text-on-tooltip)] opacity-0 shadow-md transition-opacity group-hover:visible group-hover:opacity-100 group-focus-visible:visible group-focus-visible:opacity-100 group-has-[:focus-visible]:visible group-has-[:focus-visible]:opacity-100 ${alignClasses[align]}`}
      >
        <span className="font-semibold">{info.short}.</span> {info.tooltip}
      </span>
    </span>
  );
};

export default MetricInfo;
