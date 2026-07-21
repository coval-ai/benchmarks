// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useEffect, useId, useRef, useState } from "react";
import { metricDescriptions } from "@/lib/config/metrics";

const alignClasses = {
  left: "left-0",
  center: "left-1/2 -translate-x-1/2",
  right: "right-0",
};

const MetricInfo: React.FC<{
  metric?: string;
  content?: React.ReactNode;
  align?: keyof typeof alignClasses;
  /** Placement and sizing of the tooltip panel; defaults to above the trigger. */
  panelClassName?: string;
  children: React.ReactNode;
}> = ({
  metric,
  content,
  align = "center",
  panelClassName = "bottom-full mb-1.5 w-64",
  children,
}) => {
  const id = useId();
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: PointerEvent) => {
      if (!wrapperRef.current?.contains(e.target as Node)) {
        setOpen(false);
        wrapperRef.current?.blur();
      }
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);
  const info: { short: string; tooltip?: string } | undefined = metric
    ? metricDescriptions[metric.toLowerCase() as keyof typeof metricDescriptions]
    : undefined;
  let body: React.ReactNode = content ?? null;
  if (body == null && info?.tooltip) {
    const childText = React.Children.toArray(children)
      .filter((c): c is string => typeof c === "string")
      .join("")
      .trim();
    const repeatsTrigger =
      childText.toLowerCase() === info.short.toLowerCase();
    body = repeatsTrigger ? (
      info.tooltip
    ) : (
      <>
        <span className="font-semibold">{info.short}.</span> {info.tooltip}
      </>
    );
  }
  if (!body) return <>{children}</>;
  const isElement = React.isValidElement(children);
  return (
    <span
      ref={wrapperRef}
      className={`group relative inline-block ${isElement ? "" : "-m-3.5 cursor-help p-3.5 lg:-m-2 lg:p-2"}`}
      tabIndex={isElement ? undefined : 0}
      aria-describedby={isElement ? undefined : id}
      onClick={isElement ? undefined : () => setOpen((prev) => !prev)}
      onPointerUp={
        isElement
          ? (e) => e.pointerType !== "mouse" && setOpen((prev) => !prev)
          : undefined
      }
      onBlur={isElement ? undefined : () => setOpen(false)}
    >
      {isElement
        ? React.cloneElement(children as React.ReactElement<Record<string, unknown>>, {
            "aria-describedby": id,
          })
        : children}
      <span
        id={id}
        role="tooltip"
        className={`pointer-events-none absolute z-50 ${panelClassName} max-w-[calc(100vw-1.5rem)] rounded-lg border border-border-secondary bg-surface-tooltip px-3 py-2 text-left font-sans text-xs font-normal leading-snug text-[var(--color-text-on-tooltip)] shadow-md transition-opacity group-hover:visible group-hover:opacity-100 group-focus-visible:visible group-focus-visible:opacity-100 group-has-[:focus-visible]:visible group-has-[:focus-visible]:opacity-100 ${open ? "visible opacity-100" : "invisible opacity-0"} ${alignClasses[align]}`}
      >
        {body}
      </span>
    </span>
  );
};

export default MetricInfo;
