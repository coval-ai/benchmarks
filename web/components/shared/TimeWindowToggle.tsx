// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import {
  TIME_WINDOWS,
  WINDOW_LABELS,
  type TimeWindow,
} from "@/lib/config/timeWindows";

interface TimeWindowToggleProps {
  value: TimeWindow;
  onChange: (window: TimeWindow) => void;
  className?: string;
}

const TimeWindowToggle: React.FC<TimeWindowToggleProps> = ({
  value,
  onChange,
  className = "",
}) => {
  // Radiogroup contract: one Tab stop, arrows move the selection.
  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    let step: number;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      step = 1;
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      step = -1;
    } else {
      return;
    }
    event.preventDefault();
    const currentIndex = TIME_WINDOWS.indexOf(value);
    const nextIndex =
      (currentIndex + step + TIME_WINDOWS.length) % TIME_WINDOWS.length;
    onChange(TIME_WINDOWS[nextIndex] ?? value);
    const radios =
      event.currentTarget.querySelectorAll<HTMLButtonElement>("button");
    radios[nextIndex]?.focus();
  };

  return (
    <div
      role="radiogroup"
      aria-label="Time window"
      onKeyDown={handleKeyDown}
      className={`inline-flex h-8 items-center gap-0.5 rounded-md bg-surface-toggle-inactive p-[3px] ${className}`}
    >
      {TIME_WINDOWS.map((timeWindow) => {
        const active = timeWindow === value;
        return (
          <button
            key={timeWindow}
            type="button"
            role="radio"
            aria-checked={active}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(timeWindow)}
            className={`inline-flex items-center justify-center self-stretch rounded-sm px-3 text-sm font-medium transition-colors duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40 focus-visible:ring-offset-1 ${
              active
                ? "bg-surface-primary text-text-primary shadow-sm"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            {WINDOW_LABELS[timeWindow]}
          </button>
        );
      })}
    </div>
  );
};

export default TimeWindowToggle;
