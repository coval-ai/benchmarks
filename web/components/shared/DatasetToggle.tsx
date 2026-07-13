// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { STT_DATASETS } from "@/lib/config/datasets";

interface DatasetToggleProps {
  value: string;
  onChange: (dataset: string) => void;
  className?: string;
}

const DatasetToggle: React.FC<DatasetToggleProps> = ({
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
    const currentIndex = STT_DATASETS.findIndex((d) => d.id === value);
    const nextIndex =
      (currentIndex + step + STT_DATASETS.length) % STT_DATASETS.length;
    const next = STT_DATASETS[nextIndex];
    if (!next) return;
    onChange(next.id);
    const radios =
      event.currentTarget.querySelectorAll<HTMLButtonElement>("button");
    radios[nextIndex]?.focus();
  };

  return (
    <div
      role="radiogroup"
      aria-label="Dataset"
      onKeyDown={handleKeyDown}
      className={`inline-flex h-8 items-center gap-0.5 rounded-md bg-surface-toggle-inactive p-[3px] ${className}`}
    >
      {STT_DATASETS.map((dataset) => {
        const active = dataset.id === value;
        return (
          <button
            key={dataset.id}
            type="button"
            role="radio"
            aria-checked={active}
            tabIndex={active ? 0 : -1}
            title={dataset.description}
            onClick={() => onChange(dataset.id)}
            className={`inline-flex items-center justify-center self-stretch rounded-sm px-3 text-sm font-medium transition-colors duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40 focus-visible:ring-offset-1 ${
              active
                ? "bg-surface-primary text-text-primary shadow-sm"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            {dataset.label}
          </button>
        );
      })}
    </div>
  );
};

export default DatasetToggle;
