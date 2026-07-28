// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { Info } from "lucide-react";
import { useDashboard } from "@/contexts/DashboardContext";
import { datasetLabel, isPerturbationDataset } from "@/lib/config/datasets";
import MetricInfo from "@/components/shared/MetricInfo";

// Drives the WER-column scope by default; pass value/onChange to point it at
// another surface's dataset state (e.g. the accuracy bar chart). Options come
// from whatever datasets the active benchmark's window carries.
const WerDatasetSelect: React.FC<{
  className?: string;
  label?: string;
  value?: string | null;
  onChange?: (dataset: string | null) => void;
}> = ({ className, label = "WER dataset", value, onChange }) => {
  const { werDataset, changeWerDataset, availableWerDatasets, isMobile } =
    useDashboard();
  const selected = value !== undefined ? value : werDataset;
  const select = onChange ?? changeWerDataset;

  if (availableWerDatasets.length === 0) return null;

  const fullSets = availableWerDatasets.filter((d) => !isPerturbationDataset(d));
  const perturbations = availableWerDatasets.filter(isPerturbationDataset);
  const options = (ids: string[]) =>
    ids.map((id) => (
      <option key={id} value={id}>
        {datasetLabel(id)}
      </option>
    ));

  return (
    <span
      className={`inline-flex items-center gap-2 text-xs text-text-secondary${
        className ? ` ${className}` : ""
      }`}
    >
      <MetricInfo
        content={`Scopes WER to one evaluation set.${
          perturbations.length > 0
            ? " Full datasets are distinct recordings; WildASR perturbations replay the clean utterances with one degradation applied."
            : ""
        } Pooled blends every dataset in the window.`}
        align={isMobile ? "left" : "right"}
      >
        {label}{" "}
        <Info size={12} aria-hidden="true" className="inline align-[-2px]" />
      </MetricInfo>
      <span className="relative inline-flex">
        <select
          aria-label={label}
          value={selected ?? ""}
          onChange={(e) => select(e.target.value || null)}
          className="h-11 appearance-none rounded-lg border border-border-primary bg-surface-elevated pl-2.5 pr-7 text-xs font-medium text-text-primary outline-none transition-colors hover:border-selected-border focus:border-selected-border lg:h-auto lg:py-1.5"
        >
          <option value="">All datasets (pooled)</option>
          {perturbations.length > 0 ? (
            <>
              <optgroup label="Full datasets">{options(fullSets)}</optgroup>
              <optgroup label="WildASR perturbations">
                {options(perturbations)}
              </optgroup>
            </>
          ) : (
            options(fullSets)
          )}
        </select>
        <svg
          aria-hidden
          viewBox="0 0 12 12"
          className="pointer-events-none absolute right-2 top-1/2 h-3 w-3 -translate-y-1/2 text-text-tertiary"
        >
          <path
            d="M2.5 4.5 6 8l3.5-3.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
    </span>
  );
};

export default WerDatasetSelect;
