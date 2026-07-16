// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";
import { Info } from "lucide-react";
import { useDashboard } from "@/contexts/DashboardContext";
import { useActiveTab } from "@/hooks/useActiveTab";
import { datasetLabel, isPerturbationDataset } from "@/lib/config/datasets";
import MetricInfo from "@/components/shared/MetricInfo";

const WerDatasetSelect: React.FC<{ className?: string }> = ({ className }) => {
  const activeTab = useActiveTab();
  const { werDataset, changeWerDataset, availableWerDatasets, isMobile } =
    useDashboard();

  if (activeTab !== "stt" || availableWerDatasets.length === 0) return null;

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
        content="Scopes the WER column to one evaluation set. Full datasets are distinct recordings; WildASR perturbations replay the clean utterances with one degradation applied. Pooled blends every dataset in the window."
        align={isMobile ? "left" : "right"}
      >
        WER dataset{" "}
        <Info size={12} aria-hidden="true" className="inline align-[-2px]" />
      </MetricInfo>
      <span className="relative inline-flex">
        <select
          aria-label="WER dataset"
          value={werDataset ?? ""}
          onChange={(e) => changeWerDataset(e.target.value || null)}
          className="appearance-none rounded-lg border border-border-primary bg-surface-elevated py-1.5 pl-2.5 pr-7 text-xs font-medium text-text-primary outline-none transition-colors hover:border-selected-border focus:border-selected-border"
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
