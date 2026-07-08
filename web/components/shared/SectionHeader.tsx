// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useState } from "react";

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
}

const SectionHeader: React.FC<SectionHeaderProps> = ({
  label,
  description,
  stat,
  hint,
  expandable = true,
}) => {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="flex justify-between items-start gap-8 mb-4">
      <div className="w-3/4 min-w-0">
        <h2 className="text-[0.9rem] font-light text-text-secondary mb-2">
          {label}
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
        <div className="text-right min-w-0">
          <div className="text-[0.9rem] font-light text-text-secondary mb-2">
            {stat.label}
          </div>
          <div className="font-mono text-3xl sm:text-[2.4rem] font-bold break-words">{stat.value}</div>
        </div>
      )}
    </div>
  );
};

export default SectionHeader;
