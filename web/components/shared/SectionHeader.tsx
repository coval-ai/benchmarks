// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";

interface SectionHeaderProps {
  label: string;
  description: {
    short: string;
    detailed: string;
  };
  stat?: {
    label: string;
    value: string;
  };
}

const SectionHeader: React.FC<SectionHeaderProps> = ({
  label,
  description,
  stat,
}) => (
  <div className="flex justify-between items-start gap-8 mb-4">
    <div className="w-3/4 min-w-0">
      <h2 className="text-[0.72rem] font-light text-text-secondary mb-2">
        {label}
      </h2>
      <p className="text-2xl font-medium text-text-primary mb-3">
        {description.short}
      </p>
      <p className="text-text-tertiary text-sm leading-snug">
        {description.detailed}
      </p>
    </div>
    {stat && (
      <div className="text-right min-w-0">
        <div className="text-[0.72rem] font-light text-text-secondary mb-2">
          {stat.label}
        </div>
        <div className="font-mono text-[2.4rem] font-bold break-words">{stat.value}</div>
      </div>
    )}
  </div>
);

export default SectionHeader;
