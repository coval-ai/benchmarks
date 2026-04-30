// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useState } from "react";

interface ExpandableDescriptionProps {
  description: {
    short: string;
    detailed: string;
  };
}

const ExpandableDescription: React.FC<ExpandableDescriptionProps> = ({
  description
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const toggleExpanded = () => {
    setIsExpanded((prev) => !prev);
  };

  return (
    <div className="mb-4">
      <div className="flex items-center gap-2">
        <p className="text-text-secondary">{description.short}</p>
        <button
          onClick={toggleExpanded}
          className="text-text-secondary hover:text-text-primary transition-colors p-1"
          aria-label={`${isExpanded ? "Collapse" : "Expand"} ${
            description.short
          } description`}
        >
          <svg
            className={`w-4 h-4 transition-transform ${
              isExpanded ? "rotate-180" : ""
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </button>
      </div>

      {isExpanded && (
        <div className="mt-2 pl-0 pt-3">
          <p className="text-text-secondary text-sm leading-relaxed">
            {description.detailed}
          </p>
        </div>
      )}
    </div>
  );
};

export default ExpandableDescription;
