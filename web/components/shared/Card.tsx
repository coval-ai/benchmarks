// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React from "react";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Padding classes; a prop rather than className so overrides don't depend on Tailwind class order. */
  padding?: string;
}

const Card: React.FC<CardProps> = ({
  className = "",
  padding = "p-8",
  ...props
}) => (
  <div
    className={`w-full relative z-[2] border border-border-secondary rounded-lg bg-white ${padding} ${className}`}
    {...props}
  />
);

export default Card;
