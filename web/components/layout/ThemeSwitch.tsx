// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import React, { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";

const OPTIONS = [
  { value: "light", Icon: Sun, label: "Switch to light theme" },
  { value: "dark", Icon: Moon, label: "Switch to dark theme" },
] as const;

const ThemeSwitch: React.FC<{ className?: string }> = ({ className = "" }) => {
  const [mounted, setMounted] = useState(false);
  const { resolvedTheme, setTheme } = useTheme();
  useEffect(() => setMounted(true), []);

  return (
    <div
      role="radiogroup"
      aria-label="Theme"
      className={`inline-flex h-8 items-center gap-0.5 rounded-md bg-surface-toggle-inactive p-[3px] ${className}`}
    >
      {OPTIONS.map(({ value, Icon, label }) => {
        const active = mounted && resolvedTheme === value;
        return (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={label}
            onClick={() => setTheme(value)}
            className={`inline-flex items-center justify-center self-stretch rounded-sm px-2.5 transition-colors duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40 ${
              active
                ? "bg-surface-primary text-text-primary shadow-sm"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            <Icon size={15} aria-hidden />
          </button>
        );
      })}
    </div>
  );
};

export default ThemeSwitch;
