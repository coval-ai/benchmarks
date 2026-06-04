// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import React, { useState } from "react";
import GlobalNav from "@/components/layout/GlobalNav";

// Benchmark sections, shown in the secondary nav beneath the global coval.ai nav.
const SECTIONS = [
  { href: "/tts", label: "Text-to-Speech", short: "TTS" },
  { href: "/stt", label: "Speech-to-Text", short: "STT" },
  { href: "/playground", label: "Playground", short: "Playground" }
];

const DashboardHeader: React.FC = () => {
  const pathname = usePathname();
  // True while the global nav's full-screen mobile menu is open — used to hide
  // the secondary nav's tabs so they don't paint over the overlay.
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  return (
    <>
      <GlobalNav onMobileOpenChange={setMobileNavOpen} />

      {/* Secondary nav — in-app benchmark navigation. Sits flush under the
          60px global nav. Hidden on mobile while the full-screen menu is open. */}
      <nav
        aria-label="Benchmark sections"
        className={`fixed inset-x-0 top-[60px] z-40 h-12 border-b border-border-primary bg-white ${
          mobileNavOpen ? "max-lg:hidden" : ""
        }`}
      >
        {/* Anchored to the full-width nav (not the centered container) so it
            always lines up with the logo's left margin, like the global nav. */}
        <span className="absolute left-4 top-1/2 hidden -translate-y-1/2 font-mono text-xs uppercase tracking-wider text-text-tertiary sm:inline md:left-6">
          Voice Model Benchmarks
        </span>
        <div className="absolute left-1/2 top-0 flex h-full -translate-x-1/2 items-center gap-1 px-4 md:px-6">
          {SECTIONS.map((section) => {
            const active = pathname === section.href;
            return (
              <Link
                key={section.href}
                href={section.href}
                aria-current={active ? "page" : undefined}
                aria-label={section.label}
                className={`relative flex h-full items-center px-3 text-sm tracking-wide transition-colors ${
                  active
                    ? "text-text-primary"
                    : "text-text-secondary hover:text-text-primary"
                }`}
              >
                <span className="sm:hidden">{section.short}</span>
                <span className="hidden sm:inline">{section.label}</span>
                <span
                  className={`absolute bottom-0 left-0 h-0.5 bg-text-primary transition-all duration-300 ease-out ${
                    active ? "w-full opacity-100" : "w-0 opacity-0"
                  }`}
                />
              </Link>
            );
          })}
        </div>
      </nav>
    </>
  );
};

export default DashboardHeader;
