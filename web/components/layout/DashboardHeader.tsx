// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import React from "react";
import ThemeSwitch from "@/components/layout/ThemeSwitch";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";

// Benchmark sections, shown as tabs in the upper right of the top nav bar.
const SECTIONS = [
  { href: "/overview", label: "Overview", short: "Overview" },
  { href: "/tts", label: "Text-to-Speech", short: "TTS" },
  { href: "/stt", label: "Speech-to-Text", short: "STT" },
  { href: "/playground", label: "Playground", short: "Playground" }
];

// GitHub mark, reused in the desktop header and the mobile menu.
const GithubIcon: React.FC = () => (
  <svg viewBox="0 0 16 16" width={20} height={20} aria-hidden fill="currentColor">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
  </svg>
);

const DashboardHeader: React.FC = () => {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = React.useState(false);

  // Close the mobile menu whenever the route changes.
  React.useEffect(() => {
    setMenuOpen(false);
  }, [pathname]);

  return (
    <header className="fixed inset-x-0 top-0 z-50 h-[60px] border-b border-border-primary bg-surface-primary">
      <div className="relative flex h-full items-center px-4 md:px-6">
        {/* Brand — "Voice Model Benchmarks" wordmark over a "By Coval" lockup */}
        <div className="flex flex-col justify-center leading-none">
          <Link
            href="/"
            aria-label="Voice AI Benchmarks"
            className="font-sans text-lg font-medium tracking-wide text-text-primary transition-opacity hover:opacity-80"
          >
            Voice AI Benchmarks
          </Link>
          <a
            href="https://www.coval.ai"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="By Coval"
            onClick={() =>
              capturePostHogEvent(POSTHOG_EVENTS.headerCovalLinkClicked, {
                path: pathname
              })
            }
            className="flex items-center gap-1 text-[11px] text-text-tertiary transition-opacity hover:opacity-80"
          >
            By
            <Image
              src="/coval-logo.svg"
              alt="Coval"
              width={99}
              height={22}
              priority
              className="coval-logo h-2 w-auto"
            />
          </a>
        </div>

        {/* Section tabs — centered on the page, independent of the brand width */}
        <nav
          aria-label="Benchmark sections"
          className="absolute left-1/2 top-0 hidden h-full -translate-x-1/2 items-center gap-1 lg:flex"
        >
          {SECTIONS.map((section) => {
            const active = pathname === section.href;
            return (
              <Link
                key={section.href}
                href={section.href}
                aria-current={active ? "page" : undefined}
                aria-label={section.label}
                className={`relative flex h-full items-center px-2 text-sm tracking-wide transition-colors sm:px-3 ${
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
        </nav>

        {/* Theme toggle + GitHub link (desktop only) */}
        <div className="ml-auto hidden items-center gap-4 lg:flex">
          <ThemeSwitch />
          <a
            href="https://github.com/coval-ai/benchmarks"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="View source on GitHub"
            className="flex items-center text-text-secondary transition-colors hover:text-text-primary"
          >
            <GithubIcon />
          </a>
        </div>

        {/* Hamburger — opens the mobile menu (two-bar icon, mobile only) */}
        <button
          type="button"
          onClick={() => setMenuOpen((open) => !open)}
          aria-label={menuOpen ? "Close menu" : "Open menu"}
          aria-expanded={menuOpen}
          aria-controls="mobile-menu"
          className="ml-auto flex h-11 w-11 flex-col items-center justify-center gap-1.5 text-text-primary lg:hidden"
        >
          <span
            className={`h-0.5 w-6 bg-current transition-transform duration-300 ease-out ${
              menuOpen ? "translate-y-1 rotate-45" : ""
            }`}
          />
          <span
            className={`h-0.5 w-6 bg-current transition-transform duration-300 ease-out ${
              menuOpen ? "-translate-y-1 -rotate-45" : ""
            }`}
          />
        </button>
      </div>

      {/* Mobile menu — fullscreen overlay with all global items (mobile only) */}
      <nav
        id="mobile-menu"
        aria-label="Global navigation"
        aria-hidden={!menuOpen}
        className={`fixed inset-x-0 top-[60px] bottom-0 flex flex-col gap-2 bg-surface-primary px-4 py-6 transition-all duration-300 ease-out lg:hidden ${
          menuOpen
            ? "translate-y-0 opacity-100 pointer-events-auto"
            : "-translate-y-2 opacity-0 pointer-events-none"
        }`}
      >
        {SECTIONS.map((section) => {
          const active = pathname === section.href;
          return (
            <Link
              key={section.href}
              href={section.href}
              aria-current={active ? "page" : undefined}
              tabIndex={menuOpen ? undefined : -1}
              onClick={() => setMenuOpen(false)}
              className={`flex items-center rounded-xl px-4 py-4 text-2xl tracking-wide transition-colors ${
                active
                  ? "bg-selected-bg font-medium text-text-primary"
                  : "text-text-secondary hover:bg-hover-bg hover:text-text-primary"
              }`}
            >
              {section.label}
            </Link>
          );
        })}
        <div className="mt-auto flex items-center justify-between rounded-xl px-4 py-3">
          <span className="text-2xl tracking-wide text-text-secondary">Theme</span>
          <ThemeSwitch />
        </div>
        <a
          href="https://github.com/coval-ai/benchmarks"
          target="_blank"
          rel="noopener noreferrer"
          tabIndex={menuOpen ? undefined : -1}
          onClick={() => setMenuOpen(false)}
          className="flex items-center gap-3 rounded-xl px-4 py-4 text-2xl tracking-wide text-text-secondary transition-colors hover:bg-hover-bg hover:text-text-primary"
        >
          <GithubIcon />
          GitHub
        </a>
      </nav>
    </header>
  );
};

export default DashboardHeader;
