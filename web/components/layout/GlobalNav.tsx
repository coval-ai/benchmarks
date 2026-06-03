// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import Image from "next/image";
import React, { useEffect, useState } from "react";

// Marketing chrome mirrored from www.coval.ai. Benchmarks is hosted on a
// separate origin (benchmarks.coval.ai), so coval.ai-relative paths are made
// absolute here; docs/app/trust links point at their own hosts.
const COVAL = "https://www.coval.ai";

type NavLink = { label: string; href: string; external?: boolean };
type NavGroup = { label: string; href?: string; items?: NavLink[] };

const NAV_GROUPS: NavGroup[] = [
  {
    label: "Product",
    items: [
      { label: "Overview", href: `${COVAL}/products` },
      { label: "Simulate", href: `${COVAL}/products/simulation` },
      { label: "Observe", href: `${COVAL}/products/observability` },
      { label: "Human Review", href: `${COVAL}/products/human-review` }
    ]
  },
  {
    label: "Industries",
    items: [
      { label: "Financial Services", href: `${COVAL}/industries/financial-services` },
      { label: "Healthcare", href: `${COVAL}/industries/healthcare` },
      { label: "Insurance", href: `${COVAL}/industries/insurance` }
    ]
  },
  { label: "Pricing", href: `${COVAL}/pricing` },
  {
    label: "Resources",
    items: [
      { label: "Documentation", href: "https://docs.coval.dev", external: true },
      {
        label: "API Reference",
        href: "https://docs.coval.dev/api-reference/v1/introduction",
        external: true
      },
      { label: "Trust Center", href: "https://trust.oneleet.com/coval", external: true }
    ]
  },
  {
    label: "Company",
    items: [
      { label: "About Us", href: `${COVAL}/about-us` },
      { label: "Blog", href: `${COVAL}/blog` },
      { label: "Partners", href: `${COVAL}/partners` },
      { label: "Careers", href: `${COVAL}/careers` }
    ]
  }
];

const LOGIN_HREF = "https://app.coval.dev/login";
const GET_STARTED_HREF = `${COVAL}/pricing`;

const externalProps = (external?: boolean) =>
  external ? { target: "_blank" as const, rel: "noopener noreferrer" } : {};

// Chevrons mirrored from coval.ai's mobile nav (same paths/stroke).
const ChevronRight: React.FC = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 16 16"
    fill="none"
    aria-hidden="true"
    className="shrink-0 text-[#999]"
  >
    <path
      d="M6 3l5 5-5 5"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const ChevronLeft: React.FC = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 16 16"
    fill="none"
    aria-hidden="true"
    className="shrink-0"
  >
    <path
      d="M10 3L5 8l5 5"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const DesktopGroup: React.FC<{ group: NavGroup }> = ({ group }) => {
  // Direct link (no dropdown).
  if (!group.items) {
    return (
      <a
        href={group.href}
        className="rounded-full px-3 py-2 text-sm font-medium text-[#444] hover:bg-[#e8e6e0] hover:text-[#0a0a0a] transition-colors"
      >
        {group.label}
      </a>
    );
  }

  // Hover dropdown. The menu sits flush under the trigger (pt bridge keeps the
  // hover area continuous so it doesn't close in the gap).
  return (
    <div className="group relative">
      <button
        type="button"
        aria-haspopup="menu"
        className="flex items-center rounded-full px-3 py-2 text-sm font-medium text-[#444] hover:bg-[#e8e6e0] hover:text-[#0a0a0a] transition-colors"
      >
        {group.label}
      </button>
      <div className="invisible absolute left-0 top-full pt-2 opacity-0 transition-opacity duration-150 group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100">
        <div
          role="menu"
          className="min-w-[15rem] rounded-2xl bg-surface-primary p-2 shadow-xl"
        >
          {group.items.map((item) => (
            <a
              key={item.label}
              href={item.href}
              role="menuitem"
              {...externalProps(item.external)}
              className="block rounded-xl px-3 py-2 text-sm font-medium text-[#111] hover:bg-[#f2f1ee] transition-colors"
            >
              {item.label}
            </a>
          ))}
        </div>
      </div>
    </div>
  );
};

const GlobalNav: React.FC<{
  // Notifies the parent when the mobile menu opens/closes (e.g. so the
  // secondary nav can hide its tabs behind the full-screen overlay).
  onMobileOpenChange?: (open: boolean) => void;
}> = ({ onMobileOpenChange }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  // Index of the open child panel (group with sub-items), or null for the root.
  const [activePanel, setActivePanel] = useState<number | null>(null);

  useEffect(() => {
    onMobileOpenChange?.(mobileOpen);
  }, [mobileOpen, onMobileOpenChange]);

  const closeMobile = () => {
    setMobileOpen(false);
    setActivePanel(null);
  };

  return (
    <>
    <header className="fixed inset-x-0 top-0 z-50 h-[60px] border-b border-border-primary bg-white">
      <div className="flex h-full items-center justify-between px-4 md:px-6">
        {/* Logo */}
        <a
          href={`${COVAL}/`}
          aria-label="Coval home"
          className="shrink-0 transition-opacity hover:opacity-80"
        >
          <Image
            src="/coval-logo.svg"
            alt="Coval"
            width={99}
            height={22}
            priority
            className="h-[22px] w-auto"
          />
        </a>

        {/* Desktop nav groups */}
        <nav className="hidden items-center gap-1 lg:flex">
          {NAV_GROUPS.map((group) => (
            <DesktopGroup key={group.label} group={group} />
          ))}
        </nav>

        {/* Desktop CTAs */}
        <div className="hidden items-center gap-2 lg:flex">
          <a
            href={LOGIN_HREF}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full px-3 py-2 font-sans text-sm text-[#444] hover:bg-[#e8e6e0] hover:text-[#0a0a0a] transition-colors"
          >
            Log In
          </a>
          <a
            href={GET_STARTED_HREF}
            className="rounded-full bg-text-primary px-4 py-2 font-sans text-sm text-surface-primary transition-opacity hover:opacity-85"
          >
            Get Started
          </a>
        </div>

        {/* Mobile toggle */}
        <button
          type="button"
          onClick={() => (mobileOpen ? closeMobile() : setMobileOpen(true))}
          aria-expanded={mobileOpen}
          aria-controls="nav-mobile-menu"
          aria-label="Toggle navigation menu"
          className="flex h-9 w-9 items-center justify-center rounded-lg hover:bg-hover-bg lg:hidden"
        >
          <span className="relative block h-3.5 w-5">
            <span
              className={`absolute left-0 top-0 h-0.5 w-5 bg-text-primary transition-transform ${
                mobileOpen ? "translate-y-[6px] rotate-45" : ""
              }`}
            />
            <span
              className={`absolute left-0 top-[6px] h-0.5 w-5 bg-text-primary transition-opacity ${
                mobileOpen ? "opacity-0" : ""
              }`}
            />
            <span
              className={`absolute left-0 top-[12px] h-0.5 w-5 bg-text-primary transition-transform ${
                mobileOpen ? "-translate-y-[6px] -rotate-45" : ""
              }`}
            />
          </span>
        </button>
      </div>
    </header>

      {/* Mobile menu — full-screen sliding panels, mirrored from coval.ai.
          Sits below the 60px header; parent groups slide to a child panel. */}
      <div
        id="nav-mobile-menu"
        aria-hidden={!mobileOpen}
        className={`fixed inset-x-0 bottom-0 top-[60px] z-40 overflow-hidden bg-[#f2f1ee] lg:hidden ${
          mobileOpen ? "block" : "hidden"
        }`}
      >
        <div className="relative h-full w-full overflow-hidden">
          {/* Root panel */}
          <div
            inert={activePanel !== null}
            className={`absolute inset-0 flex flex-col gap-1 overflow-y-auto px-3 py-4 transition-transform duration-[280ms] ease-[cubic-bezier(0.4,0,0.2,1)] ${
              activePanel !== null ? "-translate-x-full" : "translate-x-0"
            }`}
          >
            <nav aria-label="Mobile navigation" className="flex flex-col gap-1">
              {NAV_GROUPS.map((group, i) =>
                group.items ? (
                  <button
                    key={group.label}
                    type="button"
                    onClick={() => setActivePanel(i)}
                    className="flex w-full items-center justify-between rounded-lg px-3 py-3.5 text-left text-2xl font-medium text-[#222] transition-colors hover:bg-[#e8e6e0] hover:text-[#111]"
                  >
                    <span>{group.label}</span>
                    <ChevronRight />
                  </button>
                ) : (
                  <a
                    key={group.label}
                    href={group.href}
                    onClick={closeMobile}
                    className="flex w-full items-center justify-between rounded-lg px-3 py-3.5 text-2xl font-medium text-[#222] transition-colors hover:bg-[#e8e6e0] hover:text-[#111]"
                  >
                    {group.label}
                  </a>
                )
              )}
            </nav>
            <div className="mt-auto flex gap-2 border-t border-[#e5e5e5] px-3 pt-3">
              <a
                href={LOGIN_HREF}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-full px-4 py-2 font-sans text-sm font-medium text-[#444] transition-colors hover:bg-[#e8e6e0] hover:text-[#0a0a0a]"
              >
                Log In
              </a>
              <a
                href={GET_STARTED_HREF}
                className="rounded-full bg-[#1f1e1c] px-4 py-2 font-sans text-sm text-[#f2f1ee] transition-colors hover:bg-[#333130]"
              >
                Get Started
              </a>
            </div>
          </div>

          {/* Child panels — one per group with sub-items */}
          {NAV_GROUPS.map((group, i) =>
            group.items ? (
              <div
                key={group.label}
                aria-hidden={activePanel !== i}
                inert={activePanel !== i}
                className={`absolute inset-0 flex flex-col gap-1 overflow-y-auto px-3 py-4 transition-transform duration-[280ms] ease-[cubic-bezier(0.4,0,0.2,1)] ${
                  activePanel === i ? "translate-x-0" : "translate-x-full"
                }`}
              >
                <div className="mb-2 flex items-center gap-2 px-1">
                  <button
                    type="button"
                    onClick={() => setActivePanel(null)}
                    className="flex items-center gap-1.5 rounded-md p-2 text-[15px] font-medium text-[#666] transition-colors hover:bg-[#e8e6e0] hover:text-[#111]"
                  >
                    <ChevronLeft />
                    {group.label}
                  </button>
                </div>
                <nav className="flex flex-col gap-1">
                  {group.items.map((item) => (
                    <a
                      key={item.label}
                      href={item.href}
                      {...externalProps(item.external)}
                      onClick={closeMobile}
                      className="flex w-full items-center justify-between rounded-lg px-3 py-3.5 text-2xl font-medium text-[#222] transition-colors hover:bg-[#e8e6e0] hover:text-[#111]"
                    >
                      {item.label}
                    </a>
                  ))}
                </nav>
              </div>
            ) : null
          )}
        </div>
      </div>
    </>
  );
};

export default GlobalNav;
