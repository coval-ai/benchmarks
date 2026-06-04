// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import Link from "next/link";
import React from "react";

// Dark gradient footer mirrored from www.coval.ai (`.footer`). Layout and
// colors replicate the live site's `.footer-*` classes. Colors are hardcoded
// light-on-dark (the cream design tokens would be dark-on-dark here).

// Site navigation, mirrored from the top nav sections plus the source repo.
const FOOTER_LINKS = [
  { href: "/overview", label: "Overview" },
  { href: "/tts", label: "Text-to-Speech" },
  { href: "/stt", label: "Speech-to-Text" },
  { href: "/playground", label: "Playground" }
];

const GITHUB_URL = "https://github.com/coval-ai/benchmarks";

const DashboardFooter: React.FC = () => {
  return (
    <footer
      className="relative flex min-h-[calc(25vh-30px)] w-full flex-col overflow-hidden text-[#999]"
      style={{
        background:
          "linear-gradient(to bottom, #0f0c0a 0% 55%, #0c0908 68%, #080604 80%, #040302, #000)"
      }}
    >
      <div aria-hidden className="footer-noise" />

      {/* .footer-inner */}
      <div className="relative z-[1] flex w-full flex-1 flex-col justify-center px-4 py-8 md:px-6">
        {/* .footer-bottom */}
        <div className="flex flex-col items-start gap-3">
          <span className="font-sans text-lg font-medium tracking-wide text-[#f2f1ee]">
            Voice Model Benchmarks
          </span>

          <nav
            aria-label="Site"
            className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm"
          >
            {FOOTER_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-[#ccc] transition-colors duration-150 hover:text-[#f2f1ee]"
              >
                {link.label}
              </Link>
            ))}
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-[#ccc] transition-colors duration-150 hover:text-[#f2f1ee]"
            >
              <svg
                viewBox="0 0 16 16"
                width={16}
                height={16}
                aria-hidden
                fill="currentColor"
              >
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
              </svg>
              GitHub
            </a>
          </nav>

          <p className="text-xs text-[#555]">&copy; 2026 Coval, Inc.</p>
        </div>
      </div>
    </footer>
  );
};

export default DashboardFooter;
