// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import Image from "next/image";

// Dark gradient footer mirrored from www.coval.ai. Colors are hardcoded
// light-on-dark (the cream design tokens would be dark-on-dark here).
const COVAL = "https://www.coval.ai";

type FooterLink = { label: string; href: string; external?: boolean };
type FooterColumn = { heading: string; links: FooterLink[] };

const COLUMNS: FooterColumn[] = [
  {
    heading: "Product",
    links: [
      { label: "Overview", href: `${COVAL}/products` },
      { label: "Simulate", href: `${COVAL}/products/simulation` },
      { label: "Observe", href: `${COVAL}/products/observability` },
      { label: "Review", href: `${COVAL}/products/human-review` }
    ]
  },
  {
    heading: "Industries",
    links: [
      { label: "Healthcare", href: `${COVAL}/industries/healthcare` },
      { label: "Financial Services", href: `${COVAL}/industries/financial-services` },
      { label: "Insurance", href: `${COVAL}/industries/insurance` }
    ]
  },
  {
    heading: "Company",
    links: [
      { label: "About Us", href: `${COVAL}/about-us` },
      { label: "Blog", href: `${COVAL}/blog` },
      { label: "Partners", href: `${COVAL}/partners` },
      { label: "Careers", href: `${COVAL}/careers` }
    ]
  },
  {
    heading: "Resources",
    links: [
      { label: "Documentation", href: "https://docs.coval.dev", external: true },
      {
        label: "API",
        href: "https://docs.coval.dev/api-reference/v1/introduction",
        external: true
      },
      { label: "Trust Center", href: "https://trust.oneleet.com/coval", external: true }
    ]
  },
  {
    heading: "Community",
    links: [
      { label: "X", href: "https://x.com/covaldev", external: true },
      { label: "LinkedIn", href: "https://www.linkedin.com/company/covaldev/", external: true },
      { label: "GitHub", href: "https://github.com/coval-ai/benchmarks", external: true },
      { label: "Email", href: "mailto:brooke@coval.dev" }
    ]
  }
];

const externalProps = (external?: boolean) =>
  external ? { target: "_blank" as const, rel: "noopener noreferrer" } : {};

const DashboardFooter: React.FC = () => {
  return (
    <footer
      className="relative w-full overflow-hidden text-[#999]"
      style={{
        background:
          "linear-gradient(to bottom, #0f0c0a 0% 55%, #0c0908 68%, #080604 80%, #040302, #000)"
      }}
    >
      <div aria-hidden className="footer-noise" />
      <div className="relative z-10 mx-auto max-w-[1400px] px-6 py-16 md:px-8">
        {/* Link columns */}
        <div className="grid grid-cols-2 gap-8 sm:grid-cols-3 lg:grid-cols-5">
          {COLUMNS.map((column) => (
            <div key={column.heading}>
              <h3 className="mb-4 font-mono text-xs uppercase tracking-wider text-white/50">
                {column.heading}
              </h3>
              <ul className="space-y-3">
                {column.links.map((link) => (
                  <li key={link.label}>
                    <a
                      href={link.href}
                      {...externalProps(link.external)}
                      className="text-sm text-white/60 transition-colors hover:text-white"
                    >
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom bar */}
        <div className="mt-16 flex flex-col items-start gap-4 border-t border-white/10 pt-8 sm:flex-row sm:items-center sm:justify-between">
          <a
            href={`${COVAL}/`}
            target="_blank"
            rel="noopener noreferrer"
            className="transition-opacity hover:opacity-80"
            aria-label="Coval home"
          >
            <Image
              src="/coval-logo.svg"
              alt="Coval"
              width={90}
              height={20}
              className="h-5 w-auto invert"
            />
          </a>

          <p className="font-mono text-xs text-white/40">&copy; 2026 Coval, Inc.</p>

          <div className="flex items-center gap-6">
            <a
              href={`${COVAL}/privacy-policy`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-white/50 transition-colors hover:text-white"
            >
              Privacy Policy
            </a>
            <a
              href={`${COVAL}/terms-of-service`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-white/50 transition-colors hover:text-white"
            >
              Terms of Service
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default DashboardFooter;
