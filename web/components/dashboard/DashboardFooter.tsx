// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import Image from "next/image";

// Dark gradient footer mirrored from www.coval.ai (`.footer`). Markup, layout
// and colors replicate the live site's `.footer-*` classes exactly. Colors are
// hardcoded light-on-dark (the cream design tokens would be dark-on-dark here).
const COVAL = "https://www.coval.ai";

type FooterLink = { label: string; href: string; external?: boolean };
type FooterSection = { heading: string; links: FooterLink[] };

// Four grid columns. Resources and Community share the 4th column, stacked —
// matching coval.ai, where the Community section sits below Resources.
const COLUMNS: FooterSection[][] = [
  [
    {
      heading: "Product",
      links: [
        { label: "Overview", href: `${COVAL}/products` },
        { label: "Simulate", href: `${COVAL}/products/simulation` },
        { label: "Observe", href: `${COVAL}/products/observability` },
        { label: "Review", href: `${COVAL}/products/human-review` }
      ]
    }
  ],
  [
    {
      heading: "Industries",
      links: [
        { label: "Healthcare", href: `${COVAL}/industries/healthcare` },
        { label: "Financial Services", href: `${COVAL}/industries/financial-services` },
        { label: "Insurance", href: `${COVAL}/industries/insurance` }
      ]
    }
  ],
  [
    {
      heading: "Company",
      links: [
        { label: "About Us", href: `${COVAL}/about-us` },
        { label: "Blog", href: `${COVAL}/blog` },
        { label: "Partners", href: `${COVAL}/partners` },
        { label: "Careers", href: `${COVAL}/careers` }
      ]
    }
  ],
  [
    {
      heading: "Resources",
      links: [
        { label: "Documentation", href: "https://docs.coval.ai", external: true },
        {
          label: "API",
          href: "https://docs.coval.ai/api-reference/v1/introduction",
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
        { label: "Email", href: "mailto:brooke@coval.dev" }
      ]
    }
  ]
];

const externalProps = (external?: boolean) =>
  external ? { target: "_blank" as const, rel: "noopener noreferrer" } : {};

const DashboardFooter: React.FC = () => {
  return (
    <footer
      className="relative flex min-h-[calc(50vh-60px)] w-full flex-col justify-center overflow-hidden text-[#999]"
      style={{
        background:
          "linear-gradient(to bottom, #0f0c0a 0% 55%, #0c0908 68%, #080604 80%, #040302, #000)"
      }}
    >
      <div aria-hidden className="footer-noise" />

      {/* .footer-inner */}
      <div className="relative z-[1] mx-auto w-full max-w-[1400px] px-6 py-16 sm:px-10">
        {/* .footer-columns */}
        <nav
          aria-label="Footer navigation"
          className="grid grid-cols-2 gap-8 pb-12 md:grid-cols-4"
        >
          {COLUMNS.map((sections, columnIndex) => (
            <div key={columnIndex}>
              {sections.map((section, sectionIndex) => (
                <div key={section.heading} className={sectionIndex > 0 ? "mt-8" : undefined}>
                  <p className="mb-4 font-mono text-base font-medium uppercase tracking-[0.08em] text-[#555]">
                    {section.heading}
                  </p>
                  <ul className="flex flex-col gap-2">
                    {section.links.map((link) => (
                      <li key={link.label}>
                        <a
                          href={link.href}
                          {...externalProps(link.external)}
                          className="text-base text-[#ccc] transition-colors duration-150 hover:text-[#f2f1ee]"
                        >
                          {link.label}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          ))}
        </nav>

        {/* .footer-bottom */}
        <div className="flex flex-col items-start gap-3 pt-8">
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
              className="block h-5 w-auto invert"
            />
          </a>

          {/* .footer-legal — copyright with Privacy/Terms inline to its right */}
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <p className="font-mono text-xs text-[#555]">&copy; 2026 Coval, Inc.</p>
            <a
              href={`${COVAL}/privacy-policy`}
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-xs text-[#555] no-underline"
            >
              Privacy Policy
            </a>
            <a
              href={`${COVAL}/terms-of-service`}
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-xs text-[#555] no-underline"
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
