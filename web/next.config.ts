// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import type { NextConfig } from "next";

// Hostnames allowed to load dev resources when the dev server is opened from
// another device on the LAN (e.g. testing the mobile nav on a phone via the
// network IP). Set NEXT_ALLOWED_DEV_ORIGINS to a comma-separated list of
// hostnames (no scheme or port), e.g. "192.168.86.185,my-laptop.local".
const allowedDevOrigins = (process.env.NEXT_ALLOWED_DEV_ORIGINS ?? "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

const proxyApiTarget = process.env.BENCH_PROXY_API?.replace(/\/$/, "");
// Optional split-target for /v1/pricing only: lets a dev preview serve live
// production benchmark data while the pricing store (not yet deployed there)
// comes from a local API. Rewrites are ordered, so the specific rule wins.
const proxyPricingTarget = process.env.BENCH_PROXY_PRICING_API?.replace(/\/$/, "");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Without this, Next 16 blocks /_next/* dev resources cross-origin, so the page
  // renders via SSR but never hydrates and interactive controls appear dead.
  allowedDevOrigins,
  // Dashboard pages fetch benchmark data directly from NEXT_PUBLIC_API_URL (FastAPI).
  // No rewrites needed — FastAPI allows CORS from the browser.
  // Playground routes go through Next.js route handlers (web/app/api/**) so provider
  // API keys stay server-side and never reach the browser.
  ...(proxyApiTarget
    ? {
        async rewrites() {
          return [
            ...(proxyPricingTarget
              ? [
                  {
                    source: "/proxy-api/v1/pricing",
                    destination: `${proxyPricingTarget}/v1/pricing`,
                  },
                ]
              : []),
            { source: "/proxy-api/:path*", destination: `${proxyApiTarget}/:path*` },
          ];
        },
      }
    : {}),
  // The arena pages moved under /overview. These fire before middleware and keep the
  // query string, so shared links — including the labeler unlock link's ?access= token —
  // land on the new paths intact. Temporary on purpose: browsers cache 308s forever.
  async redirects() {
    return [
      { source: "/arena", destination: "/overview/arena", permanent: false },
      { source: "/arena/:path*", destination: "/overview/arena/:path*", permanent: false },
    ];
  },
};

export default nextConfig;
