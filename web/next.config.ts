// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Allow loading dev resources when the dev server is opened from another device
  // on the LAN (e.g. testing the mobile nav on a phone via the network IP).
  // Without this, Next 16 blocks /_next/* dev resources cross-origin, so the page
  // renders via SSR but never hydrates and interactive controls appear dead.
  allowedDevOrigins: ["192.168.86.185"],
  // Dashboard pages fetch benchmark data directly from NEXT_PUBLIC_API_URL (FastAPI).
  // No rewrites needed — FastAPI allows CORS from the browser.
  // Playground routes go through Next.js route handlers (web/app/api/**) so provider
  // API keys stay server-side and never reach the browser.
};

export default nextConfig;
