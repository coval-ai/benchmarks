// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Dashboard pages fetch benchmark data directly from NEXT_PUBLIC_API_URL (FastAPI).
  // No rewrites needed — FastAPI allows CORS from the browser.
  // Playground routes go through Next.js route handlers (web/app/api/**) so provider
  // API keys stay server-side and never reach the browser.
};

export default nextConfig;
