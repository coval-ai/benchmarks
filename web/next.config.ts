// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // FastAPI lives at NEXT_PUBLIC_API_URL; client fetches go straight there (CORS allowed by FastAPI).
  // No rewrites — keep it simple. If we later need same-origin, add a rewrite block.
  // Playground STT/TTS use same-origin route handlers (web/app/api/**) so provider keys stay server-side.
};

export default nextConfig;
