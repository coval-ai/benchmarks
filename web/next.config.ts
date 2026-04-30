import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // FastAPI lives at NEXT_PUBLIC_API_URL; client fetches go straight there (CORS allowed by FastAPI).
  // No rewrites — keep it simple. If we later need same-origin, add a rewrite block.
};

export default nextConfig;
