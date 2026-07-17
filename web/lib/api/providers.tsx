// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "next-themes";
import { useEffect, useState, type ReactNode } from "react";
import { captureInternalKeyFromUrl, stripInternalKeyFromUrl } from "@/lib/api/internalKey";

export function ApiProviders({ children }: { children: ReactNode }) {
  // Store ?internal=<key> before the first query fires (idempotent); the URL
  // cleanup must wait until after hydration or the router restores the param.
  if (typeof window !== "undefined") captureInternalKeyFromUrl();
  useEffect(() => stripInternalKeyFromUrl(), []);
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60_000,
            refetchOnWindowFocus: false,
            refetchInterval: 5 * 60_000,
            retry: 2,
          },
        },
      })
  );
  return (
    <ThemeProvider attribute="data-theme" defaultTheme="system" enableSystem>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </ThemeProvider>
  );
}
