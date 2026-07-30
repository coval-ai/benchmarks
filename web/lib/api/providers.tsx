// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "next-themes";
import { useEffect, useState, type ReactNode } from "react";
import { captureTokensFromUrl, stripTokensFromUrl } from "@/lib/api/accessTokens";

export function ApiProviders({ children }: { children: ReactNode }) {
  // Store ?internal=<key> / ?ea=<token> before the first query fires (idempotent);
  // the URL cleanup must wait until after hydration or the router restores the param.
  const tokensChanged = typeof window === "undefined" ? false : captureTokensFromUrl();
  useEffect(() => stripTokensFromUrl(), []);
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
  // A different token is a different view of the embargo, so whatever is cached was
  // fetched for someone else. Dropped rather than revalidated, and never in render.
  useEffect(() => {
    if (tokensChanged) client.clear();
  }, [client, tokensChanged]);
  return (
    <ThemeProvider attribute="data-theme" defaultTheme="system" enableSystem>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </ThemeProvider>
  );
}
