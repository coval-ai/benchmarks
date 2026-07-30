// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "next-themes";
import { useEffect, useState, type ReactNode } from "react";
import { applyTokensFromUrl } from "@/lib/api/accessTokens";

export function ApiProviders({ children }: { children: ReactNode }) {
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
  // Adopting ?internal=<key> / ?ea=<token>, dropping a previous caller's cached rows,
  // and cleaning the URL all happen here: one committed effect, so the identity change
  // cannot be lost to a replayed render, and the router cannot restore the param.
  useEffect(() => applyTokensFromUrl(() => client.clear()), [client]);
  return (
    <ThemeProvider attribute="data-theme" defaultTheme="system" enableSystem>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </ThemeProvider>
  );
}
