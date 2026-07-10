// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { usePathname } from "next/navigation";

export function useActiveTab(): "tts" | "stt" | "s2s" {
  const pathname = usePathname();
  if (pathname === "/stt") return "stt";
  if (pathname === "/s2s") return "s2s";
  return "tts";
}
