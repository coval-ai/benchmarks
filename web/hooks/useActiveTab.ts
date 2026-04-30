// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { usePathname } from "next/navigation";

export function useActiveTab(): "tts" | "stt" {
  const pathname = usePathname();
  return pathname === "/stt" ? "stt" : "tts";
}
