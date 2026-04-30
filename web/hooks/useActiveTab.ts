"use client";

import { usePathname } from "next/navigation";

export function useActiveTab(): "tts" | "stt" {
  const pathname = usePathname();
  return pathname === "/stt" ? "stt" : "tts";
}
