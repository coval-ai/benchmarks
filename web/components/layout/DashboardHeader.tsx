// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import React from "react";
import { ThemeToggle } from "@/components/ui/ThemeToggle";


const DashboardHeader: React.FC = () => {
  const pathname = usePathname();
  const isTTS = pathname === "/tts";
  const isSTT = pathname === "/stt";

  return (
    <div className="fixed top-0 left-0 right-0 z-50 h-16 bg-surface-overlay backdrop-blur-xl border-b border-border-primary">
      <div className="flex items-center justify-between h-full px-6">
        {/* Logo - Different sizes for mobile/desktop */}
        <div className="flex items-center">
          <a
            href="https://coval.dev"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:opacity-80 transition-opacity"
          >
            <Image
              src="/Coval.webp"
              alt="Coval"
              width={120}
              height={32}
              priority
              style={{ width: "auto", height: "auto" }}
              className="h-8 w-auto hidden md:block dark:invert-0 invert"
            />
            <Image
              src="/Coval.webp"
              alt="Coval"
              width={90}
              height={24}
              priority
              style={{ width: "auto", height: "auto" }}
              className="h-6 w-auto md:hidden dark:invert-0 invert"
            />
          </a>
        </div>

        <div className="flex-1 flex justify-start px-4 md:px-20">
          <div className="relative flex space-x-4 md:space-x-8">
            {/* TTS Link */}
            <Link
              href="/tts"
              aria-label="Switch to Text-to-Speech view"
              className="relative px-2 md:px-3 py-2 text-text-primary font-light tracking-wide transition-all duration-300 ease-out hover:text-text-secondary text-xs md:text-sm"
            >
              <span className="md:hidden">TTS</span>
              <span className="hidden md:inline">Text-to-Speech</span>
              <div
                className={`absolute bottom-0 left-0 h-0.5 bg-text-primary transition-all duration-500 ease-out ${
                  isTTS ? "w-full opacity-100" : "w-0 opacity-0"
                }`}
                style={{ transformOrigin: "left center" }}
              />
            </Link>

            {/* STT Link */}
            <Link
              href="/stt"
              aria-label="Switch to Speech-to-Text view"
              className="relative px-2 md:px-3 py-2 text-text-primary font-light tracking-wide transition-all duration-300 ease-out hover:text-text-secondary text-xs md:text-sm"
            >
              <span className="md:hidden">STT</span>
              <span className="hidden md:inline">Speech-to-Text</span>
              <div
                className={`absolute bottom-0 left-0 h-0.5 bg-text-primary transition-all duration-500 ease-out ${
                  isSTT ? "w-full opacity-100" : "w-0 opacity-0"
                }`}
                style={{ transformOrigin: "left center" }}
              />
            </Link>
          </div>
        </div>

        {/* Right Side - Theme Toggle */}
        <div className="flex items-center gap-3">
          <ThemeToggle />
        </div>
      </div>
    </div>
  );
};

export default DashboardHeader;
