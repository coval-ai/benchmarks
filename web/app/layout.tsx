// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import type { Metadata } from "next";
import localFont from "next/font/local";
import { Geist_Mono } from "next/font/google";
import { ApiProviders } from "@/lib/api/providers";
import "./globals.css";

// PP Mori — coval.ai brand face, self-hosted.
const ppMori = localFont({
  variable: "--font-sans",
  display: "swap",
  src: [
    { path: "./fonts/PPMori-Regular.woff2", weight: "400", style: "normal" },
    { path: "./fonts/PPMori-Medium.woff2", weight: "500", style: "normal" }
  ]
});

// Geist Mono — buttons, labels, code (matches coval.ai).
const geistMono = Geist_Mono({
  variable: "--font-mono",
  subsets: ["latin"]
});

export const metadata: Metadata = {
  title: "Benchmarks by Coval",
  description: "Coval.dev"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${ppMori.variable} ${geistMono.variable}`}>
        <div
          aria-hidden
          className="noise-overlay pointer-events-none fixed inset-0 z-[1]"
        />
        <ApiProviders>{children}</ApiProviders>
      </body>
    </html>
  );
}
