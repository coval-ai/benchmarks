// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useState, useEffect } from "react";

// PNG exports stage the card at desktop width, but charts gate their mobile
// layouts (scroll slots, tick fonts, touch targets) on the viewport, not the
// card. Exports flip this flag so every chart renders its desktop layout and
// a phone export is pixel-identical to a desktop one.
const EXPORT_STAGE_EVENT = "chart-export-stage";
let exportStaged = false;

export const setExportStaged = (staged: boolean) => {
  exportStaged = staged;
  window.dispatchEvent(new Event(EXPORT_STAGE_EVENT));
};

export const useMobileDetection = () => {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768 && !exportStaged);
    };

    checkMobile();
    window.addEventListener("resize", checkMobile);
    window.addEventListener(EXPORT_STAGE_EVENT, checkMobile);
    return () => {
      window.removeEventListener("resize", checkMobile);
      window.removeEventListener(EXPORT_STAGE_EVENT, checkMobile);
    };
  }, []);

  return isMobile;
};
