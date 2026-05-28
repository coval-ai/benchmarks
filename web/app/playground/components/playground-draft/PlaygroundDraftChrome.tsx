"use client";

import type { ReactNode } from "react";
import { useEffect, useRef } from "react";
import type { PlaygroundMode } from "../../hooks/usePlaygroundMode";

type PlaygroundDraftChromeProps = {
  mode: PlaygroundMode;
  onModeChange: (mode: PlaygroundMode) => void;
  /** True while a benchmark / results modal is open — mode tabs and shortcuts are disabled. */
  modeTabsLocked?: boolean;
  children: ReactNode;
};

/** Main title stays **Playground** (Montserrat via `layout` body); mode is only reflected in TTS/STT tabs. */
export function PlaygroundDraftChrome({
  mode,
  onModeChange,
  modeTabsLocked = false,
  children
}: PlaygroundDraftChromeProps) {
  const tablistRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = tablistRef.current;
    if (!el) return;
    const onKey = (e: KeyboardEvent) => {
      if (modeTabsLocked) return;
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      if (e.key === "ArrowLeft") onModeChange("tts");
      else onModeChange("stt");
    };
    el.addEventListener("keydown", onKey);
    return () => el.removeEventListener("keydown", onKey);
  }, [modeTabsLocked, onModeChange]);

  return (
    <div className="mx-auto w-full min-w-0 max-w-[1080px]">
      <div className="relative px-4 pb-2 pt-8 md:px-8 md:pt-10">
        <div className="mb-8 flex flex-col items-center gap-5 text-center">
          <div>
            <h1 className="font-sans text-2xl font-bold text-text-primary md:text-4xl">Playground</h1>
          </div>
          <div
            ref={tablistRef}
            className={`flex justify-center transition-opacity duration-200 ${modeTabsLocked ? "pointer-events-none opacity-40" : ""}`}
            role="tablist"
            aria-label="Playground mode"
            aria-disabled={modeTabsLocked}
          >
            <div className="relative inline-flex w-full max-w-[220px] gap-1 rounded-full border border-border-primary bg-surface-toggle-inactive p-1 sm:max-w-[240px]">
              <span
                aria-hidden
                className="pointer-events-none absolute bottom-1 left-1 top-1 rounded-full bg-surface-toggle-active transition-transform duration-300 ease-[cubic-bezier(0.33,1,0.68,1)] motion-reduce:transition-none motion-reduce:duration-0"
                style={{
                  width: "calc((100% - 8px - 4px) / 2)",
                  transform:
                    mode === "stt" ? "translateX(calc(100% + 4px))" : "translateX(0)"
                }}
              />
              <button
                type="button"
                id="tab-playground-tts"
                role="tab"
                aria-selected={mode === "tts"}
                aria-controls="playground-panel-tts"
                tabIndex={mode === "tts" ? 0 : -1}
                disabled={modeTabsLocked}
                className={`relative z-10 flex min-h-9 min-w-[100px] flex-1 items-center justify-center rounded-full px-4 py-1.5 text-xs font-medium transition-[color,background-color,box-shadow] duration-300 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-text-tertiary/40 ${
                  mode === "tts"
                    ? "bg-surface-toggle-active text-text-on-toggle-active shadow-sm"
                    : "text-text-secondary hover:text-text-primary"
                }`}
                onClick={() => {
                  onModeChange("tts");
                }}
              >
                TTS
              </button>
              <button
                type="button"
                id="tab-playground-stt"
                role="tab"
                aria-selected={mode === "stt"}
                aria-controls="playground-panel-stt"
                tabIndex={mode === "stt" ? 0 : -1}
                disabled={modeTabsLocked}
                className={`relative z-10 flex min-h-9 min-w-[100px] flex-1 items-center justify-center rounded-full px-4 py-1.5 text-xs font-medium transition-[color,background-color,box-shadow] duration-300 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-text-tertiary/40 ${
                  mode === "stt"
                    ? "bg-surface-toggle-active text-text-on-toggle-active shadow-sm"
                    : "text-text-secondary hover:text-text-primary"
                }`}
                onClick={() => {
                  onModeChange("stt");
                }}
              >
                STT
              </button>
            </div>
          </div>
        </div>
      </div>
      <div className="px-4 pb-8 pt-0 md:px-8 md:pb-10">{children}</div>
    </div>
  );
}
