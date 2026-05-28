"use client";

import "./playground-ui.css";
import DashboardFooter from "@/components/dashboard/DashboardFooter";
import DashboardHeader from "@/components/layout/DashboardHeader";
import { PlaygroundShell } from "./components/PlaygroundShell";
import { PlaygroundDraftChrome } from "./components/playground-draft/PlaygroundDraftChrome";
import { TTSPlaygroundPanel } from "./components/playground-draft/TTSPlaygroundPanel";
import { STTPlaygroundPanel } from "./components/playground-draft/STTPlaygroundPanel";
import { usePlaygroundMode } from "./hooks/usePlaygroundMode";
import { getEnabledSttModels, getEnabledTtsModels } from "@/lib/playground/providers";
import { useState } from "react";

export function PlaygroundPageClient() {
  const { mode, setMode } = usePlaygroundMode();
  const ttsList = getEnabledTtsModels();
  const sttList = getEnabledSttModels();
  const [benchmarkOverlayOpen, setBenchmarkOverlayOpen] = useState(false);

  return (
    <PlaygroundShell header={<DashboardHeader />}>
      <div className="px-4 pb-24 pt-24 md:px-6 md:pt-28">
        <PlaygroundDraftChrome
          mode={mode}
          onModeChange={setMode}
          modeTabsLocked={benchmarkOverlayOpen}
        >
          <div key={mode} className="playground-panel-switch">
            {mode === "tts" ? (
              <TTSPlaygroundPanel models={ttsList} onBenchmarkOverlayChange={setBenchmarkOverlayOpen} />
            ) : (
              <STTPlaygroundPanel models={sttList} onBenchmarkOverlayChange={setBenchmarkOverlayOpen} />
            )}
          </div>
        </PlaygroundDraftChrome>
        <DashboardFooter />
      </div>
    </PlaygroundShell>
  );
}
