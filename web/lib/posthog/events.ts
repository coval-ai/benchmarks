export const POSTHOG_EVENTS = {
  ttsPageVisited: "tts_page_visited",
  sttPageVisited: "stt_page_visited",
  ttsTimelineTooltipOpened: "tts_timeline_tooltip_opened",
  sttTimelineTooltipOpened: "stt_timeline_tooltip_opened",
  playgroundTtsVisited: "playground_tts_visited",
  playgroundSttVisited: "playground_stt_visited",
  playgroundTtsBenchmarkPressed: "playground_tts_benchmark_pressed",
  playgroundSttRecordPressed: "playground_stt_record_pressed"
} as const;

export type PlaygroundModeChangeTrigger = "click" | "keyboard";
export type PlaygroundTtsRunTrigger = "button" | "keyboard";
export type DashboardSelectionTrigger = "sidebar" | "mobile_sheet";
