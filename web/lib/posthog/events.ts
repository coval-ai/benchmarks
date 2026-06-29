// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

export const POSTHOG_EVENTS = {
  playgroundModelSelectionChanged: "playground_model_selection_changed",
  playgroundTtsBenchmarkPressed: "playground_tts_benchmark_pressed",
  playgroundSttRecordPressed: "playground_stt_record_pressed",
  playgroundBenchmarkCompleted: "playground_benchmark_completed",
  dashboardChartHovered: "dashboard_chart_hovered",
  dashboardScrollDepth: "dashboard_scroll_depth",
  playgroundTtsTypingStarted: "playground_tts_typing_started",
  dashboardHeatmapSorted: "dashboard_heatmap_sorted",
  dashboardWerBarClicked: "dashboard_wer_bar_clicked",
  dashboardChartPanned: "dashboard_chart_panned",
  sttTranscriptBrowsed: "stt_transcript_browsed",
  playgroundExamplePromptUsed: "playground_example_prompt_used",
  playgroundModeSwitched: "playground_mode_switched",
  playgroundResultPlayed: "playground_result_played",
  dashboardTimeWindowChanged: "dashboard_time_window_changed"
} as const;

export type PostHogSurface =
  | "tts_dashboard"
  | "stt_dashboard"
  | "playground"
  | "overview";
export type PostHogMode = "tts" | "stt";
export type DashboardChartId =
  | "timeline"
  | "scatter"
  | "wer_bar"
  | "box_plot"
  | "heatmap"
  | "performance_delta";
export type PlaygroundRunTrigger = "button" | "keyboard";
export type PlaygroundModeSwitchTrigger = "tab" | "keyboard";
export type TranscriptBrowseMethod = "arrow" | "scroll";
