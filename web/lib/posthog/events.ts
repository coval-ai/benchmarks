// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

export const POSTHOG_EVENTS = {
  dashboardFacetChanged: "dashboard_facet_changed",
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
  dashboardTimeWindowChanged: "dashboard_time_window_changed",
  dashboardChartShared: "dashboard_chart_shared",
  dashboardWerDatasetChanged: "dashboard_wer_dataset_changed",
  dashboardWerBarViewChanged: "dashboard_wer_bar_view_changed",
  s2sSamplePlayRequested: "s2s_sample_play_requested",
  s2sSamplePlaybackEnded: "s2s_sample_playback_ended",
  s2sSampleTickChanged: "s2s_sample_tick_changed",
  s2sSampleSeeked: "s2s_sample_seeked",
  headerCovalLinkClicked: "header_coval_link_clicked",
  pricingColumnSorted: "pricing_column_sorted",
  pricingTooltipOpened: "pricing_tooltip_opened",
  priceQualityToggleChanged: "price_quality_toggle_changed",
  priceHistoryViewed: "price_history_viewed"
} as const;

export type PostHogSurface =
  | "tts_dashboard"
  | "stt_dashboard"
  | "s2s_dashboard"
  | "playground"
  | "overview";
export type PostHogMode = "tts" | "stt" | "s2s";
export type DashboardChartId =
  | "timeline"
  | "scatter"
  | "wer_bar"
  | "instruction_bar"
  | "wer_radar"
  | "box_plot"
  | "heatmap"
  | "performance_delta"
  | "price_history";
export type PlaygroundRunTrigger = "button" | "keyboard";
export type PlaygroundModeSwitchTrigger = "tab" | "keyboard";
// The quality bar plots WER on STT/TTS and instruction adherence on S2S; the
// dashboard_wer_bar_clicked event name predates S2S, so `metric` says which
// was plotted. Only "wer" can fire today — S2S instruction bars aren't
// clickable — but the tag keeps the event honest if they ever become so.
export type QualityBarMetric = "wer" | "instruction";
// Sample panes never auto-advance (their <audio> stops on end), so a play can
// only start from the pane's Play button or a timeline-tooltip click.
export type S2SPlayTrigger = "button" | "timeline";
export type S2SSeekMethod = "slider" | "turn";
export type S2STickDirection = "older" | "newer" | "latest";
