// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import Card from "@/components/shared/Card";
import { useDashboard } from "@/contexts/DashboardContext";
import { ApiError } from "@/lib/api/client";
import {
  useS2SSampleAudioUrls,
  useS2SSampleIdsQuery,
  useS2SSampleQuery,
} from "@/lib/api/queries";
import { spokenTurns, visibleRecordings } from "@/lib/audioSamples/s2sFeed";
import { capturePostHogEvent } from "@/lib/posthog/client";
import {
  POSTHOG_EVENTS,
  type S2SPlayTrigger,
  type S2STickDirection,
} from "@/lib/posthog/events";
import { usePlaybackCoordinator } from "@/hooks/useSequencedPlayback";
import {
  SampleOutputs,
  type SampleListenInfo,
  type SampleOutputItem,
  type SampleSeekInfo,
} from "./SampleOutputs";

// Which failure case fired, for the console — the API's status when we have one,
// otherwise whatever the thrown value says.
function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    return `status ${err.status} ${err.statusText}`;
  }
  return err instanceof Error ? err.message : String(err);
}

// Day of a pinned tick, for the "showing an older day" control only.
function pinnedDayLabel(tick: string): string {
  const d = new Date(tick);
  if (Number.isNaN(d.getTime())) return tick;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
}

export function SamplesCard() {
  const {
    modelsByProvider,
    normalizeProviderName,
    s2sPlayRequest,
    selectS2SSample,
    getCurrentTimeWindow,
  } = useDashboard();
  const coordinator = usePlaybackCoordinator();
  const visibleModels = useMemo(
    () => new Set(Object.values(modelsByProvider).flat()),
    [modelsByProvider]
  );

  const indexQuery = useS2SSampleIdsQuery();
  const [windowStart, windowEnd] = getCurrentTimeWindow();
  const sampleTicks = useMemo(
    () =>
      (indexQuery.data ?? [])
        .filter((tick) => {
          const timestamp = new Date(tick).getTime();
          return timestamp >= windowStart && timestamp <= windowEnd;
        })
        .reverse(),
    [indexQuery.data, windowStart, windowEnd]
  );
  const latestTick = sampleTicks.at(-1) ?? null;
  // A timeline-tooltip click pins the card to that bucket; otherwise show latest.
  const effectiveTick = s2sPlayRequest?.tick ?? latestTick;
  const effectiveTime = effectiveTick ? new Date(effectiveTick).getTime() : NaN;
  const olderTick = [...sampleTicks]
    .reverse()
    .find((tick) => new Date(tick).getTime() < effectiveTime);
  const newerTick = sampleTicks.find(
    (tick) => new Date(tick).getTime() > effectiveTime
  );
  const manifestQuery = useS2SSampleQuery(effectiveTick);
  const manifest = manifestQuery.data ?? null;
  const displayedTick =
    manifestQuery.isPlaceholderData && manifest?.sample_id
      ? manifest.sample_id
      : effectiveTick;

  // Keep the failure cause internal (dev console only); public visitors just see
  // the generic "temporarily unavailable" copy below.
  useEffect(() => {
    if (process.env.NODE_ENV === "production") return;
    const err = manifestQuery.error ?? indexQuery.error;
    if (err) console.error(`[s2s-samples] fetch failed: ${describeError(err)}`, err);
  }, [manifestQuery.error, indexQuery.error]);

  // Every tick with recordings is playable, including the pre-multi-turn ones
  // that carry no transcript — those render as panes with a play button and no
  // turn list, so a timeline click always reaches its audio.
  const recordings = useMemo(
    () => (manifest ? visibleRecordings(manifest, visibleModels) : []),
    [manifest, visibleModels]
  );
  // Signed URLs are fetched as soon as the sample lands, not on the play click: a
  // browser will not start playback that begins after an await.
  const audioUrls = useS2SSampleAudioUrls(recordings);
  const items = useMemo<SampleOutputItem[]>(
    () =>
      recordings.map((recording, position) => ({
        provider: recording.provider,
        model: recording.model,
        url: audioUrls.urls[position] ?? "",
        audioPath: recording.audio_path,
        turns: spokenTurns(recording.turns),
      })),
    [recordings, audioUrls.urls]
  );

  const fetchError = manifestQuery.isError || indexQuery.isError || audioUrls.isError;

  // A timeline row can name a provider this bucket never recorded — an older
  // tick from before that model was benchmarked, or one absent from the page's
  // catalogue. Autoplay then finds no matching pane and stays quiet, so the
  // click reads as broken unless we say what happened.
  const requestedProvider = s2sPlayRequest?.provider;
  const requestedItem = requestedProvider
    ? items.find(
        (item) =>
          normalizeProviderName(item.provider) ===
          normalizeProviderName(requestedProvider)
      )
    : undefined;
  const requestedProviderMissing =
    requestedProvider !== undefined &&
    manifest !== null &&
    !manifestQuery.isPlaceholderData &&
    requestedItem === undefined;

  // Bucket/model of the last activation: the listen-ended flush can arrive
  // after a tick swap has already replaced the manifest, and must attribute to
  // what actually played.
  const playContextRef = useRef<{ bucket?: string; model?: string }>({});

  const handlePlay = useCallback(
    (provider: string, trigger: S2SPlayTrigger) => {
      const position = items.findIndex((item) => item.provider === provider);
      const item = position >= 0 ? items[position] : undefined;
      playContextRef.current = { bucket: manifest?.sample_id, model: item?.model };
      capturePostHogEvent(POSTHOG_EVENTS.s2sSamplePlayRequested, {
        surface: "s2s_dashboard",
        mode: "s2s",
        provider,
        model_id: item?.model,
        position: position >= 0 ? position : undefined,
        has_transcript: (item?.turns?.length ?? 0) > 0,
        trigger,
        bucket_at: manifest?.sample_id,
      });
    },
    [items, manifest?.sample_id]
  );

  const handlePlaybackEnded = useCallback(
    (provider: string, listen: SampleListenInfo) => {
      capturePostHogEvent(POSTHOG_EVENTS.s2sSamplePlaybackEnded, {
        surface: "s2s_dashboard",
        mode: "s2s",
        provider,
        model_id: playContextRef.current.model,
        bucket_at: playContextRef.current.bucket,
        trigger: listen.trigger,
        listen_pct: listen.listenPct,
        duration_seconds: listen.durationSeconds,
        completed: listen.completed,
      });
    },
    []
  );

  const handleSeeked = useCallback(
    (provider: string, seek: SampleSeekInfo) => {
      const item = items.find((i) => i.provider === provider);
      capturePostHogEvent(POSTHOG_EVENTS.s2sSampleSeeked, {
        surface: "s2s_dashboard",
        mode: "s2s",
        provider,
        model_id: item?.model,
        bucket_at: manifest?.sample_id,
        method: seek.method,
        to_seconds: Math.round(seek.toSeconds),
        turn_index: seek.turnIndex,
        turn_role: seek.turnRole,
      });
    },
    [items, manifest?.sample_id]
  );

  const handleTickSelect = useCallback(
    (direction: S2STickDirection, toTick: string) => {
      capturePostHogEvent(POSTHOG_EVENTS.s2sSampleTickChanged, {
        surface: "s2s_dashboard",
        mode: "s2s",
        direction,
        from_bucket: displayedTick,
        to_bucket: toTick,
      });
      selectS2SSample(toTick);
    },
    [displayedTick, selectS2SSample]
  );

  // Hold the previous panes until every signed URL is in: a pane whose URL is
  // still in flight would hand the shared audio element an empty src.
  const pendingUrls = recordings.length > 0 && audioUrls.isPending;
  const lastReadyItems = useRef<SampleOutputItem[]>([]);
  if (!pendingUrls) lastReadyItems.current = items;
  const displayItems = pendingUrls ? lastReadyItems.current : items;
  const transitioning = manifestQuery.isPlaceholderData || pendingUrls;

  const loading =
    indexQuery.isLoading ||
    (effectiveTick != null && manifestQuery.isLoading) ||
    (pendingUrls && displayItems.length === 0);

  useEffect(() => {
    if (s2sPlayRequest?.source === "timeline") {
      document.getElementById("s2s-samples")?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [s2sPlayRequest]);

  return (
    <Card id="s2s-samples" className="text-left min-w-0 h-full flex flex-col" padding="p-5 lg:p-8">
      <div className="mb-4">
        <h2 className="text-xl font-medium text-text-primary">
          Conversation samples
        </h2>
        <p className="mt-1 text-sm font-light text-text-tertiary">
          Latency tells you when a model speaks; these recordings let you hear
          how it handles the same scripted caller.
          <span className="mt-1 block">
            Play a model to follow its transcript. Tap a turn to jump the audio
            there.
          </span>
        </p>
      </div>

      {effectiveTick && sampleTicks.length > 1 ? (
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <div className="flex w-fit items-center gap-2 rounded-lg border border-border-secondary bg-surface-secondary/40 p-1">
            <button
              type="button"
              aria-label="Show older conversation sample"
              disabled={!olderTick}
              onClick={() => olderTick && handleTickSelect("older", olderTick)}
              className="flex size-11 items-center justify-center rounded-md border border-border-secondary text-text-secondary transition-colors hover:bg-surface-tertiary hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-text-tertiary/40 disabled:cursor-default disabled:text-text-tertiary disabled:opacity-30 lg:size-8"
            >
              <ChevronLeft className="size-4" />
            </button>
            <span className="min-w-28 text-center font-mono text-xs text-text-secondary">
              {pinnedDayLabel(displayedTick ?? effectiveTick)}
              {displayedTick === latestTick ? " · latest" : ""}
            </span>
            <button
              type="button"
              aria-label="Show newer conversation sample"
              disabled={!newerTick}
              onClick={() => newerTick && handleTickSelect("newer", newerTick)}
              className="flex size-11 items-center justify-center rounded-md border border-border-secondary text-text-secondary transition-colors hover:bg-surface-tertiary hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-text-tertiary/40 disabled:cursor-default disabled:text-text-tertiary disabled:opacity-30 lg:size-8"
            >
              <ChevronRight className="size-4" />
            </button>
          </div>
          {effectiveTick !== latestTick ? (
            <button
              type="button"
              onClick={() => latestTick && handleTickSelect("latest", latestTick)}
              className="min-h-11 rounded-full border border-border-primary px-3 text-xs text-text-secondary transition-colors hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-text-tertiary/40 lg:min-h-8"
            >
              Return to latest
            </button>
          ) : null}
        </div>
      ) : null}

      {loading ? (
        <div className="h-40 animate-pulse rounded-lg bg-surface-secondary" />
      ) : fetchError ? (
        <p className="py-8 text-center text-sm text-text-tertiary">
          Samples are temporarily unavailable.
        </p>
      ) : displayItems.length === 0 ? (
        <p className="py-8 text-center text-sm text-text-tertiary">
          {!manifest && s2sPlayRequest
            ? "No sample recorded for this point."
            : "Samples appear here after the next benchmark run."}
        </p>
      ) : (
        <div
          aria-busy={transitioning}
          className={`flex flex-1 flex-col gap-4 transition-opacity ${
            transitioning ? "pointer-events-none opacity-50" : ""
          }`}
        >
          {requestedProviderMissing && requestedProvider ? (
            <p className="text-[11px] text-text-tertiary">
              No recording for {normalizeProviderName(requestedProvider)} at this point.
            </p>
          ) : null}
          <SampleOutputs
            items={displayItems}
            normalizeProvider={normalizeProviderName}
            onPlay={handlePlay}
            onSeeked={handleSeeked}
            onPlaybackEnded={handlePlaybackEnded}
            playRequest={
              requestedItem && !transitioning
                ? { provider: requestedItem.provider, nonce: s2sPlayRequest!.nonce }
                : null
            }
            coordinator={coordinator}
          />
        </div>
      )}
    </Card>
  );
}
