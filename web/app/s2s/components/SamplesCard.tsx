// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import Card from "@/components/shared/Card";
import { useDashboard } from "@/contexts/DashboardContext";
import { SampleFetchError } from "@/lib/audioSamples/createSampleFeed";
import { s2sSampleFeed, visibleRecordings } from "@/lib/audioSamples/s2sFeed";
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

// Which failure case fired, for the console — network/http/parse vs an
// unclassified error, with the status when we have one.
function describeError(err: unknown): string {
  if (err instanceof SampleFetchError) {
    return err.status != null ? `${err.kind} (status ${err.status})` : err.kind;
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

  const indexQuery = s2sSampleFeed.useIndexQuery();
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
  const manifestQuery = s2sSampleFeed.useManifestQuery(effectiveTick);
  const manifest = manifestQuery.data ?? null;
  const displayedTick =
    manifestQuery.isPlaceholderData && manifest?.bucket_at
      ? manifest.bucket_at
      : effectiveTick;

  const fetchError = manifestQuery.isError || indexQuery.isError;
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
  const items = useMemo<SampleOutputItem[]>(() => {
    if (!manifest) return [];
    return visibleRecordings(manifest, visibleModels).map((r) => ({
      provider: r.provider,
      model: r.model,
      url: s2sSampleFeed.objectUrl(r.object),
      turns: r.turns,
    }));
  }, [manifest, visibleModels]);

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
      playContextRef.current = { bucket: manifest?.bucket_at, model: item?.model };
      capturePostHogEvent(POSTHOG_EVENTS.s2sSamplePlayRequested, {
        surface: "s2s_dashboard",
        mode: "s2s",
        provider,
        model_id: item?.model,
        position: position >= 0 ? position : undefined,
        has_transcript: (item?.turns?.length ?? 0) > 0,
        trigger,
        bucket_at: manifest?.bucket_at,
      });
    },
    [items, manifest?.bucket_at]
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
        bucket_at: manifest?.bucket_at,
        method: seek.method,
        to_seconds: Math.round(seek.toSeconds),
        turn_index: seek.turnIndex,
        turn_role: seek.turnRole,
      });
    },
    [items, manifest?.bucket_at]
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

  const loading =
    indexQuery.isLoading || (effectiveTick != null && manifestQuery.isLoading);

  useEffect(() => {
    if (s2sPlayRequest?.source === "timeline") {
      document.getElementById("s2s-samples")?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [s2sPlayRequest]);

  return (
    <Card id="s2s-samples" className="text-left min-w-0 h-full flex flex-col" padding="p-5 lg:p-8">
      <div className="mb-4 flex items-baseline justify-between gap-2">
        <h2 className="text-xl font-medium text-text-primary">
          Conversation samples
        </h2>
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
      ) : items.length === 0 ? (
        <p className="py-8 text-center text-sm text-text-tertiary">
          {!manifest && s2sPlayRequest
            ? "No sample recorded for this point."
            : "Samples appear here after the next benchmark run."}
        </p>
      ) : (
        <div className="flex flex-1 flex-col gap-4">
          {requestedProviderMissing && requestedProvider ? (
            <p className="text-[11px] text-text-tertiary">
              No recording for {normalizeProviderName(requestedProvider)} at this point.
            </p>
          ) : null}
          <SampleOutputs
            items={items}
            normalizeProvider={normalizeProviderName}
            onPlay={handlePlay}
            onSeeked={handleSeeked}
            onPlaybackEnded={handlePlaybackEnded}
            playRequest={
              requestedItem && !manifestQuery.isPlaceholderData
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
