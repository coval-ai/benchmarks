// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useCallback, useEffect, useMemo } from "react";
import Card from "@/components/shared/Card";
import { useDashboard } from "@/contexts/DashboardContext";
import { SampleFetchError } from "@/lib/audioSamples/createSampleFeed";
import { isMultiTurn, s2sSampleFeed, visibleRecordings } from "@/lib/audioSamples/s2sFeed";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";
import { usePlaybackCoordinator } from "@/hooks/useSequencedPlayback";
import { SampleOutputs, type SampleOutputItem } from "./SampleOutputs";

// Which failure case fired, for the console — network/http/parse vs an
// unclassified error, with the status when we have one.
function describeError(err: unknown): string {
  if (err instanceof SampleFetchError) {
    return err.status != null ? `${err.kind} (status ${err.status})` : err.kind;
  }
  return err instanceof Error ? err.message : String(err);
}

export function SamplesCard() {
  const { modelsByProvider, normalizeProviderName, s2sPlayRequest } = useDashboard();
  const coordinator = usePlaybackCoordinator();
  const visibleProviders = useMemo(
    () => new Set(Object.keys(modelsByProvider)),
    [modelsByProvider]
  );

  const indexQuery = s2sSampleFeed.useIndexQuery();
  const latestTick = indexQuery.data?.[0] ?? null;
  // A timeline-tooltip click pins the card to that bucket; otherwise show latest.
  const effectiveTick = s2sPlayRequest?.tick ?? latestTick;
  const manifestQuery = s2sSampleFeed.useManifestQuery(effectiveTick);
  const manifest = manifestQuery.data ?? null;
  const hasNoConversation = manifest != null && !isMultiTurn(manifest);

  const fetchError = manifestQuery.isError || indexQuery.isError;
  // Keep the failure cause internal (dev console only); public visitors just see
  // the generic "temporarily unavailable" copy below.
  useEffect(() => {
    if (process.env.NODE_ENV === "production") return;
    const err = manifestQuery.error ?? indexQuery.error;
    if (err) console.error(`[s2s-samples] fetch failed: ${describeError(err)}`, err);
  }, [manifestQuery.error, indexQuery.error]);

  const items = useMemo<SampleOutputItem[]>(() => {
    if (!manifest || hasNoConversation) return [];
    return visibleRecordings(manifest, visibleProviders).map((r) => ({
      provider: r.provider,
      model: r.model,
      url: s2sSampleFeed.objectUrl(r.object),
      turns: r.turns,
    }));
  }, [manifest, hasNoConversation, visibleProviders]);

  const handlePlay = useCallback(
    (provider: string) => {
      capturePostHogEvent(POSTHOG_EVENTS.s2sSamplePlayRequested, {
        surface: "s2s_dashboard",
        mode: "s2s",
        provider,
        bucket_at: manifest?.bucket_at,
      });
    },
    [manifest?.bucket_at]
  );

  const loading =
    indexQuery.isLoading || (effectiveTick != null && manifestQuery.isLoading);

  return (
    <Card className="text-left min-w-0 h-full flex flex-col" padding="p-5 lg:p-8">
      <div className="mb-3 text-[0.9rem] font-light text-text-secondary">
        Conversation samples
      </div>

      {loading ? (
        <div className="h-40 animate-pulse rounded-lg bg-surface-secondary" />
      ) : fetchError ? (
        <p className="py-8 text-center text-sm text-text-tertiary">
          Samples are temporarily unavailable.
        </p>
      ) : items.length === 0 ? (
        <p className="py-8 text-center text-sm text-text-tertiary">
          {hasNoConversation || (!manifest && s2sPlayRequest)
            ? "No conversation recorded for this point."
            : "Samples appear here after the next benchmark run."}
        </p>
      ) : (
        <div className="flex flex-1 flex-col gap-4">
          <SampleOutputs
            items={items}
            normalizeProvider={normalizeProviderName}
            onPlay={handlePlay}
            playRequest={
              s2sPlayRequest
                ? { provider: s2sPlayRequest.provider, nonce: s2sPlayRequest.nonce }
                : null
            }
            coordinator={coordinator}
          />
        </div>
      )}
    </Card>
  );
}
