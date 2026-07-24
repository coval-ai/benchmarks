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
import { SampleInput } from "./SampleInput";
import { SampleOutputs, type SampleOutputItem } from "./SampleOutputs";

function tickLabel(tick: string): string {
  const d = new Date(tick);
  if (Number.isNaN(d.getTime())) return tick;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

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
  const multiTurn = manifest ? isMultiTurn(manifest) : false;

  const fetchError = manifestQuery.isError || indexQuery.isError;
  // Keep the failure cause internal (dev console only); public visitors just see
  // the generic "temporarily unavailable" copy below.
  useEffect(() => {
    if (process.env.NODE_ENV === "production") return;
    const err = manifestQuery.error ?? indexQuery.error;
    if (err) console.error(`[s2s-samples] fetch failed: ${describeError(err)}`, err);
  }, [manifestQuery.error, indexQuery.error]);

  const items = useMemo<SampleOutputItem[]>(() => {
    if (!manifest) return [];
    return visibleRecordings(manifest, visibleProviders).map((r) => ({
      provider: r.provider,
      model: r.model,
      url: s2sSampleFeed.objectUrl(r.object),
      turns: r.turns,
    }));
  }, [manifest, visibleProviders]);

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
      <div className="mb-3 flex items-baseline justify-between gap-2">
        <div className="text-[0.9rem] font-light text-text-secondary">
          Conversation samples
        </div>
        {manifest ? (
          <span className="flex items-baseline gap-2 font-mono text-xs text-text-tertiary">
            {manifest.persona_name ? (
              <span className="text-text-secondary">{manifest.persona_name}</span>
            ) : null}
            <span>{tickLabel(manifest.bucket_at)}</span>
          </span>
        ) : null}
      </div>

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
          {!multiTurn ? (
            <SampleInput
              transcript={manifest?.transcript ?? null}
              inputAudioUrl={manifest?.input_audio_url ?? null}
              coordinator={coordinator}
            />
          ) : null}
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

      {!multiTurn ? (
        <p className="mt-4 text-[10px] text-text-tertiary">
          Prompts from the SLURP dataset (CC BY-NC 4.0).
        </p>
      ) : null}
    </Card>
  );
}
