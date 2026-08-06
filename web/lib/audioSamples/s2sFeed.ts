// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

// S2S sample helpers. Reading the samples themselves lives in lib/api (the bucket
// is private; the API decides what a caller may see), so what remains here is the
// page-side shaping the API does not do.

import type {
  S2SSampleApiResponse,
  S2SSampleRecording,
  S2SSampleTurn,
} from "@/lib/api/client";
import { toModelKey } from "../utils/formatters";

// The page's turn shape. Deliberately not the wire type: Coval omits an offset as
// null, and the components treat "no offset" as undefined, so the two collapse
// here rather than at every use site.
export interface S2STurn {
  index: number;
  role: string;
  content: string;
  start_offset?: number;
  end_offset?: number;
}

// Manifests written before the sampler filtered them carry the persona's
// `end_conversation` tool record as a `role: "tool"` turn whose content is raw
// JSON. Drop anything that isn't spoken dialogue so it can't render as a caller
// turn — those manifests stay readable for their 30-day retention.
const SPOKEN_ROLES: ReadonlySet<string> = new Set(["user", "assistant"]);

export function spokenTurns(turns: readonly S2SSampleTurn[] | undefined): S2STurn[] {
  return (turns ?? [])
    .filter((turn) => SPOKEN_ROLES.has(turn.role))
    .map((turn) => ({
      index: turn.index,
      role: turn.role,
      content: turn.content,
      start_offset: turn.start_offset ?? undefined,
      end_offset: turn.end_offset ?? undefined,
    }));
}

// Drop recordings whose model isn't visible on the page (disabled catalogue).
// Keyed by (provider, model): one provider can carry several S2S models. This is
// separate from the embargo filter the API applies — a model can be visible to
// this caller and still be switched off on the page.
export function visibleRecordings(
  sample: S2SSampleApiResponse,
  visibleModels: ReadonlySet<string>
): S2SSampleRecording[] {
  return sample.recordings.filter((recording) =>
    visibleModels.has(toModelKey(recording.provider, recording.model))
  );
}
