"use client";

import { useCallback, useMemo } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { capturePostHogEvent } from "@/lib/posthog/client";
import { POSTHOG_EVENTS } from "@/lib/posthog/events";

export type PlaygroundMode = "tts" | "stt";

const MODE_QUERY = "playground_mode";

function playgroundModeFromQueryValue(raw: string | null): PlaygroundMode {
  if (raw === "stt") return "stt";
  if (raw === "tts" || raw === null) return "tts";

  if (process.env.NODE_ENV === "development") {
    console.warn(
      `[usePlaygroundMode] unexpected ${MODE_QUERY}="${raw}", falling back to "tts"`
    );
  }
  return "tts";
}

/**
 * Syncs playground TTS/STT mode to the URL (`?playground_mode=tts|stt`).
 *
 * **Suspense:** `useSearchParams()` must run under a parent `<Suspense>` (see
 * `app/playground/page.tsx`). Do not mount a tree that calls this hook without
 * that boundary, or Next.js can warn or fail during prerender.
 */
export function usePlaygroundMode(): {
  mode: PlaygroundMode;
  setMode: (next: PlaygroundMode) => void;
} {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const mode = useMemo<PlaygroundMode>(() => {
    const raw = searchParams.get(MODE_QUERY);
    return playgroundModeFromQueryValue(raw);
  }, [searchParams]);

  const setMode = useCallback(
    (next: PlaygroundMode) => {
      if (next === mode) return;

      capturePostHogEvent(POSTHOG_EVENTS.playgroundModeSwitched, {
        surface: "playground",
        from: mode,
        to: next
      });

      // Preserve unrelated query params when switching mode; only
      // `playground_mode` is updated.
      const params = new URLSearchParams(searchParams.toString());
      params.set(MODE_QUERY, next);
      const qs = params.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [mode, pathname, router, searchParams]
  );

  return { mode, setMode };
}
