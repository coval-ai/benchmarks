import posthog from "posthog-js";

type PostHogPropertyValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | Array<string | number | boolean | null>;

export type PostHogProperties = Record<string, PostHogPropertyValue>;

function isPostHogConfigured(): boolean {
  return Boolean(process.env.NEXT_PUBLIC_POSTHOG_TOKEN);
}

export function capturePostHogEvent(event: string, properties?: PostHogProperties): void {
  if (!isPostHogConfigured()) return;

  try {
    posthog.capture(event, properties);
  } catch {
    // Never let analytics failures affect the benchmark UI.
  }
}
