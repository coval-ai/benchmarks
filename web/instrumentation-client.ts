import posthog from "posthog-js";

const posthogToken = process.env.NEXT_PUBLIC_POSTHOG_TOKEN;

function shouldEnablePostHogDebug(): boolean {
  if (typeof window === "undefined") return false;

  const searchParams = new URLSearchParams(window.location.search);
  return (
    process.env.NEXT_PUBLIC_POSTHOG_DEBUG === "true" ||
    searchParams.get("posthog_debug") === "1"
  );
}

if (posthogToken) {
  posthog.init(posthogToken, {
    api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST ?? "https://us.i.posthog.com",
    defaults: "2026-01-30",
    autocapture: false,
    disable_session_recording: true,
    loaded: (instance) => {
      if (!shouldEnablePostHogDebug()) return;

      instance.debug(true);

      if (typeof window !== "undefined") {
        (
          window as Window & {
            __covalPostHog?: unknown;
          }
        ).__covalPostHog = instance;
      }
    }
  });

  posthog.register({ product: "public-benchmarks" });
}
