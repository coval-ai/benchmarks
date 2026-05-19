# PostHog Integration into Coval Benchmarks Website

This note captures a practical PostHog setup plan for the Coval Benchmarks web
app and narrows the event plan to the handful of interactions that matter most.

The goal is not to track everything. The goal is to track where users spend the
most time actively interacting:

- choosing models to compare
- running TTS benchmarks
- recording and submitting STT benchmarks
- switching between TTS and STT playground modes
- reviewing benchmark results

## Recommended Branch Split

Keep the PostHog work in separate branches so setup and event instrumentation
can be reviewed independently.

### Branch 1: bootstrap only

Create a branch for SDK installation and baseline wiring only:

```bash
git switch main
git pull
git switch -c chore/posthog-bootstrap
```

Scope:

- install `posthog-js`
- add env vars
- add `instrumentation-client.ts`
- add shared client PostHog helpers
- do not add feature events yet

### Branch 2: product events

Create a second branch for actual analytics instrumentation:

```bash
git switch main
git switch -c feat/posthog-core-events
```

Scope:

- add client-side usage events
- add server-side route-handler events
- keep event names and shared properties centralized

## Stack-Specific Setup Plan

This app is a `Next.js 16` App Router app using `pnpm`.

The important shape of the app is:

- dashboards live in `web/app/tts` and `web/app/stt`
- the interactive playground lives in `web/app/playground`
- TTS playground execution happens in `web/app/api/playground/tts/route.ts`
- STT playground execution happens in `web/app/api/stt/benchmark/route.ts`

That means the right integration is:

- client SDK for page-level and interaction events
- server SDK for route-handler success/failure and provider metrics

## Implementation Steps

### 1. Install SDKs

From `web/`:

```bash
pnpm add posthog-js
```

Use `posthog-js` for browser events.

### 2. Add environment variables

Add these to local env and Vercel:

```env
NEXT_PUBLIC_POSTHOG_TOKEN=phc_xxx
NEXT_PUBLIC_POSTHOG_HOST=https://us.i.posthog.com
```

Also add them to `web/.env.example` so the setup is documented for the repo.

Notes:

- use `NEXT_PUBLIC_` because the browser SDK needs access
- if the PostHog project is in EU, use `https://eu.i.posthog.com`
- do not hardcode the token in source files

### 3. Initialize PostHog in Next.js

For this stack, the correct client entry point is root-level
`web/instrumentation-client.ts`.

Recommended shape:

```ts
import posthog from "posthog-js";

posthog.init(process.env.NEXT_PUBLIC_POSTHOG_TOKEN!, {
  api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST,
  defaults: "2026-01-30",
});
```

This is the lightest-weight Next 16 setup and matches the App Router model.

### 4. Add a client helper and event registry

Create a small analytics layer instead of calling `posthog.capture()` directly
throughout the app.

Recommended files:

- `web/lib/posthog/client.ts`
- `web/lib/posthog/events.ts`

Purpose:

- centralize event names
- centralize shared properties
- avoid typos and event drift
- make future cleanup easier

### 5. Start anonymous

There is no login flow in the current web app, so do not block rollout on
`identify()`.

Start with anonymous events first.

If the site later gets authentication or a stable workspace/user id, add:

```ts
posthog.identify(stableUserId);
```

### 6. Protect privacy

Do not send raw user content to PostHog.

Avoid sending:

- raw prompt text
- raw transcripts
- raw audio

Prefer safe derived properties:

- `text_length`
- `audio_duration_ms`
- `selected_model_ids`
- `selected_model_count`
- `provider`
- `model_id`
- `success_count`
- `failure_count`
- `ttfa_ms`
- `ttct_ms`
- `error_code`

### 7. Verify the integration

Use a simple verification flow:

1. load `/tts`, `/stt`, and `/playground`
2. run a TTS benchmark
3. run an STT benchmark
4. confirm the browser events appear in PostHog live events

## Current Browser Event Plan

The current implemented browser-event set is intentionally smaller and simpler
than the original draft above.

Track only these 8 browser events:

1. `tts_page_visited`
2. `stt_page_visited`
3. `tts_timeline_tooltip_opened`
4. `stt_timeline_tooltip_opened`
5. `playground_tts_visited`
6. `playground_stt_visited`
7. `playground_tts_benchmark_pressed`
8. `playground_stt_record_pressed`

Why these 8:

- they give one page-visit event for each benchmark dashboard
- they capture one high-signal chart interaction for each dashboard
- they capture one page-visit event for each playground mode
- they capture the primary action button on each playground mode

This keeps the analytics surface intentionally small and makes verification in
PostHog much easier than the earlier, more granular draft.
