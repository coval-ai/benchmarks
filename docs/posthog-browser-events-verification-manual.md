# PostHog Browser Events Verification Manual

This note describes how to verify the current browser-side PostHog event set for
the Coval Benchmarks website.

The current target is **exactly 8 browser events**. Older custom events are not
part of the current spec.

## Current 8 events

The website should emit these 8 browser events:

1. `tts_page_visited`
2. `stt_page_visited`
3. `tts_timeline_tooltip_opened`
4. `stt_timeline_tooltip_opened`
5. `playground_tts_visited`
6. `playground_stt_visited`
7. `playground_tts_benchmark_pressed`
8. `playground_stt_record_pressed`

These are wired in:

- `web/app/tts/tts-dashboard.tsx`
- `web/app/stt/stt-dashboard.tsx`
- `web/components/visualizations/TimelineChart.tsx`
- `web/components/charts/tooltips/TimelineTooltip.tsx`
- `web/app/playground/playground-page-client.tsx`
- `web/app/playground/components/playground-draft/TTSPlaygroundPanel.tsx`
- `web/app/playground/components/playground-draft/STTPlaygroundPanel.tsx`

## Scope of this manual

This manual is for verifying the real UI paths directly.

It is focused on:

- actual dashboard page visits
- actual timeline tooltip openings
- actual playground button presses

It is **not** based on synthetic smoke events.

## Preconditions

Before testing, make sure:

- `web/.env` or `web/.env.local` contains:

```env
NEXT_PUBLIC_POSTHOG_TOKEN=<your token>
NEXT_PUBLIC_POSTHOG_HOST=https://us.i.posthog.com
```

- the dev server has been restarted after any env change
- you are looking at the correct PostHog project for that token
- you open test pages with `?posthog_debug=1` so `posthog-js` logs outgoing
  events in the browser console

## One-time setup

Open:

- `http://localhost:3000/playground?posthog_debug=1`

In the browser console, run these one at a time:

```js
window.__covalPostHog
```

```js
window.__covalPostHog.get_distinct_id()
```

```js
window.__covalPostHog.get_session_id()
```

Keep the session id handy. It is the cleanest way to isolate your own events in
PostHog Live Events.

### If `window.__covalPostHog` is `undefined`

Most likely causes:

- the URL is missing `?posthog_debug=1`
- `NEXT_PUBLIC_POSTHOG_DEBUG` is not set to `true`
- the dev server was not restarted after env edits
- the page was not reloaded after restart

In that case:

1. restart the dev server
2. reopen the page with `?posthog_debug=1`
3. fully reload the page
4. rerun:

```js
window.__covalPostHog
```

### If Safari shows something like `Zn { ... }`

That is normal. Safari is showing the minified class name for the PostHog
client object.

To verify it is the right object, run:

```js
typeof window.__covalPostHog
```

```js
typeof window.__covalPostHog?.capture
```

```js
typeof window.__covalPostHog?.get_distinct_id
```

They should return `"object"` and `"function"` values.

## What to check for each event

For each manual interaction below, check these layers:

### 1. Browser console

With `?posthog_debug=1` enabled, you should see lines like:

- `send "tts_page_visited"`
- `send "playground_tts_benchmark_pressed"`

### 2. Browser network

In DevTools Network, you should see a browser event ingestion request to a
PostHog endpoint like:

- `https://us.i.posthog.com/i/v0/e/`

It should return `200`.

### 3. PostHog Live Events

Filter by:

- exact event name
- and ideally your current browser `Session ID`

For all 8 current events, `Library` should be:

- `posthog-js`

## Manual checks by event

### 1. `tts_page_visited`

How to trigger:

1. open `/tts?posthog_debug=1`
2. wait for the page to finish rendering

What to verify:

- console shows `send "tts_page_visited"`
- PostHog shows `event = tts_page_visited`
- `Library = posthog-js`

Expected useful properties:

- `surface = "tts_dashboard"`
- `path = "/tts"`

### 2. `stt_page_visited`

How to trigger:

1. open `/stt?posthog_debug=1`
2. wait for the page to finish rendering

What to verify:

- console shows `send "stt_page_visited"`
- PostHog shows `event = stt_page_visited`
- `Library = posthog-js`

Expected useful properties:

- `surface = "stt_dashboard"`
- `path = "/stt"`

### 3. `tts_timeline_tooltip_opened`

How to trigger:

1. open `/tts?posthog_debug=1`
2. hover one point on the first line chart
3. wait until the tooltip appears with the model ranking list

What to verify:

- console shows `send "tts_timeline_tooltip_opened"`
- PostHog shows `event = tts_timeline_tooltip_opened`
- `Library = posthog-js`

Expected useful properties:

- `surface = "tts_dashboard"`
- `tooltip_timestamp`
- `visible_model_count`
- `selected_model_count`

### 4. `stt_timeline_tooltip_opened`

How to trigger:

1. open `/stt?posthog_debug=1`
2. hover one point on the first line chart
3. wait until the tooltip appears with the model ranking list

What to verify:

- console shows `send "stt_timeline_tooltip_opened"`
- PostHog shows `event = stt_timeline_tooltip_opened`
- `Library = posthog-js`

Expected useful properties:

- `surface = "stt_dashboard"`
- `tooltip_timestamp`
- `visible_model_count`
- `selected_model_count`

### 5. `playground_tts_visited`

How to trigger:

1. open `/playground?posthog_debug=1`
2. make sure TTS is the active mode

What to verify:

- console shows `send "playground_tts_visited"`
- PostHog shows `event = playground_tts_visited`
- `Library = posthog-js`

Expected useful properties:

- `mode = "tts"`
- `path = "/playground"`

### 6. `playground_stt_visited`

How to trigger:

1. open `/playground?posthog_debug=1`
2. switch to STT mode

What to verify:

- console shows `send "playground_stt_visited"`
- PostHog shows `event = playground_stt_visited`
- `Library = posthog-js`

Expected useful properties:

- `mode = "stt"`
- `path = "/playground"`

### 7. `playground_tts_benchmark_pressed`

How to trigger:

1. open `/playground?posthog_debug=1`
2. stay in TTS mode
3. enter text
4. make sure at least one model is selected
5. click the `Benchmark` button

What to verify:

- console shows `send "playground_tts_benchmark_pressed"`
- PostHog shows `event = playground_tts_benchmark_pressed`
- `Library = posthog-js`

Expected useful properties:

- `mode = "tts"`
- `text_length`
- `selected_model_ids`
- `selected_model_count`
- `used_example_prompt`

### 8. `playground_stt_record_pressed`

How to trigger:

1. open `/playground?posthog_debug=1`
2. switch to STT mode
3. make sure at least one model is selected
4. click the `Record` button

What to verify:

- console shows `send "playground_stt_record_pressed"`
- PostHog shows `event = playground_stt_record_pressed`
- `Library = posthog-js`

Expected useful properties:

- `mode = "stt"`
- `selected_model_ids`
- `selected_model_count`

## Quick full pass

To cover the full 8-event set quickly:

1. open `/tts?posthog_debug=1`
2. confirm `tts_page_visited`
3. hover the first chart and confirm `tts_timeline_tooltip_opened`
4. open `/stt?posthog_debug=1`
5. confirm `stt_page_visited`
6. hover the first chart and confirm `stt_timeline_tooltip_opened`
7. open `/playground?posthog_debug=1`
8. confirm `playground_tts_visited`
9. click `Benchmark` in TTS mode and confirm `playground_tts_benchmark_pressed`
10. switch to STT and confirm `playground_stt_visited`
11. click `Record` and confirm `playground_stt_record_pressed`

## Can this be checked without manual interaction?

Partially.

Not completely with the current test setup.

### What is automated today

Currently automated:

- unit coverage for the shared browser helper in
  `web/lib/posthog/client.test.ts`

This proves:

- `posthog-js` can be called from the browser
- analytics failures are swallowed safely

This does **not** prove:

- the real UI hooks fire from actual interactions
- every real interaction is visible in PostHog

## Best non-manual plan

### Plan A: Hook/component tests

Add focused tests that mock `capturePostHogEvent()` and assert that the real UI
paths call it.

Examples:

- assert `/tts` mount triggers `tts_page_visited`
- assert `/stt` mount triggers `stt_page_visited`
- assert opening the first chart tooltip triggers the correct tooltip event
- assert the TTS benchmark button triggers `playground_tts_benchmark_pressed`
- assert the STT record button triggers `playground_stt_record_pressed`

Pros:

- fast
- deterministic
- good regression coverage

Cons:

- does not prove browser ingestion or PostHog UI visibility

### Plan B: Browser E2E tests with network assertions

Add Playwright tests that:

1. load the real app locally
2. perform the actual user interactions
3. intercept browser requests to PostHog
4. assert that the outgoing payload includes the expected event name

Pros:

- verifies the real UI hooks
- verifies browser network emission
- reduces manual effort substantially

Cons:

- more setup
- still does not prove the shared PostHog UI shows the event immediately

### Plan C: Keep one short manual spot check

Even with automation, keep one quick manual check in the real PostHog project:

1. trigger one event locally
2. filter by session id or a unique marker
3. confirm it appears in Live Events

That validates the external hosted environment, not just local code behavior.
