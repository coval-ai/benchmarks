# Playground

The playground is a public, browser-based way to try Coval's enabled speech
providers without exposing provider credentials to the client.

## Architecture

- `/playground` is a Next.js App Router page with TTS and STT modes.
- TTS calls the same-origin route handler at `POST /api/playground/tts`.
- STT records microphone audio in the browser, converts it to 16 kHz PCM, and
  posts it to `POST /api/playground/stt`. The route accepts a JSON-encoded
  `modelIds` array and fans out to providers internally so one click consumes
  one daily-quota item regardless of how many models were selected.
- Both route handlers run on the server (`runtime = "nodejs"`) and call provider
  APIs with server-only `PLAYGROUND_*` environment variables.
- The browser receives only generated audio, transcripts, timing metrics, and
  normalized error payloads. Provider API keys are never sent to the browser.

The benchmark dashboards still read historical results from the FastAPI backend
through `NEXT_PUBLIC_API_URL`. The interactive playground does not depend on
FastAPI playground routes.

## Environment

Set these variables in the hosting environment for the Next.js app:

```bash
PLAYGROUND_ELEVENLABS_API_KEY=
PLAYGROUND_CARTESIA_API_KEY=
PLAYGROUND_DEEPGRAM_API_KEY=
PLAYGROUND_RIME_API_KEY=
PLAYGROUND_ASSEMBLYAI_API_KEY=
PLAYGROUND_SPEECHMATICS_API_KEY=
```

Do not prefix provider credentials with `NEXT_PUBLIC_`.

## TTS Flow

1. The UI sends `{ model_id, text }` to `/api/playground/tts`.
2. The route validates the model against `web/lib/playground/providers.ts`.
3. The route calls the selected provider server-side.
4. The route returns WAV audio and exposes `X-TTFA-Ms` when available.
5. The client measures total request latency and displays TTFA, total latency,
   and characters per second.

## STT Flow

1. The UI captures microphone audio with `getUserMedia` at the device's native
   rate and resamples to 16 kHz mono PCM in the browser.
2. On record-stop, the UI posts one multipart request to `/api/playground/stt`
   with the PCM blob and a JSON-encoded list of selected `modelIds`.
3. The route validates every id against the allowlist derived from
   `getEnabledSttModels()`, consumes one daily-quota item, and fans out to each
   provider via `Promise.all`.
4. The route returns `{ results: STTResponse[] }` — one entry per requested
   model. Each is either a success row (`transcript`, `ttfaMs`, `audioToFinalMs`)
   or a per-model error row that does not bring down peers in the batch.
5. `ttfaMs` is "time from server-received audio to first transcript token", not
   live streaming TTFT — the browser uploads the whole clip before transcription
   starts. UI labels reflect this.

## STT Methodology Caveats

The playground STT comparison is a **controlled-condition offline-upload
benchmark**, not a live-streaming TTFT benchmark. For runner-paced realtime
numbers, see the benchmark dashboard at `/stt`.

Planned follow-up: server-side real-time pacing of audio to each provider WS

## Public Repo Notes

- Provider model ids, endpoint URLs, sample rates, and voice ids are integration
  metadata, not secrets.
- API keys live only in server-side environment variables.
- Local `.env`, `.env.local`, and generated deployment outputs should not be
  committed.
- The playground intentionally avoids durable storage; result state stays in the
  browser for the current session.

## Public Launch Gates

All five must be `done` before this surface is OSS-ready. Today, the route
handlers are defense-in-depth for honest browsers, not the abuse boundary.

- **Signed sessions** — `done`. HMAC-SHA256 token in an HttpOnly cookie
  (`__playground_session`) minted on `/playground` server-component render,
  required on every `POST /api/playground/*`. The cookie is set with two
  Path scopes (`/playground` and `/api/playground`) so the page can read it
  on reload without leaking it to dashboard routes. Rate limiter keys on
  the session `sid`, not on IP. See `web/lib/playground/session.ts`.
  Rotation: set `PLAYGROUND_SESSION_SECRET_PREVIOUS` to the current secret,
  set `PLAYGROUND_SESSION_SECRET` to a fresh one, deploy. Existing cookies
  verify against either for ~24h; drop `PREVIOUS` after.
- **Durable quota** — `not yet wired`. `concurrent` / `daily` Maps in
  `security.ts` are per-instance and reset on cold start; move to Upstash Redis
  keyed on the signed session.
- **CAPTCHA** — `not yet wired`. Gate session-token issuance behind Cloudflare
  Turnstile so scripted callers fail before any provider call.
- **Per-provider spend budgets** — `not yet wired`. Add an absolute server-side
  daily $-cap per provider that returns `RATE_LIMITED` once hit, independent of
  the IP-level counters.
- **Abuse response procedure** — `not yet wired`. Document on-call, alert
  routing, IP/session block path, and provider-key rotation without redeploy.
