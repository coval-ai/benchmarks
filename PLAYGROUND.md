# Playground

The playground is a public, browser-based way to try Coval's enabled speech
providers without exposing provider credentials to the client.

## Architecture

- `/playground` is a Next.js App Router page with TTS and STT modes.
- TTS calls the same-origin route handler at `POST /api/playground/tts`.
- STT records microphone audio in the browser, converts it to 16 kHz PCM, and
  posts it to `POST /api/stt/benchmark`.
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

1. The UI captures microphone audio with `getUserMedia`.
2. Audio is converted to mono 16 kHz PCM in the browser.
3. The UI posts the PCM blob and selected model id to `/api/stt/benchmark`.
4. The route validates the model against its allowlist and calls the provider
   server-side.
5. The route returns a transcript and provider timing metadata.

## Public Repo Notes

- Provider model ids, endpoint URLs, sample rates, and voice ids are integration
  metadata, not secrets.
- API keys live only in server-side environment variables.
- Local `.env`, `.env.local`, and generated deployment outputs should not be
  committed.
- The playground intentionally avoids durable storage; result state stays in the
  browser for the current session.
