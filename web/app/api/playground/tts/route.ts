export const runtime = "nodejs";
export const maxDuration = 60;

import { getTtsModelById } from "@/lib/playground/providers";
import { getSessionFromRequest } from "@/lib/playground/session";
import {
  isAllowedOrigin,
  tryAcquireSession,
  releaseSession,
  tryConsumeDailyQuota,
} from "@/lib/playground/security";
import WebSocket from "@/lib/stt/ws";

const SAMPLE_RATE = 24_000;
const MAX_TEXT_LENGTH = 500;
const PROVIDER_TIMEOUT_MS = 55_000;
const MAX_AUDIO_BYTES = 4 * 1024 * 1024;

export async function POST(req: Request) {
  if (!isAllowedOrigin(req.headers.get("origin"))) {
    return Response.json({ error: "Forbidden.", code: "FORBIDDEN" }, { status: 403 });
  }
  const session = getSessionFromRequest(req);
  if (!session) {
    return Response.json({ error: "Unauthorized.", code: "UNAUTHORIZED" }, { status: 401 });
  }
  if (!(await tryAcquireSession(session.sid))) {
    return Response.json({ error: "Too many concurrent sessions.", code: "RATE_LIMITED" }, { status: 429 });
  }

  try {
    return await handle(req, session.sid);
  } finally {
    await releaseSession(session.sid);
  }
}

async function handle(req: Request, sid: string) {
  const rawLen = req.headers.get("content-length");
  const contentLength = rawLen != null ? Number(rawLen) : NaN;
  if (!Number.isFinite(contentLength) || contentLength < 0 || contentLength > 4_096) {
    return Response.json({ error: "Request too large.", code: "PAYLOAD_TOO_LARGE" }, { status: 413 });
  }
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return Response.json(
      { error: "Invalid JSON body.", code: "VALIDATION_ERROR" },
      { status: 400 }
    );
  }

  const { model_id, text } = body as { model_id?: string; text?: string };

  if (!model_id || typeof model_id !== "string") {
    return Response.json(
      { error: "model_id is required.", code: "VALIDATION_ERROR" },
      { status: 400 }
    );
  }
  if (!text || typeof text !== "string" || !text.trim()) {
    return Response.json(
      { error: "Text cannot be empty.", code: "TEXT_EMPTY" },
      { status: 400 }
    );
  }
  if (text.trim().length > MAX_TEXT_LENGTH) {
    return Response.json(
      { error: `Text exceeds maximum length (${MAX_TEXT_LENGTH}).`, code: "TEXT_TOO_LONG" },
      { status: 400 }
    );
  }

  const config = getTtsModelById(model_id);
  if (!config?.enabled) {
    return Response.json(
      { error: "Unknown or disabled modelId.", code: "INVALID_MODEL" },
      { status: 400 }
    );
  }

  // Daily cap is the last gate so malformed or rejected requests don't burn quota.
  if (!(await tryConsumeDailyQuota(sid, "tts"))) {
    return Response.json({ error: "Daily quota exceeded.", code: "RATE_LIMITED" }, { status: 429 });
  }

  try {
    const { chunks, ttfaMs } = await synthesize(
      config.provider,
      config.model,
      config.voice,
      text.trim()
    );

    if (chunks.length === 0) {
      throw new Error("Provider returned no audio data.");
    }

    const wav = buildWav(chunks);
    const headers: Record<string, string> = {
      "Content-Type": "audio/wav",
      "Cache-Control": "no-store",
      "Access-Control-Expose-Headers": "X-TTFA-Ms",
    };
    if (ttfaMs !== null) {
      headers["X-TTFA-Ms"] = ttfaMs.toFixed(2);
    }
    const responseBody = new ArrayBuffer(wav.byteLength);
    new Uint8Array(responseBody).set(wav);
    return new Response(responseBody, { headers });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Synthesis failed.";
    console.error("[playground/tts] provider error", { detail });
    return Response.json({ error: "Audio synthesis failed.", code: "UPSTREAM_ERROR" }, { status: 502 });
  }
}

// ── routing ──────────────────────────────────────────────────────────────────

async function synthesize(
  provider: string,
  model: string,
  voice: string,
  text: string
): Promise<{ chunks: Buffer[]; ttfaMs: number | null }> {
  switch (provider) {
    case "elevenlabs": return synthesizeElevenLabs(model, voice, text);
    case "cartesia":   return synthesizeCartesia(model, voice, text);
    case "deepgram":   return synthesizeDeepgram(model, text);
    case "rime":       return synthesizeRime(model, voice, text);
    default:
      throw new Error(`No adapter for provider '${provider}'`);
  }
}

// ── providers ─────────────────────────────────────────────────────────────────

// ElevenLabs: streaming PCM via /stream endpoint, binary chunks over HTTP
async function synthesizeElevenLabs(model: string, voiceId: string, text: string) {
  const apiKey = process.env.PLAYGROUND_ELEVENLABS_API_KEY;
  if (!apiKey) throw new Error("PLAYGROUND_ELEVENLABS_API_KEY not set");

  const t0 = performance.now();
  const res = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream?output_format=pcm_24000`,
    {
      method: "POST",
      headers: {
        "xi-api-key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text, model_id: model }),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    }
  );
  if (!res.ok) throw new Error(`ElevenLabs ${res.status}: ${await res.text()}`);
  return readBinaryStream(res.body!, t0);
}

function synthesizeCartesia(model: string, voiceId: string, text: string) {
  const apiKey = process.env.PLAYGROUND_CARTESIA_API_KEY;
  if (!apiKey) throw new Error("PLAYGROUND_CARTESIA_API_KEY not set");

  const url = "wss://api.cartesia.ai/tts/websocket?cartesia_version=2024-06-10";

  return new Promise<{ chunks: Buffer[]; ttfaMs: number | null }>((resolve, reject) => {
    const ws = new WebSocket(url, { headers: { "X-API-Key": apiKey } });

    let settled = false;
    let opened = false;
    let t0 = 0;
    let ttfaMs: number | null = null;
    let gotDone = false;
    const chunks: Buffer[] = [];
    let total = 0;

    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      fn();
    };

    const timer = setTimeout(() => {
      ws.close();
      settle(() => reject(new Error("Cartesia: timeout")));
    }, PROVIDER_TIMEOUT_MS);
    const clearTimer = () => clearTimeout(timer);

    ws.on("open", () => {
      opened = true;
      t0 = performance.now();
      ws.send(
        JSON.stringify({
          model_id: model,
          transcript: text,
          voice: { mode: "id", id: voiceId },
          output_format: { container: "raw", encoding: "pcm_s16le", sample_rate: SAMPLE_RATE },
          language: "en",
          continue: false,
          context_id: crypto.randomUUID(),
        }),
      );
    });

    ws.on("message", (raw: Buffer | string) => {
      const msgText = Buffer.isBuffer(raw) ? raw.toString("utf8") : String(raw);
      let evt: { type?: string; data?: string; error?: string; message?: string };
      try {
        evt = JSON.parse(msgText) as typeof evt;
      } catch {
        return;
      }
      if (evt.type === "chunk" && evt.data) {
        if (ttfaMs === null) ttfaMs = performance.now() - t0;
        const buf = Buffer.from(evt.data, "base64");
        total += buf.length;
        if (total > MAX_AUDIO_BYTES) {
          clearTimer();
          ws.close();
          settle(() => reject(new Error("Provider audio exceeded cap")));
          return;
        }
        chunks.push(buf);
      } else if (evt.type === "done") {
        gotDone = true;
        clearTimer();
        ws.close();
        settle(() => resolve({ chunks, ttfaMs }));
      } else if (evt.type === "error") {
        settle(() => {
          clearTimer();
          ws.close();
          reject(new Error(`Cartesia: ${evt.message ?? evt.error ?? "error"}`));
        });
      }
    });

    ws.on("close", () => {
      clearTimer();
      settle(() => {
        if (!opened) {
          reject(new Error("Cartesia: connection closed before open"));
          return;
        }
        if (!gotDone) {
          reject(new Error("Cartesia: connection closed before done"));
          return;
        }
      });
    });

    ws.on("error", (err: Error) => {
      clearTimer();
      settle(() => reject(err));
    });
  });
}

function synthesizeDeepgram(model: string, text: string) {
  const apiKey = process.env.PLAYGROUND_DEEPGRAM_API_KEY;
  if (!apiKey) throw new Error("PLAYGROUND_DEEPGRAM_API_KEY not set");

  const params = new URLSearchParams({
    encoding: "linear16",
    sample_rate: String(SAMPLE_RATE),
    model,
  });
  const url = `wss://api.deepgram.com/v1/speak?${params.toString()}`;

  return new Promise<{ chunks: Buffer[]; ttfaMs: number | null }>((resolve, reject) => {
    const ws = new WebSocket(url, { headers: { Authorization: `Token ${apiKey}` } });

    let settled = false;
    let opened = false;
    let t0 = 0;
    let ttfaMs: number | null = null;
    let gotFlushed = false;
    const chunks: Buffer[] = [];
    let total = 0;

    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      fn();
    };

    const timer = setTimeout(() => {
      ws.close();
      settle(() => reject(new Error("Deepgram: timeout")));
    }, PROVIDER_TIMEOUT_MS);
    const clearTimer = () => clearTimeout(timer);

    ws.on("open", () => {
      opened = true;
      ws.send(JSON.stringify({ type: "Speak", text }));
      t0 = performance.now();
      ws.send(JSON.stringify({ type: "Flush" }));
    });

    ws.on("message", (raw: Buffer | string) => {
      if (Buffer.isBuffer(raw) && raw.length > 0 && raw[0] !== 0x7b) {
        if (ttfaMs === null && t0 > 0) ttfaMs = performance.now() - t0;
        total += raw.length;
        if (total > MAX_AUDIO_BYTES) {
          clearTimer();
          ws.close();
          settle(() => reject(new Error("Provider audio exceeded cap")));
          return;
        }
        chunks.push(raw);
        return;
      }
      const msgText = Buffer.isBuffer(raw) ? raw.toString("utf8") : String(raw);
      let msg: { type?: string; description?: string };
      try {
        msg = JSON.parse(msgText) as typeof msg;
      } catch {
        return;
      }
      if (msg.type === "Flushed") {
        gotFlushed = true;
        clearTimer();
        ws.close();
        settle(() => resolve({ chunks, ttfaMs }));
      }
    });

    ws.on("close", () => {
      clearTimer();
      settle(() => {
        if (!opened) {
          reject(new Error("Deepgram: connection closed before open"));
          return;
        }
        if (!gotFlushed) {
          reject(new Error("Deepgram: connection closed before Flushed"));
          return;
        }
      });
    });

    ws.on("error", (err: Error) => {
      clearTimer();
      settle(() => reject(err));
    });
  });
}

function synthesizeRime(model: string, voice: string, text: string) {
  const apiKey = process.env.PLAYGROUND_RIME_API_KEY;
  if (!apiKey) throw new Error("PLAYGROUND_RIME_API_KEY not set");

  const params = new URLSearchParams({
    modelId: model,
    speaker: voice || "luna",
    audioFormat: "pcm",
    samplingRate: String(SAMPLE_RATE),
    segment: "never",
  });
  const url = `wss://users-ws.rime.ai/ws3?${params.toString()}`;

  return new Promise<{ chunks: Buffer[]; ttfaMs: number | null }>((resolve, reject) => {
    const ws = new WebSocket(url, { headers: { Authorization: `Bearer ${apiKey}` } });

    let settled = false;
    let opened = false;
    let t0 = 0;
    let ttfaMs: number | null = null;
    let gotDone = false;
    const chunks: Buffer[] = [];
    let total = 0;

    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      fn();
    };

    const timer = setTimeout(() => {
      ws.close();
      settle(() => reject(new Error("Rime: timeout")));
    }, PROVIDER_TIMEOUT_MS);
    const clearTimer = () => clearTimeout(timer);

    ws.on("open", () => {
      opened = true;
      t0 = performance.now();
      ws.send(JSON.stringify({ text }));
      ws.send(JSON.stringify({ operation: "eos" }));
    });

    ws.on("message", (raw: Buffer | string) => {
      const msgText = Buffer.isBuffer(raw) ? raw.toString("utf8") : String(raw);
      let msg: { type?: string; data?: string; message?: string };
      try {
        msg = JSON.parse(msgText) as typeof msg;
      } catch {
        return;
      }
      if (msg.type === "chunk" && msg.data) {
        if (ttfaMs === null) ttfaMs = performance.now() - t0;
        const buf = Buffer.from(msg.data, "base64");
        total += buf.length;
        if (total > MAX_AUDIO_BYTES) {
          clearTimer();
          ws.close();
          settle(() => reject(new Error("Provider audio exceeded cap")));
          return;
        }
        chunks.push(buf);
      } else if (msg.type === "done") {
        gotDone = true;
        clearTimer();
        ws.close();
        settle(() => resolve({ chunks, ttfaMs }));
      } else if (msg.type === "error") {
        settle(() => {
          clearTimer();
          ws.close();
          reject(new Error(`Rime: ${msg.message ?? "error"}`));
        });
      }
    });

    ws.on("close", () => {
      clearTimer();
      settle(() => {
        if (!opened) {
          reject(new Error("Rime: connection closed before open"));
          return;
        }
        if (!gotDone) {
          reject(new Error("Rime: connection closed before done"));
          return;
        }
      });
    });

    ws.on("error", (err: Error) => {
      clearTimer();
      settle(() => reject(err));
    });
  });
}

// ── shared helpers ────────────────────────────────────────────────────────────

// Reads a binary ReadableStream chunk by chunk.
// TTFA = t0 (set by caller before the fetch) → first non-empty chunk received.
async function readBinaryStream(
  body: ReadableStream<Uint8Array>,
  t0: number
): Promise<{ chunks: Buffer[]; ttfaMs: number | null }> {
  const reader = body.getReader();
  const chunks: Buffer[] = [];
  let ttfaMs: number | null = null;
  let total = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value && value.length > 0) {
      if (ttfaMs === null) ttfaMs = performance.now() - t0;
      total += value.length;
      if (total > MAX_AUDIO_BYTES) {
        await reader.cancel();
        throw new Error("Provider audio exceeded cap");
      }
      chunks.push(Buffer.from(value));
    }
  }
  return { chunks, ttfaMs };
}

// Prepends a 44-byte WAV header to raw PCM data.
// All providers output 24 kHz, 16-bit signed LE, mono — header constants match.
function buildWav(chunks: Buffer[]): Buffer {
  const pcm = Buffer.concat(chunks);
  const dataLen = pcm.byteLength;
  const header = Buffer.alloc(44);

  header.write("RIFF", 0);
  header.writeUInt32LE(36 + dataLen, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);              // fmt chunk size (PCM = 16)
  header.writeUInt16LE(1, 20);               // audio format (1 = PCM)
  header.writeUInt16LE(1, 22);               // channels (1 = mono)
  header.writeUInt32LE(SAMPLE_RATE, 24);     // sample rate
  header.writeUInt32LE(SAMPLE_RATE * 2, 28); // byte rate (rate × channels × bytes/sample)
  header.writeUInt16LE(2, 32);               // block align (channels × bytes/sample)
  header.writeUInt16LE(16, 34);              // bits per sample
  header.write("data", 36);
  header.writeUInt32LE(dataLen, 40);

  return Buffer.concat([header, pcm]);
}
