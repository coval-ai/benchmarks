export const runtime = "nodejs";
export const maxDuration = 60;

import { callDeepgram } from "@/lib/stt/deepgram";
import { callElevenLabs } from "@/lib/stt/elevenlabs";
import { callAssemblyAI } from "@/lib/stt/assemblyai";
import { callSpeechmatics } from "@/lib/stt/speechmatics";
import { getEnabledSttModels } from "@/lib/playground/providers";
import type { STTResponse } from "@/lib/stt/types";
import {
  isAllowedOrigin,
  getClientIp,
  tryAcquireSession,
  releaseSession,
  tryConsumeDailyQuota,
} from "@/lib/playground/security";

const ALLOWED_MODELS = new Set(getEnabledSttModels().map((m) => m.id));

// 60 s × 16 kHz × 16-bit mono = 1.92 MB; reject larger uploads before buffering.
const MAX_PCM_BYTES = 60 * 16_000 * 2;

export async function POST(req: Request): Promise<Response> {
  if (!isAllowedOrigin(req.headers.get("origin"))) {
    return Response.json(
      { modelId: "", error: "Forbidden.", code: "FORBIDDEN" } satisfies STTResponse,
      { status: 403 },
    );
  }
  const ip = getClientIp(req);
  if (!tryAcquireSession(ip)) {
    return Response.json(
      { modelId: "", error: "Too many concurrent sessions.", code: "RATE_LIMITED" } satisfies STTResponse,
      { status: 429 },
    );
  }

  try {
    return await handle(req, ip);
  } finally {
    releaseSession(ip);
  }
}

async function handle(req: Request, ip: string): Promise<Response> {
  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return Response.json(
      { modelId: "", error: "Invalid multipart form data.", code: "VALIDATION_ERROR" } satisfies STTResponse,
      { status: 400 },
    );
  }
  const modelId = String(form.get("modelId") ?? "");

  if (!ALLOWED_MODELS.has(modelId)) {
    return Response.json(
      { modelId, error: "Unknown or disabled model.", code: "INVALID_MODEL" } satisfies STTResponse,
      { status: 400 },
    );
  }

  const audioPart = form.get("audio");
  if (!(audioPart instanceof Blob)) {
    return Response.json(
      { modelId, error: "Missing audio field.", code: "MISSING_AUDIO" } satisfies STTResponse,
      { status: 400 },
    );
  }
  if (audioPart.size > MAX_PCM_BYTES) {
    return Response.json(
      { modelId, error: "Audio exceeds max duration.", code: "AUDIO_TOO_LARGE" } satisfies STTResponse,
      { status: 413 },
    );
  }

  // Daily cap is the last gate so malformed or rejected requests don't burn quota.
  if (!tryConsumeDailyQuota(ip)) {
    return Response.json(
      { modelId, error: "Daily quota exceeded.", code: "RATE_LIMITED" } satisfies STTResponse,
      { status: 429 },
    );
  }

  const pcm = await audioPart.arrayBuffer();
  const sep = modelId.indexOf(":");
  const provider = modelId.slice(0, sep);
  const model = modelId.slice(sep + 1);

  try {
    let transcript: string;
    let ttftMs: number | null = null;
    let audioToFinalMs = 0;

    switch (provider) {
      case "speechmatics": {
        const r = await callSpeechmatics(
          pcm,
          model as "default" | "enhanced",
          requireEnv("PLAYGROUND_SPEECHMATICS_API_KEY"),
        );
        ({ transcript, ttftMs, audioToFinalMs } = r);
        break;
      }
      case "deepgram": {
        const r = await callDeepgram(pcm, model, requireEnv("PLAYGROUND_DEEPGRAM_API_KEY"));
        ({ transcript, ttftMs, audioToFinalMs } = r);
        break;
      }
      case "elevenlabs": {
        const r = await callElevenLabs(pcm, requireEnv("PLAYGROUND_ELEVENLABS_API_KEY"));
        ({ transcript, ttftMs, audioToFinalMs } = r);
        break;
      }
      case "assemblyai": {
        const r = await callAssemblyAI(pcm, requireEnv("PLAYGROUND_ASSEMBLYAI_API_KEY"));
        ({ transcript, ttftMs, audioToFinalMs } = r);
        break;
      }
      default:
        return Response.json(
          { modelId, error: "Unhandled provider.", code: "INVALID_MODEL" } satisfies STTResponse,
          { status: 400 },
        );
    }

    return Response.json({
      modelId,
      transcript,
      ttftMs,
      audioToFinalMs,
    } satisfies STTResponse);

  } catch (err) {
    const error = err instanceof Error ? err.message : "Provider error";
    return Response.json(
      { modelId, error, code: "PROVIDER_ERROR" } satisfies STTResponse,
      { status: 502 },
    );
  }
}

function requireEnv(name: string): string {
  const val = process.env[name];
  if (!val) throw new Error(`${name} is not configured`);
  return val;
}
