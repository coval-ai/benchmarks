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
const MAX_PCM_BYTES = 60 * 16_000 * 2;
const MAX_MODELS_PER_REQUEST = 16;

type BatchError = { error: string; code: string };
type BatchSuccess = { results: STTResponse[] };

export async function POST(req: Request): Promise<Response> {
  if (!isAllowedOrigin(req.headers.get("origin"))) {
    return Response.json(
      { error: "Forbidden.", code: "FORBIDDEN" } satisfies BatchError,
      { status: 403 },
    );
  }
  const ip = getClientIp(req);
  if (!tryAcquireSession(ip)) {
    return Response.json(
      { error: "Too many concurrent sessions.", code: "RATE_LIMITED" } satisfies BatchError,
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
  const rawLen = req.headers.get("content-length");
  const contentLength = rawLen != null ? Number(rawLen) : NaN;
  if (!Number.isFinite(contentLength) || contentLength < 0 || contentLength > MAX_PCM_BYTES + 4_096) {
    return Response.json(
      { error: "Audio exceeds max duration.", code: "AUDIO_TOO_LARGE" } satisfies BatchError,
      { status: 413 },
    );
  }

  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return Response.json(
      { error: "Invalid multipart form data.", code: "VALIDATION_ERROR" } satisfies BatchError,
      { status: 400 },
    );
  }

  const modelIdsRaw = form.get("modelIds");
  if (typeof modelIdsRaw !== "string" || !modelIdsRaw) {
    return Response.json(
      { error: "modelIds field is required.", code: "VALIDATION_ERROR" } satisfies BatchError,
      { status: 400 },
    );
  }
  let modelIds: string[];
  try {
    const parsed = JSON.parse(modelIdsRaw);
    if (!Array.isArray(parsed) || parsed.some((m: unknown) => typeof m !== "string")) {
      throw new Error("not a string array");
    }
    modelIds = parsed as string[];
  } catch {
    return Response.json(
      { error: "modelIds must be a JSON string array.", code: "VALIDATION_ERROR" } satisfies BatchError,
      { status: 400 },
    );
  }
  if (modelIds.length === 0) {
    return Response.json(
      { error: "modelIds cannot be empty.", code: "VALIDATION_ERROR" } satisfies BatchError,
      { status: 400 },
    );
  }
  if (modelIds.length > MAX_MODELS_PER_REQUEST) {
    return Response.json(
      { error: "Too many models per request.", code: "VALIDATION_ERROR" } satisfies BatchError,
      { status: 400 },
    );
  }
  for (const id of modelIds) {
    if (!ALLOWED_MODELS.has(id)) {
      return Response.json(
        { error: `Unknown or disabled model: ${id}`, code: "INVALID_MODEL" } satisfies BatchError,
        { status: 400 },
      );
    }
  }

  const audioPart = form.get("audio");
  if (!(audioPart instanceof Blob)) {
    return Response.json(
      { error: "Missing audio field.", code: "MISSING_AUDIO" } satisfies BatchError,
      { status: 400 },
    );
  }
  if (audioPart.size > MAX_PCM_BYTES) {
    return Response.json(
      { error: "Audio exceeds max duration.", code: "AUDIO_TOO_LARGE" } satisfies BatchError,
      { status: 413 },
    );
  }

  // One click = one quota item, regardless of how many models were selected.
  if (!tryConsumeDailyQuota(ip)) {
    return Response.json(
      { error: "Daily quota exceeded.", code: "RATE_LIMITED" } satisfies BatchError,
      { status: 429 },
    );
  }

  const pcm = await audioPart.arrayBuffer();
  const results = await Promise.all(modelIds.map((id) => transcribeOne(id, pcm)));

  return Response.json({ results } satisfies BatchSuccess);
}

async function transcribeOne(modelId: string, pcm: ArrayBuffer): Promise<STTResponse> {
  const sep = modelId.indexOf(":");
  const provider = modelId.slice(0, sep);
  const model = modelId.slice(sep + 1);
  try {
    let r: { transcript: string; ttfaMs: number | null; audioToFinalMs: number };
    switch (provider) {
      case "speechmatics":
        r = await callSpeechmatics(
          pcm,
          model as "default" | "enhanced",
          requireEnv("PLAYGROUND_SPEECHMATICS_API_KEY"),
        );
        break;
      case "deepgram":
        r = await callDeepgram(pcm, model, requireEnv("PLAYGROUND_DEEPGRAM_API_KEY"));
        break;
      case "elevenlabs":
        r = await callElevenLabs(pcm, requireEnv("PLAYGROUND_ELEVENLABS_API_KEY"));
        break;
      case "assemblyai":
        r = await callAssemblyAI(pcm, requireEnv("PLAYGROUND_ASSEMBLYAI_API_KEY"));
        break;
      default:
        return { modelId, error: "Unhandled provider.", code: "INVALID_MODEL" };
    }
    return {
      modelId,
      transcript: r.transcript,
      ttfaMs: r.ttfaMs,
      audioToFinalMs: r.audioToFinalMs,
    };
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Provider error";
    console.error("[playground/stt] provider error", { modelId, detail });
    return { modelId, error: "Transcription failed.", code: "PROVIDER_ERROR" };
  }
}

function requireEnv(name: string): string {
  const val = process.env[name];
  if (!val) throw new Error(`${name} is not configured`);
  return val;
}
