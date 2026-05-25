"use client";

import { useCallback, useRef, useState } from "react";
import { startAudioCapture, type AudioCaptureHandle } from "@/lib/playground/audio-capture";
import { getEnabledSttModels } from "@/lib/playground/providers";
import { isSTTError, type STTResponse } from "@/lib/stt/types";

export type BenchmarkPhase = "idle" | "recording" | "submitting" | "complete" | "error";

export type ModelResult = {
  transcript: string;
  ttftMs: number | null;
  audioToFinalMs: number;
};

export type STTBenchmarkSession = {
  phase: BenchmarkPhase;
  audioDurationMs: number | null;
  results: Map<string, ModelResult>;
  errors: Map<string, string>;
  sessionError: string | null;
  start: (modelIds: string[]) => Promise<void>;
  stop: () => void;
  reset: () => void;
};

export type STTBenchmarkCompletionSummary = {
  audioDurationMs: number;
  modelIds: string[];
  successes: Array<{
    modelId: string;
    ttftMs: number | null;
    audioToFinalMs: number;
  }>;
  failures: Array<{
    modelId: string;
    errorMessage: string;
  }>;
};

type STTBenchmarkOutcome =
  | {
      status: "success";
      modelId: string;
      ttftMs: number | null;
      audioToFinalMs: number;
    }
  | {
      status: "error";
      modelId: string;
      errorMessage: string;
    };

const SAMPLE_RATE = 16_000;
const MAX_DURATION_MS = 60_000;

export function useSTTBenchmark(options?: {
  onComplete?: (summary: STTBenchmarkCompletionSummary) => void;
}): STTBenchmarkSession {
  const onComplete = options?.onComplete;
  const [phase, setPhase] = useState<BenchmarkPhase>("idle");
  const [audioDurationMs, setAudioDurationMs] = useState<number | null>(null);
  const [results, setResults] = useState<Map<string, ModelResult>>(new Map());
  const [errors, setErrors] = useState<Map<string, string>>(new Map());
  const [sessionError, setSessionError] = useState<string | null>(null);

  const captureRef = useRef<AudioCaptureHandle | null>(null);
  const chunksRef = useRef<ArrayBuffer[]>([]);
  const modelIdsRef = useRef<string[]>([]);
  const autoStopRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const startingRef = useRef(false);
  const stopRequestedRef = useRef(false);

  const submitPcm = useCallback(async (chunks: ArrayBuffer[]) => {
    setPhase("submitting");

    // Concatenate all Int16 chunks from the AudioWorklet into one buffer
    const totalBytes = chunks.reduce((sum, b) => sum + b.byteLength, 0);
    const combined = new Uint8Array(totalBytes);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(new Uint8Array(chunk), offset);
      offset += chunk.byteLength;
    }

    // Derive duration from sample count — more reliable than wall-clock timing
    const durMs = Math.round(((totalBytes / 2) / SAMPLE_RATE) * 1000);
    setAudioDurationMs(durMs);

    if (totalBytes === 0 || durMs <= 0) {
      setSessionError("No audio captured.");
      setPhase("error");
      return;
    }

    const pcmBlob = new Blob([combined], { type: "audio/l16" });
    const allowedModelIds = new Set(getEnabledSttModels().map((m) => m.id));
    const modelIds = modelIdsRef.current.filter((id) => allowedModelIds.has(id));

    if (modelIds.length === 0) {
      setSessionError("Select at least one model.");
      setPhase("error");
      return;
    }

    const outcomes = await Promise.all(
      modelIds.map(async (modelId) => {
        const form = new FormData();
        form.set("modelId", modelId);
        form.set("audio", pcmBlob, "audio.pcm");
        form.set("audioDurationMs", String(durMs));

        // Abort slightly after the server's maxDuration so the client doesn't hang
        const controller = new AbortController();
        const abortTimer = setTimeout(() => controller.abort(), 65_000);

        try {
          const res = await fetch("/api/stt/benchmark", {
            method: "POST",
            body: form,
            signal: controller.signal
          });
          clearTimeout(abortTimer);

          if (!res.ok) {
            // Route always returns JSON errors; Vercel infrastructure errors (504) return HTML
            const errData = await res.json().catch(() => null) as STTResponse | null;
            const message =
              errData && isSTTError(errData) ? errData.error : `Request failed (HTTP ${res.status})`;
            setErrors((prev) => new Map(prev).set(modelId, message));
            return {
              status: "error" as const,
              modelId,
              errorMessage: message
            };
          }

          const data = (await res.json()) as STTResponse;
          if (isSTTError(data)) {
            setErrors((prev) => new Map(prev).set(modelId, data.error));
            return {
              status: "error" as const,
              modelId,
              errorMessage: data.error
            };
          } else {
            const transcript = data.transcript.trim() || "No transcript returned.";
            setResults((prev) =>
              new Map(prev).set(modelId, {
                transcript,
                ttftMs: data.ttftMs,
                audioToFinalMs: data.audioToFinalMs,
              }),
            );
            return {
              status: "success" as const,
              modelId,
              ttftMs: data.ttftMs,
              audioToFinalMs: data.audioToFinalMs,
            };
          }
        } catch (err) {
          clearTimeout(abortTimer);
          const message =
            err instanceof Error && err.name === "AbortError"
              ? "Request timed out"
              : "Network error";
          setErrors((prev) => new Map(prev).set(modelId, message));
          return {
            status: "error" as const,
            modelId,
            errorMessage: message
          };
        }
      }),
    );

    const successes = outcomes.filter(
      (outcome): outcome is Extract<STTBenchmarkOutcome, { status: "success" }> =>
        outcome.status === "success"
    );
    const failures = outcomes.filter(
      (outcome): outcome is Extract<STTBenchmarkOutcome, { status: "error" }> =>
        outcome.status === "error"
    );

    onComplete?.({
      audioDurationMs: durMs,
      modelIds,
      successes,
      failures
    });

    setPhase("complete");
  }, [onComplete]);

  const stop = useCallback(() => {
    if (autoStopRef.current) {
      clearTimeout(autoStopRef.current);
      autoStopRef.current = null;
    }
    if (!captureRef.current) {
      // Enter can be released before getUserMedia/AudioWorklet startup resolves.
      if (startingRef.current) stopRequestedRef.current = true;
      return;
    }
    stopRequestedRef.current = false;
    captureRef.current.stop();
    captureRef.current = null;
    void submitPcm(chunksRef.current);
  }, [submitPcm]);

  const start = useCallback(async (modelIds: string[]) => {
    if (captureRef.current || startingRef.current) return;
    stopRequestedRef.current = false;
    startingRef.current = true;
    chunksRef.current = [];
    modelIdsRef.current = modelIds;
    setPhase("recording");
    setResults(new Map());
    setErrors(new Map());
    setSessionError(null);
    setAudioDurationMs(null);

    try {
      captureRef.current = await startAudioCapture((chunk) => {
        chunksRef.current.push(chunk);
      });
    } catch (err) {
      startingRef.current = false;
      stopRequestedRef.current = false;
      setSessionError(err instanceof Error ? err.message : "Microphone access denied.");
      setPhase("error");
      return;
    }
    startingRef.current = false;

    // Hard cap — stop automatically at 60 s
    autoStopRef.current = setTimeout(stop, MAX_DURATION_MS);
    if (stopRequestedRef.current) stop();
  }, [stop]);

  const reset = useCallback(() => {
    if (autoStopRef.current) {
      clearTimeout(autoStopRef.current);
      autoStopRef.current = null;
    }
    startingRef.current = false;
    stopRequestedRef.current = false;
    captureRef.current?.stop();
    captureRef.current = null;
    chunksRef.current = [];
    modelIdsRef.current = [];
    setPhase("idle");
    setResults(new Map());
    setErrors(new Map());
    setSessionError(null);
    setAudioDurationMs(null);
  }, []);

  return { phase, audioDurationMs, results, errors, sessionError, start, stop, reset };
}
