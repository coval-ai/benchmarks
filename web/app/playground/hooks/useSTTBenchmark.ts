"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { startAudioCapture, type AudioCaptureHandle } from "@/lib/playground/audio-capture";
import { getEnabledSttModels } from "@/lib/playground/providers";
import { isSTTError, type STTResponse } from "@/lib/stt/types";

export type BenchmarkPhase = "idle" | "recording" | "submitting" | "complete" | "error";

export type ModelResult = {
  transcript: string;
  ttfaMs: number | null;
  audioToFinalMs: number;
};

export type STTBenchmarkSession = {
  phase: BenchmarkPhase;
  analyser: AnalyserNode | null;
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
    ttfaMs: number | null;
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
      ttfaMs: number | null;
      audioToFinalMs: number;
    }
  | {
      status: "error";
      modelId: string;
      errorMessage: string;
    };

const SAMPLE_RATE = 16_000;
const MAX_DURATION_MS = 50_000;

export function useSTTBenchmark(options?: {
  onComplete?: (summary: STTBenchmarkCompletionSummary) => void;
}): STTBenchmarkSession {
  const onComplete = options?.onComplete;
  const [phase, setPhase] = useState<BenchmarkPhase>("idle");
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
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
  const controllerRef = useRef<AbortController | null>(null);
  const startTokenRef = useRef(0);

  useEffect(() => {
    const startToken = startTokenRef;
    return () => {
      startToken.current++;
      if (autoStopRef.current) clearTimeout(autoStopRef.current);
      controllerRef.current?.abort();
      captureRef.current?.stop();
    };
  }, []);

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

    const form = new FormData();
    form.set("audio", pcmBlob, "audio.pcm");
    form.set("modelIds", JSON.stringify(modelIds));
    form.set("audioDurationMs", String(durMs));

    const controller = new AbortController();
    controllerRef.current = controller;
    const abortTimer = setTimeout(() => controller.abort(), 65_000);

    let batch: { results: STTResponse[] } | null = null;
    let topLevelError: string | null = null;
    try {
      const res = await fetch("/api/playground/stt", {
        method: "POST",
        body: form,
        signal: controller.signal,
      });
      clearTimeout(abortTimer);

      if (!res.ok) {
        const errData = (await res.json().catch(() => null)) as { error?: string } | null;
        topLevelError = errData?.error ?? `Request failed (HTTP ${res.status})`;
      } else {
        batch = (await res.json()) as { results: STTResponse[] };
      }
    } catch (err) {
      clearTimeout(abortTimer);
      topLevelError =
        err instanceof Error && err.name === "AbortError" ? "Request timed out" : "Network error";
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
    }

    let outcomes: STTBenchmarkOutcome[];
    if (topLevelError) {
      outcomes = modelIds.map((modelId) => ({
        status: "error" as const,
        modelId,
        errorMessage: topLevelError,
      }));
      setErrors((prev) => {
        const next = new Map(prev);
        for (const id of modelIds) next.set(id, topLevelError);
        return next;
      });
    } else {
      const results = batch?.results ?? [];
      outcomes = results.map((data) => {
        if (isSTTError(data)) {
          setErrors((prev) => new Map(prev).set(data.modelId, data.error));
          return {
            status: "error" as const,
            modelId: data.modelId,
            errorMessage: data.error,
          };
        }
        const transcript = data.transcript.trim() || "No transcript returned.";
        setResults((prev) =>
          new Map(prev).set(data.modelId, {
            transcript,
            ttfaMs: data.ttfaMs,
            audioToFinalMs: data.audioToFinalMs,
          }),
        );
        return {
          status: "success" as const,
          modelId: data.modelId,
          ttfaMs: data.ttfaMs,
          audioToFinalMs: data.audioToFinalMs,
        };
      });

      const received = new Set(results.map((r) => r.modelId));
      const missing = modelIds.filter((id) => !received.has(id));
      if (missing.length > 0) {
        const message = "Provider response missing.";
        setErrors((prev) => {
          const next = new Map(prev);
          for (const id of missing) next.set(id, message);
          return next;
        });
        outcomes.push(
          ...missing.map((id) => ({
            status: "error" as const,
            modelId: id,
            errorMessage: message,
          })),
        );
      }
    }

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
    setAnalyser(null);
    void submitPcm(chunksRef.current);
  }, [submitPcm]);

  const start = useCallback(async (modelIds: string[]) => {
    if (captureRef.current || startingRef.current) return;
    const token = ++startTokenRef.current;
    stopRequestedRef.current = false;
    startingRef.current = true;
    chunksRef.current = [];
    modelIdsRef.current = modelIds;
    setPhase("recording");
    setResults(new Map());
    setErrors(new Map());
    setSessionError(null);
    setAudioDurationMs(null);

    let handle: AudioCaptureHandle;
    try {
      handle = await startAudioCapture((chunk) => {
        chunksRef.current.push(chunk);
      });
    } catch (err) {
      if (startTokenRef.current !== token) return;
      startingRef.current = false;
      stopRequestedRef.current = false;
      setSessionError(err instanceof Error ? err.message : "Microphone access denied.");
      setPhase("error");
      return;
    }

    if (startTokenRef.current !== token) {
      handle.stop();
      return;
    }
    captureRef.current = handle;
    startingRef.current = false;
    setAnalyser(handle.analyser);

    // Hard cap — stop automatically at MAX_DURATION_MS
    autoStopRef.current = setTimeout(stop, MAX_DURATION_MS);
    if (stopRequestedRef.current) stop();
  }, [stop]);

  const reset = useCallback(() => {
    startTokenRef.current++;
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
    setAnalyser(null);
    setPhase("idle");
    setResults(new Map());
    setErrors(new Map());
    setSessionError(null);
    setAudioDurationMs(null);
  }, []);

  return { phase, analyser, audioDurationMs, results, errors, sessionError, start, stop, reset };
}
