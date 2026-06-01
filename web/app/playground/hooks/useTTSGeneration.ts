"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { postPlaygroundTts, PlaygroundTtsError } from "@/lib/api/client";
import type { PlaygroundApiError } from "@/lib/playground/schemas";

export type TtsGenerationStatus = "idle" | "loading" | "success" | "error";

export type TtsGenerationRow = {
  status: TtsGenerationStatus;
  audioUrl: string | null;
  ttfaMs: number | null;
  totalMs: number | null;
  error: PlaygroundApiError | null;
};

const IDLE_ROW: TtsGenerationRow = {
  status: "idle",
  audioUrl: null,
  ttfaMs: null,
  totalMs: null,
  error: null,
};

export type TtsRunSummary = {
  overallDurationMs: number;
  successes: Array<{
    modelId: string;
    ttfaMs: number | null;
    totalMs: number;
  }>;
  failures: Array<{
    modelId: string;
    errorCode: PlaygroundApiError["code"];
    errorMessage: string;
  }>;
};

type TtsRunOutcome =
  | {
      status: "success";
      modelId: string;
      ttfaMs: number | null;
      totalMs: number;
    }
  | {
      status: "error";
      modelId: string;
      errorCode: PlaygroundApiError["code"];
      errorMessage: string;
    }
  | null;

export function useTTSGeneration() {
  const [rows, setRows] = useState<Record<string, TtsGenerationRow>>({});
  const controllersRef = useRef<Record<string, AbortController>>({});
  const runTokenRef = useRef(0);

  const setLoading = useCallback((modelId: string) => {
    setRows((prev) => {
      const prevRow = prev[modelId] ?? IDLE_ROW;
      return {
        ...prev,
        [modelId]: { ...prevRow, status: "loading", error: null },
      };
    });
  }, []);

  const runBenchmark = useCallback(async (text: string, modelIds: string[]): Promise<TtsRunSummary | null> => {
    const trimmed = text.trim();
    if (!trimmed || modelIds.length === 0) return null;
    const myToken = ++runTokenRef.current;
    const startedAt = performance.now();

    const outcomes = await Promise.all(
      modelIds.map(async (modelId) => {
        const existing = controllersRef.current[modelId];
        existing?.abort();
        const controller = new AbortController();
        controllersRef.current[modelId] = controller;
        setLoading(modelId);
        try {
          const response = await postPlaygroundTts(
            { modelId, text: trimmed },
            { signal: controller.signal }
          );
          if (runTokenRef.current !== myToken) return null;
          const url = URL.createObjectURL(response.audioBlob);
          setRows((prev) => {
            const previousUrl = prev[modelId]?.audioUrl;
            if (previousUrl) URL.revokeObjectURL(previousUrl);
            return {
              ...prev,
              [modelId]: {
                status: "success",
                audioUrl: url,
                ttfaMs: response.ttfaMs,
                totalMs: response.totalMs,
                error: null,
              },
            };
          });
          return {
            status: "success" as const,
            modelId,
            ttfaMs: response.ttfaMs,
            totalMs: response.totalMs,
          };
        } catch (error) {
          if ((error as Error).name === "AbortError") {
            return null;
          }
          if (runTokenRef.current !== myToken) return null;
          const normalized: PlaygroundApiError =
            error instanceof PlaygroundTtsError
              ? error.payload
              : { code: "UPSTREAM_ERROR", error: "Failed to generate audio." };
          setRows((prev) => ({
            ...prev,
            [modelId]: {
              ...(prev[modelId] ?? IDLE_ROW),
              status: "error",
              error: normalized,
            },
          }));
          return {
            status: "error" as const,
            modelId,
            errorCode: normalized.code,
            errorMessage: normalized.error
          };
        } finally {
          if (controllersRef.current[modelId] === controller) {
            delete controllersRef.current[modelId];
          }
        }
      })
    );

    if (runTokenRef.current !== myToken) return null;

    const successes = outcomes.filter(
      (outcome): outcome is Extract<Exclude<TtsRunOutcome, null>, { status: "success" }> =>
        outcome?.status === "success"
    );
    const failures = outcomes.filter(
      (outcome): outcome is Extract<Exclude<TtsRunOutcome, null>, { status: "error" }> =>
        outcome?.status === "error"
    );

    return {
      overallDurationMs: performance.now() - startedAt,
      successes,
      failures
    };
  }, [setLoading]);

  const cancelAll = useCallback(() => {
    runTokenRef.current += 1;
    for (const controller of Object.values(controllersRef.current)) controller.abort();
    controllersRef.current = {};
  }, []);

  const resetRows = useCallback(() => {
    setRows((prev) => {
      for (const row of Object.values(prev)) {
        if (row.audioUrl) URL.revokeObjectURL(row.audioUrl);
      }
      return {};
    });
  }, []);

  const hasInFlight = Object.values(rows).some((row) => row.status === "loading");

  const rowsRef = useRef(rows);
  rowsRef.current = rows;
  useEffect(() => {
    return () => {
      runTokenRef.current += 1;
      for (const controller of Object.values(controllersRef.current)) controller.abort();
      controllersRef.current = {};
      for (const row of Object.values(rowsRef.current)) {
        if (row.audioUrl) URL.revokeObjectURL(row.audioUrl);
      }
    };
  }, []);

  return { rows, runBenchmark, cancelAll, resetRows, hasInFlight };
}
