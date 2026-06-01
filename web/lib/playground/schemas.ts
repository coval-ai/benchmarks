// ---------------------------------------------------------------------------
// STT WebSocket message types (server → browser)
// ---------------------------------------------------------------------------

/** Partial or final transcript forwarded by the STT proxy. */
export type STTTranscriptMessage = {
  type: "partial" | "final";
  modelId: string;
  text: string;
  /** Server-side monotonic ms — for relative sequencing only, not absolute latency. */
  timestampMs: number;
};

/** One provider failed to connect before the session started. */
export type STTProviderFailedMessage = {
  type: "PROVIDER_CONNECTION_FAILED";
  modelId: string;
  reason: string;
};

/** A previously-live provider disconnected mid-session. */
export type STTProviderDisconnectedMessage = {
  type: "PROVIDER_DISCONNECTED";
  modelId: string;
};

export type STTServerMessage =
  | STTTranscriptMessage
  | STTProviderFailedMessage
  | STTProviderDisconnectedMessage;

// ---------------------------------------------------------------------------
// TTS / generic playground error types
// ---------------------------------------------------------------------------

export type PlaygroundErrorCode =
  | "INVALID_MODEL"
  | "TEXT_EMPTY"
  | "TEXT_TOO_LONG"
  | "VALIDATION_ERROR"
  | "RATE_LIMITED"
  | "FORBIDDEN"
  | "UPSTREAM_ERROR";

export type PlaygroundApiError = {
  error: string;
  code: PlaygroundErrorCode;
};

/** Some APIs wrap error payloads as ``{ "detail": <payload> }``. */
function extractPlaygroundErrorPayload(body: unknown): { error?: string; code: string } | null {
  if (!body || typeof body !== "object") return null;
  const b = body as Record<string, unknown>;

  const fromObject = (o: Record<string, unknown>): { error?: string; code: string } | null => {
    const code = o.code;
    if (typeof code !== "string") return null;
    const err = o.error;
    return { code, error: typeof err === "string" ? err : undefined };
  };

  const top = fromObject(b);
  if (top) return top;

  const d = b.detail;
  if (d && typeof d === "object" && !Array.isArray(d)) {
    return fromObject(d as Record<string, unknown>);
  }

  return null;
}

function isTrustedPlaygroundCode(code: string): code is PlaygroundErrorCode {
  return (
    code === "INVALID_MODEL" ||
    code === "TEXT_EMPTY" ||
    code === "TEXT_TOO_LONG" ||
    code === "VALIDATION_ERROR" ||
    code === "UPSTREAM_ERROR" ||
    code === "RATE_LIMITED" ||
    code === "FORBIDDEN"
  );
}

export function normalizePlaygroundError(status: number, body: unknown): PlaygroundApiError {
  if (status === 429) {
    return { error: "Rate limited. Please try again shortly.", code: "RATE_LIMITED" };
  }

  const extracted = extractPlaygroundErrorPayload(body);
  if (extracted && isTrustedPlaygroundCode(extracted.code)) {
    return { code: extracted.code, error: extracted.error ?? "Request failed." };
  }

  if (status === 422) {
    return { error: "Request failed validation.", code: "VALIDATION_ERROR" };
  }

  return { error: "Upstream request failed.", code: "UPSTREAM_ERROR" };
}
