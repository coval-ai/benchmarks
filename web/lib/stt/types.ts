export type STTSuccess = {
  modelId: string;
  transcript: string;
  ttfaMs: number | null;
  audioToFinalMs: number;
};

export type STTError = {
  modelId: string;
  error: string;
  code:
    | "INVALID_MODEL"
    | "MISSING_AUDIO"
    | "PROVIDER_ERROR"
    | "RATE_LIMITED"
    | "FORBIDDEN"
    | "VALIDATION_ERROR"
    | "AUDIO_TOO_LARGE";
};

export type STTResponse = STTSuccess | STTError;

export function isSTTError(r: STTResponse): r is STTError {
  return "error" in r;
}
