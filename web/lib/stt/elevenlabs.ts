import WebSocket from "@/lib/stt/ws";

const WS_URL =
  "wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime";
const CHUNK_BYTES = 8_192;
const SESSION_TIMEOUT_MS = 55_000;

const _ERROR_TYPES = new Set([
  "scribe_error",
  "scribe_auth_error",
  "scribe_quota_exceeded_error",
  "scribe_throttled_error",
  "scribe_unaccepted_terms_error",
  "scribe_rate_limited_error",
  "scribe_queue_overflow_error",
  "scribe_resource_exhausted_error",
  "scribe_session_time_limit_exceeded_error",
  "scribe_input_error",
  "scribe_chunk_size_exceeded_error",
  "scribe_insufficient_audio_activity_error",
  "scribe_transcriber_error",
]);

export type ElevenLabsResult = {
  transcript: string;
  ttftMs: number | null;
  audioToFinalMs: number;
};

export function callElevenLabs(
  pcm: ArrayBuffer,
  apiKey: string,
): Promise<ElevenLabsResult> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(WS_URL, { headers: { "xi-api-key": apiKey } });
    const buf = Buffer.from(pcm);
    let settled = false;
    let sessionReady = false;
    let audioSent = false;
    let t0 = 0;
    let ttftMs: number | null = null;
    let lastCommitTime = 0;
    const committed: string[] = [];
    let lastPartial = "";

    const handshakeTimer = setTimeout(() => {
      ws.close();
      settle(() => reject(new Error("ElevenLabs: session_started timeout")));
    }, 5_000);

    const sessionTimer = setTimeout(() => {
      ws.close();
      settle(() => reject(new Error("ElevenLabs: timeout")));
    }, SESSION_TIMEOUT_MS);

    const clearTimers = () => {
      clearTimeout(handshakeTimer);
      clearTimeout(sessionTimer);
    };

    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      clearTimers();
      fn();
    };

    const clearHandshakeTimer = () => clearTimeout(handshakeTimer);

    const sendAudioAndCommit = () => {
      t0 = performance.now();
      let isFirstChunk = true;
      for (let i = 0; i < buf.length; i += CHUNK_BYTES) {
        const slice = buf.subarray(i, Math.min(i + CHUNK_BYTES, buf.length));
        if (slice.length === 0) continue;
        const b64 = slice.toString("base64");
        const message: {
          message_type: "input_audio_chunk";
          audio_base_64: string;
          sample_rate?: number;
        } = { message_type: "input_audio_chunk", audio_base_64: b64 };
        if (isFirstChunk) {
          message.sample_rate = 16_000;
          isFirstChunk = false;
        }
        ws.send(JSON.stringify(message));
      }
      ws.send(
        JSON.stringify({
          message_type: "input_audio_chunk",
          audio_base_64: "",
          commit: true,
        }),
      );
      audioSent = true;
    };

    ws.on("message", (raw) => {
      const text = Buffer.isBuffer(raw) ? raw.toString("utf8") : String(raw);
      let msg: {
        message_type?: string;
        text?: string;
        message?: string;
      };
      try {
        msg = JSON.parse(text) as typeof msg;
      } catch {
        return;
      }
      const mt = msg.message_type ?? "";
      if (_ERROR_TYPES.has(mt)) {
        clearHandshakeTimer();
        settle(() => {
          ws.close();
          reject(new Error(`ElevenLabs: ${mt}${msg.message ? `: ${msg.message}` : ""}`));
        });
        return;
      }
      if (mt === "session_started") {
        clearHandshakeTimer();
        sessionReady = true;
        sendAudioAndCommit();
        return;
      }
      if (mt === "partial_transcript") {
        const p = typeof msg.text === "string" ? msg.text.trim() : "";
        if (p) {
          if (ttftMs === null && t0 > 0) ttftMs = Math.round(performance.now() - t0);
          lastPartial = p;
        }
        return;
      }
      if (mt === "committed_transcript" || mt === "committed_transcript_with_timestamps") {
        const t = typeof msg.text === "string" ? msg.text.trim() : "";
        if (t && !committed.includes(t)) {
          committed.push(t);
          lastCommitTime = performance.now();
        }
      }
    });

    ws.on("close", () => {
      clearHandshakeTimer();
      settle(() => {
        if (!sessionReady) {
          reject(new Error("ElevenLabs: connection closed before session_started"));
          return;
        }
        const joined = committed.join(" ").trim();
        const transcript = joined || lastPartial;
        const audioToFinalMs = lastCommitTime > 0
          ? Math.round(lastCommitTime - t0)
          : (audioSent && t0 > 0 ? Math.round(performance.now() - t0) : 0);
        resolve({ transcript, ttftMs, audioToFinalMs });
      });
    });

    ws.on("error", (err) => {
      clearHandshakeTimer();
      settle(() => reject(err));
    });
  });
}

