import WebSocket from "@/lib/stt/ws";

// Mirrors the runner's providers/stt/gradium.py:
// setup (pcm_16000) → ready → base64 audio messages → flush → settle → end_of_stream.
// The flush forces the model's lookahead buffer (~10 × 80 ms frames) to emit;
// FLUSH_SETTLE_MS gives the server time to send the flushed text before we close.
const WS_URL = "wss://api.gradium.ai/api/speech/asr";
const PCM_CHUNK_BYTES = 3_200; // 16 kHz × 16-bit × mono × 0.1 s
const READY_TIMEOUT_MS = 10_000;
const SESSION_TIMEOUT_MS = 55_000;
const FLUSH_SETTLE_MS = 2_000;

export type GradiumResult = {
  transcript: string;
  ttfaMs: number | null;
  audioToFinalMs: number;
};

export function callGradium(
  pcm: ArrayBuffer,
  model: string,
  apiKey: string,
): Promise<GradiumResult> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(WS_URL, { headers: { "x-api-key": apiKey } });

    const buf = Buffer.from(pcm);
    const texts: string[] = [];
    let t0 = 0;
    let ttfaMs: number | null = null;
    let lastTextTime = 0;
    let settled = false;
    let ready = false;
    let gotEos = false;
    let eosTimer: ReturnType<typeof setTimeout> | undefined;

    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(readyTimer);
      clearTimeout(sessionTimer);
      clearTimeout(eosTimer);
      fn();
    };

    const readyTimer = setTimeout(() => {
      ws.close();
      settle(() => reject(new Error("Gradium: ready timeout")));
    }, READY_TIMEOUT_MS);

    const sessionTimer = setTimeout(() => {
      ws.close();
      settle(() => reject(new Error("Gradium: timeout")));
    }, SESSION_TIMEOUT_MS);

    ws.on("open", () => {
      // Unlike the runner, the TTFA clock starts at ws open (handshake included)
      // so timings are comparable with the other playground adapters.
      t0 = performance.now();
      ws.send(
        JSON.stringify({
          type: "setup",
          model_name: model,
          input_format: "pcm_16000",
          json_config: JSON.stringify({ language: "en" }),
        }),
      );
    });

    ws.on("message", (raw) => {
      let msg: { type?: string; text?: string; message?: string };
      try {
        msg = JSON.parse(raw.toString()) as typeof msg;
      } catch {
        return;
      }

      if (msg.type === "ready") {
        ready = true;
        clearTimeout(readyTimer);
        for (let i = 0; i < buf.length; i += PCM_CHUNK_BYTES) {
          ws.send(
            JSON.stringify({
              type: "audio",
              audio: buf.subarray(i, i + PCM_CHUNK_BYTES).toString("base64"),
            }),
          );
        }
        ws.send(JSON.stringify({ type: "flush", flush_id: 1 }));
        eosTimer = setTimeout(() => {
          // No-op callback: the socket may be closing when the timer fires,
          // and ws emits an error event for callback-less sends in that state.
          ws.send(JSON.stringify({ type: "end_of_stream" }), () => {});
        }, FLUSH_SETTLE_MS);
        return;
      }

      if (msg.type === "text") {
        const text = (msg.text ?? "").trim();
        if (text) {
          if (ttfaMs === null && t0 > 0) ttfaMs = Math.round(performance.now() - t0);
          texts.push(text);
          lastTextTime = performance.now();
        }
        return;
      }

      if (msg.type === "end_of_stream") {
        gotEos = true;
        ws.close();
        settle(resolveWithTranscript);
        return;
      }

      if (msg.type === "error") {
        settle(() => reject(new Error(`Gradium: ${msg.message ?? "error"}`)));
        ws.close();
      }
    });

    // The server's end_of_stream echo is just an early-exit signal (same as the
    // runner): a close after flushed text still carries a usable transcript.
    function resolveWithTranscript() {
      const audioToFinalMs =
        lastTextTime > 0
          ? Math.round(lastTextTime - t0)
          : t0 > 0
            ? Math.round(performance.now() - t0)
            : 0;
      resolve({ transcript: texts.join(" ").trim(), ttfaMs, audioToFinalMs });
    }

    ws.on("close", () => {
      settle(() => {
        if (!ready) {
          reject(new Error("Gradium: connection closed before ready"));
          return;
        }
        if (!gotEos && texts.length === 0) {
          reject(new Error("Gradium: connection closed before end_of_stream"));
          return;
        }
        resolveWithTranscript();
      });
    });

    ws.on("error", (err) => settle(() => reject(err)));
  });
}
