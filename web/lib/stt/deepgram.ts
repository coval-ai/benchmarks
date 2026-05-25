import WebSocket from "@/lib/stt/ws";

// 16 kHz × 16-bit × 1 channel × 0.1s = 3200 bytes. Same for nova and flux —
// both endpoints accept linear16/16k/mono streamed in 100ms PCM frames.
const DEEPGRAM_PCM_CHUNK_BYTES = 3_200;

export type DeepgramResult = {
  transcript: string;
  ttftMs: number | null;
  audioToFinalMs: number;
};

export async function callDeepgram(
  pcm: ArrayBuffer,
  model: string,
  apiKey: string,
): Promise<DeepgramResult> {
  if (model === "flux-general-en") {
    return callDeepgramFlux(pcm, apiKey);
  }
  // nova-2 / nova-3 — v1 WebSocket protocol, mirrors runner's
  // DeepgramProvider._build_websocket_url so playground TTFT/audio-to-final
  // numbers are comparable with the benchmark dashboard.
  return callDeepgramNova(pcm, model, apiKey);
}

function callDeepgramNova(
  pcm: ArrayBuffer,
  model: string,
  apiKey: string,
): Promise<DeepgramResult> {
  const params = new URLSearchParams({
    sample_rate: "16000",
    encoding: "linear16",
    channels: "1",
    interim_results: "true",
    vad_events: "true",
    no_delay: "true",
    model,
  });
  const url = `wss://api.deepgram.com/v1/listen?${params.toString()}`;
  const buf = Buffer.from(pcm);

  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url, {
      headers: { Authorization: `Token ${apiKey}` },
    });

    let settled = false;
    let sentAudio = false;
    let t0 = 0;
    let ttftMs: number | null = null;
    const finalSegments: string[] = [];
    let lastInterim = "";
    let lastFinalTime = 0;

    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      fn();
    };

    const timer = setTimeout(() => {
      ws.close();
      settle(() => reject(new Error("Deepgram Nova: timeout")));
    }, 55_000);
    const clearTimer = () => clearTimeout(timer);

    ws.on("open", () => {
      t0 = performance.now();
      sentAudio = true;
      for (let i = 0; i < buf.length; i += DEEPGRAM_PCM_CHUNK_BYTES) {
        const slice = buf.subarray(i, Math.min(i + DEEPGRAM_PCM_CHUNK_BYTES, buf.length));
        if (slice.length > 0) ws.send(slice);
      }
      ws.send(JSON.stringify({ type: "CloseStream" }));
    });

    ws.on("message", (raw) => {
      const text = Buffer.isBuffer(raw) ? raw.toString("utf8") : String(raw);
      if (!text.trimStart().startsWith("{")) return;

      let msg: {
        type?: string;
        description?: string;
        speech_final?: boolean;
        is_final?: boolean;
        transcript?: string;
        channel?: {
          alternatives?: Array<{
            transcript?: string;
            words?: Array<{ word?: string; punctuated_word?: string }>;
          }>;
        };
      };
      try {
        msg = JSON.parse(text) as typeof msg;
      } catch {
        return;
      }

      if (msg.type === "Error") {
        settle(() => {
          clearTimer();
          ws.close();
          reject(new Error(`Deepgram Nova: ${msg.description ?? "error"}`));
        });
        return;
      }

      if (msg.type !== "Results") return;

      const transcript = extractDeepgramTranscript(msg);
      if (!transcript) return;

      if (ttftMs === null && t0 > 0) ttftMs = Math.round(performance.now() - t0);

      if (msg.speech_final) {
        finalSegments.push(transcript);
        lastFinalTime = performance.now();
      } else {
        lastInterim = transcript;
      }
    });

    ws.on("close", () => {
      clearTimer();
      settle(() => {
        if (!sentAudio) {
          reject(new Error("Deepgram Nova: connection closed before audio send"));
          return;
        }
        const transcript =
          finalSegments.length > 0 ? finalSegments.join(" ").trim() : lastInterim.trim();
        const audioToFinalMs = lastFinalTime > 0
          ? Math.round(lastFinalTime - t0)
          : (t0 > 0 ? Math.round(performance.now() - t0) : 0);
        resolve({ transcript, ttftMs, audioToFinalMs });
      });
    });

    ws.on("error", (err) => {
      clearTimer();
      settle(() => reject(err));
    });
  });
}

function callDeepgramFlux(pcm: ArrayBuffer, apiKey: string): Promise<DeepgramResult> {
  const params = new URLSearchParams({
    model: "flux-general-en",
    encoding: "linear16",
    sample_rate: "16000",
  });
  const url = `wss://api.deepgram.com/v2/listen?${params.toString()}`;
  const buf = Buffer.from(pcm);

  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url, {
      headers: { Authorization: `Token ${apiKey}` },
    });

    let settled = false;
    let connectedSeen = false;
    let sentAudio = false;
    let t0 = 0;
    let ttftMs: number | null = null;
    const endTurnTexts: string[] = [];
    let lastTranscript = "";

    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      fn();
    };

    const timer = setTimeout(() => {
      ws.close();
      settle(() => reject(new Error("Deepgram Flux: timeout")));
    }, 55_000);

    const clearTimer = () => clearTimeout(timer);

    ws.on("message", (raw) => {
      const text = Buffer.isBuffer(raw) ? raw.toString("utf8") : String(raw);
      if (!text.trimStart().startsWith("{")) return;

      let msg: {
        type?: string;
        event?: string;
        transcript?: string;
        description?: string;
        code?: string;
        channel?: {
          alternatives?: Array<{
            transcript?: string;
            words?: Array<{ word?: string; punctuated_word?: string }>;
          }>;
        };
      };
      try {
        msg = JSON.parse(text) as typeof msg;
      } catch {
        return;
      }

      switch (msg.type) {
        case "Connected": {
          if (sentAudio) return;
          sentAudio = true;
          connectedSeen = true;
          t0 = performance.now();
          for (let i = 0; i < buf.length; i += DEEPGRAM_PCM_CHUNK_BYTES) {
            const slice = buf.subarray(i, Math.min(i + DEEPGRAM_PCM_CHUNK_BYTES, buf.length));
            if (slice.length > 0) ws.send(slice);
          }
          ws.send(JSON.stringify({ type: "CloseStream" }));
          return;
        }
        case "TurnInfo": {
          const transcript = extractDeepgramTranscript(msg);
          if (transcript) {
            if (ttftMs === null && t0 > 0) ttftMs = Math.round(performance.now() - t0);
            lastTranscript = transcript;
          }
          if (msg.event === "EndOfTurn" && transcript) {
            endTurnTexts.push(transcript);
          }
          return;
        }
        case "Results": {
          const transcript = extractDeepgramTranscript(msg);
          if (transcript) {
            if (ttftMs === null && t0 > 0) ttftMs = Math.round(performance.now() - t0);
            lastTranscript = transcript;
          }
          return;
        }
        case "Error": {
          settle(() => {
            clearTimer();
            ws.close();
            reject(new Error(`Deepgram Flux: ${msg.description ?? msg.code ?? "error"}`));
          });
          return;
        }
        default:
          return;
      }
    });

    ws.on("close", () => {
      clearTimer();
      settle(() => {
        if (!connectedSeen) {
          reject(new Error("Deepgram Flux: connection closed before Connected (check API key)"));
          return;
        }
        const transcript =
          endTurnTexts.length > 0 ? endTurnTexts.join(" ").trim() : lastTranscript.trim();
        const audioToFinalMs = t0 > 0 ? Math.round(performance.now() - t0) : 0;
        resolve({ transcript, ttftMs, audioToFinalMs });
      });
    });

    ws.on("error", (err) => {
      clearTimer();
      settle(() => reject(err));
    });
  });
}

function extractDeepgramTranscript(msg: {
  transcript?: string;
  channel?: {
    alternatives?: Array<{
      transcript?: string;
      words?: Array<{ word?: string; punctuated_word?: string }>;
    }>;
  };
}): string {
  if (typeof msg.transcript === "string" && msg.transcript.trim()) {
    return msg.transcript.trim();
  }
  const alt = msg.channel?.alternatives?.[0];
  if (!alt) return "";
  const words = alt.words ?? [];
  const wordText = words
    .map((w) => (w.punctuated_word ?? w.word ?? "").trim())
    .filter(Boolean)
    .join(" ");
  return wordText || (alt.transcript ?? "").trim();
}
