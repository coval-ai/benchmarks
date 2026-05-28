import WebSocket from "@/lib/stt/ws";

const WS_URL =
  "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&speech_model=universal-streaming-english";
const CHUNK_BYTES = 8_192;
const SESSION_TIMEOUT_MS = 55_000;

export type AssemblyAIResult = {
  transcript: string;
  ttfaMs: number | null;
  audioToFinalMs: number;
};

export function callAssemblyAI(pcm: ArrayBuffer, apiKey: string): Promise<AssemblyAIResult> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(WS_URL, { headers: { Authorization: apiKey } });
    const buf = Buffer.from(pcm);
    const turns: string[] = [];
    let settled = false;
    let t0 = 0;
    let ttfaMs: number | null = null;
    let gotTermination = false;

    const timer = setTimeout(() => {
      ws.close();
      settle(() => reject(new Error("AssemblyAI: timeout")));
    }, SESSION_TIMEOUT_MS);

    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn();
    };

    ws.on("open", () => {
      t0 = performance.now();
      for (let i = 0; i < buf.length; i += CHUNK_BYTES) {
        ws.send(buf.subarray(i, i + CHUNK_BYTES));
      }
      ws.send(JSON.stringify({ type: "Terminate" }));
    });

    ws.on("message", (raw) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(raw.toString()) as Record<string, unknown>;
      } catch {
        return;
      }
      if (msg["type"] === "Turn") {
        const text = extractText(msg);
        if (text) {
          if (ttfaMs === null && t0 > 0) ttfaMs = Math.round(performance.now() - t0);
          if (msg["end_of_turn"] === true) turns.push(text);
        }
      } else if (msg["type"] === "Termination") {
        gotTermination = true;
        const audioToFinalMs = t0 > 0 ? Math.round(performance.now() - t0) : 0;
        settle(() => resolve({ transcript: turns.join(" ").trim(), ttfaMs, audioToFinalMs }));
      }
    });

    ws.on("close", () =>
      settle(() => {
        if (t0 === 0) {
          reject(new Error("AssemblyAI: connection closed before audio send"));
          return;
        }
        if (!gotTermination) {
          reject(new Error("AssemblyAI: connection closed before Termination"));
          return;
        }
      }),
    );
    ws.on("error", (err) => settle(() => reject(err)));
  });
}

function extractText(msg: Record<string, unknown>): string {
  if (typeof msg["transcript"] === "string" && msg["transcript"].trim()) {
    return msg["transcript"].trim();
  }
  const words = msg["words"];
  if (!Array.isArray(words)) return "";
  return words
    .map((w: unknown) => {
      if (!w || typeof w !== "object") return "";
      const t = (w as Record<string, unknown>)["text"];
      return typeof t === "string" ? t.trim() : "";
    })
    .filter(Boolean)
    .join(" ");
}
