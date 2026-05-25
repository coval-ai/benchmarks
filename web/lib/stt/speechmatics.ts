import WebSocket from "@/lib/stt/ws";

const WS_URL = "wss://wus.rt.speechmatics.com/v2";
const CHUNK_BYTES = 8_192;
const RECOGNITION_START_TIMEOUT_MS = 10_000;
const SESSION_TIMEOUT_MS = 55_000;

export type SpeechmaticsResult = {
  transcript: string;
  ttftMs: number | null;
  audioToFinalMs: number;
};

export function callSpeechmatics(
  pcm: ArrayBuffer,
  model: "default" | "enhanced",
  apiKey: string,
): Promise<SpeechmaticsResult> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(WS_URL, {
      headers: { Authorization: `Bearer ${apiKey}` },
    });

    const buf = Buffer.from(pcm);
    const finals: string[] = [];
    let seqNo = 0;
    let t0 = 0;
    let ttftMs: number | null = null;
    let settled = false;
    let recognitionStarted = false;
    let gotEndOfTranscript = false;

    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(startTimer);
      clearTimeout(sessionTimer);
      fn();
    };

    const startTimer = setTimeout(() => {
      ws.close();
      settle(() => reject(new Error("Speechmatics: RecognitionStarted timeout")));
    }, RECOGNITION_START_TIMEOUT_MS);

    const sessionTimer = setTimeout(() => {
      ws.close();
      settle(() => reject(new Error("Speechmatics: timeout")));
    }, SESSION_TIMEOUT_MS);

    ws.on("open", () => {
      ws.send(
        JSON.stringify({
          message: "StartRecognition",
          transcription_config: {
            language: "en",
            enable_partials: false,
            operating_point: model === "enhanced" ? "enhanced" : "standard",
          },
          audio_format: {
            type: "raw",
            encoding: "pcm_s16le",
            sample_rate: 16_000,
          },
        }),
      );
    });

    ws.on("message", (raw) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(raw.toString()) as Record<string, unknown>;
      } catch {
        return;
      }

      const type = typeof msg["message"] === "string" ? msg["message"] : "";

      if (type === "RecognitionStarted") {
        recognitionStarted = true;
        clearTimeout(startTimer);
        t0 = performance.now();
        for (let i = 0; i < buf.length; i += CHUNK_BYTES) {
          ws.send(buf.subarray(i, i + CHUNK_BYTES));
          seqNo++;
        }
        ws.send(JSON.stringify({ message: "EndOfStream", last_seq_no: seqNo }));
        return;
      }

      if (type === "AddTranscript") {
        const text = extractText(msg);
        if (text) {
          if (ttftMs === null && t0 > 0) ttftMs = Math.round(performance.now() - t0);
          finals.push(text);
        }
        return;
      }

      if (type === "EndOfTranscript") {
        gotEndOfTranscript = true;
        const audioToFinalMs = t0 > 0 ? Math.round(performance.now() - t0) : 0;
        ws.close();
        settle(() => resolve({ transcript: finals.join(" ").trim(), ttftMs, audioToFinalMs }));
        return;
      }

      if (type === "Error") {
        const reason =
          typeof msg["reason"] === "string" ? msg["reason"] : "unknown error";
        settle(() => reject(new Error(`Speechmatics: ${reason}`)));
        ws.close();
      }
    });

    ws.on("close", () => {
      settle(() => {
        if (!recognitionStarted) {
          reject(new Error("Speechmatics: connection closed before RecognitionStarted"));
          return;
        }
        if (!gotEndOfTranscript) {
          reject(new Error("Speechmatics: connection closed before EndOfTranscript"));
          return;
        }
      });
    });

    ws.on("error", (err) => settle(() => reject(err)));
  });
}

function extractText(msg: Record<string, unknown>): string {
  const results = msg["results"];
  if (!Array.isArray(results)) return "";
  return results
    .flatMap((r: unknown) => {
      if (!r || typeof r !== "object") return [];
      const alts = (r as Record<string, unknown>)["alternatives"];
      return Array.isArray(alts) ? alts : [];
    })
    .map((a: unknown) => {
      if (!a || typeof a !== "object") return "";
      const content = (a as Record<string, unknown>)["content"];
      return typeof content === "string" ? content.trim() : "";
    })
    .filter(Boolean)
    .join(" ")
    .trim();
}
