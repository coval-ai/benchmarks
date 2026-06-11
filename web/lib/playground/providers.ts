/**
 * Central allowlist for Playground TTS/STT models. Client code should send
 * stable `id` values only; server routes resolve provider-specific fields.
 *
 * Keep this list intentionally small and public: every enabled row must have a
 * server-side adapter in `web/app/api/**` and a matching server-only env var.
 *
 * This is a curated subset of runner-supported models, not a mirror. The parity
 * check in `providers.test.ts` asserts every model here exists in the runner's
 * `_VALID_MODELS`; the runner may support more models than the playground exposes.
 */

export type TtsProviderId =
  | "cartesia"
  | "elevenlabs"
  | "deepgram"
  | "rime"
  | "gradium";

export type SttProviderId =
  | "deepgram"
  | "assemblyai"
  | "elevenlabs"
  | "speechmatics";

export type TtsModelConfig = {
  id: string;
  provider: TtsProviderId;
  label: string;
  model: string;
  voice: string;
  enabled: boolean;
};

export type SttModelConfig = {
  id: string;
  provider: SttProviderId;
  label: string;
  model: string;
  /** Provider endpoint metadata for display/debugging; keys stay server-side. */
  url: string;
  sampleRate: number;
  enabled: boolean;
};

export const ttsModels: TtsModelConfig[] = [
  {
    id: "elevenlabs:eleven_flash_v2_5:default",
    provider: "elevenlabs",
    label: "Eleven Flash v2.5",
    model: "eleven_flash_v2_5",
    voice: "IKne3meq5aSn9XLyUdCD",
    enabled: true
  },
  {
    id: "elevenlabs:eleven_multilingual_v2:default",
    provider: "elevenlabs",
    label: "Eleven Multilingual v2",
    model: "eleven_multilingual_v2",
    voice: "IKne3meq5aSn9XLyUdCD",
    enabled: true
  },
  {
    id: "elevenlabs:eleven_turbo_v2_5:default",
    provider: "elevenlabs",
    label: "Eleven Turbo v2.5",
    model: "eleven_turbo_v2_5",
    voice: "IKne3meq5aSn9XLyUdCD",
    enabled: true
  },
  {
    id: "cartesia:sonic-3:default",
    provider: "cartesia",
    label: "Sonic 3",
    model: "sonic-3",
    voice: "f786b574-daa5-4673-aa0c-cbe3e8534c02",
    enabled: true
  },
  {
    id: "deepgram:aura-2-thalia-en:default",
    provider: "deepgram",
    label: "Aura 2 Thalia (en)",
    model: "aura-2-thalia-en",
    voice: "aura-2-thalia-en",
    enabled: true
  },
  {
    id: "rime:arcana:luna",
    provider: "rime",
    label: "Arcana",
    model: "arcana",
    voice: "luna",
    enabled: true
  },
  {
    id: "rime:mistv3:luna",
    provider: "rime",
    label: "Mist v3",
    model: "mistv3",
    voice: "luna",
    enabled: true
  },
  {
    id: "gradium:default:emma",
    provider: "gradium",
    label: "Default",
    model: "default",
    voice: "YTpq7expH9539ERJ",
    enabled: true
  }
];

export const sttModels: SttModelConfig[] = [
  {
    id: "deepgram:nova-2",
    provider: "deepgram",
    label: "Nova-2",
    model: "nova-2",
    url: "wss://api.deepgram.com/v1/listen?sample_rate=16000&encoding=linear16&channels=1&interim_results=true&vad_events=true&no_delay=true&model=nova-2",
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "deepgram:nova-3",
    provider: "deepgram",
    label: "Nova-3",
    model: "nova-3",
    url: "wss://api.deepgram.com/v1/listen?sample_rate=16000&encoding=linear16&channels=1&interim_results=true&vad_events=true&no_delay=true&model=nova-3",
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "deepgram:flux-general-en",
    provider: "deepgram",
    label: "Flux",
    model: "flux-general-en",
    url: "wss://api.deepgram.com/v2/listen?model=flux-general-en&sample_rate=16000&encoding=linear16",
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "elevenlabs:scribe_v2_realtime",
    provider: "elevenlabs",
    label: "Scribe v2 Realtime",
    model: "scribe_v2_realtime",
    url: "wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime",
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "assemblyai:universal-streaming",
    provider: "assemblyai",
    label: "AssemblyAI",
    model: "universal-streaming",
    url: "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&speech_model=universal-streaming-english",
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "speechmatics:default",
    provider: "speechmatics",
    label: "Speechmatics (default)",
    model: "default",
    url: "wss://wus.rt.speechmatics.com/v2",
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "speechmatics:enhanced",
    provider: "speechmatics",
    label: "Speechmatics (enhanced)",
    model: "enhanced",
    url: "wss://wus.rt.speechmatics.com/v2",
    sampleRate: 16_000,
    enabled: true
  }
];

export function getEnabledTtsModels(): TtsModelConfig[] {
  return ttsModels.filter((m) => m.enabled);
}

export function getEnabledSttModels(): SttModelConfig[] {
  return sttModels.filter((m) => m.enabled);
}

export function getTtsModelById(id: string): TtsModelConfig | undefined {
  return ttsModels.find((m) => m.id === id);
}

export function getSttModelById(id: string): SttModelConfig | undefined {
  return sttModels.find((m) => m.id === id);
}
