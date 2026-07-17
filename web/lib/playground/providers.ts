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
  | "speechmatics"
  | "gradium";

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
    id: "cartesia:sonic-3.5:default",
    provider: "cartesia",
    label: "Sonic 3.5",
    model: "sonic-3.5",
    voice: "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
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
    label: "Gradium TTS",
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
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "deepgram:nova-3",
    provider: "deepgram",
    label: "Nova-3",
    model: "nova-3",
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "deepgram:flux-general-en",
    provider: "deepgram",
    label: "Flux",
    model: "flux-general-en",
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "elevenlabs:scribe_v2_realtime",
    provider: "elevenlabs",
    label: "Scribe v2 Realtime",
    model: "scribe_v2_realtime",
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "assemblyai:universal-streaming",
    provider: "assemblyai",
    label: "AssemblyAI",
    model: "universal-streaming",
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "speechmatics:default",
    provider: "speechmatics",
    label: "Speechmatics (default)",
    model: "default",
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "speechmatics:enhanced",
    provider: "speechmatics",
    label: "Speechmatics (enhanced)",
    model: "enhanced",
    sampleRate: 16_000,
    enabled: true
  },
  {
    id: "gradium:default",
    provider: "gradium",
    label: "Gradium STT",
    model: "default",
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
