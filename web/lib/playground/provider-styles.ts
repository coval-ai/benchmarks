import type { SttModelConfig, TtsModelConfig } from "./providers";
import { getModelColor, getProviderColor } from "@/lib/utils/colors";

export type ProviderVisual = {
  /** Swatch / dot — same keying as benchmark charts (`modelColors`) */
  dot: string;
  border: string;
  /** Provider label — same as benchmark `providerColors` */
  nameColor: string;
};

function hexToRgba(hex: string, alpha: number): string {
  const normalized = hex.replace("#", "");
  const isValidHex6 = /^[0-9a-fA-F]{6}$/.test(normalized);
  if (!isValidHex6) {
    if (process.env.NODE_ENV === "development") {
      console.warn("[getPlaygroundModelVisual] expected #RRGGBB model hex, got:", hex);
    }
    return `rgba(107, 114, 128, ${alpha})`;
  }
  const r = Number.parseInt(normalized.slice(0, 2), 16);
  const g = Number.parseInt(normalized.slice(2, 4), 16);
  const b = Number.parseInt(normalized.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

/** Canonical provider display strings for pills, modals, and benchmark-aligned labels. */
const PROVIDER_DISPLAY: Record<string, string> = {
  elevenlabs: "ElevenLabs",
  cartesia: "Cartesia",
  deepgram: "Deepgram",
  rime: "Rime",
  assemblyai: "AssemblyAI",
  speechmatics: "Speechmatics"
};

/** User-facing copy when the map is missing an id (prod only — never leak raw internal ids). */
export const PLAYGROUND_PROVIDER_LABEL_FALLBACK_PROD = "Provider";

/**
 * Human-readable provider label for playground UI. Keys must stay in lockstep with
 * `TtsProviderId` / `SttProviderId` in `providers.ts` — add the id here in the **same PR** as a new
 * provider (no runtime “auto” generation; CI can later assert every allowlisted id exists here).
 *
 * - **Development:** `console.error` + label `[UNMAPPED: …]` so QA immediately sees the gap.
 * - **Production:** generic `PLAYGROUND_PROVIDER_LABEL_FALLBACK_PROD` so end users never see raw ids.
 */
export function formatProviderDisplayName(providerId: string): string {
  if (providerId in PROVIDER_DISPLAY) {
    return PROVIDER_DISPLAY[providerId as keyof typeof PROVIDER_DISPLAY]!;
  }

  const msg =
    `[Playground] UNMAPPED provider id "${providerId}". ` +
    `Add it to PROVIDER_DISPLAY in provider-styles.ts in the same change set as providers/config.`;

  if (process.env.NODE_ENV === "development") {
    console.error(msg);
    return `[UNMAPPED: ${providerId}]`;
  }

  return PLAYGROUND_PROVIDER_LABEL_FALLBACK_PROD;
}

/** Per-model colors match `web/lib/config/colors.ts` + `getModelColor` / `getProviderColor`. */
export function getPlaygroundModelVisual(m: TtsModelConfig | SttModelConfig): ProviderVisual {
  const dot = getModelColor(m.model);
  const nameColor = getProviderColor(formatProviderDisplayName(m.provider));
  return {
    dot,
    border: hexToRgba(dot, 0.55),
    nameColor
  };
}

/**
 * Short label for **pills and tag-cloud sort only**. Does not change `m.id`, `m.model`, `m.voice`,
 * or any field the runner/API uses — those stay exactly as in `providers.ts`.
 */
export function getTtsPillLabel(m: TtsModelConfig): string {
  if (m.provider === "deepgram" && m.label.includes("Thalia")) {
    return "Aura 2 Thalia";
  }
  return m.label;
}

/** Display-only; `m.id` / URLs / `model` are unchanged for STT backends. */
export function getSttPillLabel(m: SttModelConfig): string {
  if (m.id === "speechmatics:enhanced") return "Speechmatics+";
  if (m.id === "speechmatics:default") return "Speechmatics";
  if (m.id === "elevenlabs:scribe_v2_realtime") return "Scribe v2 Realtime";
  return m.label;
}
