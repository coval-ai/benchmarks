"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import DashboardHeader from "@/components/layout/DashboardHeader";
import { CymaticLoader } from "@/components/shared/CymaticLoader";
import { ARENA_DOMAINS, type ArenaDomain } from "@/lib/arena/domains";
import { getBattleSource } from "@/lib/arena/source";
import type {
  BlindBattle,
  ExamplePrompt,
  Outcome,
  Reveal,
  RevealedModel,
} from "@/lib/arena/types";
import { AudioPlayer } from "./components/AudioPlayer";

const MIN_CHARS = 3;
const MAX_CHARS = 500;

// Brand focus state — the app-wide ring, so nothing here falls back to the
// browser's off-palette blue outline.
const FOCUS = "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-text-tertiary/40";
const FIELD =
  "w-full rounded-xl border border-border-primary bg-surface-elevated outline-none transition-colors focus:border-selected-border focus:ring-2 focus:ring-text-tertiary/40";
// b1: PP Mori Semibold, 14px, 1% letter spacing (500 is the heaviest PP Mori weight we self-host).
const BTN = `min-h-11 font-sans text-sm font-medium tracking-[0.01em] ${FOCUS}`;

// Post-vote only: standings shown before a vote anchor the judgment, and votes are what
// tightens the provisional CIs. Flip to false to pull the link entirely —
// /overview/arena/leaderboard stays reachable by URL either way.
const SHOW_LEADERBOARD_LINK = true;

export default function ArenaPage() {
  const source = getBattleSource();
  const voterId = useVoterId();

  const [text, setText] = useState("");
  const [domain, setDomain] = useState<ArenaDomain | "">("");
  const [battle, setBattle] = useState<BlindBattle | null>(null);
  const [battleDomain, setBattleDomain] = useState<ArenaDomain | "">("");
  const [vote, setVote] = useState<Outcome | null>(null);
  const [reveal, setReveal] = useState<Reveal | null>(null);
  const [recorded, setRecorded] = useState(false);
  const [autoAdvance, setAutoAdvance] = useState(false);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState<"a" | "b" | null>(null);
  const [autoPlay, setAutoPlay] = useState(false);
  const nextBattleRef = useRef<HTMLButtonElement>(null);
  const runToken = useRef(0);
  const autoAdvanceRef = useRef(autoAdvance);
  autoAdvanceRef.current = autoAdvance;

  useEffect(() => {
    try {
      setAutoAdvance(localStorage.getItem("arena_auto_advance") === "1");
    } catch {
      // Storage blocked (private mode, sandboxed iframe, etc.): session-only default.
    }
  }, []);

  useEffect(() => {
    if (recorded && !loading) nextBattleRef.current?.focus();
  }, [recorded, loading]);

  const persistAutoAdvance = (on: boolean) => {
    setAutoAdvance(on);
    try {
      localStorage.setItem("arena_auto_advance", on ? "1" : "0");
    } catch {
      // Storage blocked: the toggle still works for this session.
    }
  };

  const stopAutoAdvance = () => {
    runToken.current += 1; // drop the in-flight next battle
    if (battle) {
      // Realign the inputs with the battle still on screen — quickBattle may
      // have already written the aborted example into them.
      setText(battle.prompt);
      setDomain(battleDomain);
    }
    setLoading(false);
    persistAutoAdvance(false);
  };

  const generate = async (promptText: string, promptDomain: ArenaDomain | "") => {
    const trimmed = promptText.trim();
    if (trimmed.length < MIN_CHARS || !promptDomain || loading) return;
    const token = ++runToken.current;
    setLoading(true);
    setError(null);
    try {
      const b = await source.createBattle(trimmed, promptDomain);
      if (token !== runToken.current) return; // a newer action superseded this one
      setBattle(b);
      setBattleDomain(promptDomain);
      setVote(null);
      setReveal(null);
      setRecorded(false);
      // With auto-advance on, play the new battle hands-free: A starts now,
      // B takes over when A ends (see onEnded below).
      setActive(autoAdvanceRef.current ? "a" : null);
      setAutoPlay(autoAdvanceRef.current);
    } catch {
      if (token === runToken.current) setError("Couldn't generate audio. Please try again.");
    } finally {
      if (token === runToken.current) setLoading(false);
    }
  };

  const applyExample = async () => {
    setError(null);
    try {
      const example = await source.getExamplePrompt();
      setText(example.text);
      setDomain(example.domain);
    } catch {
      setError("Couldn't fetch an example. Please try again.");
    }
  };

  const quickBattle = async () => {
    const token = ++runToken.current;
    setError(null);
    let example: ExamplePrompt;
    try {
      example = await source.getExamplePrompt();
    } catch {
      if (token === runToken.current) setError("Couldn't fetch an example. Please try again.");
      return;
    }
    if (token !== runToken.current) return; // Stop (or a newer action) superseded this
    setText(example.text);
    setDomain(example.domain);
    void generate(example.text, example.domain);
  };

  const castVote = async (outcome: Outcome) => {
    if (!battle || vote || submitting) return;
    setSubmitting(true);
    setVote(outcome);
    setActive(null); // stop both players
    setAutoPlay(false);
    try {
      await source.submitVote({ battleId: battle.battleId, outcome, voterId });
      try {
        setReveal(await source.reveal(battle.battleId, voterId));
      } catch {
        setReveal(null); // vote is recorded; identities just unavailable
      }
      setRecorded(true);
      if (autoAdvanceRef.current) void quickBattle();
    } catch {
      setVote(null); // let them retry
      setError("Couldn't record your vote. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  const chainEnded = (side: "a" | "b") => {
    // Hand A off to B only while auto-playing. Every other ending releases focus, because
    // `active` also drives the per-card playing dot — leaving it set kept a card lit with
    // nothing playing once a manual listen finished.
    if (autoPlay && side === "a") {
      setActive("b");
      return;
    }
    setAutoPlay(false);
    setActive(null);
  };

  const trimmed = text.trim();
  const dirty = battle !== null && (trimmed !== battle.prompt || domain !== battleDomain);
  const canVote = battle !== null && !recorded && !dirty;

  return (
    <>
      <DashboardHeader />
      <main className="min-h-screen bg-surface-primary px-6 pb-24 pt-32 text-text-primary">
        <div className="mx-auto flex max-w-[760px] flex-col gap-8">
          <h1 className="text-center font-sans text-2xl font-medium tracking-tight sm:text-3xl">
            Which voice sounds more natural?
          </h1>

          <section className="flex flex-col gap-3">
            <div className="relative">
              <select
                value={domain}
                onChange={(e) => setDomain(e.target.value as ArenaDomain | "")}
                aria-label="Domain"
                className={`${FIELD} appearance-none px-4 py-3 pr-10 font-mono text-sm`}
              >
                <option value="" disabled>
                  Select a domain *
                </option>
                {ARENA_DOMAINS.map((d) => (
                  <option key={d.value} value={d.value}>
                    {d.label}
                  </option>
                ))}
              </select>
              <svg
                aria-hidden
                viewBox="0 0 12 12"
                className="pointer-events-none absolute right-4 top-1/2 h-3 w-3 -translate-y-1/2 text-text-tertiary"
              >
                <path
                  d="M2.5 4.5 6 8l3.5-3.5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <textarea
              value={text}
              maxLength={MAX_CHARS}
              onChange={(e) => setText(e.target.value)}
              placeholder="Describe a scenario or write text to synthesize…"
              rows={4}
              className={`${FIELD} resize-none p-4 font-sans text-base leading-relaxed`}
            />
            <div className="flex items-center justify-between text-sm text-text-tertiary">
              <button
                type="button"
                onClick={() => void applyExample()}
                className={`${BTN} flex items-center rounded-md underline underline-offset-2 hover:text-text-secondary`}
              >
                Use an example
              </button>
              <span className="font-mono">
                {text.length}/{MAX_CHARS}
              </span>
            </div>
          </section>

          <div
            key={battle?.battleId ?? "pending"}
            className="grid grid-cols-1 gap-2 sm:grid-cols-[1fr_44px_1fr] sm:gap-0 sm:items-stretch"
          >
            <BattleCard
              side="a"
              blindTitle="Model A"
              revealed={recorded && reveal ? reveal.a : null}
              picked={recorded ? vote : null}
              isActive={active === "a"}
              src={battle?.audioA ?? null}
              onActivate={() => setActive("a")}
              autoPlay={autoPlay}
              onEnded={() => chainEnded("a")}
            />
            <div className="flex items-center justify-center font-mono text-xs text-text-tertiary">
              VS
            </div>
            <BattleCard
              side="b"
              blindTitle="Model B"
              revealed={recorded && reveal ? reveal.b : null}
              picked={recorded ? vote : null}
              isActive={active === "b"}
              src={battle?.audioB ?? null}
              onActivate={() => setActive("b")}
              autoPlay={autoPlay}
              onEnded={() => chainEnded("b")}
            />
          </div>

          <section className="flex min-h-[52px] flex-col gap-3" aria-live="polite">
            {recorded && loading ? (
              <div className="flex items-center justify-center gap-3">
                <p className="font-mono text-sm text-text-secondary">
                  ✓ Recorded — loading the next battle…
                </p>
                <button
                  type="button"
                  onClick={stopAutoAdvance}
                  className={`${BTN} rounded-full border border-border-primary px-4 text-text-secondary hover:bg-hover-bg`}
                >
                  Stop
                </button>
              </div>
            ) : canVote && !loading ? (
              <>
                <div className="grid grid-cols-[1fr_0.7fr_1fr] gap-3">
                  <VoteButton label="Model A" onClick={() => castVote("A_WIN")} disabled={submitting} />
                  <VoteButton label="Tie" onClick={() => castVote("TIE")} disabled={submitting} />
                  <VoteButton label="Model B" onClick={() => castVote("B_WIN")} disabled={submitting} />
                </div>
                <label className="flex min-h-11 cursor-pointer items-center gap-2 self-center font-mono text-xs text-text-secondary">
                  <input
                    type="checkbox"
                    className={`h-5 w-5 accent-text-primary ${FOCUS}`}
                    checked={autoAdvance}
                    onChange={() => persistAutoAdvance(!autoAdvance)}
                  />
                  Auto-advance
                </label>
              </>
            ) : recorded && !dirty && !loading ? (
              <div className="flex flex-col items-center gap-4">
                <p className="font-mono text-sm text-text-secondary">✓ Battle recorded</p>
                <div className="flex flex-wrap items-center justify-center gap-3">
                  <button
                    ref={nextBattleRef}
                    type="button"
                    onClick={() => void quickBattle()}
                    className={`${BTN} rounded-full bg-surface-toggle-active px-6 text-text-on-toggle-active`}
                  >
                    Another battle
                  </button>
                  <label className="flex min-h-11 cursor-pointer items-center gap-2 font-mono text-xs text-text-secondary">
                    <input
                      type="checkbox"
                      className={`h-5 w-5 accent-text-primary ${FOCUS}`}
                      checked={autoAdvance}
                      onChange={() => persistAutoAdvance(!autoAdvance)}
                    />
                    Auto-advance
                  </label>
                  {SHOW_LEADERBOARD_LINK && (
                    <Link
                      href="/overview/arena/leaderboard"
                      className={`${BTN} flex items-center rounded-full border border-border-primary px-6 text-text-secondary hover:bg-hover-bg`}
                    >
                      View leaderboard
                    </Link>
                  )}
                </div>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => void generate(text, domain)}
                disabled={trimmed.length < MIN_CHARS || !domain || loading}
                className={`${BTN} inline-flex items-center gap-2 self-start rounded-full bg-surface-toggle-active px-6 text-text-on-toggle-active disabled:opacity-40`}
              >
                {loading && <CymaticLoader size={16} animated />}
                {loading ? "Generating…" : "Generate speech"}
              </button>
            )}
            {error && <p className="text-center font-sans text-sm text-accent-rust">{error}</p>}
          </section>
        </div>
      </main>
    </>
  );
}

function makeVoterId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `anon-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  }
}

function useVoterId(): string {
  const ref = useRef<string | null>(null);
  if (ref.current === null && typeof window !== "undefined") {
    try {
      const stored = localStorage.getItem("arena_voter_id");
      if (stored) {
        ref.current = stored;
      } else {
        const fresh = makeVoterId();
        localStorage.setItem("arena_voter_id", fresh);
        ref.current = fresh;
      }
    } catch {
      // Storage blocked (private mode, sandboxed iframe, etc.): ephemeral id.
      ref.current = makeVoterId();
    }
  }
  return ref.current ?? "";
}

function BattleCard({
  side,
  blindTitle,
  revealed,
  picked,
  isActive,
  src,
  onActivate,
  autoPlay,
  onEnded,
}: {
  side: "a" | "b";
  blindTitle: string;
  revealed: RevealedModel | null;
  picked: Outcome | null;
  isActive: boolean;
  src: string | null;
  onActivate: () => void;
  autoPlay: boolean;
  onEnded: () => void;
}) {
  const won = picked === (side === "a" ? "A_WIN" : "B_WIN");
  const tie = picked === "TIE";
  const highlighted = won || tie;
  return (
    <div
      className={`flex flex-col gap-4 rounded-xl border bg-surface-elevated p-5 ${
        highlighted ? "border-selected-border bg-selected-bg" : "border-border-primary"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="flex items-center gap-2">
          {/* Doubles as the playing indicator — the only cue outside the play glyph itself. */}
          <span
            className={`mt-1 h-2 w-2 shrink-0 self-start rounded-full ${
              isActive ? "bg-text-primary" : "bg-text-tertiary"
            }`}
          />
          {revealed ? (
            <span className="flex flex-col leading-tight">
              <span className="font-sans text-sm text-text-primary">{revealed.model}</span>
              <span className="font-mono text-xs text-text-tertiary">{revealed.provider}</span>
            </span>
          ) : (
            <span className="font-mono text-sm">{blindTitle}</span>
          )}
        </span>
        {picked !== null && (
          <span className="shrink-0 font-mono text-xs text-text-secondary">
            {won ? "YOUR PICK" : tie ? "TIE" : ""}
          </span>
        )}
      </div>
      <AudioPlayer
        src={src}
        label={`Model ${side.toUpperCase()}`}
        isActive={isActive}
        onActivate={onActivate}
        autoPlay={autoPlay}
        onEnded={onEnded}
      />
    </div>
  );
}

function VoteButton({
  label,
  onClick,
  disabled,
}: {
  label: string;
  onClick: () => void;
  disabled: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`${BTN} rounded-xl border border-border-primary bg-surface-elevated py-3 hover:border-selected-border hover:bg-selected-bg disabled:opacity-40`}
    >
      {label}
    </button>
  );
}
