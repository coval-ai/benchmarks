"use client";

import { useEffect, useRef, useState } from "react";
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
    if (!autoPlay) return;
    if (side === "a") {
      setActive("b");
    } else {
      setAutoPlay(false);
      setActive(null);
    }
  };

  const trimmed = text.trim();
  const dirty = battle !== null && (trimmed !== battle.prompt || domain !== battleDomain);
  const canVote = battle !== null && !recorded && !dirty;

  return (
    <main className="min-h-screen bg-surface-primary px-6 pb-24 pt-32 text-text-primary">
      <div className="mx-auto flex max-w-[760px] flex-col gap-8">
        <h1 className="text-center font-sans text-2xl">Which voice sounds more natural?</h1>

        <section className="flex flex-col gap-3">
          <select
            value={domain}
            onChange={(e) => setDomain(e.target.value as ArenaDomain | "")}
            aria-label="Domain"
            className="w-full appearance-none rounded-xl border border-border-primary bg-surface-elevated px-4 py-3 font-sans text-sm outline-none focus:border-selected-border"
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
          <textarea
            value={text}
            maxLength={MAX_CHARS}
            onChange={(e) => setText(e.target.value)}
            placeholder="Describe a scenario or write text to synthesize…"
            rows={4}
            className="w-full resize-none rounded-xl border border-border-primary bg-surface-elevated p-4 font-sans text-base leading-relaxed outline-none focus:border-selected-border"
          />
          <div className="flex items-center justify-between text-sm text-text-tertiary">
            <button
              type="button"
              onClick={() => void applyExample()}
              className="font-sans underline underline-offset-2 hover:text-text-secondary"
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
          className="grid grid-cols-[1fr_44px_1fr] items-stretch"
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
                className="rounded-full border border-border-primary px-4 py-1.5 font-mono text-xs text-text-secondary hover:bg-hover-bg"
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
              <label className="flex cursor-pointer items-center gap-2 self-center font-mono text-xs text-text-secondary">
                <input
                  type="checkbox"
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
                  className="rounded-full bg-surface-toggle-active px-6 py-2.5 font-mono text-sm text-text-on-toggle-active"
                >
                  Another battle
                </button>
                <label className="flex cursor-pointer items-center gap-2 font-mono text-xs text-text-secondary">
                  <input
                    type="checkbox"
                    checked={autoAdvance}
                    onChange={() => persistAutoAdvance(!autoAdvance)}
                  />
                  Auto-advance
                </label>
                <a
                  href="/arena/leaderboard"
                  className="rounded-full border border-border-primary px-6 py-2.5 font-mono text-sm text-text-secondary hover:bg-hover-bg"
                >
                  View leaderboard
                </a>
                <a
                  href="/arena/admin"
                  className="rounded-full border border-border-primary px-6 py-2.5 font-mono text-sm text-text-secondary hover:bg-hover-bg"
                >
                  Monitoring
                </a>
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => void generate(text, domain)}
              disabled={trimmed.length < MIN_CHARS || !domain || loading}
              className="self-start rounded-full bg-surface-toggle-active px-6 py-2.5 font-mono text-sm text-text-on-toggle-active disabled:opacity-40"
            >
              {loading ? "Generating…" : "Generate speech"}
            </button>
          )}
          {error && <p className="text-center font-sans text-sm text-accent-rust">{error}</p>}
        </section>
      </div>
    </main>
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
          <span className="mt-1 h-2 w-2 shrink-0 self-start rounded-full bg-text-tertiary" />
          {revealed ? (
            <span className="flex flex-col leading-tight">
              <span className="font-sans text-sm text-text-primary">{revealed.model}</span>
              <span className="font-mono text-xs text-text-tertiary">{revealed.provider}</span>
            </span>
          ) : (
            <span className="font-sans text-sm">{blindTitle}</span>
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
      className="rounded-xl border border-border-primary bg-surface-elevated py-3 font-mono text-sm hover:border-selected-border hover:bg-selected-bg disabled:opacity-40"
    >
      {label}
    </button>
  );
}
