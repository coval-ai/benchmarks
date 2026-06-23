"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getBattleSource } from "@/lib/arena/source";
import type { BlindBattle, Outcome, Reveal, RevealedModel } from "@/lib/arena/types";
import { AudioPlayer } from "./components/AudioPlayer";

const MIN_CHARS = 3;
const MAX_CHARS = 500;
const EXAMPLES = [
  "Thanks for calling — I can help you with that refund right away.",
  "The northern lights danced across the sky in ribbons of green and violet.",
  "Honestly? I'd grab the earlier train. Traffic downtown is brutal today.",
];

export default function ArenaPage() {
  const source = getBattleSource();
  const voterId = useVoterId();

  const [step, setStep] = useState<1 | 2 | 4>(1);
  const [text, setText] = useState("");
  const [battle, setBattle] = useState<BlindBattle | null>(null);
  const [vote, setVote] = useState<Outcome | null>(null);
  const [reveal, setReveal] = useState<Reveal | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState<"a" | "b" | null>(null);
  const newBattleRef = useRef<HTMLButtonElement>(null);
  const runToken = useRef(0);

  useEffect(() => {
    if (step === 4) newBattleRef.current?.focus();
  }, [step]);

  const reset = useCallback(() => {
    runToken.current += 1; // invalidate any in-flight createBattle
    setStep(1);
    setText("");
    setBattle(null);
    setVote(null);
    setReveal(null);
    setError(null);
    setActive(null);
  }, []);

  const generate = async () => {
    const trimmed = text.trim();
    if (trimmed.length < MIN_CHARS || loading) return;
    const token = ++runToken.current;
    setLoading(true);
    setError(null);
    try {
      const b = await source.createBattle(trimmed);
      if (token !== runToken.current) return; // a newer action superseded this one
      setBattle(b);
      setStep(2);
    } catch {
      if (token === runToken.current) setError("Couldn't generate audio. Please try again.");
    } finally {
      if (token === runToken.current) setLoading(false);
    }
  };

  const castVote = async (outcome: Outcome) => {
    if (!battle || vote || submitting) return;
    setSubmitting(true);
    setVote(outcome);
    setActive(null); // stop both players
    try {
      await source.submitVote({ battleId: battle.battleId, outcome, voterId });
      try {
        setReveal(await source.reveal(battle.battleId));
      } catch {
        setReveal(null); // vote is recorded; identities just unavailable
      }
      setStep(4);
    } catch {
      setVote(null); // let them retry
      setError("Couldn't record your vote. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="min-h-screen bg-surface-primary px-6 pb-24 pt-32 text-text-primary">
      <div className="mx-auto flex max-w-[760px] flex-col gap-8">
        <h1 className="text-center font-sans text-2xl">Which voice sounds more natural?</h1>

        <div
          key={battle?.battleId ?? "pending"}
          className="grid grid-cols-[1fr_44px_1fr] items-stretch"
        >
          <BattleCard
            side="a"
            blindTitle="Model A"
            revealed={step === 4 && reveal ? reveal.a : null}
            picked={step === 4 ? vote : null}
            isActive={active === "a"}
            src={battle?.audioA ?? null}
            onActivate={() => setActive("a")}
          />
          <div className="flex items-center justify-center font-mono text-xs text-text-tertiary">
            VS
          </div>
          <BattleCard
            side="b"
            blindTitle="Model B"
            revealed={step === 4 && reveal ? reveal.b : null}
            picked={step === 4 ? vote : null}
            isActive={active === "b"}
            src={battle?.audioB ?? null}
            onActivate={() => setActive("b")}
          />
        </div>

        {step === 1 && (
          <section className="flex flex-col gap-3">
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
                onClick={() => setText(EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)] ?? "")}
                className="font-sans underline underline-offset-2 hover:text-text-secondary"
              >
                Use an example
              </button>
              <span className="font-mono">
                {text.length}/{MAX_CHARS}
              </span>
            </div>
            <button
              type="button"
              onClick={generate}
              disabled={text.trim().length < MIN_CHARS || loading}
              className="self-start rounded-full bg-surface-toggle-active px-6 py-2.5 font-mono text-sm text-text-on-toggle-active disabled:opacity-40"
            >
              {loading ? "Generating…" : "Generate speech"}
            </button>
            {error && <p className="font-sans text-sm text-accent-rust">{error}</p>}
          </section>
        )}

        {battle && step !== 4 && (
          <section className="flex flex-col gap-3">
            <div className="grid grid-cols-[1fr_0.7fr_1fr] gap-3">
              <VoteButton label="Model A" onClick={() => castVote("A_WIN")} disabled={submitting} />
              <VoteButton label="Tie" onClick={() => castVote("TIE")} disabled={submitting} />
              <VoteButton label="Model B" onClick={() => castVote("B_WIN")} disabled={submitting} />
            </div>
            {error && <p className="text-center font-sans text-sm text-accent-rust">{error}</p>}
          </section>
        )}

        {step === 4 && (
          <div className="flex flex-col items-center gap-4" aria-live="polite">
            <p className="font-mono text-sm text-text-secondary">✓ Battle recorded</p>
            <div className="flex gap-3">
              <button
                ref={newBattleRef}
                type="button"
                onClick={reset}
                className="rounded-full bg-surface-toggle-active px-6 py-2.5 font-mono text-sm text-text-on-toggle-active"
              >
                New battle
              </button>
              <a
                href="/arena/leaderboard"
                className="rounded-full border border-border-primary px-6 py-2.5 font-mono text-sm text-text-secondary hover:bg-hover-bg"
              >
                View leaderboard
              </a>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

function useVoterId(): string {
  const [id, setId] = useState("");
  useEffect(() => {
    let existing = localStorage.getItem("arena_voter_id");
    if (!existing) {
      existing = crypto.randomUUID();
      localStorage.setItem("arena_voter_id", existing);
    }
    setId(existing);
  }, []);
  return id;
}

function BattleCard({
  side,
  blindTitle,
  revealed,
  picked,
  isActive,
  src,
  onActivate,
}: {
  side: "a" | "b";
  blindTitle: string;
  revealed: RevealedModel | null;
  picked: Outcome | null;
  isActive: boolean;
  src: string | null;
  onActivate: () => void;
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
