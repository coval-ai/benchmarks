# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed the ``arena`` schema with sample battles + votes for local development.

A throwaway fixture (replaced by real audio generation later) that unblocks the
API/UI/rating work. Four models, all six pairings (a connected comparison
graph), and a deterministic vote pattern that yields a clear latent ranking.

Run against a LOCAL dev database only::

    DATABASE_URL=postgresql://postgres:postgres@localhost:5432/benchmarks \\
        python -m coval_bench.db.seed_arena [--reset]
"""

from __future__ import annotations

import argparse
import asyncio
from urllib.parse import urlsplit

from coval_bench.config import get_settings
from coval_bench.db.arena_store import ArenaPool, ArenaStore
from coval_bench.db.conn import lifespan_pool
from coval_bench.db.models import Battle, VoteOutcome, VoterType

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_MODELS: list[tuple[str, str]] = [
    ("cartesia", "sonic-3.5"),
    ("elevenlabs", "eleven_flash_v2_5"),
    ("openai", "gpt-4o-mini-tts"),
    ("deepgram", "aura-2"),
]

_DOMAINS: list[str] = ["general", "support", "narration"]

_PROMPTS: list[str] = [
    "Your appointment is confirmed for Tuesday at 9 AM.",
    "I'm sorry, I didn't quite catch that — could you repeat it?",
    "The package was delivered to the front desk this afternoon.",
    "Let me pull up your account; this will just take a moment.",
]

# Seven votes per battle. Model A is the lower-indexed (stronger) model, so it
# wins most: a clear latent ranking and a connected graph for the rating engine.
_VOTE_PATTERN: list[VoteOutcome] = [
    VoteOutcome.A_WIN,
    VoteOutcome.A_WIN,
    VoteOutcome.A_WIN,
    VoteOutcome.A_WIN,
    VoteOutcome.A_WIN,
    VoteOutcome.B_WIN,
    VoteOutcome.TIE,
]


def _assert_local(database_url: str) -> None:
    """Refuse to seed/reset against anything but a local DB.

    Prod connects via a Cloud SQL unix socket (empty host), so requiring an
    explicit local host blocks prod while allowing ``…@localhost:5432/…``.
    """
    host = urlsplit(database_url).hostname
    if host not in _LOCAL_HOSTS:
        raise SystemExit(
            f"refusing to seed/reset arena: DB host {host!r} is not local "
            f"({sorted(_LOCAL_HOSTS)}). This script is dev-only."
        )


def _sample_url(provider: str, model: str) -> str:
    return f"https://samples.coval.dev/arena/{provider}-{model}.wav"


async def _clear(pool: ArenaPool) -> None:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM arena.leaderboard_snapshots")
            await cur.execute("DELETE FROM arena.votes")
            await cur.execute("DELETE FROM arena.battles")
        await conn.commit()


async def run_seed(*, reset: bool = False) -> None:
    """Insert the sample battles + votes. Skips if already seeded (unless reset)."""
    settings = get_settings()
    _assert_local(settings.database_url)
    async with lifespan_pool(settings) as pool:
        store = ArenaStore(pool)

        if reset:
            await _clear(pool)
        elif await store.list_battles(limit=1):
            print("arena already seeded — skipping (pass --reset to reseed)")
            return

        pairs = [(i, j) for i in range(len(_MODELS)) for j in range(i + 1, len(_MODELS))]

        battle_count = 0
        vote_count = 0
        for idx, (i, j) in enumerate(pairs):
            provider_a, model_a = _MODELS[i]
            provider_b, model_b = _MODELS[j]
            battle = await store.insert_battle(
                Battle(
                    provider_a=provider_a,
                    model_a=model_a,
                    provider_b=provider_b,
                    model_b=model_b,
                    domain=_DOMAINS[idx % len(_DOMAINS)],
                    prompt_text=_PROMPTS[idx % len(_PROMPTS)],
                    audio_a_url=_sample_url(provider_a, model_a),
                    audio_b_url=_sample_url(provider_b, model_b),
                )
            )
            battle_count += 1
            if battle.id is None:  # pragma: no cover — id is set by RETURNING
                raise RuntimeError("insert_battle returned a battle without an id")

            for vote_idx, outcome in enumerate(_VOTE_PATTERN):
                await store.upsert_vote(
                    battle_id=battle.id,
                    outcome=outcome,
                    voter_type=VoterType.LABELER,
                    voter_id=f"labeler-{vote_idx + 1}",
                )
                vote_count += 1

        print(f"seeded {battle_count} battles, {vote_count} votes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the local arena schema with sample battles and votes."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete existing arena rows before seeding",
    )
    args = parser.parse_args()
    asyncio.run(run_seed(reset=args.reset))


if __name__ == "__main__":
    main()
