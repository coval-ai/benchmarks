# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""JudgeJev: word errors that would change how a voice agent understands the speaker.

The normalizer and alignment WER uses find every place the transcript differs
from the reference. TypeSafe's Jev model then answers one yes/no question per
difference, and the ones it says yes to are counted over the reference words.
Jev never produces a score: it only says which differences matter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx
import jiwer
from pydantic import BaseModel

from coval_bench.metrics.wer import normalize_text

JEV_URL = "https://api.typesafe.ai/v1/systemone"
# Pinned: a floating alias would move scores without a code change.
JEV_MODEL = "jev-1.14.0"

# A difference counts as an error when Jev's probability of "yes" reaches the cutoff.
ERROR_CUTOFF = 0.5
LENIENT_CUTOFF = 0.7
STRICT_CUTOFF = 0.3

_CONTEXT_WORDS = 3
_TIMEOUT_S = 10.0
_ATTEMPTS = 2
_RETRYABLE = frozenset({429, 500, 502, 503, 529})

_INSTRUCTIONS = (
    "`reference` is what a speaker actually said and `transcript` is a speech-to-text system's "
    "output for the same audio. `changes[{index}]` is one place where they differ: "
    "`reference_words` became `transcript_words` (an empty string means nothing was there). "
    "Judge only this one difference, in the context of the whole sentence. Does it change what "
    "the speaker asked for or stated, so that a voice agent acting on the transcript would do "
    "or say something different?"
)
_YES = (
    "The transcript says something different from what was spoken: a different word with a "
    "different meaning (card -> car, left -> right, games -> gangs), a different number or "
    "amount (fifteen -> fifty), a different person or place, a name that became a different "
    "name or was garbled into something that sounds different (Liggins -> Wiggins, Solanki -> "
    "Salanki, Lafasto -> Lefaso), a dropped or added negation, content that is missing or "
    "invented, or words turned into nonsense. A real word replaced by a different word or by a "
    "non-word also counts, even when only a letter or two differs (plans -> plants, inspires -> "
    "requires, hamburger -> hamburg, while -> well), and so does any added or removed 'not'. "
    "If the transcript's version is not a recognizable word at all (earthquakes -> eartsekes, "
    "warranty -> waryty, description -> descrinon), count it: the transcriber did not get the "
    "word, even if a reader could guess it."
)
_NO = (
    "The transcript says the same thing. This covers formatting, punctuation, hyphenation and "
    "compounding (dragon flies -> dragonflies); digits versus number words; contractions; "
    "fillers, stutters and repeated words; real spelling variants that are pronounced the "
    "same, of the same word (flak -> flack) or the same name (Layton -> Leighton, Allison -> "
    "Alison, Karl -> Carl), but not names that sound different; singular versus plural and "
    "possessives (friend -> friends); verb tense or form (had -> has, survived -> survive, "
    "easier -> easy); and dropped, added or swapped articles and small function words (the -> "
    "a, a missing 'the' or 'have') when the sentence still means the same. Also harmless: a "
    "word missing from the very end of a sentence that is already cut off mid-thought; a "
    "restated equivalent that was left out (80 kilometers 50 miles -> 80 kilometers); and a "
    "near-synonym that leaves the request or statement the same (grants insight -> provides "
    "insight)."
)

AskJev = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class Difference(BaseModel):
    """One run of words where the normalized transcript departs from the reference."""

    reference_words: str
    transcript_words: str
    where_in_reference: str
    # What WER charges for this span.
    weight: int


class JudgeJevResult(BaseModel):
    rate: float
    percentage: float
    errors: int
    strict_errors: int
    lenient_errors: int
    expected_errors: float
    differences: int
    reference_words: int


def find_differences(reference: str, hypothesis: str) -> tuple[int, list[Difference]]:
    """Reference word count and the differing spans, adjacent edits merged into one."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref or not hyp:
        words = hyp or ref
        if not words:
            return 0, []
        joined = " ".join(words)
        missing = Difference(
            reference_words=joined if ref else "",
            transcript_words=joined if hyp else "",
            where_in_reference=f"[{joined if ref else ''}]",
            weight=len(words),
        )
        return len(ref), [missing]

    spans: list[list[int]] = []
    current: list[int] | None = None
    for chunk in jiwer.process_words(" ".join(ref), " ".join(hyp)).alignments[0]:
        if chunk.type == "equal":
            current = None
            continue
        size = max(chunk.ref_end_idx - chunk.ref_start_idx, chunk.hyp_end_idx - chunk.hyp_start_idx)
        if current is None:
            current = [chunk.ref_start_idx, chunk.ref_end_idx, chunk.hyp_start_idx, 0, 0]
            spans.append(current)
        current[1], current[3] = chunk.ref_end_idx, chunk.hyp_end_idx
        current[4] += size

    differences = []
    for ref_start, ref_end, hyp_start, hyp_end, weight in spans:
        span = " ".join(ref[ref_start:ref_end])
        before = " ".join(ref[max(0, ref_start - _CONTEXT_WORDS) : ref_start])
        after = " ".join(ref[ref_end : ref_end + _CONTEXT_WORDS])
        differences.append(
            Difference(
                reference_words=span,
                transcript_words=" ".join(hyp[hyp_start:hyp_end]),
                where_in_reference=f"{before} [{span}] {after}".strip(),
                weight=weight,
            )
        )
    return len(ref), differences


def build_request(
    reference: str, hypothesis: str, differences: Sequence[Difference]
) -> dict[str, Any]:
    return {
        "model": JEV_MODEL,
        "state": {
            "reference": reference,
            "transcript": hypothesis,
            "changes": [d.model_dump(exclude={"weight"}) for d in differences],
        },
        "questions": {
            f"c{index}": {
                "type": "noul",
                "instructions": _INSTRUCTIONS.replace("{index}", str(index)),
                "criteria": {"true": _YES, "false": _NO},
            }
            for index in range(len(differences))
        },
    }


def score(
    reference_words: int, differences: Sequence[Difference], probabilities: Sequence[float]
) -> JudgeJevResult:
    """Count the differences Jev called real, each at the word count WER gives it."""

    def errors_at(cutoff: float) -> int:
        return sum(d.weight for d, p in zip(differences, probabilities, strict=True) if p >= cutoff)

    errors = errors_at(ERROR_CUTOFF)
    # An empty reference follows compute_wer: the count stands in for the rate.
    rate = errors / reference_words if reference_words else float(errors)
    return JudgeJevResult(
        rate=rate,
        percentage=rate * 100.0,
        errors=errors,
        strict_errors=errors_at(STRICT_CUTOFF),
        lenient_errors=errors_at(LENIENT_CUTOFF),
        expected_errors=sum(d.weight * p for d, p in zip(differences, probabilities, strict=True)),
        differences=len(differences),
        reference_words=reference_words,
    )


async def compute_judgejev(reference: str, hypothesis: str, ask: AskJev) -> JudgeJevResult:
    """Score one transcript. ``ask`` posts a Jev request and returns the decoded response.

    Jev is only asked when both sides have words and they differ: identical
    text has nothing to judge, and an empty side is wrong in full.
    """
    reference_words, differences = find_differences(reference, hypothesis)
    one_side_empty = not normalize_text(reference) or not normalize_text(hypothesis)
    if not differences or one_side_empty:
        return score(reference_words, differences, [1.0] * len(differences))
    answers = (await ask(build_request(reference, hypothesis, differences)))["answers"]
    probabilities = [float(answers[f"c{index}"]["noul"]) for index in range(len(differences))]
    return score(reference_words, differences, probabilities)


async def ask_jev(body: dict[str, Any], *, api_key: str) -> dict[str, Any]:
    """POST one request to Jev, retrying once on a rate limit or server error."""
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        for attempt in range(_ATTEMPTS):
            response = await client.post(
                JEV_URL, json=body, headers={"Authorization": f"Bearer {api_key}"}
            )
            if response.status_code in _RETRYABLE and attempt + 1 < _ATTEMPTS:
                await asyncio.sleep(1.0)
                continue
            response.raise_for_status()
            decoded: dict[str, Any] = response.json()
            return decoded
    raise RuntimeError("unreachable")
