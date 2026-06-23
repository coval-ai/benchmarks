# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Example prompts per domain for seeding demo battles.

The arena's real prompts are user-written; these are a small fixture so a labeler
has battles to vote on before any user prompts arrive. Keys are the launch domains
and are stored verbatim as a battle's ``domain``.
"""

from __future__ import annotations

EXAMPLE_PROMPTS: dict[str, list[str]] = {
    "customer service": [
        "Thanks for holding — I've found your order and it'll ship today.",
        "I'm sorry for the trouble; let me refund that charge right now.",
        "Your appointment is confirmed for Tuesday at nine in the morning.",
    ],
    "insurance": [
        "Your claim has been approved and payment will arrive within five business days.",
        "This policy covers water damage but excludes flooding from natural disasters.",
        "To file a claim, I'll need your policy number and the date of the incident.",
    ],
    "healthcare": [
        "Your test results came back normal, so no follow-up is needed.",
        "Please take this medication twice daily with food.",
        "The doctor will see you now; the visit usually takes about fifteen minutes.",
    ],
}
