# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The mock tool service the benchmarked agents call.

Every orchestration platform under test points its tool definitions at this
service, so a tool call costs the same on all of them and returns the same
answer. That is what makes a tool-using conversation comparable across
platforms: the agent's behaviour varies, the world it acts on does not.

The pieces:

``fixtures``
    The seeded world, loaded from the suite's private ``mock-tools.json``.
``resolver``
    Which seed answers a given call: exact on shared keys, then most keys
    matched, then a fuzzy fallback.
``dispatch``
    Handlers built *from* ``tool-definitions.json``, so adding a tool to the
    contract adds a route, and no tool gets a hand-written one.
``keyterms``
    Domain vocabulary read back out of the fixtures for STT keyterm prompting.

The tool-call log lives at ``coval_bench.db.mock_tool_store``, beside the other
stores, because it is database access rather than mock behaviour.
"""
