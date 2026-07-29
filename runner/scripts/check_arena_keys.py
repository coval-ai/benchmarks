# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Assert arena roster and benchmarks-api key mounts agree, in both directions.

Reads ``envs/prod/provider_keys.tf`` from ``coval-ai/benchmark-infra`` (passed
as a path; the CI workflow fetches it) — the single place provider keys are
declared — and extracts the env-var names mounted on the benchmarks-api
service (entries with ``api = true``). Fails if any provider the arena would
synthesize lacks its key on the service (the cross-repo parity gate), and also
fails when an ACTIVE provider is opted out with ``arena_enabled=False`` even
though its key IS mounted — a stale opt-out that silently keeps the provider
off the arena. Opted-out providers without a ``PROVIDER_ENV`` mapping (e.g.
ADC-authenticated ones) are skipped: there is no key mount to check them
against.

Usage: python scripts/check_arena_keys.py <path-to-envs/prod/provider_keys.tf>
"""

from __future__ import annotations

import sys

import hcl2

from coval_bench.arena.pairing import active_tts_models
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import MODEL_REGISTRY, ModelStatus
from coval_bench.registries.provider_keys import PROVIDER_ENV


def _api_mounted_env_names(node: object) -> set[str]:
    """Env-var names of ``provider_keys`` entries with ``api = true``, found anywhere."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "provider_keys" and isinstance(value, dict):
                for entry in value.values():
                    env = entry.get("env") if isinstance(entry, dict) else None
                    if isinstance(env, str) and entry.get("api") is True:
                        # python-hcl2 preserves literal quotes on string values,
                        # e.g. '"OPENAI_API_KEY"'.
                        found.add(env.strip('"'))
            else:
                found |= _api_mounted_env_names(value)
    elif isinstance(node, list):
        for item in node:
            found |= _api_mounted_env_names(item)
    return found


def main(tf_path: str) -> int:
    with open(tf_path) as fp:
        tree = hcl2.load(fp)
    mounted = _api_mounted_env_names(tree)

    if not mounted:
        print(f"ERROR: no api = true provider_keys entries found in {tf_path}")
        return 1

    required: set[str] = set()
    unmapped: set[str] = set()
    for m in active_tts_models():
        env_var = PROVIDER_ENV.get(m.provider)
        if env_var is None:
            unmapped.add(m.provider)
        else:
            required.add(env_var)

    if unmapped:
        print(f"ERROR: no PROVIDER_ENV entry for arena providers: {sorted(unmapped)}")
        return 1

    if not required:
        print("ERROR: arena roster is empty — no ACTIVE, arena_enabled TTS providers to verify.")
        return 1

    missing = sorted(required - mounted)
    if missing:
        print(
            f"ERROR: arena-eligible but NOT mounted on benchmarks-api: {missing}\n"
            "Declare them in coval-ai/benchmark-infra envs/prod/provider_keys.tf "
            "with api = true, then re-run this check."
        )
        return 1

    stale_opt_outs = sorted(
        {
            m.provider
            for m in MODEL_REGISTRY
            if m.benchmark is Benchmark.TTS
            and m.status is ModelStatus.ACTIVE
            and not m.arena_enabled
            and PROVIDER_ENV.get(m.provider) in mounted
        }
    )
    if stale_opt_outs:
        print(
            f"ERROR: key mounted on benchmarks-api but arena_enabled=False: {stale_opt_outs}\n"
            "The opt-out is stale — remove arena_enabled=False from these providers' "
            "ACTIVE models in registries/models.py (or unmount the key if the "
            "exclusion is intentional)."
        )
        return 1

    print(f"OK: all {len(required)} arena provider keys are mounted on benchmarks-api")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
