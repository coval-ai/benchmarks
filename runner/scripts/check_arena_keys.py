# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Assert every arena-eligible TTS provider's API key is mounted on benchmarks-api.

Reads the ``coval-ai/benchmark-infra`` Cloud Run service module (passed as a
path; the CI workflow fetches it) and structurally extracts the env-var names it
mounts from Secret Manager. Fails if any provider the arena would synthesize
lacks its key on the service — the cross-repo parity gate.

Usage: python scripts/check_arena_keys.py <path-to-cloud_run_service/main.tf>
"""

from __future__ import annotations

import sys

import hcl2

from coval_bench.arena.pairing import active_tts_models
from coval_bench.registries.provider_keys import PROVIDER_ENV


def _has_secret_ref(node: object) -> bool:
    if isinstance(node, dict):
        if "secret_key_ref" in node:
            return True
        return any(_has_secret_ref(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_secret_ref(item) for item in node)
    return False


def _secret_backed_env_names(node: object) -> set[str]:
    """Env-var names bound to a Secret Manager secret, found anywhere in the tree."""
    found: set[str] = set()
    if isinstance(node, dict):
        name = node.get("name")
        if isinstance(name, str) and _has_secret_ref(node.get("value_source")):
            # python-hcl2 preserves literal quotes on string values, e.g. '"OPENAI_API_KEY"'.
            found.add(name.strip('"'))
        for value in node.values():
            found |= _secret_backed_env_names(value)
    elif isinstance(node, list):
        for item in node:
            found |= _secret_backed_env_names(item)
    return found


def main(tf_path: str) -> int:
    with open(tf_path) as fp:
        tree = hcl2.load(fp)
    mounted = _secret_backed_env_names(tree)

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
            "Mount them in coval-ai/benchmark-infra "
            "(modules/cloud_run_service/main.tf env{} block + "
            "envs/prod/cloud_run_service.tf secret_ids + the secret_manager secret), "
            "then re-run this check."
        )
        return 1

    print(f"OK: all {len(required)} arena provider keys are mounted on benchmarks-api")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
