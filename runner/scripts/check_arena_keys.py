# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Assert arena roster and benchmarks-api key mounts agree, in both directions.

Reads a ``coval-ai/benchmark-infra`` file (passed as a path; the CI workflow
fetches it) and structurally extracts the env-var names mounted on
benchmarks-api from Secret Manager. Two infra layouts are understood:
``envs/prod/provider_keys.tf`` (one map entry per key; ``api = true`` marks
the mount) and the older ``modules/cloud_run_service/main.tf`` (inline env
blocks with ``secret_key_ref``). Fails if any provider the arena would
synthesize lacks its key on the service — the cross-repo parity gate, and the
only failing condition. An ACTIVE provider opted out with
``arena_enabled=False`` while its key IS mounted only warns: the registry owns
arena membership, so a deliberate exclusion must not need an infra change to
keep CI green. Opted-out providers without a ``PROVIDER_ENV`` mapping (e.g.
ADC-authenticated ones) are skipped: there is no key mount to check them
against.

Usage: python scripts/check_arena_keys.py <path-to-infra-tf-file>
"""

from __future__ import annotations

import sys

import hcl2

from coval_bench.arena.pairing import active_tts_models
from coval_bench.registries.benchmarks import Benchmark
from coval_bench.registries.models import MODEL_REGISTRY, ModelStatus
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


def _provider_map_env_names(node: object) -> set[str]:
    """Env-var names of ``provider_keys`` map entries flagged ``api = true``."""
    found: set[str] = set()
    if isinstance(node, dict):
        entries = node.get("provider_keys")
        if isinstance(entries, dict):
            for entry in entries.values():
                if not isinstance(entry, dict):
                    continue
                env = entry.get("env")
                # python-hcl2 preserves literal quotes on string values, and
                # parses bare `true` as bool — normalise both.
                if isinstance(env, str) and str(entry.get("api")).strip('"').lower() == "true":
                    found.add(env.strip('"'))
        for value in node.values():
            found |= _provider_map_env_names(value)
    elif isinstance(node, list):
        for item in node:
            found |= _provider_map_env_names(item)
    return found


def main(tf_path: str) -> int:
    with open(tf_path) as fp:
        tree = hcl2.load(fp)
    mounted = _secret_backed_env_names(tree) | _provider_map_env_names(tree)

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
            "Add the key to envs/prod/provider_keys.tf in coval-ai/benchmark-infra "
            "with api = true, then re-run this check."
        )
        return 1

    mounted_opt_outs = sorted(
        {
            m.provider
            for m in MODEL_REGISTRY
            if m.benchmark is Benchmark.TTS
            and m.status is ModelStatus.ACTIVE
            and not m.arena_enabled
            and PROVIDER_ENV.get(m.provider) in mounted
        }
    )
    if mounted_opt_outs:
        print(
            f"::warning::key mounted on benchmarks-api but arena_enabled=False: "
            f"{mounted_opt_outs}. Fine if the exclusion is deliberate. If it is not, "
            "drop arena_enabled=False in registries/models.py, or unmount the key "
            "to free the slot."
        )

    print(f"OK: all {len(required)} arena provider keys are mounted on benchmarks-api")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
