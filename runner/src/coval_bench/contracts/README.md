# Contracts

What every variant of the voice orchestration benchmark is given, and the one
place a pinned value is written down.

## Why this lives inside the package

`Dockerfile.api` builds from `runner/` and copies only `pyproject.toml`,
`uv.lock`, `README.md`, `src/` and `alembic.ini`. The mock-tool router runs
inside that container and reads `tool-definitions.json` and the fixtures at
request time, so a contract at the repo root would not exist when it is needed.
Everything here ships in the wheel and is read with `importlib.resources`, the
same way `datasets/manifests` is.

## Layout

```
contracts/
  stack.json          the pinned component layer, identical for every variant
  dental/
    system-prompt.txt      \  the agent under test. byte-identical across
    first-message.txt       |  variants, covered by the contract hash
    tool-definitions.json  /
    _source/               pulled from the platform, never edited by hand
    _private/              the evaluator. hashed, never committed
  platforms/
    <platform>-agent.json  the only permitted difference, published for audit
```

## The two rules

1. **`stack.json` and the public `dental/` files are byte-identical across every variant.** A
   platform config may not restate a pinned value; it references this. If two
   platforms could drift apart, the comparison is void.
2. **`platforms/` holds the declared difference.** Each file records how one
   vendor was wired, so a reader can audit the tested configuration rather than
   infer it from results.

Transport is in neither. It is declared per variant on the registry row, because
it is the one layer not pinned to a single value.

## Rationale keys

JSON has no comments, so rationale sits beside the value it explains under keys
starting with `_`:

```json
"llm": {
  "provider": "openai",
  "model": "gpt-4.1",
  "temperature": 0,
  "_why": "Pinnable on every platform under test, so model choice cannot explain a difference."
}
```

Models accept `_`-prefixed extras and reject anything else, so a mistyped key
still fails loudly:

```
{"modle": "gpt-4.1"}  ->  ValidationError: field 'model' missing
```

That is the failure worth catching. A mistyped *value* is caught later, when the
vendor rejects it at apply time. A mistyped *key* would silently drop the pin and
leave the platform on its own default, which is invisible in the results.

`stack_as_dict()` strips these before anything is sent to a vendor.

## The contract hash

`contract_sha256("dental")` is one SHA-256 over `stack.json` plus the suite
files, in a fixed order. It is published beside every result, so a number always
points at an exact contract. It covers `stack.json` because changing a pin
changes what the agent is just as surely as changing its prompt.

Missing files are skipped, so the hash is meaningful before the contract is
complete and changes when a file is added.

## `_source/`

Raw dumps from `coval-bench pull-contract`. Committed on purpose: re-running the
pull produces a diff, and a diff means a vendor changed something underneath us.
That is drift detection for free.

Promoting a `_source/` file into a contract file is a deliberate human step, so a
re-pull can never silently rewrite a contract that a published run already used.

### What is published, and what is not

The agent definition is public; the evaluator is not.

| File | Published | Why |
|---|---|---|
| `system-prompt.txt`, `first-message.txt`, `tool-definitions.json` | yes | The agent under test. A reader must be able to audit what the agent was told in order to trust a result |
| `_source/coval-agent.json`, `coval-prompt.txt`, `test-set.json` | yes, identifiers withheld | Provenance and coverage-level methodology. No scenario specifics: no patient names, dates or availability |
| `_private/` | **no, gitignored as a directory** | The evaluator. `input_str` is the caller's script, `expected_behaviors` are the assertions, and `mock-tools.json` encodes the expected availability and patient identities. Publishing them would let a platform optimise for the test rather than the task |

Identifiers are withheld because this repo is public. `redact_identifiers()`
replaces `id`, `name`, `customer_agent_id`, `phone_number` and `endpoint` with
`[REDACTED]` as the dump is written, so a re-pull cannot reintroduce them. The
dial target is the reason it matters: it is a live endpoint, and publishing it
lets anyone call the assistant. `display_name` and `voice_id` are deliberately
kept, being methodology rather than identity.

The contract hash is unaffected: `contract_sha256` covers `stack.json` and the suite files, never `_source/`.

Two things to know about what is in there:

- `coval-prompt.txt` is Coval's `prompt` field, which for a `MODEL_TYPE_VOICE`
  agent holds the **benchmark specification**, not the prompt the agent runs. The
  running prompt lives with the vendor and arrives as `vapi-prompt.txt`.
- The scan before committing found no credential-shaped fields, all emails on
  `@example.com`, and every phone number in the 555 block except
  `602-556-0182`, which is the deliberate confusable for IR03 (phone digit
  repair) and is fictional.

Platform configs are redacted on write: any key matching
`api_key|secret|token|password|bearer|authorization` becomes `[REDACTED]` before
the file is created, and the puller prints what it redacted. `credentialIds` is
deliberately not matched, being an account-scoped reference that `apply` needs
and that carries no key material.
