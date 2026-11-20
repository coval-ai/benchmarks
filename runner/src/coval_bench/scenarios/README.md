# Contracts

What every variant of the voice orchestration benchmark is given, and the one
place a pinned value is written down.

## Why this lives inside the package

`Dockerfile.api` builds from `runner/` and copies only `pyproject.toml`,
`uv.lock`, `README.md`, `src/` and `alembic.ini`. The API loads each active
mock-tool suite's tools and fixtures at startup. Committed contract files ship
in the wheel and are read with `importlib.resources`, just like dataset manifests.

## Layout

```
scenarios/
  stack.json          the pinned component layer, identical for every variant
  personas.json       the caller personas: label, description, order, dataset id
  annotated.py        the shared contract-model base
  bank/
    system-prompt.txt      the Ultra Bank agent, authored here and served by the
    tool-definitions.json  LLM proxy; the test set and judge stay in Coval
  id_roulette/
    _private/              tools.json: Corral's tools and answers, published from the
                           assets in benchmark-infra, never committed
  platforms/
    <platform>-agent.json  the only permitted difference, published for audit
```

## The two rules

1. **The pinned stack and the suite contract are shared across variants.** A
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

`contract_sha256("id_roulette")` is one SHA-256 over `stack.json` plus the suite
files, in a fixed order. It is published beside every result, so a number always
points at an exact contract. It covers `stack.json` because changing a pin
changes what the agent is just as surely as changing its prompt.

Missing files are skipped, so the hash is meaningful before the contract is
complete and changes when a file is added.

## What is published, and what is not

The agent definition is public; the evaluator is not.

| File | Published | Why |
|---|---|---|
| `system-prompt.txt`, `first-message.txt`, `tool-definitions.json` | yes | The agent under test. A reader must be able to audit what the agent was told in order to trust a result |
| `_private/` | **no, gitignored as a directory** | The evaluator. `input_str` is the caller's script, `expected_behaviors` are the assertions, and `mock-tools.json` encodes the expected availability and patient identities. Publishing them would let a platform optimise for the test rather than the task |
