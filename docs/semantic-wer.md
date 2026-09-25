# Semantic WER (draft library)

This repository provides a deterministic, platform-neutral Semantic WER scorer
as a library. It compares reviewed semantic units against word alignments,
assigning level 1/2/3 units weights 0/1/3. Identifiers use span-aligned,
exact canonical matching; componentized units also check polarity.

## Annotation contract

Annotations mark atomic, meaning-bearing spans in the reference transcript.
Each unit includes its text and character offsets, semantic kind and role,
canonical value, level, and any narrowly reviewed accepted forms. Units must
match the reference text and may not overlap. A logical unit whose meaningful
words are separated may instead declare ordered, non-overlapping components;
it is still scored once.

The level describes the unit's structural role, not a subjective estimate of
real-world harm:

| Level | Name | Assignment test | Weight |
|---|---|---|---:|
| 1 | Surface | Changing it preserves the proposition or requested action. | 0 |
| 2 | Qualifier | Removing it loses a soft preference, optional constraint, metadata, or useful context while preserving the base action and required fields. | 1 |
| 3 | Core | Changing it changes the action, target, truth value, polarity, or a required argument. | 3 |

Apply the tests in order. First ask whether a plausible replacement, removal,
role reversal, or polarity flip would change the requested action, answer,
target, truth value, or a required argument; if so, label the unit core. If not,
ask whether removal loses an explicit soft preference or useful detail; if so,
label it qualifier. Everything else is surface. This counterfactual is review
evidence used to assign the frozen label; it is not evaluated by the runtime
scorer.

Hard constraints such as `only`, `must`, `exactly`, and `do not` normally make
the affected unit core. Soft cues such as `prefer`, `ideally`, `if possible`,
and `about` normally indicate a qualifier. Quantities, units, dates, times,
money, measurements, identifiers, and safety constraints are core unless the
utterance explicitly makes them optional.

Annotation proposals may be produced by a strict-schema model, but a human
must validate the complete transcript, add missing units, and edit or remove
incorrect proposals before benchmark use. The public schema intentionally
omits reviewer identity, timestamps, rationales, source run IDs, audio hashes,
and raw model proposals; private benchmark workflows retain that evidence.

## Scoring contract

The scorer returns a primary pooled error percentage and normalized value rows
for error/reference weights, core/qualifier/surface counts, semantic
successes, and utterances. Per-sample core and qualifier percentages are
diagnostics and must not be averaged as aggregate values.

`SemanticAnnotationSet` is deliberately neutral: it carries schema and
normalizer versions, dataset identity, and public samples. Reviewer identity,
timestamps, notes, source run IDs, audio hashes, and review evidence belong in
private benchmark harnesses.

For each annotated unit, an error contributes its level weight to `error_weight`
and every unit contributes to `reference_weight`; the primary percentage is
`100 * error_weight / reference_weight`. Level 1 surface units have weight 0,
level 2 qualifiers weight 1, and level 3 core units weight 3. Zero denominators
produce `0` rather than NaN. Aggregates pool raw weights and counts across
utterances, including standard WER counts; they never average per-utterance
ratios. Rows use `primary` with unit `percent`, weight operands with unit
`weight`, and all other sufficient statistics with unit `count`; row roles are
`primary` or `component`.

Each unit has one deterministic outcome: `correct`, `substituted`, or
`deleted`. The scorer maps raw character spans to tokens produced by the pinned
WER normalizer, aligns the normalized reference and hypothesis, and compares
the aligned hypothesis span with the normalized reference or an explicit
accepted form. Componentized units require all components and reject an added
polarity token. Identifiers use symbol-preserving canonicalization for spoken
letter names and spoken or literal dashes and stars, then require the exact
canonical identifier. General synonym matching and embedding similarity are
not used.

`SemanticWERResult.normalized_value_rows()` emits one observation's values;
`SemanticWERAggregate.normalized_value_rows()` emits the same keys after raw
weights and counts have been pooled across observations. Both use this shape:

```text
metric, metric_version, value_key, unit, value, value_role
```

The value keys are `primary`, `error_weight`, `reference_weight`,
`core_errors`, `core_units`, `qualifier_errors`, `qualifier_units`,
`surface_errors`, `surface_units`, `semantic_successes`, and `utterances`.

The scorer is deterministic but depends on the pinned WER normalizer and word
alignment. It does not infer hypothesis-only semantic insertions, and
componentized annotations must have non-overlapping spans. Identifier matching
canonicalizes spoken punctuation and letter names, then requires an exact
span-aligned value. Attached characters and adjacent numeric, symbol, or
single-letter identifier tokens remain errors. Other adjacent inserted words
stay outside the semantic span and are measured by standard WER; the draft
does not infer whether an arbitrary multi-letter word was intended as another
identifier segment.

Library availability is not metric activation. This draft does not register a
Metric enum, runner/orchestrator job, scheduler, database writer or migration,
API route, dashboard display, or Terraform resource. No production benchmark
uses Semantic WER until those paths are separately reviewed and enabled.

Examples and tests use invented text and identifiers.
