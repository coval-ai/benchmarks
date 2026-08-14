# Supported storage operations

The normalized storage layer supports explicit domain operations rather than
generic CRUD.

| Operation | Behavior |
|---|---|
| Create observation | Store an immutable benchmark observation with its run, dataset sample, provider, model, benchmark, capture outcome, and available audio or transport metadata. |
| Attach preprocessing artifact | Attach an immutable, versioned word- or phoneme-timestamp output with explicit producer name, provider, model, and version. |
| Queue metric evaluation | Create a named metric-evaluation variant in the `queued` state, atomically freezing its ordered, role-aware preprocessing inputs. |
| Start metric evaluation | Transition a queued evaluation to `running`. |
| Complete metric evaluation | Atomically mark an evaluation `succeeded` and store its metric values and output artifacts. |
| Fail metric evaluation | Mark an evaluation `failed` and store the failure information. |
| Replay completion | Accept an exact retry of an already-succeeded evaluation without duplicating values or artifacts. Reject a retry whose contents differ. |
| Refresh bucket rollups | Recompute normalized aggregate values from succeeded metric evaluations. |
| Read bucket rollups | Query aggregate values by bucket time, provider, model, benchmark, dataset, metric, metric version, evaluation variant, value key, and unit. |

Evaluation lifecycle:

```text
queued → running
queued → failed
running → succeeded
running → failed
```

Observations, preprocessing artifacts, evaluation inputs, metric values, and metric artifacts are
immutable after insertion. Evaluation inputs are linked only while an evaluation is queued, must
belong to the same observation, and retain the exact lineage used by replayed completion. Succeeded and failed evaluations are terminal.
Completing an evaluation and inserting its values and artifacts happens in one
transaction, so partial results cannot be persisted.
