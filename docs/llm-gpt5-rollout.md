# GPT-5 reasoning comparison

These are two configurations of the same upstream model:

| Registry provider | Registry model | Upstream model | reasoning_effort |
| --- | --- | --- | --- |
| openai | gpt-5-minimal | gpt-5 | minimal |
| openai | gpt-5-medium | gpt-5 | medium |

Both use the shared Ultra Bank prompt, tools, test set, persona, and instruction
judge. This measures the LLM component of a cascade; speech recognition and
speech synthesis are outside this benchmark. GPT-5 supports both settings in
the [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-5).

## Deployment and registration

1. Deploy the API and `llm-sync` runner with model-specific routing and ingestion
   before registering either variant. The previous provider-only implementation
   cannot safely distinguish multiple OpenAI entries.
2. Using the [admin authentication instructions](../runner/README.md#api-auth),
   POST each variant separately to `/v1/admin/models`. Use the following payload,
   replacing `model` with `gpt-5-medium` for the second entry:

   ```json
   {
     "modality": "LLM",
     "provider": "openai",
     "model": "gpt-5-minimal",
     "creator": "openai",
     "source": "official-api",
     "licensing": "proprietary",
     "region": "us",
     "arena_enabled": false,
     "collected": true,
     "published": false
   }
   ```

3. Execute `llm-sync`. It creates a distinct Coval agent, run template, and
   schedule for each variant. Newly created schedules defer ingestion until the
   next sync. Existing GPT-4.1 and Gemini agent identities remain stable.
4. Launch fresh runs from both new Ultra Bank run templates. Once finished,
   execute `llm-sync` again to ingest the results.
5. Verify each variant has its own run and model label in normalized records
   and aggregate results, including instruction score and TTFT. Keep entries
   unpublished until the comparison has been reviewed.

Agent endpoints select the registry model with `?benchmark_model=gpt-5-minimal`
or `?benchmark_model=gpt-5-medium`. The request body's `model` field remains the
Coval session identifier. The provider-only OpenAI endpoint still selects
GPT-4.1. No database migration is required.

## Rollback

Set both new registry entries to `collected=false` and run the compatible sync
to disable their schedules before rolling back the API or runner to a
provider-only version. Preserve the variant names on historical results.
