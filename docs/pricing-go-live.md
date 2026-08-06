# Pricing go-live runbook (BENCH-631 epic)

Merge order: the six branches are stacked and must merge in ticket order
(632 → 633 → 634 → 635 → 636 → 637); each PR contains only its own commit.

## Deploy sequence

Prod auto-deploys main on a frequent weekday cron, so steps 1–2 run
**before merging** — both are purely additive and invisible to the currently
deployed code, which removes the merge→deploy race entirely.

1. **Migrate prod BEFORE merging** — migrations `0015`–`0018` must be in
   place before the new runner/API images go live: the new writer's INSERT
   names the new usage columns and `finish_run` names the judge/cost columns,
   so old-schema + new-code fails every run. (Old code + new schema is fine —
   all four migrations are plain `CREATE TABLE`/`ADD COLUMN`, no matview
   drop/recreate, no lock-order hazard, no backfill.) Apply via
   cloud-sql-proxy the usual way, from the 637 branch checkout:
   `coval-bench db migrate` against the proxied DSN.
2. **Seed the ratesheet (also pre-merge)** — `uv run python
   scripts/seed_pricing.py` against the proxied DSN (the local-host guard
   passes through the proxy; the script is idempotent and only ever
   supersedes). Expect 69 rows inserted and a 6-model gap list (baseten ×2
   dedicated, alibaba console-only, deepdub enterprise, fluxions + deepgram
   flux TTS unreleased) — those are known and deliberate;
   `scripts/check_pricing_coverage.py` will keep flagging them until they're
   priced or paused.
3. **Merge the six PRs in order** — the scheduled deploy (or a manual
   ci-cd-switchboard "Deploy Service" run) rolls out runner + API. Then
   verify `GET /v1/pricing?benchmark=STT|TTS|S2S` returns entries with
   `as_of` + `source_url` and that early-access models (murf falcon-2, xai
   grok-voice, modulate english-v2) are absent without an EA token.
4. **Web deploys AFTER the API is live** — `vercel-build` runs `pnpm codegen`
   against the live API's `/openapi.json` at build time, and the new
   components reference the pricing types, so a web build that runs before
   the new API is live fails its typecheck (safe: the site stays on the
   previous version — just redeploy once step 3 is verified). At runtime the
   ordering is free either way: against an API without `/v1/pricing` every
   pricing surface hides itself (verified live).

## Day-one expectations

- Duration- and character-billed models show normalized prices immediately
  after the seed.
- **Token-billed models (openai gpt-4o-transcribe/mini) show "—" until the
  runner has captured ≥50 usage samples in the trailing 7 days** — the
  measured conversion needs BENCH-632's usage columns populated. At the
  30-minute cadence that's within the first day. gpt-4o-mini-tts and soniox
  tts-rt-v1 stay "—" longer: their providers don't report token usage yet.
- `runs.total_cost_usd` starts populating on the first post-deploy run;
  internal spend queries are in `docs/cost-tracking.md`.

## Infra follow-ups (benchmark-infra, not this repo)

- New weekly Cloud Run Job: same image, command `coval-bench update-prices`;
  cadence mirrored by `PRICING_UPDATE_PERIOD_SECONDS` (default 604800).
- Secrets on that job: `OPENAI_API_KEY` (extraction), `LINEAR_API_KEY`
  (review items; Slack webhook `PRICING_REVIEW_SLACK_WEBHOOK` is the
  fallback, log-only without either).
- Confirm `PRICING_EXTRACTION_MODEL` (default `gpt-5-mini`) is a valid model
  id on the account before the first scheduled run: `coval-bench
  update-prices --dry-run --provider deepgram` exercises the whole path
  without writing.
- Optional: wire `scripts/check_pricing_coverage.py` into CI once the six
  known gaps are priced or the models paused (it exits non-zero today by
  design).

## Data provenance

Every seeded rate was verified against the provider's public pricing page on
2026-08-06 (see each row's `source_url`/`evidence`); deepgram, elevenlabs,
openai, and assemblyai were independently re-verified against the live pages
before this runbook was written. Plan-based rates (cartesia, hume, gradium,
lmnt, speechify, gladia) carry their plan assumption on the row and in the
price tooltip.
