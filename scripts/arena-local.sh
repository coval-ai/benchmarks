#!/usr/bin/env bash
# Local Voice Arena stack: Postgres + API (docker compose) + Next.js web (pnpm).
# Run from anywhere inside the worktree.
#
#   ./arena-local.sh up            # db + migrate + api on :8000
#   ./arena-local.sh web           # ensure web env, then pnpm dev on :3000 (foreground)
#   ./arena-local.sh all           # up -> snapshot -> web  (one shot; the normal way to start)
#   ./arena-local.sh snapshot      # compute a leaderboard ratings snapshot (needs votes)
#   ./arena-local.sh status        # health + battle/vote counts
#   ./arena-local.sh down          # stop the docker stack (keeps the pgdata volume)
#
# The voting UI generates battles on demand from your prompts. Env files are only
# appended to, never overwritten; if a key already exists with a different value
# the script stops and asks you to reconcile.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

ENV_FILE="$ROOT/.env"
WEB_ENV="$ROOT/web/.env.local"
API_URL="http://localhost:8000"

log()  { printf '\033[36m▸ %s\033[0m\n' "$*" >&2; }
warn() { printf '\033[33m! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# read KEY's non-empty value from a dotenv file; empty if absent or set-but-empty
env_get() { [ -f "$1" ] && sed -n "s/^$2=\(..*\)/\1/p" "$1" | head -1 || true; }

# ensure KEY=VALUE in a file. Absent -> append. Present-but-empty -> fill in place
# (avoids duplicate lines and the API/web value drift they cause). Present with a
# different non-empty value -> stop and let the user reconcile.
env_ensure() {
  local file="$1" key="$2" val="$3" existing tmp
  [ -f "$file" ] || { mkdir -p "$(dirname "$file")"; touch "$file"; }
  existing="$(env_get "$file" "$key")"
  if [ -n "$existing" ]; then
    [ "$existing" = "$val" ] || die "$key already set to a different value in ${file#"$ROOT"/}; reconcile it by hand."
    return 0
  fi
  if grep -q "^$key=" "$file"; then
    tmp="$(mktemp)"; sed "s|^$key=.*|$key=$val|" "$file" > "$tmp" && mv "$tmp" "$file"
  else
    printf '%s=%s\n' "$key" "$val" >> "$file"
  fi
  log "set $key in ${file#"$ROOT"/}"
}

# Resolve one shared labeler key: prefer an existing value in either file, else generate.
resolve_labeler_key() {
  local k
  k="$(env_get "$ENV_FILE" ARENA_LABELER_KEY)"
  [ -z "$k" ] && k="$(env_get "$WEB_ENV" ARENA_LABELER_KEY)"
  [ -z "$k" ] && k="local-$(openssl rand -hex 8 2>/dev/null || echo devkey$RANDOM)"
  printf '%s' "$k"
}

ensure_backend_env() {
  if [ ! -f "$ENV_FILE" ] && [ -f "$ROOT/.env.example" ]; then
    cp "$ROOT/.env.example" "$ENV_FILE"
    log "created .env from .env.example — add >=2 TTS provider keys before generating battles"
  fi
  local key; key="$(resolve_labeler_key)"
  env_ensure "$ENV_FILE" ARENA_LABELER_KEY "$key"
  # Writable clip dir (the app default 'arena-audio' is relative -> non-root
  # container user can't create it under /app) and an absolute base URL so the
  # browser (on :3000) can fetch clips the API serves on :8000.
  env_ensure "$ENV_FILE" ARENA_AUDIO_DIR /tmp/arena-audio
  env_ensure "$ENV_FILE" ARENA_AUDIO_BASE_URL "$API_URL"
}

ensure_web_env() {
  local key; key="$(resolve_labeler_key)"
  env_ensure "$WEB_ENV" NEXT_PUBLIC_ARENA_SOURCE api
  env_ensure "$WEB_ENV" ARENA_API_URL "$API_URL"
  env_ensure "$WEB_ENV" ARENA_LABELER_KEY "$key"
}

wait_for_api() {
  log "waiting for API on $API_URL/healthz ..."
  for _ in $(seq 1 30); do
    if curl -fsS "$API_URL/healthz" >/dev/null 2>&1; then log "API is up"; return 0; fi
    sleep 1
  done
  die "API did not come up — check: docker compose logs api"
}

cmd_up() {
  command -v docker >/dev/null || die "docker not found"
  ensure_backend_env
  log "starting Postgres"
  docker compose up -d db
  log "running migrations (alembic upgrade head)"
  docker compose run --rm migrate
  log "starting API"
  docker compose up -d api
  wait_for_api
}

cmd_snapshot() { docker compose run --rm runner coval-bench arena snapshot; }

cmd_status() {
  # Match how compose resolves these: shell env wins, then .env, then default.
  local pg_user pg_db
  pg_user="${POSTGRES_USER:-$(env_get "$ENV_FILE" POSTGRES_USER)}"; pg_user="${pg_user:-postgres}"
  pg_db="${POSTGRES_DB:-$(env_get "$ENV_FILE" POSTGRES_DB)}"; pg_db="${pg_db:-benchmarks}"
  if curl -fsS "$API_URL/healthz" >/dev/null 2>&1; then log "API: up"; else warn "API: down"; fi
  docker compose exec -T db psql -U "$pg_user" -d "$pg_db" \
    -c "SELECT (SELECT count(*) FROM arena.battles) AS battles, (SELECT count(*) FROM arena.votes) AS votes;" \
    2>/dev/null || warn "could not query db (is it up? are migrations applied?)"
}

cmd_web() {
  command -v pnpm >/dev/null || die "pnpm not found"
  ensure_web_env
  cd "$ROOT/web"
  log "syncing web dependencies"
  pnpm install
  log "starting Next.js dev — open http://localhost:3000/arena"
  pnpm dev
}

cmd_down() { docker compose down; }

case "${1:-}" in
  up)       cmd_up ;;
  snapshot) cmd_snapshot ;;
  status)   cmd_status ;;
  web)      cmd_web ;;
  down)     cmd_down ;;
  all)      cmd_up; cmd_snapshot || warn "snapshot skipped (no votes yet)"; cmd_web ;;
  *)        sed -n '2,14p' "$0"; exit 2 ;;
esac
