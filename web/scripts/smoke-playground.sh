#!/usr/bin/env bash
#
# Smoke test for the public playground API surface (/api/playground/{tts,stt}).
#
# Exercises the security + validation ladder against a running dev server using
# FAKE provider keys — no real provider calls succeed, so no spend is incurred.
# The auth/validation checks (403/401/413/400) are deterministic regardless of
# keys; the upstream checks only assert the request reached the provider stage.
#
# Usage:
#   1. In web/.env.local set PLAYGROUND_SESSION_SECRET to any 32+ byte base64
#      value and the PLAYGROUND_*_API_KEY vars to fake values (e.g. "fake-key").
#   2. Run the dev server:  pnpm dev
#   3. In another shell:    bash web/scripts/smoke-playground.sh
#
# Override the target with BASE_URL, e.g.
#   BASE_URL=http://localhost:3001 bash web/scripts/smoke-playground.sh
#
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:3000}"
ORIGIN="${ORIGIN:-$BASE_URL}"
COOKIE_NAME="__playground_session"

JAR="$(mktemp)"
AUDIO_SMALL="$(mktemp)"
AUDIO_BIG="$(mktemp)"
trap 'rm -f "$JAR" "$AUDIO_SMALL" "$AUDIO_BIG"' EXIT

head -c 32000 /dev/zero > "$AUDIO_SMALL"
head -c 2000000 /dev/zero > "$AUDIO_BIG"

PASS=0
FAIL=0

check() {
  local name="$1" expected="$2" actual="$3"
  if [ "$actual" = "$expected" ]; then
    printf '  \033[32mPASS\033[0m  %-44s (%s)\n' "$name" "$actual"
    PASS=$((PASS + 1))
  else
    printf '  \033[31mFAIL\033[0m  %-44s (expected %s, got %s)\n' "$name" "$expected" "$actual"
    FAIL=$((FAIL + 1))
  fi
}

check_in() {
  local name="$1" actual="$2"
  shift 2
  for code in "$@"; do
    if [ "$actual" = "$code" ]; then
      printf '  \033[32mPASS\033[0m  %-44s (%s)\n' "$name" "$actual"
      PASS=$((PASS + 1))
      return
    fi
  done
  printf '  \033[31mFAIL\033[0m  %-44s (expected one of %s, got %s)\n' "$name" "$*" "$actual"
  FAIL=$((FAIL + 1))
}

echo "Target: $BASE_URL (origin: $ORIGIN)"
echo

echo "Preflight: mint session cookie via GET /playground"
page_code="$(curl -s -c "$JAR" -o /dev/null -w '%{http_code}' "$BASE_URL/playground")"
if [ "$page_code" != "200" ] || ! grep -q "$COOKIE_NAME" "$JAR"; then
  echo "  cannot obtain a session cookie (GET /playground -> $page_code)."
  echo "  Is 'pnpm dev' running and PLAYGROUND_SESSION_SECRET set in web/.env.local?"
  exit 1
fi
echo "  cookie acquired"
echo

echo "TTS  POST /api/playground/tts"
code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/playground/tts" \
  -H 'Content-Type: application/json' -d '{"model_id":"cartesia:sonic-3:default","text":"hi"}')"
check "no Origin -> 403 FORBIDDEN" 403 "$code"

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/playground/tts" \
  -H "Origin: $ORIGIN" -H 'Content-Type: application/json' -d '{"model_id":"cartesia:sonic-3:default","text":"hi"}')"
check "valid Origin, no cookie -> 401 UNAUTHORIZED" 401 "$code"

big_text="$(head -c 5000 /dev/zero | tr '\0' 'a')"
code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/playground/tts" \
  -H "Origin: $ORIGIN" -b "$JAR" -H 'Content-Type: application/json' \
  -d "{\"model_id\":\"cartesia:sonic-3:default\",\"text\":\"$big_text\"}")"
check "body > 4096 bytes -> 413 PAYLOAD_TOO_LARGE" 413 "$code"

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/playground/tts" \
  -H "Origin: $ORIGIN" -b "$JAR" -H 'Content-Type: application/json' -d '{"text":"hi"}')"
check "missing model_id -> 400 VALIDATION_ERROR" 400 "$code"

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/playground/tts" \
  -H "Origin: $ORIGIN" -b "$JAR" -H 'Content-Type: application/json' \
  -d '{"model_id":"cartesia:sonic-3:default","text":"hello"}')"
check_in "valid request reaches provider (200 real key / 502 fake)" "$code" 200 502
echo

echo "STT  POST /api/playground/stt"
code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/playground/stt" \
  -F 'modelIds=["deepgram:nova-2"]' -F "audio=@$AUDIO_SMALL")"
check "no Origin -> 403 FORBIDDEN" 403 "$code"

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/playground/stt" \
  -H "Origin: $ORIGIN" -F 'modelIds=["deepgram:nova-2"]' -F "audio=@$AUDIO_SMALL")"
check "valid Origin, no cookie -> 401 UNAUTHORIZED" 401 "$code"

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/playground/stt" \
  -H "Origin: $ORIGIN" -b "$JAR" -F "audio=@$AUDIO_BIG" -F 'modelIds=["deepgram:nova-2"]')"
check "audio > 1.88 MB -> 413 AUDIO_TOO_LARGE" 413 "$code"

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/playground/stt" \
  -H "Origin: $ORIGIN" -b "$JAR" -F "audio=@$AUDIO_SMALL")"
check "missing modelIds -> 400 VALIDATION_ERROR" 400 "$code"

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/playground/stt" \
  -H "Origin: $ORIGIN" -b "$JAR" -F 'modelIds=["bogus:model"]' -F "audio=@$AUDIO_SMALL")"
check "unknown model -> 400 INVALID_MODEL" 400 "$code"

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/playground/stt" \
  -H "Origin: $ORIGIN" -b "$JAR" -F 'modelIds=["deepgram:nova-2"]' -F "audio=@$AUDIO_SMALL")"
check "valid request -> 200 (per-model error rows on fake keys)" 200 "$code"
echo

echo "Result: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
