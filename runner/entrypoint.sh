#!/bin/sh
# Args from a pre-BENCH-762 caller already spell the whole command; run them
# verbatim so the image never depends on benchmark-infra deploying in lockstep.
case "$1" in
  python | coval-bench) exec "$@" ;;
esac
exec python -m coval_bench "$@"
