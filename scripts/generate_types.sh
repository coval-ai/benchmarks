#!/usr/bin/env bash
# Generate TypeScript types for the benchmarks API from its OpenAPI spec.
# Output is pushed to the generated-types branch by CI for benchmarks-web.
#
# Prerequisites:
#   - uv (runner/ is synced with --frozen)
#   - Node.js 22+ and npm (for openapi-typescript)
#
# Usage:
#   bash scripts/generate_types.sh                  # default output to generated/
#   bash scripts/generate_types.sh ./custom-output   # custom output directory
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_ROOT="${1:-${REPO_ROOT}/generated}"
SPEC_PATH="${OUTPUT_ROOT}/openapi.json"
TYPES_DIR="${OUTPUT_ROOT}/types/api"

# Same versions benchmarks-web pins, so schema.ts is byte-identical on both sides.
OPENAPI_TYPESCRIPT_VERSION=7.13.0
TYPESCRIPT_VERSION=5.9.3

mkdir -p "${TYPES_DIR}"

echo "==> Writing OpenAPI spec from create_app()..."
uv run --frozen --project "${REPO_ROOT}/runner" python - "${SPEC_PATH}" <<'PY'
import json
import sys
import tempfile
from pathlib import Path

from coval_bench.api.app import create_app
from coval_bench.config import Settings

app = create_app(Settings(_env_file=None, arena_audio_dir=Path(tempfile.mkdtemp())))
# Serialized the way Starlette serves /openapi.json, so the sha256 matches the consumer's.
spec = json.dumps(app.openapi(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
Path(sys.argv[1]).write_text(spec, encoding="utf-8")
PY

echo ""
echo "==> Converting OpenAPI spec to TypeScript..."

TYPEGEN_NODE_ROOT="$(mktemp -d)"
cleanup_typegen_node_root() {
  rm -rf "${TYPEGEN_NODE_ROOT}"
}
trap cleanup_typegen_node_root EXIT

npm install \
  --prefix "${TYPEGEN_NODE_ROOT}" \
  --no-save \
  --no-audit \
  --no-fund \
  --silent \
  "openapi-typescript@${OPENAPI_TYPESCRIPT_VERSION}" \
  "typescript@${TYPESCRIPT_VERSION}"

cat > "${TYPEGEN_NODE_ROOT}/generate.mjs" <<'JS'
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import openapiTS, { astToString } from "openapi-typescript";

const [specPath, typesDir] = process.argv.slice(2);
const raw = readFileSync(specPath, "utf8");
const sha = createHash("sha256").update(raw).digest("hex");
const ts = astToString(await openapiTS(JSON.parse(raw)));
writeFileSync(`${typesDir}/schema.ts`, ts, "utf8");
writeFileSync(
  `${typesDir}/.openapi-sha`,
  `source: coval-ai/benchmarks scripts/generate_types.sh\nsha256: ${sha}\n`,
  "utf8",
);
console.log(`  types/api/schema.ts (spec sha256 ${sha.slice(0, 12)})`);
JS

node "${TYPEGEN_NODE_ROOT}/generate.mjs" "${SPEC_PATH}" "${TYPES_DIR}"

echo ""
echo "==> Done! TypeScript types written to ${TYPES_DIR}"
