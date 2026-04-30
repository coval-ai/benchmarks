import { writeFile, mkdir } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import openapiTS, { astToString } from "openapi-typescript";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const OUT_PATH = resolve(__dirname, "../lib/api/generated/schema.ts");

async function main() {
  const url = new URL("/openapi.json", API_URL);
  console.warn(`Fetching ${url}...`);
  const ast = await openapiTS(url);
  const ts = astToString(ast);
  await mkdir(dirname(OUT_PATH), { recursive: true });
  await writeFile(OUT_PATH, ts, "utf8");
  console.warn(`Wrote ${OUT_PATH}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
