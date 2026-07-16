export const runtime = "nodejs";

import { arenaAccessOk } from "@/lib/arena/guard";
import { arenaRunnerFetch } from "@/lib/arena/runner";
import type { ExamplePrompt } from "@/lib/arena/types";

interface ExamplePromptOut {
  prompt: string;
  domain: ExamplePrompt["domain"];
}

export async function GET(req: Request) {
  if (!(await arenaAccessOk())) return new Response(null, { status: 404 });
  let res: Response;
  try {
    res = await arenaRunnerFetch(
      "/v1/arena/example-prompt",
      undefined,
      req.headers.get("x-forwarded-for"),
    );
  } catch {
    return Response.json({ error: "Example fetch failed." }, { status: 502 });
  }
  if (!res.ok) return Response.json({ error: "Example fetch failed." }, { status: res.status });
  const p = (await res.json()) as ExamplePromptOut;
  const example: ExamplePrompt = { text: p.prompt, domain: p.domain };
  return Response.json(example);
}
