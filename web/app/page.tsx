// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { redirect } from "next/navigation";

// The query string must survive this hop: shared links carry access tokens
// (?ea=<token>, ?internal=<key>) that unlock early-access models, and a bare
// redirect("/overview") would strip them before the client ever stores them.
export default async function Home({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = new URLSearchParams();
  for (const [key, values] of Object.entries(await searchParams)) {
    if (values === undefined) continue;
    for (const value of Array.isArray(values) ? values : [values]) {
      params.append(key, value);
    }
  }
  const qs = params.toString();
  redirect(qs ? `/overview?${qs}` : "/overview");
}
