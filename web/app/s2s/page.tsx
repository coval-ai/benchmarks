// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { notFound } from "next/navigation";
import { Suspense } from "react";
import { S2SDashboard } from "./s2s-dashboard";

export default function Page() {
  // Gated rollout: /s2s is unreachable until NEXT_PUBLIC_S2S_ENABLED is set.
  // It is intentionally left out of the nav and Overview until launch.
  if (process.env.NEXT_PUBLIC_S2S_ENABLED !== "true") {
    notFound();
  }
  return (
    <Suspense fallback={null}>
      <S2SDashboard />
    </Suspense>
  );
}
