// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from "react";
import { S2SDashboard } from "./s2s-dashboard";

export default function Page() {
  return (
    <Suspense fallback={null}>
      <S2SDashboard />
    </Suspense>
  );
}
