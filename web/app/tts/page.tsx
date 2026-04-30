// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from "react";
import { TTSDashboard } from "./tts-dashboard";

export default function Page() {
  return (
    <Suspense fallback={null}>
      <TTSDashboard />
    </Suspense>
  );
}
