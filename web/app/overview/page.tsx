// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from "react";
import { OverviewPageClient } from "./overview-page-client";

export default function Page() {
  return (
    <Suspense fallback={null}>
      <OverviewPageClient />
    </Suspense>
  );
}
