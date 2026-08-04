// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { BarDataPoint } from "../../../types/benchmark.types";
import CustomBarTooltip from "./BarTooltip";

const point: BarDataPoint = {
  model: "soniox/stt-rt-v5",
  averageWER: 6.6,
  provider: "Soniox",
  breakdown: { substitutions: 4.1, deletions: 1.5, insertions: 1.0 },
};

const render = (p: BarDataPoint) =>
  renderToStaticMarkup(
    <CustomBarTooltip
      active
      payload={[{ dataKey: "averageWER", value: p.averageWER, payload: p }] as never}
      label={p.model}
    />
  );

describe("CustomBarTooltip", () => {
  it("lists each error type under the total", () => {
    const html = render(point);
    expect(html).toContain("Average WER: 6.6%");
    expect(html).toContain("Substitutions: 4.1%");
    expect(html).toContain("Deletions: 1.5%");
    expect(html).toContain("Insertions: 1.0%");
  });

  it("falls back to the total alone when the API sends no split", () => {
    const html = render({ ...point, breakdown: undefined });
    expect(html).toContain("Average WER: 6.6%");
    expect(html).not.toContain("Substitutions");
    expect(html).not.toContain("Deletions");
    expect(html).not.toContain("Insertions");
  });
});
